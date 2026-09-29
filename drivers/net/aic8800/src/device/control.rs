use alloc::{boxed::Box, collections::VecDeque, vec, vec::Vec};

use super::{AicError, ControlRequest, Entropy, Pmk};
use crate::{
    lmac::{
        ME_CHAN_CONFIG_CFM, ME_CHAN_CONFIG_REQ, ME_CONFIG_CFM, ME_CONFIG_REQ,
        ME_SET_CONTROL_PORT_CFM, ME_SET_CONTROL_PORT_REQ, ME_STA_ADD_CFM, ME_STA_ADD_REQ,
        MM_ADD_IF_CFM, MM_ADD_IF_REQ, MM_KEY_ADD_CFM, MM_KEY_ADD_REQ, MM_RESET_CFM, MM_RESET_REQ,
        SM_CONNECT_CFM, SM_CONNECT_REQ, SM_DISCONNECT_CFM, SM_DISCONNECT_REQ, TASK_ME, TASK_MM,
        TASK_SM, add_interface_payload, channel_config_payload, connect_payload,
        control_port_payload, disconnect_payload, key_add_payload, me_config_payload,
        me_sta_add_payload,
    },
    wpa2::{HandshakeAction, Wpa2Handshake, WpaError},
};

const MAX_SSID_LENGTH: usize = 32;
const CONTROL_COMMAND_CAPACITY: usize = 8;
const CONTROL_COMMAND_BYTE_CAPACITY: usize = 4 * 1024;

pub(super) struct ControlCommand {
    pub message_id: u16,
    pub destination: u16,
    pub expected_message_id: u16,
    pub payload: Vec<u8>,
}

pub(super) struct ControlState {
    pub commands: VecDeque<ControlCommand>,
    pub command_bytes: usize,
    pub operation: ControlOperation,
}

pub(super) enum ControlOperation {
    Commands,
    Connect(Box<PendingConnect>),
    Disconnect,
    /// Internal command queue of a running access point, used to register
    /// associated clients with the firmware via `ME_STA_ADD`.  It carries no
    /// waiting WEXT caller, so its completion must not raise `ControlComplete`.
    ApStationAdd,
}

pub(super) struct PendingConnect {
    pub pmk: Option<Pmk>,
    pub entropy: Option<Entropy>,
    pub phase: ConnectPhase,
    handshake: Option<Wpa2Handshake>,
    pending_m4: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ConnectPhase {
    /// Clear a possible association left in firmware across a warm host reset.
    /// Linux keeps this state in cfg80211; the bare-metal owner must establish
    /// the same invariant explicitly before issuing a new connect request.
    Resetting,
    AwaitConfirmation,
    AwaitIndication,
    AwaitHandshake,
    InstallingKeys(u8),
    AwaitM4Transmit,
    AwaitControlPort,
}

pub(super) enum ControlEffect {
    None,
    TransmitEapol(Vec<u8>),
}

impl ControlState {
    fn has_command_capacity(&self, additional_items: usize, additional_bytes: usize) -> bool {
        self.commands
            .len()
            .checked_add(additional_items)
            .is_some_and(|items| items <= CONTROL_COMMAND_CAPACITY)
            && self
                .command_bytes
                .checked_add(additional_bytes)
                .is_some_and(|bytes| bytes <= CONTROL_COMMAND_BYTE_CAPACITY)
    }

    fn push_command(&mut self, command: ControlCommand) -> Result<(), AicError> {
        let command_bytes = command.payload.len();
        if !self.has_command_capacity(1, command_bytes) {
            return Err(AicError::ControlQueueFull);
        }
        self.command_bytes += command_bytes;
        self.commands.push_back(command);
        Ok(())
    }

    /// Whether this control state is the access point's internal station-add
    /// queue rather than a caller-driven connect/disconnect/scan operation.
    pub(super) fn is_ap_station_add(&self) -> bool {
        matches!(self.operation, ControlOperation::ApStationAdd)
    }

    pub(super) fn push_station_add(&mut self, command: ControlCommand) -> Result<(), AicError> {
        self.push_command(command)
    }

    /// Creates the access point's internal control state carrying a single
    /// `ME_STA_ADD`.  The interface only reports `Busy` to a new user request
    /// while this transient state is alive, which clears once its confirmation
    /// is handled.
    pub(super) fn ap_station_add(command: ControlCommand) -> Self {
        let command_bytes = command.payload.len();
        Self {
            commands: VecDeque::from([command]),
            command_bytes,
            operation: ControlOperation::ApStationAdd,
        }
    }

    pub(super) fn pop_command(&mut self) -> Option<ControlCommand> {
        let command = self.commands.pop_front()?;
        self.command_bytes -= command.payload.len();
        Some(command)
    }

    pub(super) fn assign_ap_interface(&mut self, index: u8) -> Result<(), AicError> {
        use crate::lmac::{APM_SET_BEACON_IE_REQ, APM_START_REQ, MM_ADD_IF_REQ, MM_RESET_REQ};
        // After a MAC reset the reset confirmation has already been consumed,
        // so the ADD_IF command is at the front; when no reset was issued the
        // ADD_IF is already the head of the queue.
        let front_is_reset =
            self.commands.front().map(|command| command.message_id) == Some(MM_RESET_REQ);
        let add_if_pos = if front_is_reset { 1 } else { 0 };
        if self
            .commands
            .get(add_if_pos)
            .map(|command| command.message_id)
            != Some(MM_ADD_IF_REQ)
        {
            return Err(AicError::CompletionMismatch);
        }
        self.pop_command();
        for command in &mut self.commands {
            match command.message_id {
                APM_SET_BEACON_IE_REQ => command.payload[0] = index,
                APM_START_REQ => command.payload[APM_START_VIF_INDEX] = index,
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) fn confirm_ap_start(&mut self, payload: &[u8]) -> Result<u8, AicError> {
        let command = self.commands.front().ok_or(AicError::CompletionMismatch)?;
        let bcmc_index =
            crate::lmac::parse_ap_start(payload, command.payload[APM_START_VIF_INDEX])?;
        self.pop_command();
        Ok(bcmc_index)
    }

    pub(super) fn accept_connect_indication(
        &mut self,
        station_index: u8,
        bssid: [u8; 6],
        local_mac: [u8; 6],
    ) -> Result<(), AicError> {
        let ControlOperation::Connect(connect) = &self.operation else {
            return Err(AicError::CompletionMismatch);
        };
        if connect.phase != ConnectPhase::AwaitIndication {
            return Err(AicError::CompletionMismatch);
        }
        if connect.pmk.is_none() {
            if !self.has_command_capacity(1, 2) {
                return Err(AicError::ControlQueueFull);
            }
            self.push_command(ControlCommand {
                message_id: ME_SET_CONTROL_PORT_REQ,
                destination: TASK_ME,
                expected_message_id: ME_SET_CONTROL_PORT_CFM,
                payload: control_port_payload(station_index, true).to_vec(),
            })?;
            let ControlOperation::Connect(connect) = &mut self.operation else {
                return Err(AicError::CompletionMismatch);
            };
            connect.phase = ConnectPhase::AwaitControlPort;
            Ok(())
        } else {
            let ControlOperation::Connect(connect) = &mut self.operation else {
                return Err(AicError::CompletionMismatch);
            };
            let entropy = connect.entropy.take().ok_or(AicError::EntropyUnavailable)?;
            let pmk = connect.pmk.take().ok_or(AicError::WpaProtocol)?;
            connect.handshake = Some(Wpa2Handshake::new(
                *pmk.bytes(),
                bssid,
                local_mac,
                *entropy.bytes(),
            ));
            connect.phase = ConnectPhase::AwaitHandshake;
            Ok(())
        }
    }

    pub(super) fn process_eapol(
        &mut self,
        interface_index: u8,
        station_index: u8,
        eapol: &[u8],
    ) -> Result<ControlEffect, AicError> {
        let action = {
            let ControlOperation::Connect(connect) = &mut self.operation else {
                return Err(AicError::CompletionMismatch);
            };
            if connect.phase != ConnectPhase::AwaitHandshake {
                return Err(AicError::WpaProtocol);
            }
            log::info!("[wifi] WPA2 EAPOL frame consumed by handshake state machine");
            connect
                .handshake
                .as_mut()
                .ok_or(AicError::WpaProtocol)?
                .process(eapol)
                .map_err(map_wpa_error)?
        };
        match action {
            HandshakeAction::SendM2(frame) => Ok(ControlEffect::TransmitEapol(frame)),
            HandshakeAction::InstallKeys(keys) => {
                if !self.has_command_capacity(2, 88) {
                    return Err(AicError::ControlQueueFull);
                }
                self.push_command(ControlCommand {
                    message_id: MM_KEY_ADD_REQ,
                    destination: TASK_MM,
                    expected_message_id: MM_KEY_ADD_CFM,
                    payload: key_add_payload(
                        interface_index,
                        station_index,
                        true,
                        0,
                        &keys.temporal_key,
                    )?
                    .to_vec(),
                })?;
                self.push_command(ControlCommand {
                    message_id: MM_KEY_ADD_REQ,
                    destination: TASK_MM,
                    expected_message_id: MM_KEY_ADD_CFM,
                    payload: key_add_payload(
                        interface_index,
                        u8::MAX,
                        false,
                        keys.group_key_index,
                        &keys.group_key,
                    )?
                    .to_vec(),
                })?;
                let ControlOperation::Connect(connect) = &mut self.operation else {
                    return Err(AicError::CompletionMismatch);
                };
                connect.pending_m4 = Some(keys.m4.clone());
                connect.phase = ConnectPhase::InstallingKeys(2);
                Ok(ControlEffect::None)
            }
        }
    }

    pub(super) fn accept_key_confirmation(&mut self) -> Result<Option<Vec<u8>>, AicError> {
        let ControlOperation::Connect(connect) = &mut self.operation else {
            return Err(AicError::CompletionMismatch);
        };
        let ConnectPhase::InstallingKeys(remaining) = connect.phase else {
            return Err(AicError::CompletionMismatch);
        };
        if remaining > 1 {
            connect.phase = ConnectPhase::InstallingKeys(remaining - 1);
            Ok(None)
        } else {
            connect.phase = ConnectPhase::AwaitM4Transmit;
            connect
                .pending_m4
                .take()
                .map(Some)
                .ok_or(AicError::WpaProtocol)
        }
    }

    pub(super) fn accept_m4_transmit(&mut self, station_index: u8) -> Result<(), AicError> {
        let ControlOperation::Connect(connect) = &self.operation else {
            return Err(AicError::CompletionMismatch);
        };
        if connect.phase != ConnectPhase::AwaitM4Transmit {
            return Err(AicError::CompletionMismatch);
        }
        if !self.has_command_capacity(1, 2) {
            return Err(AicError::ControlQueueFull);
        }
        self.push_command(ControlCommand {
            message_id: ME_SET_CONTROL_PORT_REQ,
            destination: TASK_ME,
            expected_message_id: ME_SET_CONTROL_PORT_CFM,
            payload: control_port_payload(station_index, true).to_vec(),
        })?;
        let ControlOperation::Connect(connect) = &mut self.operation else {
            return Err(AicError::CompletionMismatch);
        };
        connect.phase = ConnectPhase::AwaitControlPort;
        Ok(())
    }
}

fn map_wpa_error(error: WpaError) -> AicError {
    match error {
        WpaError::Mic => AicError::WpaMic,
        WpaError::ReplayCounter => AicError::WpaReplay,
        WpaError::Rsn => AicError::WpaRsn,
        WpaError::InvalidKeyData | WpaError::KeyUnwrap | WpaError::GtkMissing => {
            AicError::WpaKeyData
        }
        WpaError::FrameTooShort
        | WpaError::InvalidLength
        | WpaError::InvalidDescriptor
        | WpaError::UnexpectedMessage
        | WpaError::InvalidState => AicError::WpaProtocol,
    }
}

pub(super) fn build(
    request: ControlRequest,
    mac: [u8; 6],
    interface_index: Option<u8>,
) -> Result<ControlState, AicError> {
    let mut commands = VecDeque::new();
    let operation = match request {
        ControlRequest::Scan { ssid } => {
            if ssid
                .as_ref()
                .is_some_and(|ssid| ssid.len() > MAX_SSID_LENGTH)
            {
                return Err(AicError::InvalidControlRequest);
            }
            commands.push_back(scan_command(ssid.as_deref()));
            ControlOperation::Commands
        }
        ControlRequest::Connect { ssid, pmk, entropy } => {
            if ssid.is_empty() || ssid.len() > MAX_SSID_LENGTH {
                return Err(AicError::InvalidControlRequest);
            }
            let entropy = if pmk.is_none() {
                None
            } else {
                Some(entropy.ok_or(AicError::EntropyUnavailable)?)
            };
            // A prior MM_RESET_REQ (issued by the AP flow) clears every
            // firmware VIF including the STA interface. Re-register the STA
            // VIF before connecting so SM_CONNECT_REQ references a valid
            // firmware slot; the confirmation handler patches the real
            // firmware-assigned index into the connect command.
            if interface_index.is_none() {
                commands.push_back(ControlCommand {
                    message_id: MM_ADD_IF_REQ,
                    destination: TASK_MM,
                    expected_message_id: MM_ADD_IF_CFM,
                    payload: add_interface_payload(mac, 0).to_vec(),
                });
            }
            let interface_index = interface_index.unwrap_or(0);
            commands.push_back(ControlCommand {
                message_id: SM_DISCONNECT_REQ,
                destination: TASK_SM,
                expected_message_id: SM_DISCONNECT_CFM,
                payload: disconnect_payload(interface_index).to_vec(),
            });
            commands.push_back(ControlCommand {
                message_id: SM_CONNECT_REQ,
                destination: TASK_SM,
                expected_message_id: SM_CONNECT_CFM,
                payload: connect_payload(&ssid, pmk.is_some(), interface_index),
            });
            ControlOperation::Connect(Box::new(PendingConnect {
                pmk,
                entropy,
                phase: ConnectPhase::Resetting,
                handshake: None,
                pending_m4: None,
            }))
        }
        ControlRequest::Disconnect => {
            let interface_index = interface_index.ok_or(AicError::InvalidControlRequest)?;
            commands.push_back(ControlCommand {
                message_id: SM_DISCONNECT_REQ,
                destination: TASK_SM,
                expected_message_id: SM_DISCONNECT_CFM,
                payload: disconnect_payload(interface_index).to_vec(),
            });
            ControlOperation::Disconnect
        }
        ControlRequest::StartOpenAccessPoint { ssid, channel } => {
            if ssid.is_empty() || ssid.len() > MAX_SSID_LENGTH || !(1..=14).contains(&channel) {
                return Err(AicError::InvalidControlRequest);
            }
            log::info!(
                "[wifi] building AP commands: ssid=\"{}\" channel={} \
                 mac={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                core::str::from_utf8(&ssid).unwrap_or("<invalid>"),
                channel,
                mac[0],
                mac[1],
                mac[2],
                mac[3],
                mac[4],
                mac[5]
            );
            // The access point rides a full MAC bring-up, the sequence that
            // board logs show lets a station associate and the control
            // transaction finish (log20/log29): `MM_RESET` + `ME_CONFIG`
            // re-establish the started MAC, then the channel, then the AP VIF
            // built by `open_access_point_commands` below. Dropping the
            // `MM_RESET`/`ME_CONFIG` pair to "ride boot's RF" (log30-log32)
            // left the AP control transaction unable to complete and no client
            // able to associate, so the pair is restored here. `MM_RESET` also
            // makes its confirmation clear any stale station link, so AP data
            // frames resolve through the AP VIF rather than a dead uplink.
            commands.push_back(ControlCommand {
                message_id: MM_RESET_REQ,
                destination: TASK_MM,
                expected_message_id: MM_RESET_CFM,
                payload: Vec::new(),
            });
            commands.push_back(ControlCommand {
                message_id: ME_CONFIG_REQ,
                destination: TASK_ME,
                expected_message_id: ME_CONFIG_CFM,
                payload: me_config_payload().to_vec(),
            });
            commands.push_back(ControlCommand {
                message_id: ME_CHAN_CONFIG_REQ,
                destination: TASK_ME,
                expected_message_id: ME_CHAN_CONFIG_CFM,
                payload: channel_config_payload().to_vec(),
            });
            let _ = interface_index;
            commands.extend(open_access_point_commands(&ssid, channel, mac));
            ControlOperation::Commands
        }
        ControlRequest::Cancel | ControlRequest::Shutdown => {
            return Err(AicError::InvalidControlRequest);
        }
    };
    let command_bytes = commands.iter().map(|command| command.payload.len()).sum();
    if commands.len() > CONTROL_COMMAND_CAPACITY || command_bytes > CONTROL_COMMAND_BYTE_CAPACITY {
        return Err(AicError::ControlQueueFull);
    }
    Ok(ControlState {
        commands,
        command_bytes,
        operation,
    })
}

/// Builds the `ME_STA_ADD_REQ` control command registering `peer` (identified
/// by its MAC, the AID handed back in its association response and the legacy
/// rates it advertised) on the access-point VIF `interface_index`.
pub(super) fn ap_sta_add_command(
    peer: [u8; 6],
    rates: &[u8],
    aid: u16,
    interface_index: u8,
) -> ControlCommand {
    ControlCommand {
        message_id: ME_STA_ADD_REQ,
        destination: TASK_ME,
        expected_message_id: ME_STA_ADD_CFM,
        payload: me_sta_add_payload(peer, rates, aid, interface_index),
    }
}

fn scan_command(ssid: Option<&[u8]>) -> ControlCommand {
    const CHANNELS: [u16; 14] = [
        2412, 2417, 2422, 2427, 2432, 2437, 2442, 2447, 2452, 2457, 2462, 2467, 2472, 2484,
    ];
    const CHANNEL_SIZE: usize = 8;
    const CHANNEL_MAX: usize = 42;
    const SSID_SIZE: usize = 33;
    const SSID_MAX: usize = 2;
    let mut payload = vec![0; 376];
    for (index, frequency) in CHANNELS.into_iter().enumerate() {
        let offset = index * CHANNEL_SIZE;
        payload[offset..offset + 2].copy_from_slice(&frequency.to_le_bytes());
        payload[offset + 4] = 30;
    }
    let ssid_offset = CHANNEL_MAX * CHANNEL_SIZE;
    let ssid_count = if let Some(ssid) = ssid {
        payload[ssid_offset] = ssid.len() as u8;
        payload[ssid_offset + 1..ssid_offset + 1 + ssid.len()].copy_from_slice(ssid);
        1
    } else {
        0
    };
    let bssid_offset = ssid_offset + SSID_MAX * SSID_SIZE + 1;
    payload[bssid_offset..bssid_offset + 6].fill(0xff);
    let tail = bssid_offset + 8;
    payload[tail + 6] = 0;
    payload[tail + 7] = CHANNELS.len() as u8;
    payload[tail + 8] = ssid_count;
    ControlCommand {
        message_id: 0x1000,
        destination: 4,
        expected_message_id: 0x1001,
        payload,
    }
}

/// Offset of the `vif_idx` field inside the natural-aligned vendor
/// `struct apm_start_req` payload built by [`open_access_point_commands`].
///
/// Vendor layout (lmac_msg.h, natural alignment): `mac_rateset` (length + 12
/// rates = 13 bytes), one pad byte, `mac_chan_def` { u16 freq; u8 band;
/// u8 flags; s8 tx_power } at 14..19, `center_freq1/2` u32 at 20..28,
/// `ch_width` at 28, pad, `bcn_addr` u32 at 32, `bcn_len`/`tim_oft`/
/// `bcn_int` u16 at 36..42, pad, `flags` u32 at 44, `ctrl_port_ethertype`
/// u16 at 48, `tim_len` at 50, `vif_idx` at 51 (struct size 52).
const APM_START_VIF_INDEX: usize = 51;

fn open_access_point_commands(ssid: &[u8], channel: u8, mac: [u8; 6]) -> VecDeque<ControlCommand> {
    use crate::lmac::{
        APM_SET_BEACON_IE_CFM, APM_SET_BEACON_IE_REQ, APM_START_CFM, APM_START_REQ, MM_ADD_IF_CFM,
        MM_ADD_IF_REQ, MM_SET_FILTER_CFM, MM_SET_FILTER_REQ, TASK_MM,
    };
    let mut commands = VecDeque::new();
    let add_interface = crate::lmac::add_interface_payload(mac, 2).to_vec();
    commands.push_back(ControlCommand {
        message_id: MM_ADD_IF_REQ,
        destination: TASK_MM,
        expected_message_id: MM_ADD_IF_CFM,
        payload: add_interface,
    });
    commands.push_back(ControlCommand {
        message_id: MM_SET_FILTER_REQ,
        destination: TASK_MM,
        expected_message_id: MM_SET_FILTER_CFM,
        payload: crate::lmac::filter_payload().to_vec(),
    });
    let (beacon, tim_offset) = open_beacon(ssid, channel, mac);
    log::info!(
        "[wifi] AP beacon: ssid=\"{}\" channel={} len={} tim_offset={} content={:02x?}",
        core::str::from_utf8(ssid).unwrap_or("<invalid>"),
        channel,
        beacon.len(),
        tim_offset,
        beacon
    );
    // Vendor `struct apm_set_bcn_ie_req` layout (natural alignment): vif_idx
    // at 0, one pad byte, bcn_ie_len u16 at 2, bcn_ie at 4. The vif index
    // comes from the ADD_IF confirmation and is pinned in by the mailbox
    // completion path.
    let mut beacon_request = vec![0; 516];
    beacon_request[2..4].copy_from_slice(&(beacon.len() as u16).to_le_bytes());
    beacon_request[4..4 + beacon.len()].copy_from_slice(&beacon);
    commands.push_back(ControlCommand {
        message_id: APM_SET_BEACON_IE_REQ,
        destination: 7,
        expected_message_id: APM_SET_BEACON_IE_CFM,
        payload: beacon_request,
    });
    let mut start = vec![0; 52];
    start[0] = 4;
    start[1..5].copy_from_slice(&[0x82, 0x84, 0x8b, 0x96]);
    let frequency = if channel == 14 {
        2484
    } else {
        2407 + u16::from(channel) * 5
    };
    start[14..16].copy_from_slice(&frequency.to_le_bytes());
    start[18] = 20;
    start[20..24].copy_from_slice(&u32::from(frequency).to_le_bytes());
    start[36..38].copy_from_slice(&(beacon.len() as u16).to_le_bytes());
    start[38..40].copy_from_slice(&(tim_offset as u16).to_le_bytes());
    start[40..42].copy_from_slice(&100u16.to_le_bytes());
    start[48..50].copy_from_slice(&0x888eu16.to_be_bytes());
    start[50] = 6;
    log::info!(
        "[wifi] APM_START_REQ: freq={} MHz beacon_len={} tim_offset={} bcn_int=100 payload={:02x?}",
        frequency,
        beacon.len(),
        tim_offset,
        &start[..]
    );
    commands.push_back(ControlCommand {
        message_id: APM_START_REQ,
        destination: 7,
        expected_message_id: APM_START_CFM,
        payload: start,
    });
    commands
}

fn open_beacon(ssid: &[u8], channel: u8, mac: [u8; 6]) -> (Vec<u8>, usize) {
    let mut beacon = Vec::new();
    beacon.extend_from_slice(&[0x80, 0, 0, 0]);
    beacon.extend_from_slice(&[0xff; 6]);
    beacon.extend_from_slice(&mac);
    beacon.extend_from_slice(&mac);
    beacon.extend_from_slice(&[0, 0]);
    beacon.extend_from_slice(&[0; 8]);
    beacon.extend_from_slice(&100u16.to_le_bytes());
    beacon.extend_from_slice(&crate::ap::AP_CAPABILITY.to_le_bytes());
    beacon.extend_from_slice(&[0, ssid.len() as u8]);
    beacon.extend_from_slice(ssid);
    beacon.extend_from_slice(&[1, 8, 0x82, 0x84, 0x8b, 0x96, 0x0c, 0x12, 0x18, 0x24]);
    beacon.extend_from_slice(&[3, 1, channel]);
    let tim_offset = beacon.len();
    beacon.extend_from_slice(&[5, 4, 0, 1, 0, 0]);
    (beacon, tim_offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_connect_requires_owned_entropy() {
        assert_eq!(
            build(
                ControlRequest::Connect {
                    ssid: b"network".to_vec(),
                    pmk: Some(Pmk::new([3; 32])),
                    entropy: None,
                },
                [0; 6],
                Some(0),
            )
            .err(),
            Some(AicError::EntropyUnavailable)
        );
        assert!(
            build(
                ControlRequest::Connect {
                    ssid: b"network".to_vec(),
                    pmk: Some(Pmk::new([3; 32])),
                    entropy: Some(Entropy::new([7; 32])),
                },
                [0; 6],
                Some(0)
            )
            .is_ok()
        );
    }

    #[test]
    fn connect_request_carries_the_firmware_interface_index() {
        let state = build(
            ControlRequest::Connect {
                ssid: b"network".to_vec(),
                pmk: Some(Pmk::new([3; 32])),
                entropy: Some(Entropy::new([7; 32])),
            },
            [2, 0, 0, 0, 0, 1],
            Some(6),
        )
        .unwrap();

        let connect = state
            .commands
            .iter()
            .find(|command| command.message_id == SM_CONNECT_REQ)
            .expect("connect command follows the stale-association reset");
        assert_eq!(connect.payload[61], 6);
        assert_eq!(
            state.commands.front().unwrap().message_id,
            SM_DISCONNECT_REQ
        );
    }

    #[test]
    fn disconnect_waits_for_confirmation_not_the_async_indication() {
        let state = build(ControlRequest::Disconnect, [2, 0, 0, 0, 0, 1], Some(6)).unwrap();
        let command = state.commands.front().unwrap();

        assert_eq!(command.message_id, SM_DISCONNECT_REQ);
        assert_eq!(command.expected_message_id, SM_DISCONNECT_CFM);
        assert_eq!(command.payload, [3, 0, 6, 0]);
    }
}
