use alloc::{vec, vec::Vec};
use core::time::Duration;

use super::*;
use crate::{
    lmac::{
        SM_CONNECT_IND, SM_DISCONNECT_IND, parse_connect_indication, parse_disconnect_indication,
    },
    protocol::{BLOCK_SIZE, ethernet_tx_frame, mgmt_tx_frame},
    registers::ReceiveLength,
    rx::{ParsedFrame, parse_fifo},
};

const IO_RETRY: Duration = Duration::from_millis(1);
// Access points have no peer traffic to raise CARD_INT while they idle, so a
// bare interrupt wait lets the firmware's per-beacon indication and beacon-TX
// completion queue back up until the transmitter stalls. Wake on this interval
// (well inside the ~102 ms beacon period) to drain one bounded scan.
const AP_RX_POLL: Duration = Duration::from_millis(50);
// The firmware reports packet buffers, not SDIO blocks. Keep two buffers
// available for commands, as in the vendor DATA_FLOW_CTRL_THRESH contract.
const DATA_TX_RESERVED_CREDITS: u8 = 2;
const INTERNAL_TX_CAPACITY: usize = 2;
const INTERNAL_TX_BYTE_CAPACITY: usize = 8 * 1024;
const ETHERTYPE_EAPOL: [u8; 2] = [0x88, 0x8e];

impl AicDevice {
    pub(super) fn drive_ready(&mut self, now: MonotonicTime) -> AicAction {
        if self.mailbox_timed_out(now) {
            return self.drive_mailbox(now);
        }
        if self.lifecycle.mailbox.is_some() {
            // LMAC confirmations arrive on the command/data FIFO and are
            // announced through the level-sensitive CARD_INT source.  Once a
            // mailbox write has completed, drain one bounded receive scan
            // before waiting for the next interrupt; otherwise a pending
            // confirmation would leave the mailbox parked while the control
            // command at the front of the queue is submitted again.
            if self.mailbox_waiting_for_receive()
                && let Some(action) = self.drive_receive_scan()
            {
                return action;
            }
            return self.drive_mailbox(now);
        }
        if let Some(control) = self.lifecycle.control.as_ref()
            && let Some(command) = control.commands.front()
        {
            let message_id = command.message_id;
            let destination = command.destination;
            let expected = command.expected_message_id;
            let payload = command.payload.clone();
            self.begin_lmac_mailbox(message_id, destination, &payload, expected, now);
            return self.drive_mailbox(now);
        }
        // Deliver terminal/control events before starting another level-triggered
        // receive scan.  CARD_INT may remain asserted while the firmware drains
        // queued traffic; scanning first would indefinitely postpone the
        // ControlComplete event that releases the Linux WEXT caller.
        if let Some(event) = self.take_priority_event() {
            return AicAction::Event(event);
        }
        if let Some(action) = self.drive_receive_scan() {
            return action;
        }
        self.prepare_next_transmit();
        if let Some(active) = self.data.active_tx.as_mut() {
            if let Some(deadline) = active.retry_at {
                if now < deadline {
                    return AicAction::WaitForInterruptUntil(deadline);
                }
                active.retry_at = None;
            }
            // DC bypasses flow control only for its separate command mailbox.
            // Every data packet must obtain firmware capacity before CMD53.
            return self.emit(
                IoPurpose::TransmitFlow,
                read_byte(self.data_function(), self.registers().flow_control),
            );
        }
        // A running access point keeps emitting beacons whose per-beacon
        // indications and TX-completion credits land on the receive FIFO. The
        // firmware halts the beacon transmitter once that queue is not
        // drained, and with no associated peer nothing else re-asserts
        // CARD_INT, so parking on a bare interrupt wait permanently silences
        // the AP. Arm one bounded receive scan and wake on a timer instead:
        // each wake drains at most one scan and then releases the owner lock
        // until the next interval, so the queue keeps draining without
        // starving callers the way an unparked rescan loop would.
        if self.data.ap.is_some() {
            self.request_receive_scan();
            return AicAction::WaitForInterruptUntil(now.after(AP_RX_POLL));
        }
        AicAction::WaitForInterrupt
    }

    fn take_priority_event(&mut self) -> Option<AicEvent> {
        let index = self
            .data
            .events
            .iter()
            .position(|event| !matches!(event, AicEvent::Receive(_)))?;
        self.data.remove_event(index)
    }

    pub(super) fn request_receive_scan(&mut self) {
        // Firmware startup owns both SDIO functions until a mailbox
        // confirmation is waiting.  The controller reports CARD_INT as a
        // level source, so treating that status as a data-plane receive event
        // during function setup would continually preempt the startup FSM.
        // The first confirmation interrupt is the sole startup exception; it
        // arms one bounded scan and subsequent level samples are coalesced.
        if self.lifecycle.state == AicState::Starting && !self.startup_confirmation_waiting() {
            return;
        }
        if self.io.receive.active {
            // CARD_INT is level-triggered.  The in-flight scan drains every
            // enabled function; rearm_and_check() observes a source that
            // remains asserted and schedules the next scan after this one
            // completes, so a second scan must never be queued here.
            return;
        }
        log::debug!("[wifi] CARD_INT: arming receive scan");
        self.io.receive.active = true;
        self.io.receive.next_path = 0;
    }

    pub(super) fn drive_receive_scan(&mut self) -> Option<AicAction> {
        if !self.io.receive.active {
            return None;
        }
        if let Some(path) = self.receive_path(usize::from(self.io.receive.next_path)) {
            let function = self.receive_function(path);
            return Some(self.emit(
                IoPurpose::ReceiveCount(path),
                read_byte(function, self.registers().block_count),
            ));
        }
        self.io.receive.active = false;
        self.io.receive.next_path = 0;
        None
    }

    pub(super) fn consume_receive_count(
        &mut self,
        path: RxPath,
        response: SdioResponse,
    ) -> Result<(), AicError> {
        let count = expect_byte(response)?;
        match self.registers().receive_length(count) {
            ReceiveLength::Empty => self.advance_receive_path(),
            ReceiveLength::OtherInterrupt => {
                log::info!("[wifi] rx scan {path:?}: other interrupt (count={count:#04x})");
                // The vendor D80 IRQ handler acknowledges the dev-to-host soft
                // IRQ here: read the interrupt-pending register, clear bit 0,
                // and write it back. Without the acknowledgement the pending
                // bit keeps CARD_INT asserted, which both starves the owner
                // loop and leaves the firmware waiting for its interrupt to
                // be consumed.
                self.io.next = Some((
                    IoPurpose::ReceiveOtherAck(path),
                    read_byte(
                        self.receive_function(path),
                        self.registers()
                            .sleep_status
                            .expect("v3 interrupt status implies a sleep-status register"),
                    ),
                ));
            }
            ReceiveLength::Blocks(blocks) => {
                log::debug!(
                    "[wifi] rx scan {path:?}: {} blocks ({} bytes, count={count:#04x})",
                    blocks,
                    usize::from(blocks) * BLOCK_SIZE
                );
                self.io.next = Some((
                    IoPurpose::ReceiveData(path),
                    read_fifo(
                        self.receive_function(path),
                        self.registers().read_fifo,
                        usize::from(blocks) * BLOCK_SIZE,
                    ),
                ));
            }
            ReceiveLength::ByteMode => {
                log::info!("[wifi] rx scan {path:?}: byte mode (count={count:#04x})");
                self.io.next = Some((
                    IoPurpose::ReceiveByteLength(path),
                    read_byte(
                        self.receive_function(path),
                        self.registers().byte_mode_length,
                    ),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn consume_receive_other_ack(
        &mut self,
        path: RxPath,
        response: SdioResponse,
    ) -> Result<(), AicError> {
        let pending = expect_byte(response)?;
        self.io.next = Some((
            IoPurpose::ReceiveOtherClear(path),
            write_byte(
                self.receive_function(path),
                self.registers()
                    .sleep_status
                    .expect("v3 interrupt status implies a sleep-status register"),
                pending & !1,
            ),
        ));
        Ok(())
    }

    pub(super) fn consume_receive_other_clear(
        &mut self,
        path: RxPath,
        response: SdioResponse,
    ) -> Result<(), AicError> {
        // The write is issued with read-after-write; the read-back byte is the
        // register's pre-write value and is not compared.
        let _ = expect_byte(response)?;
        // The vendor D80 handler re-reads the interrupt status after the soft
        // IRQ acknowledgement; stay on the same path until it reads empty.
        self.io.next = Some((
            IoPurpose::ReceiveCount(path),
            read_byte(self.receive_function(path), self.registers().block_count),
        ));
        Ok(())
    }

    pub(super) fn consume_receive_byte_length(
        &mut self,
        path: RxPath,
        response: SdioResponse,
    ) -> Result<(), AicError> {
        let units = expect_byte(response)?;
        if units == 0 || units > 128 {
            return Err(AicError::InvalidRxByteLength { units });
        }
        self.io.next = Some((
            IoPurpose::ReceiveData(path),
            read_fifo(
                self.receive_function(path),
                self.registers().read_fifo,
                usize::from(units) * 4,
            ),
        ));
        Ok(())
    }

    pub(super) fn consume_receive_data(
        &mut self,
        path: RxPath,
        response: SdioResponse,
    ) -> Result<(), AicError> {
        let receive_data = expect_data(response)?;
        log::debug!(
            "[wifi] rx data {path:?}: {} bytes raw: {:02x?}",
            receive_data.len(),
            &receive_data[..receive_data.len().min(64)]
        );
        let frames =
            parse_fifo(&receive_data, self.mailbox_confirmation_id()).map_err(|error| {
                let header_length = receive_data.len().min(24);
                let mut header = [0; 24];
                header[..header_length].copy_from_slice(&receive_data[..header_length]);
                let header_words = [
                    u64::from_le_bytes(header[0..8].try_into().expect("fixed header word")),
                    u64::from_le_bytes(header[8..16].try_into().expect("fixed header word")),
                    u64::from_le_bytes(header[16..24].try_into().expect("fixed header word")),
                ];
                log::error!(
                    "malformed AIC RX frame on {path:?}: transfer={} header={:02x?}",
                    receive_data.len(),
                    &receive_data[..header_length]
                );
                AicError::MalformedRxFrame {
                    offset: error.offset,
                    packet_type: error.packet_type,
                    declared_length: error.declared_length,
                    available_length: error.available_length,
                    header_words,
                }
            })?;
        for frame in frames {
            match frame {
                ParsedFrame::Data {
                    frame,
                    decryption_status,
                } => {
                    let Some(frames) = decapsulate_data_frames(&frame, decryption_status) else {
                        let fc = u16::from_le_bytes([
                            frame.first().copied().unwrap_or(0),
                            frame.get(1).copied().unwrap_or(0),
                        ]);
                        let ftype = (fc >> 2) & 0x3;
                        let subtype = (fc >> 4) & 0xf;
                        let le16 = |off: usize| {
                            frame
                                .get(off..off + 2)
                                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        };
                        // Read-only diagnostic decode of 802.11 management
                        // frames so AP association events can be correlated
                        // without touching the control plane. Addresses follow
                        // the standard 24-byte management header layout.
                        let dst = frame.get(4..10).unwrap_or(&[]);
                        let src = frame.get(10..16).unwrap_or(&[]);
                        let bssid = frame.get(16..22).unwrap_or(&[]);
                        if ftype == 0 {
                            match subtype {
                                11 => log::info!(
                                    "[wifi] rx AUTH dst={dst:02x?} src={src:02x?} \
                                     bssid={bssid:02x?} algo={:#04x} seq={:?} status={:#04x} \
                                     len={}",
                                    le16(24).unwrap_or(0),
                                    le16(26),
                                    le16(28).unwrap_or(0),
                                    frame.len(),
                                ),
                                0 => log::info!(
                                    "[wifi] rx ASSOC_REQ dst={dst:02x?} src={src:02x?} \
                                     bssid={bssid:02x?} cap={:#04x} len={}",
                                    le16(24).unwrap_or(0),
                                    frame.len(),
                                ),
                                1 => log::info!(
                                    "[wifi] rx ASSOC_RESP dst={dst:02x?} src={src:02x?} \
                                     bssid={bssid:02x?} cap={:#04x} status={:#04x} aid={:?}",
                                    le16(24).unwrap_or(0),
                                    le16(26).unwrap_or(0),
                                    le16(28),
                                ),
                                10 | 12 => log::info!(
                                    "[wifi] rx {} dst={dst:02x?} src={src:02x?} \
                                     bssid={bssid:02x?} reason={:#04x}",
                                    if subtype == 12 { "DEAUTH" } else { "DISASSOC" },
                                    le16(24).unwrap_or(0),
                                ),
                                4 => log::info!(
                                    "[wifi] rx PROBE_REQ dst={dst:02x?} src={src:02x?} \
                                     bssid={bssid:02x?} len={}",
                                    frame.len(),
                                ),
                                _ => log::debug!(
                                    "[wifi] rx mgmt subtype={subtype} dst={dst:02x?} \
                                     src={src:02x?} bssid={bssid:02x?} len={}",
                                    frame.len(),
                                ),
                            }
                        } else {
                            log::info!(
                                "[wifi] rx non-data frame fc={fc:#06x} type={ftype} \
                                 subtype={subtype} dec_status={} len={} raw={:02x?}",
                                decryption_status,
                                frame.len(),
                                &frame[..frame.len().min(48)]
                            );
                        }
                        if ftype == 0 {
                            self.maybe_respond_ap_mgmt(&frame, subtype);
                        }
                        continue;
                    };
                    for frame in frames {
                        if frame.get(12..14) == Some(&ETHERTYPE_EAPOL) {
                            self.consume_eapol(&frame)?;
                        } else {
                            let dst = frame.get(..6).unwrap_or(&[]);
                            let src = frame.get(6..12).unwrap_or(&[]);
                            log::info!(
                                "[wifi] rx data frame \
                                 dst={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} \
                                 src={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} len={}",
                                dst[0],
                                dst[1],
                                dst[2],
                                dst[3],
                                dst[4],
                                dst[5],
                                src[0],
                                src[1],
                                src[2],
                                src[3],
                                src[4],
                                src[5],
                                frame.len()
                            );
                            self.data.push_event(AicEvent::Receive(frame))?;
                        }
                    }
                }
                ParsedFrame::Confirmation {
                    message_id,
                    payload,
                } => {
                    self.accept_mailbox_confirmation(message_id, payload)?;
                }
                ParsedFrame::DataConfirmation => {
                    log::trace!("AIC firmware data confirmation");
                }
                ParsedFrame::FirmwarePrint { length, content } => {
                    let text = core::str::from_utf8(&content)
                        .unwrap_or("<binary>")
                        .trim_end_matches('\0');
                    log::info!("[wifi] AIC firmware trace ({length} bytes): \"{text}\"");
                }
                ParsedFrame::Indication {
                    message_id: SM_CONNECT_IND,
                    payload,
                } => {
                    // Firmware can leave an asynchronous association result in
                    // the FIFO across host restart. Startup has no connection
                    // transaction: its owner is handed to the network runtime
                    // before a new Connect request can be submitted. Do not
                    // interpret the old status or install its peer identity.
                    if self.lifecycle.state == AicState::Starting {
                        log::debug!(
                            "[wifi] discarded pre-connection association indication during startup"
                        );
                        continue;
                    }
                    let indication = parse_connect_indication(&payload)?;
                    log::info!(
                        "[wifi] association complete; learned firmware vif={} station={}",
                        indication.interface_index,
                        indication.station_index
                    );
                    self.data.link.install_peer(
                        indication.interface_index,
                        indication.station_index,
                        indication.bssid,
                    )?;
                    let control = self
                        .lifecycle
                        .control
                        .as_mut()
                        .ok_or(AicError::CompletionMismatch)?;
                    let local_mac = self
                        .data
                        .link
                        .mac_address()
                        .ok_or(AicError::InvalidMacAddress)?;
                    control.accept_connect_indication(
                        indication.station_index,
                        indication.bssid,
                        local_mac,
                    )?;
                }
                ParsedFrame::Indication {
                    message_id: SM_DISCONNECT_IND,
                    payload,
                } => {
                    let indication = parse_disconnect_indication(&payload)?;
                    if self.data.link.interface_index() != Some(indication.interface_index) {
                        return Err(AicError::MalformedResponse);
                    }
                    self.data.link.clear_peer();
                    self.data.clear_internal_tx();
                    let resetting = self.lifecycle.control.as_ref().is_some_and(|control| {
                        matches!(&control.operation,
                            super::control::ControlOperation::Connect(connect)
                                if connect.phase == super::control::ConnectPhase::Resetting)
                    });
                    if !resetting && self.lifecycle.control.take().is_some() {
                        self.data
                            .push_event(AicEvent::ControlFailed(AicError::Disconnected {
                                reason_code: indication.reason_code,
                            }))?;
                    }
                }
                ParsedFrame::Indication {
                    message_id,
                    payload,
                } => {
                    log::info!(
                        "[wifi] unhandled firmware indication id={message_id:#06x}, payload={} \
                         bytes: {:02x?}",
                        payload.len(),
                        payload
                    );
                }
            }
        }
        self.io.next = Some((
            IoPurpose::ReceiveCount(path),
            read_byte(self.receive_function(path), self.registers().block_count),
        ));
        Ok(())
    }

    fn advance_receive_path(&mut self) {
        self.io.receive.next_path = self.io.receive.next_path.saturating_add(1);
    }

    pub(super) fn consume_transmit_flow(
        &mut self,
        response: SdioResponse,
        now: MonotonicTime,
    ) -> Result<(), AicError> {
        let credits = self.registers().flow_credits(expect_byte(response)?);
        let active = self
            .data
            .active_tx
            .as_mut()
            .ok_or(AicError::CompletionMismatch)?;
        if credits <= DATA_TX_RESERVED_CREDITS {
            log::debug!("[wifi] tx flow: credits={credits} below reserved, retrying");
            active.retry_at = Some(now.after(IO_RETRY));
            return Ok(());
        }
        let frame = active.wire_frame.clone();
        self.io.next = Some((
            IoPurpose::TransmitData,
            write_fifo(self.data_function(), self.registers().write_fifo, frame),
        ));
        Ok(())
    }

    pub(super) fn consume_transmit_data(&mut self, response: SdioResponse) -> Result<(), AicError> {
        expect_unit(response)?;
        let active = self
            .data
            .active_tx
            .take()
            .ok_or(AicError::CompletionMismatch)?;
        match active.completion {
            super::owner::TxCompletion::User(token) => {
                self.data.push_event(AicEvent::TransmitComplete(token))?
            }
            super::owner::TxCompletion::Internal(super::owner::InternalTxKind::M2) => {}
            super::owner::TxCompletion::Internal(super::owner::InternalTxKind::ApMgmt) => {
                log::info!("[wifi] ap mgmt tx confirmed");
            }
            super::owner::TxCompletion::Internal(super::owner::InternalTxKind::M4) => {
                let (station_index, _) = self.data.link.peer().ok_or(AicError::WpaProtocol)?;
                self.lifecycle
                    .control
                    .as_mut()
                    .ok_or(AicError::CompletionMismatch)?
                    .accept_m4_transmit(station_index)?;
            }
        }
        Ok(())
    }

    fn prepare_next_transmit(&mut self) {
        if self.data.active_tx.is_some() {
            return;
        }
        let uses_header_crc = self.transport_uses_header_crc();
        // Host-originated frames (station EAPOL and access-point management
        // responses) carry their own firmware indices, so they are emitted
        // before the peer-dependent gate below; an access point has no
        // associated peer to resolve through `tx_indices`.
        if let Some(internal) = self.data.pop_internal_tx() {
            let built = match &internal.frame {
                super::owner::InternalTxFrame::Ethernet(ethernet) => ethernet_tx_frame(
                    ethernet,
                    internal.interface_index,
                    internal.station_index,
                    uses_header_crc,
                ),
                super::owner::InternalTxFrame::Mgmt(raw) => {
                    mgmt_tx_frame(raw, internal.interface_index, uses_header_crc)
                }
            };
            let Ok(wire_frame) = built else {
                return;
            };
            if matches!(&internal.frame, super::owner::InternalTxFrame::Mgmt(_)) {
                log::info!(
                    "[wifi] ap mgmt tx emit wire_len={} if={}",
                    wire_frame.len(),
                    internal.interface_index,
                );
            }
            self.data.active_tx = Some(ActiveTx {
                retry_at: None,
                completion: super::owner::TxCompletion::Internal(internal.kind),
                wire_frame,
            });
            return;
        }
        let Some((interface_index, station_index)) = self.resolve_next_tx_indices() else {
            return;
        };
        let Some(frame) =
            self.data
                .tx
                .take_wire_frame(interface_index, station_index, uses_header_crc)
        else {
            return;
        };
        match frame {
            Ok((token, wire_frame)) => {
                if self.data.ap.is_some() {
                    log::debug!(
                        "[wifi] ap data tx emit if={interface_index} sta={station_index} \
                         wire_len={}",
                        wire_frame.len(),
                    );
                }
                self.data.active_tx = Some(ActiveTx {
                    retry_at: None,
                    completion: super::owner::TxCompletion::User(token),
                    wire_frame,
                });
            }
            Err(token) => self
                .data
                .events
                .push_back(AicEvent::TransmitComplete(token)),
        }
    }

    /// Resolves the firmware interface and station indices the next host frame
    /// must be transmitted with.  A running access point owns the radio: it
    /// sends on the AP VIF and picks the destination client's firmware station
    /// from the association table, falling back to the BC/MC station the
    /// firmware returned in `APM_START_CFM` for a broadcast or unregistered
    /// destination.  The firmware has no data-transmit station at the "no
    /// station" marker, so addressing a broadcast frame to a real BC/MC station
    /// is what lets a client's DHCP reply actually reach the air.  A station
    /// with no access point sends everything through its one associated peer,
    /// so the link supplies a fixed pair.
    fn resolve_next_tx_indices(&self) -> Option<(u8, u8)> {
        // The access point is checked first on purpose.  After a STA->AP switch
        // a station peer can still be parked in `link`, and routing client
        // frames through that stale uplink would silently deliver them to the
        // old BSS and starve the DHCP reply.
        if let Some(ap) = self.data.ap.as_ref() {
            let dest = self.data.tx.front_destination()?;
            return Some((
                ap.interface_index,
                ap.station_index_for(&dest).unwrap_or(ap.bcmc_index),
            ));
        }
        self.data.link.tx_indices()
    }

    fn consume_eapol(&mut self, ethernet: &[u8]) -> Result<(), AicError> {
        if ethernet.len() < 14 {
            return Err(AicError::WpaProtocol);
        }
        let local_mac = self
            .data
            .link
            .mac_address()
            .ok_or(AicError::InvalidMacAddress)?;
        let (station_index, bssid) = self.data.link.peer().ok_or(AicError::WpaProtocol)?;
        let interface_index = self
            .data
            .link
            .interface_index()
            .ok_or(AicError::WpaProtocol)?;
        if ethernet[..6] != local_mac || ethernet[6..12] != bssid {
            return Err(AicError::WpaProtocol);
        }
        let effect = self
            .lifecycle
            .control
            .as_mut()
            .ok_or(AicError::WpaProtocol)?
            .process_eapol(interface_index, station_index, &ethernet[14..])?;
        if let super::control::ControlEffect::TransmitEapol(frame) = effect {
            self.queue_internal_eapol(super::owner::InternalTxKind::M2, frame)?;
        }
        Ok(())
    }

    pub(super) fn queue_internal_eapol(
        &mut self,
        kind: super::owner::InternalTxKind,
        eapol: Vec<u8>,
    ) -> Result<(), AicError> {
        let ethernet_length = 14usize
            .checked_add(eapol.len())
            .ok_or(AicError::TxQueueFull)?;
        if self.data.internal_tx.len() >= INTERNAL_TX_CAPACITY
            || self
                .data
                .internal_tx_bytes
                .checked_add(ethernet_length)
                .is_none_or(|bytes| bytes > INTERNAL_TX_BYTE_CAPACITY)
        {
            return Err(AicError::TxQueueFull);
        }
        let local_mac = self
            .data
            .link
            .mac_address()
            .ok_or(AicError::InvalidMacAddress)?;
        let (station_index, bssid) = self.data.link.peer().ok_or(AicError::WpaProtocol)?;
        let interface_index = self
            .data
            .link
            .interface_index()
            .ok_or(AicError::WpaProtocol)?;
        let mut ethernet = Vec::with_capacity(ethernet_length);
        ethernet.extend_from_slice(&bssid);
        ethernet.extend_from_slice(&local_mac);
        ethernet.extend_from_slice(&ETHERTYPE_EAPOL);
        ethernet.extend_from_slice(&eapol);
        self.data.internal_tx_bytes += ethernet.len();
        self.data.internal_tx.push_back(super::owner::InternalTx {
            kind,
            interface_index,
            station_index,
            frame: super::owner::InternalTxFrame::Ethernet(ethernet),
        });
        Ok(())
    }

    /// Queues a raw 802.11 management-frame response (authentication or
    /// association) for transmission on the access-point interface.
    fn queue_internal_mgmt(&mut self, interface_index: u8, raw_frame: Vec<u8>) {
        if self.data.internal_tx.len() >= INTERNAL_TX_CAPACITY
            || self.data.internal_tx_bytes.saturating_add(raw_frame.len())
                > INTERNAL_TX_BYTE_CAPACITY
        {
            return;
        }
        self.data.internal_tx_bytes += raw_frame.len();
        self.data.internal_tx.push_back(super::owner::InternalTx {
            kind: super::owner::InternalTxKind::ApMgmt,
            interface_index,
            station_index: u8::MAX,
            frame: super::owner::InternalTxFrame::Mgmt(raw_frame),
        });
    }

    /// Answers an access-point authentication or association request that the
    /// firmware forwarded to us.  Only frames addressed to this AP's own MAC
    /// are answered, so unrelated management traffic stays untouched.
    fn maybe_respond_ap_mgmt(&mut self, frame: &[u8], subtype: u16) {
        let (Some(our_mac), Some(interface_index)) = (
            self.data.link.mac_address(),
            self.data.ap.as_ref().map(|ap| ap.interface_index),
        ) else {
            return;
        };
        if frame.get(4..10) != Some(&our_mac[..]) {
            return;
        }
        let src = frame.get(10..16).unwrap_or(&[]);
        let aid = match subtype {
            11 => None,
            0 => Some(
                self.data
                    .ap
                    .as_mut()
                    .map(|ap| ap.next_aid())
                    .unwrap_or_default(),
            ),
            _ => return,
        };
        let response = match (subtype, aid) {
            (11, _) => crate::ap::auth_response(frame),
            (0, Some(aid)) => crate::ap::assoc_response(frame, aid),
            _ => None,
        };
        match response {
            Some(raw_frame) => {
                log::info!(
                    "[wifi] ap queued {} resp src={src:02x?} if={interface_index} len={}",
                    if subtype == 11 { "AUTH" } else { "ASSOC" },
                    raw_frame.len(),
                );
                self.queue_internal_mgmt(interface_index, raw_frame);
                if let Some(aid) = aid {
                    self.queue_ap_station_add(frame, aid, interface_index);
                }
            }
            None if matches!(subtype, 0 | 11) => {
                log::info!(
                    "[wifi] ap skipped {} resp build src={src:02x?}",
                    if subtype == 11 { "AUTH" } else { "ASSOC" },
                );
            }
            None => {}
        }
    }

    /// Registers a freshly associated client with the firmware by queueing an
    /// `ME_STA_ADD` on the control mailbox, mirroring the vendor's
    /// `add_station` step that hostapd performs after sending the association
    /// response.  It is never submitted directly from this receive callback:
    /// it is placed on `lifecycle.control.commands`, which the owner drains
    /// through the mailbox state machine.  While a user control operation owns
    /// the control state the registration is skipped; the client retransmits
    /// its association and is picked up once the queue frees.
    fn queue_ap_station_add(&mut self, request: &[u8], aid: u16, interface_index: u8) {
        let Some(peer): Option<[u8; 6]> = request
            .get(10..16)
            .map(|mac| mac.try_into().expect("six-byte slice converts to array"))
        else {
            return;
        };
        let rates = crate::ap::assoc_supported_rates(request);
        let command = super::control::ap_sta_add_command(peer, &rates, aid, interface_index);
        if let Some(control) = self.lifecycle.control.as_mut() {
            if !control.is_ap_station_add() {
                return;
            }
            if control.push_station_add(command).is_err() {
                return;
            }
        } else {
            self.lifecycle.control = Some(super::control::ControlState::ap_station_add(command));
        }
        if let Some(ap) = self.data.ap.as_mut() {
            ap.queue_station_add(peer);
        }
        log::info!(
            "[wifi] ap queued ME_STA_ADD peer={peer:02x?} aid={aid} rates={}",
            rates.len(),
        );
    }
}

/// Converts the firmware's 802.11 MPDU (after its 60-byte hardware header)
/// into the Ethernet frame expected by the network stack.  The Linux AIC
/// driver performs the same operation in `rwnx_rxdataind_aicwf`: management
/// frames are consumed by the firmware control path, while station data is
/// stripped of its MAC/crypto/LLC headers before delivery.
fn decapsulate_data_frames(frame: &[u8], decryption_status: u8) -> Option<Vec<Vec<u8>>> {
    if frame.len() < 24 {
        return None;
    }
    let frame_control = u16::from_le_bytes([frame[0], frame[1]]);
    if (frame_control >> 2) & 0x3 != 2 {
        return None;
    }
    let to_ds = frame_control & 0x0100 != 0;
    let from_ds = frame_control & 0x0200 != 0;
    let qos = ((frame_control >> 4) & 0x0f) >= 8;
    let has_ht_control = frame_control & 0x8000 != 0;
    let address4_len = usize::from(to_ds && from_ds) * 6;
    let qos_offset = 24 + address4_len;
    let is_amsdu = qos && frame.get(qos_offset).is_some_and(|value| value & 0x80 != 0);
    let header_len = qos_offset + usize::from(qos) * 2 + usize::from(has_ht_control) * 4;
    let crypto_len = match decryption_status {
        0 => 0,
        1 => 4,
        2 | 3 => 8,
        7 => 18,
        // The data path intentionally supports only the cipher suites that
        // the firmware reports to this station driver.  Do not guess a
        // header length for newer/unsupported suites.
        _ => return None,
    };
    let payload = header_len.checked_add(crypto_len)?;
    if frame.len() < payload {
        return None;
    }

    if is_amsdu {
        return decapsulate_amsdu(&frame[payload..]);
    }

    let (destination, source) = match (to_ds, from_ds) {
        (false, false) => (&frame[4..10], &frame[10..16]),
        (true, false) => (&frame[16..22], &frame[10..16]),
        (false, true) => (&frame[4..10], &frame[16..22]),
        (true, true) => (&frame[16..22], &frame[24..30]),
    };
    ethernet_from_llc(destination, source, &frame[payload..]).map(|frame| vec![frame])
}

fn decapsulate_amsdu(aggregate: &[u8]) -> Option<Vec<Vec<u8>>> {
    let mut frames = Vec::new();
    let mut offset = 0usize;
    while offset < aggregate.len() {
        let header_end = offset.checked_add(14)?;
        if header_end > aggregate.len() {
            return None;
        }
        let msdu_len =
            u16::from_be_bytes([aggregate[offset + 12], aggregate[offset + 13]]) as usize;
        let end = header_end.checked_add(msdu_len)?;
        if msdu_len < 8 || end > aggregate.len() {
            return None;
        }
        frames.push(ethernet_from_llc(
            &aggregate[offset..offset + 6],
            &aggregate[offset + 6..offset + 12],
            &aggregate[header_end..end],
        )?);
        if end == aggregate.len() {
            break;
        }
        let subframe_len = 14usize.checked_add(msdu_len)?;
        let aligned_len = subframe_len.checked_add(3)? & !3;
        offset = offset.checked_add(aligned_len)?;
        if offset >= aggregate.len() {
            return None;
        }
    }
    (!frames.is_empty()).then_some(frames)
}

fn ethernet_from_llc(destination: &[u8], source: &[u8], llc: &[u8]) -> Option<Vec<u8>> {
    if destination.len() != 6
        || source.len() != 6
        || llc.len() < 8
        || llc[..6] != [0xaa, 0xaa, 0x03, 0, 0, 0]
    {
        return None;
    }
    let mut ethernet = Vec::with_capacity(12 + llc.len() - 6);
    ethernet.extend_from_slice(destination);
    ethernet.extend_from_slice(source);
    ethernet.extend_from_slice(&llc[6..]);
    Some(ethernet)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{
        common::{ChipVariant, SDIO_TYPE_CFG_CMD_RSP, SDIO_TYPE_CFG_PRINT, SDIO_TYPE_DATA},
        rx::{RX_BYTE_CAPACITY, RX_CAPACITY},
    };

    fn indication_fifo(message_id: u16, payload: &[u8]) -> Vec<u8> {
        let packet_len = 12 + payload.len();
        let mut fifo = vec![0; 4 + packet_len.div_ceil(4) * 4];
        fifo[..2].copy_from_slice(&(packet_len as u16).to_le_bytes());
        fifo[2] = SDIO_TYPE_CFG_CMD_RSP;
        fifo[4..6].copy_from_slice(&message_id.to_le_bytes());
        fifo[10..12].copy_from_slice(&(payload.len() as u16).to_le_bytes());
        fifo[16..16 + payload.len()].copy_from_slice(payload);
        fifo
    }

    fn data_fifo(marker: u8) -> Vec<u8> {
        const FRAME_LENGTH: usize = 24 + 6 + 2 + 1;
        const HARDWARE_HEADER: usize = 60;
        let mut fifo = vec![0; HARDWARE_HEADER + FRAME_LENGTH];
        fifo[..2].copy_from_slice(&(FRAME_LENGTH as u16).to_le_bytes());
        fifo[2] = SDIO_TYPE_DATA;
        let frame = &mut fifo[HARDWARE_HEADER..];
        // AP -> station data MPDU: address 1 is the Ethernet destination,
        // address 2 is the transmitter/source, followed by LLC/SNAP.
        frame[..2].copy_from_slice(&0x0208u16.to_le_bytes());
        frame[4] = marker;
        frame[5..10].copy_from_slice(&[0x10, 0x11, 0x12, 0x13, 0x14]);
        frame[10..16].copy_from_slice(&[0x20, 0x21, 0x22, 0x23, 0x24, 0x25]);
        frame[16..22].copy_from_slice(&[0x30, 0x31, 0x32, 0x33, 0x34, 0x35]);
        frame[24..30].copy_from_slice(&[0xaa, 0xaa, 0x03, 0, 0, 0]);
        frame[30..32].copy_from_slice(&[0x08, 0x00]);
        frame[32] = marker;
        fifo
    }

    fn data_mpdu(frame_control: u16, payload: &[u8]) -> Vec<u8> {
        let qos = ((frame_control >> 4) & 0x0f) >= 8;
        let header_len = 24 + usize::from(qos) * 2;
        let mut frame = vec![0; header_len + 8 + payload.len()];
        frame[..2].copy_from_slice(&frame_control.to_le_bytes());
        frame[4..10].copy_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        frame[10..16].copy_from_slice(&[0x11, 0x12, 0x13, 0x14, 0x15, 0x16]);
        frame[16..22].copy_from_slice(&[0x21, 0x22, 0x23, 0x24, 0x25, 0x26]);
        if qos {
            frame[24..26].copy_from_slice(&0u16.to_le_bytes());
        }
        frame[header_len..header_len + 6].copy_from_slice(&[0xaa, 0xaa, 0x03, 0, 0, 0]);
        frame[header_len + 6..header_len + 8].copy_from_slice(&[0x08, 0x00]);
        frame[header_len + 8..].copy_from_slice(payload);
        frame
    }

    #[test]
    fn startup_ignores_unowned_connect_results_and_keeps_mailbox_confirmation() {
        for status in [1u16, 0] {
            let mut device = AicDevice::new(ChipVariant::Aic8800DC).unwrap();
            device.start(MonotonicTime::default()).unwrap();
            device.lifecycle.mailbox =
                Some(super::super::mailbox::MailboxState::confirmation_for_test(
                    MonotonicTime::from_nanos(5_000_000_000),
                ));
            let mut payload = vec![0; 11];
            payload[..2].copy_from_slice(&status.to_le_bytes());
            let mut fifo = indication_fifo(SM_CONNECT_IND, &payload);
            fifo.extend(indication_fifo(2, &[]));
            device.io.pending = Some(PendingIo {
                id: 7,
                purpose: IoPurpose::ReceiveData(RxPath::Command),
            });
            let action = device.advance(AicInput {
                now: MonotonicTime::default(),
                event: Some(AicInputEvent::Sdio(SdioCompletion {
                    request_id: 7,
                    result: Ok(SdioResponse::Data(fifo)),
                })),
            });
            assert!(
                matches!(action, AicAction::SubmitSdio(_)),
                "unowned connection result stopped startup: {action:?}"
            );
            assert_eq!(device.state(), AicState::Starting);
            assert!(device.data.link.peer().is_none());
            assert_eq!(
                device.accept_mailbox_confirmation(2, Vec::new()),
                Err(AicError::CompletionMismatch)
            );
        }
    }

    #[test]
    fn active_connect_rejection_is_not_discarded_as_a_startup_indication() {
        let mut device = ready_transmitter(ChipVariant::Aic8800DC, 60);
        let mut control = super::super::control::build(
            ControlRequest::Connect {
                ssid: b"network".to_vec(),
                pmk: None,
                entropy: None,
            },
            [2, 0, 0, 0, 0, 1],
            Some(0),
        )
        .unwrap();
        if let super::super::control::ControlOperation::Connect(connect) = &mut control.operation {
            connect.phase = super::super::control::ConnectPhase::AwaitIndication;
        }
        control.commands.clear();
        device.lifecycle.control = Some(control);
        let mut payload = vec![0; 11];
        payload[0] = 1;
        assert_eq!(
            device.consume_receive_data(
                RxPath::Command,
                SdioResponse::Data(indication_fifo(SM_CONNECT_IND, &payload)),
            ),
            Err(AicError::FirmwareRejected {
                message_id: SM_CONNECT_IND,
                status: 1
            })
        );
    }

    #[test]
    fn successful_connect_indication_publishes_firmware_vif_and_station_indices() {
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.data.link.install_mac([2, 0, 0, 0, 0, 1]).unwrap();
        device.data.link.install_interface(2).unwrap();
        let mut control = super::super::control::build(
            ControlRequest::Connect {
                ssid: b"network".to_vec(),
                pmk: None,
                entropy: None,
            },
            [2, 0, 0, 0, 0, 1],
            Some(2),
        )
        .unwrap();
        if let super::super::control::ControlOperation::Connect(connect) = &mut control.operation {
            connect.phase = super::super::control::ConnectPhase::AwaitIndication;
        }
        control.commands.clear();
        device.lifecycle.control = Some(control);
        let mut payload = vec![0; 11];
        payload[2..8].copy_from_slice(&[2, 1, 2, 3, 4, 5]);
        payload[9] = 2;
        payload[10] = 7;
        let fifo = indication_fifo(SM_CONNECT_IND, &payload);

        device
            .consume_receive_data(RxPath::Command, SdioResponse::Data(fifo))
            .unwrap();

        assert_eq!(device.data.link.tx_indices(), Some((2, 7)));
    }

    #[test]
    fn asynchronous_disconnect_clears_the_learned_peer() {
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.data.link.install_mac([2, 0, 0, 0, 0, 1]).unwrap();
        device.data.link.install_interface(2).unwrap();
        device
            .data
            .link
            .install_peer(2, 7, [2, 1, 2, 3, 4, 5])
            .unwrap();
        let payload = [3, 0, 2, 0, 0, 0];

        device
            .consume_receive_data(
                RxPath::Command,
                SdioResponse::Data(indication_fifo(crate::lmac::SM_DISCONNECT_IND, &payload)),
            )
            .unwrap();

        assert_eq!(device.data.link.peer(), None);
    }

    #[test]
    fn mailbox_confirmation_survives_control_budget_exhaustion() {
        const PRINT_PACKET_LENGTH: usize = 8;
        const PRINT_AGGREGATE_LENGTH: usize = 4 + PRINT_PACKET_LENGTH;
        const RESPONSE_PACKET_LENGTH: usize = 12;
        const RESPONSE_AGGREGATE_LENGTH: usize = 4 + RESPONSE_PACKET_LENGTH;
        const EXPECTED_MESSAGE_ID: u16 = 2;

        let response_offset = crate::rx::CONTROL_RX_CAPACITY * PRINT_AGGREGATE_LENGTH;
        let mut fifo = vec![0; response_offset + RESPONSE_AGGREGATE_LENGTH];
        for index in 0..crate::rx::CONTROL_RX_CAPACITY {
            let offset = index * PRINT_AGGREGATE_LENGTH;
            fifo[offset..offset + 2].copy_from_slice(&(PRINT_PACKET_LENGTH as u16).to_le_bytes());
            fifo[offset + 2] = SDIO_TYPE_CFG_PRINT;
        }
        fifo[response_offset..response_offset + 2]
            .copy_from_slice(&(RESPONSE_PACKET_LENGTH as u16).to_le_bytes());
        fifo[response_offset + 2] = SDIO_TYPE_CFG_CMD_RSP;
        fifo[response_offset + 4..response_offset + 6]
            .copy_from_slice(&EXPECTED_MESSAGE_ID.to_le_bytes());

        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.mailbox = Some(MailboxState::confirmation_for_test(
            MonotonicTime::from_nanos(10),
        ));
        device.io.receive.active = true;

        device
            .consume_receive_data(RxPath::Command, SdioResponse::Data(fifo))
            .unwrap();

        assert_eq!(device.mailbox_confirmation_id(), None);
        assert!(!device.mailbox_waiting_for_receive());
    }

    #[test]
    fn receive_events_do_not_stall_after_the_first_bounded_window() {
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;

        for marker in 0..=RX_CAPACITY {
            device
                .consume_receive_data(RxPath::Command, SdioResponse::Data(data_fifo(marker as u8)))
                .unwrap();
            let event = device.data.pop_event();
            assert!(
                matches!(event, Some(AicEvent::Receive(frame)) if frame[0] == marker as u8),
                "receive event {marker} was lost after the bounded window"
            );
        }
    }

    #[test]
    fn control_completion_precedes_a_persistent_card_interrupt_scan() {
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.io.receive.active = true;
        device.data.push_event(AicEvent::ControlComplete).unwrap();

        assert!(matches!(
            device.drive_ready(MonotonicTime::from_nanos(0)),
            AicAction::Event(AicEvent::ControlComplete)
        ));
        assert!(device.io.receive.active);
    }

    #[test]
    fn idle_access_point_parks_on_a_timed_receive_poll() {
        use crate::device::owner::ApState;
        let mut device = AicDevice::new(ChipVariant::Aic8800DC).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.data.ap = Some(ApState::new(3));

        let now = MonotonicTime::from_nanos(1_000_000);
        let AicAction::WaitForInterruptUntil(deadline) = device.drive_ready(now) else {
            panic!("an idle access point must wake on a timer, not a bare interrupt");
        };
        // One bounded scan is armed for the next wake and the owner parks on
        // the poll deadline, releasing the lock instead of rescanning in a
        // tight loop that would starve control submissions.
        assert!(device.io.receive.active);
        assert!(deadline.as_nanos() > now.as_nanos());
    }

    #[test]
    fn transmit_completion_precedes_a_receive_backlog() {
        let mut device = AicDevice::new(ChipVariant::Aic8800DC).unwrap();
        device.lifecycle.state = AicState::Ready;
        for _ in 0..RX_CAPACITY {
            device.data.push_event(AicEvent::Receive(vec![0])).unwrap();
        }
        device
            .data
            .events
            .push_back(AicEvent::TransmitComplete(TxToken::new(1)));

        assert!(matches!(
            device.drive_ready(MonotonicTime::default()),
            AicAction::Event(AicEvent::TransmitComplete(token)) if token == TxToken::new(1)
        ));
    }

    #[test]
    fn receive_event_queue_obeys_item_and_byte_limits() {
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        for _ in 0..RX_CAPACITY {
            device
                .data
                .push_event(AicEvent::Receive(vec![0; 2048]))
                .unwrap();
        }
        device
            .data
            .push_event(AicEvent::Receive(vec![0; 2048]))
            .unwrap();

        assert_eq!(device.data.events.len(), RX_CAPACITY);
        assert_eq!(device.data.event_bytes, RX_BYTE_CAPACITY);

        device.data.push_event(AicEvent::ControlComplete).unwrap();
        assert_eq!(device.data.events.len(), RX_CAPACITY);
        assert!(
            device
                .data
                .events
                .iter()
                .any(|event| matches!(event, AicEvent::ControlComplete))
        );
    }

    #[test]
    fn management_frames_are_not_exposed_as_ethernet_events() {
        let management = vec![
            0x80, 0x00, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10,
        ];
        assert_eq!(decapsulate_data_frames(&management, 0), None);
    }

    fn mgmt_header(frame_control: u16, dst: [u8; 6], src: [u8; 6]) -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(&frame_control.to_le_bytes());
        frame.extend_from_slice(&[0, 0]); // duration
        frame.extend_from_slice(&dst);
        frame.extend_from_slice(&src);
        frame.extend_from_slice(&dst); // addr3 = BSS
        frame.extend_from_slice(&[0, 0]); // sequence control
        frame
    }

    #[test]
    fn ap_answers_authentication_and_association_without_a_peer() {
        use crate::{
            ap,
            device::owner::{ApState, InternalTxFrame},
        };

        let our_mac = [0x50, 0x31, 0x23, 0x23, 0x22, 0xf8];
        let client = [0x7c, 0x50, 0x79, 0x36, 0xf0, 0x15];
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.data.link.install_mac(our_mac).unwrap();
        device.data.ap = Some(ApState::new(3));
        // An access point never installs a station peer, so the transmit path
        // cannot resolve firmware indices through `tx_indices`.
        assert!(device.data.link.tx_indices().is_none());

        let mut auth = mgmt_header(0x00b0, our_mac, client);
        auth.extend_from_slice(&[0, 0, 1, 0, 0, 0]); // algorithm, transaction 1, status
        device.maybe_respond_ap_mgmt(&auth, 11);
        let response = device
            .data
            .internal_tx
            .pop_front()
            .expect("authentication request is answered");
        assert_eq!(response.interface_index, 3);
        assert!(matches!(
            response.frame,
            InternalTxFrame::Mgmt(frame)
                if frame == ap::auth_response(&auth).expect("valid request")
        ));

        // A request that is not addressed to this access point stays ignored.
        device.maybe_respond_ap_mgmt(&mgmt_header(0x00b0, [0; 6], client), 11);
        assert!(device.data.internal_tx.is_empty());

        let mut assoc = mgmt_header(0x0000, our_mac, client);
        assoc.extend_from_slice(&[0x21, 0x00, 0, 0]); // capability + listen interval
        device.maybe_respond_ap_mgmt(&assoc, 0);
        let response = device
            .data
            .internal_tx
            .pop_front()
            .expect("association request is answered");
        assert!(matches!(
            response.frame,
            InternalTxFrame::Mgmt(frame)
                if frame == ap::assoc_response(&assoc, 1).expect("valid request")
        ));

        // A management response must be transmitted even though no station
        // peer is associated, proving the AP transmit path bypasses
        // `tx_indices`.
        device.maybe_respond_ap_mgmt(&auth, 11);
        device.prepare_next_transmit();
        assert!(device.data.active_tx.is_some());
    }

    #[test]
    fn ap_association_queues_a_station_add_command_for_the_peer() {
        use crate::{
            device::owner::ApState,
            lmac::{ME_STA_ADD_REQ, TASK_ME},
        };

        let our_mac = [0x50, 0x31, 0x23, 0x23, 0x22, 0xf8];
        let client = [0x7c, 0x50, 0x79, 0x36, 0xf0, 0x15];
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.data.link.install_mac(our_mac).unwrap();
        device.data.ap = Some(ApState::new(3));

        let mut assoc = mgmt_header(0x0000, our_mac, client);
        assoc.extend_from_slice(&[0x21, 0x00, 0, 0]); // capability + listen interval
        assoc.extend_from_slice(&[1, 4, 0x82, 0x84, 0x8b, 0x96]); // supported rates
        device.maybe_respond_ap_mgmt(&assoc, 0);

        // The association response is still emitted as a management frame, and
        // registering the peer is queued on the control mailbox rather than
        // submitted straight from this receive callback.
        assert!(device.data.internal_tx.front().is_some());
        let control = device
            .lifecycle
            .control
            .as_ref()
            .expect("association opens a station-add control state");
        assert!(control.is_ap_station_add());
        let command = control.commands.front().expect("one queued command");
        assert_eq!(command.message_id, ME_STA_ADD_REQ);
        assert_eq!(command.destination, TASK_ME);
        let payload = &command.payload;
        assert_eq!(&payload[0..6], &client);
        assert_eq!(payload[6], 4); // supported-rate count
        assert_eq!(&payload[7..11], &[0x82, 0x84, 0x8b, 0x96]);
        assert_eq!(&payload[124..126], &1u16.to_le_bytes()); // allocated aid
        assert_eq!(payload[129], 3); // AP interface index

        // The peer is tracked so the pending confirmation can be attributed,
        // and its firmware station index becomes resolvable for transmit.
        let ap = device.data.ap.as_mut().unwrap();
        assert_eq!(ap.complete_station_add(5), Some(client));
        assert_eq!(ap.station_index_for(&client), Some(5));
        assert_eq!(ap.station_index_for(&[0, 0, 0, 0, 0, 0]), None);
    }

    #[test]
    fn ap_addresses_host_frames_to_the_registered_station() {
        use crate::device::owner::ApState;

        let ap_mac = [0x50, 0x31, 0x23, 0x23, 0x22, 0xf8];
        let client = [0x7c, 0x50, 0x79, 0x36, 0xf0, 0x15];

        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.data.link.install_mac(ap_mac).unwrap();
        let mut ap = ApState::new(3);
        ap.queue_station_add(client);
        assert_eq!(ap.complete_station_add(5), Some(client));
        device.data.ap = Some(ap);

        // A unicast frame for the associated client is sent on the AP VIF with
        // the firmware station index that client's `ME_STA_ADD` returned.
        let mut ethernet = client.to_vec();
        ethernet.extend_from_slice(&ap_mac);
        ethernet.extend_from_slice(&[0x08, 0x00]); // IPv4
        ethernet.extend_from_slice(&[0xab; 40]);
        device.data.tx.enqueue(TxToken::new(1), ethernet).unwrap();
        device.prepare_next_transmit();
        let wire = &device.data.active_tx.as_ref().unwrap().wire_frame;
        // `struct hostdesc` follows the 4-byte SDIO header: destination MAC at
        // offset 8, interface index at 24, station index at 25.
        assert_eq!(&wire[4 + 8..4 + 14], &client);
        assert_eq!(wire[4 + 24], 3);
        assert_eq!(wire[4 + 25], 5);

        // A broadcast frame (an ARP request, or the DHCP reply to a client
        // that has no address yet) has no single station.  It must go through
        // the firmware BC/MC station returned by the AP start confirmation; the
        // firmware has no data-transmit station at the "no station" marker, so
        // falling back to that marker would drop the frame instead of sending
        // it over the air.
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.data.link.install_mac(ap_mac).unwrap();
        let mut ap = ApState::new(3);
        ap.set_bcmc(8);
        device.data.ap = Some(ap);
        let mut broadcast = [0xff; 6].to_vec();
        broadcast.extend_from_slice(&ap_mac);
        broadcast.extend_from_slice(&[0x08, 0x06]); // ARP
        broadcast.extend_from_slice(&[0xab; 40]);
        device.data.tx.enqueue(TxToken::new(2), broadcast).unwrap();
        device.prepare_next_transmit();
        let wire = &device.data.active_tx.as_ref().unwrap().wire_frame;
        assert_eq!(wire[4 + 24], 3);
        assert_eq!(wire[4 + 25], 8);
    }

    #[test]
    fn ap_client_frames_ignore_a_stale_station_link() {
        use crate::device::owner::ApState;

        let ap_mac = [0x50, 0x31, 0x23, 0x23, 0x22, 0xf8];
        let client = [0x7c, 0x50, 0x79, 0x36, 0xf0, 0x15];

        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.data.link.install_mac(ap_mac).unwrap();
        // Simulate the STA->AP switch: the previous association is still
        // parked in `link`, so `tx_indices()` would otherwise hijack the reply.
        device.data.link.install_interface(1).unwrap();
        device.data.link.install_peer(1, 7, ap_mac).unwrap();
        assert_eq!(device.data.link.tx_indices(), Some((1, 7)));
        let mut ap = ApState::new(3);
        ap.queue_station_add(client);
        assert_eq!(ap.complete_station_add(5), Some(client));
        device.data.ap = Some(ap);

        let mut ethernet = client.to_vec();
        ethernet.extend_from_slice(&ap_mac);
        ethernet.extend_from_slice(&[0x08, 0x00]); // IPv4
        ethernet.extend_from_slice(&[0xab; 40]);
        device.data.tx.enqueue(TxToken::new(1), ethernet).unwrap();
        device.prepare_next_transmit();
        let wire = &device.data.active_tx.as_ref().unwrap().wire_frame;
        // The frame must go out on the AP VIF with the client's station index,
        // never the stale station link (interface 1, station 7).
        assert_eq!(&wire[4 + 8..4 + 14], &client);
        assert_eq!(wire[4 + 24], 3);
        assert_eq!(wire[4 + 25], 5);
    }

    #[test]
    fn qos_data_mpdu_is_decapsulated_to_ethernet() {
        let frame = data_mpdu(0x0288, &[1, 2, 3]);
        let [ethernet] = decapsulate_data_frames(&frame, 0)
            .expect("valid QoS data")
            .try_into()
            .expect("one MSDU");
        assert_eq!(&ethernet[..6], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(&ethernet[6..12], &[0x21, 0x22, 0x23, 0x24, 0x25, 0x26]);
        assert_eq!(&ethernet[12..], &[0x08, 0x00, 1, 2, 3]);
    }

    #[test]
    fn client_to_ds_frame_decapsulates_for_the_access_point() {
        // A client->access-point data MPDU sets ToDS only: the Ethernet
        // destination is address 3 and the source is address 2 (the client).
        let frame = data_mpdu(0x0188, &[1, 2, 3]);
        let [ethernet] = decapsulate_data_frames(&frame, 0)
            .expect("valid ToDS data")
            .try_into()
            .expect("one MSDU");
        assert_eq!(&ethernet[..6], &[0x21, 0x22, 0x23, 0x24, 0x25, 0x26]); // addr3 = DA
        assert_eq!(&ethernet[6..12], &[0x11, 0x12, 0x13, 0x14, 0x15, 0x16]); // addr2 = client
        assert_eq!(&ethernet[12..], &[0x08, 0x00, 1, 2, 3]);
    }

    #[test]
    fn qos_amsdu_subframes_are_decapsulated_to_ethernet() {
        let mut frame = data_mpdu(0x0288, &[]);
        frame.truncate(26);
        frame[24] = 0x80;
        frame.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        frame.extend_from_slice(&[0x21, 0x22, 0x23, 0x24, 0x25, 0x26]);
        frame.extend_from_slice(&11u16.to_be_bytes());
        frame.extend_from_slice(&[0xaa, 0xaa, 0x03, 0, 0, 0, 0x08, 0x00, 9, 8, 7]);
        frame.extend_from_slice(&[0; 3]);
        frame.extend_from_slice(&[6, 5, 4, 3, 2, 1]);
        frame.extend_from_slice(&[0x26, 0x25, 0x24, 0x23, 0x22, 0x21]);
        frame.extend_from_slice(&10u16.to_be_bytes());
        frame.extend_from_slice(&[0xaa, 0xaa, 0x03, 0, 0, 0, 0x86, 0xdd, 6, 5]);

        let ethernet = decapsulate_data_frames(&frame, 0).expect("valid A-MSDU subframes");
        assert_eq!(
            ethernet[0],
            [
                1, 2, 3, 4, 5, 6, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x08, 0x00, 9, 8, 7
            ]
        );
        assert_eq!(
            ethernet[1],
            [
                6, 5, 4, 3, 2, 1, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, 0x86, 0xdd, 6, 5
            ]
        );
    }

    #[test]
    fn encrypted_mpdu_skips_the_ccmp_header_before_llc() {
        let mut frame = data_mpdu(0x0208, &[9, 8]);
        let llc = 24;
        frame.splice(llc..llc, [0, 1, 2, 3, 4, 5, 6, 7]);
        let [ethernet] = decapsulate_data_frames(&frame, 3)
            .expect("valid CCMP data")
            .try_into()
            .expect("one MSDU");
        assert_eq!(&ethernet[12..], &[0x08, 0x00, 9, 8]);
    }

    #[test]
    fn single_function_profile_is_probed_once_per_card_interrupt() {
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.request_receive_scan();

        let Some(AicAction::SubmitSdio(count)) = device.drive_receive_scan() else {
            panic!("expected the shared command/data function count")
        };
        assert!(matches!(
            count.kind,
            SdioRequestKind::ReadByte { function, .. } if function.get() == 1
        ));
        device.io.pending = None;
        device
            .consume_receive_count(RxPath::Command, SdioResponse::Byte(0))
            .unwrap();

        assert!(device.drive_receive_scan().is_none());
        assert!(!device.io.receive.active);
    }

    #[test]
    fn dc_byte_mode_interrupt_reads_the_length_register_before_the_fifo() {
        let mut device = AicDevice::new(ChipVariant::Aic8800DC).unwrap();

        device
            .consume_receive_count(RxPath::Command, SdioResponse::Byte(64))
            .unwrap();

        assert!(matches!(
            device.io.next,
            Some((
                IoPurpose::ReceiveByteLength(RxPath::Command),
            SdioRequestKind::ReadByte { function, address },
        )) if function.get() == 2 && address.get() == 0x02
        ));
    }

    fn ready_transmitter(chip: ChipVariant, frame_len: usize) -> AicDevice {
        let mut device = AicDevice::new(chip).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.data.link.install_mac([2, 0, 0, 0, 0, 1]).unwrap();
        device.data.link.install_interface(0).unwrap();
        device
            .data
            .link
            .install_peer(0, 0, [2, 1, 2, 3, 4, 5])
            .unwrap();
        device
            .data
            .tx
            .enqueue(TxToken::new(1), vec![0; frame_len])
            .unwrap();
        device
    }

    #[test]
    fn transmit_backoff_services_card_irq_without_retrying_credits_early() {
        let mut device = ready_transmitter(ChipVariant::Aic8800D80, 1414);
        let now = MonotonicTime::default();
        let AicAction::SubmitSdio(flow) = device.advance(AicInput::tick(now)) else {
            panic!("expected credit read")
        };
        let wait = device.advance(complete(&flow, SdioResponse::Byte(0), now));
        let deadline = now.after(IO_RETRY);
        assert_eq!(wait, AicAction::WaitForInterruptUntil(deadline));
        let AicAction::SubmitSdio(rx) = device.advance(AicInput {
            now,
            event: Some(AicInputEvent::Irq(IrqSnapshot {
                sequence: 1,
                card_interrupt: true,
                transfer_complete: false,
                error: None,
            })),
        }) else {
            panic!("RX must run during TX backoff")
        };
        assert!(matches!(rx.kind, SdioRequestKind::ReadByte { address, .. }
            if address.get() == device.registers().block_count));
        assert_eq!(
            device.advance(complete(&rx, SdioResponse::Byte(0), now)),
            AicAction::WaitForInterruptUntil(deadline)
        );
        let AicAction::SubmitSdio(flow) = device.advance(AicInput::tick(deadline)) else {
            panic!("credit retry must resume at its original deadline")
        };
        let AicAction::SubmitSdio(write) =
            device.advance(complete(&flow, SdioResponse::Byte(128), deadline))
        else {
            panic!("D80 full-byte credit must permit transmission")
        };
        assert!(matches!(write.kind, SdioRequestKind::Write { .. }));
        assert_eq!(
            device.advance(complete(&write, SdioResponse::Unit, deadline)),
            AicAction::Event(AicEvent::TransmitComplete(TxToken::new(1)))
        );
    }

    #[test]
    fn dc_data_tx_checks_firmware_credits_before_writing() {
        let mut device = ready_transmitter(ChipVariant::Aic8800DC, 60);
        let AicAction::SubmitSdio(request) = device.advance(AicInput {
            now: MonotonicTime::default(),
            event: None,
        }) else {
            panic!("expected a DC data credit read")
        };
        assert!(matches!(request.kind,
            SdioRequestKind::ReadByte { function, address }
            if function.get() == 1 && address.get() == 0x0a));
    }

    #[test]
    fn data_tx_retains_packet_until_credit_reserve_is_available() {
        // One full-sized packet consumes one firmware buffer, not three
        // 512-byte SDIO blocks. Two buffers remain reserved for commands.
        for chip in [ChipVariant::Aic8800D80, ChipVariant::Aic8800DC] {
            let mut device = ready_transmitter(chip, 1414);
            let mut now = MonotonicTime::default();
            let mut action = device.advance(AicInput { now, event: None });
            for credits in [0, 1, 2, 3] {
                let AicAction::SubmitSdio(request) = action else {
                    panic!("expected a fresh credit read")
                };
                assert!(matches!(request.kind, SdioRequestKind::ReadByte { .. }));
                action = device.advance(AicInput {
                    now,
                    event: Some(AicInputEvent::Sdio(SdioCompletion {
                        request_id: request.id,
                        result: Ok(SdioResponse::Byte(credits)),
                    })),
                });
                if credits <= 2 {
                    let AicAction::WaitForInterruptUntil(deadline) = action else {
                        panic!("data TX must retain the packet while firmware buffers are reserved")
                    };
                    assert!(device.data.active_tx.is_some());
                    assert!(device.data.events.is_empty());
                    assert!(matches!(
                        device.advance(AicInput { now, event: None }),
                        AicAction::WaitForInterruptUntil(_)
                    ));
                    now = deadline;
                    action = device.advance(AicInput { now, event: None });
                }
            }
            let AicAction::SubmitSdio(write) = action else {
                panic!("three packet credits must permit one full-sized data frame")
            };
            assert!(
                matches!(&write.kind, SdioRequestKind::Write { bytes, .. } if bytes.len() == 1536)
            );
            let complete = device.advance(AicInput {
                now,
                event: Some(AicInputEvent::Sdio(SdioCompletion {
                    request_id: write.id,
                    result: Ok(SdioResponse::Unit),
                })),
            });
            assert!(
                matches!(complete, AicAction::Event(AicEvent::TransmitComplete(token)) if token == TxToken::new(1))
            );
            assert!(device.data.active_tx.is_none());
            assert!(matches!(
                device.advance(AicInput { now, event: None }),
                AicAction::WaitForInterrupt
            ));
            let next = device.advance(AicInput {
                now,
                event: Some(AicInputEvent::Tx {
                    token: TxToken::new(2),
                    frame: vec![0; 60],
                }),
            });
            assert!(
                matches!(
                    next,
                    AicAction::SubmitSdio(SdioRequest {
                        kind: SdioRequestKind::ReadByte { .. },
                        ..
                    })
                ),
                "each packet requires a fresh firmware credit check"
            );
        }
    }

    fn complete(request: &SdioRequest, response: SdioResponse, now: MonotonicTime) -> AicInput {
        AicInput {
            now,
            event: Some(AicInputEvent::Sdio(SdioCompletion {
                request_id: request.id,
                result: Ok(response),
            })),
        }
    }

    #[test]
    fn v3_other_interrupt_acknowledges_the_dev_to_host_soft_irq() {
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.request_receive_scan();

        let AicAction::SubmitSdio(count) =
            device.advance(AicInput::tick(MonotonicTime::from_nanos(0)))
        else {
            panic!("expected the receive count read")
        };
        assert!(matches!(
            count.kind,
            SdioRequestKind::ReadByte { function, address } if function.get() == 1
                && address.get() == device.registers().block_count
        ));

        let AicAction::SubmitSdio(ack) = device.advance(complete(
            &count,
            SdioResponse::Byte(0x83),
            MonotonicTime::from_nanos(1),
        )) else {
            panic!("expected the interrupt-pending ack read after an OTHER interrupt")
        };
        assert!(matches!(
            ack.kind,
            SdioRequestKind::ReadByte { function, address } if function.get() == 1
                && address.get() == device.registers().sleep_status.expect("v3 sleep status")
        ));

        let AicAction::SubmitSdio(clear) = device.advance(complete(
            &ack,
            SdioResponse::Byte(0x11),
            MonotonicTime::from_nanos(2),
        )) else {
            panic!("expected the soft-irq clear write after the pending read")
        };
        assert!(matches!(
            clear.kind,
            SdioRequestKind::WriteByte {
                function,
                address,
                value: 0x10,
                ..
            } if function.get() == 1
                && address.get() == device.registers().sleep_status.expect("v3 sleep status")
        ));

        let _ = device.advance(complete(
            &clear,
            SdioResponse::Byte(0x00),
            MonotonicTime::from_nanos(3),
        ));
    }

    #[test]
    fn v1_receive_counts_never_trigger_the_v3_other_interrupt_ack() {
        let mut device = AicDevice::new(ChipVariant::Aic8800DC).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.request_receive_scan();

        let AicAction::SubmitSdio(count) =
            device.advance(AicInput::tick(MonotonicTime::from_nanos(0)))
        else {
            panic!("expected the receive count read")
        };
        let AicAction::SubmitSdio(next) = device.advance(complete(
            &count,
            SdioResponse::Byte(0x83),
            MonotonicTime::from_nanos(1),
        )) else {
            panic!("expected the byte-mode length read")
        };
        assert!(matches!(
            next.kind,
            SdioRequestKind::ReadByte { address, .. } if address.get() == device.registers().byte_mode_length
        ));
    }

    #[test]
    fn v3_other_ack_re_reads_the_same_path_count_until_empty() {
        let mut device = AicDevice::new(ChipVariant::Aic8800D80).unwrap();
        device.lifecycle.state = AicState::Ready;
        device.request_receive_scan();

        let AicAction::SubmitSdio(count) =
            device.advance(AicInput::tick(MonotonicTime::from_nanos(0)))
        else {
            panic!("expected the receive count read")
        };
        let AicAction::SubmitSdio(ack) = device.advance(complete(
            &count,
            SdioResponse::Byte(0x83),
            MonotonicTime::from_nanos(1),
        )) else {
            panic!("expected the interrupt-pending ack read")
        };
        let AicAction::SubmitSdio(clear) = device.advance(complete(
            &ack,
            SdioResponse::Byte(0x11),
            MonotonicTime::from_nanos(2),
        )) else {
            panic!("expected the soft-irq clear write")
        };
        // The vendor D80 handler re-reads the interrupt status after the soft
        // IRQ acknowledgement; the scan must stay on the same path instead of
        // advancing so queued data is drained before the scan ends.
        let AicAction::SubmitSdio(recount) = device.advance(complete(
            &clear,
            SdioResponse::Byte(0x00),
            MonotonicTime::from_nanos(3),
        )) else {
            panic!("expected the same-path count re-read after the soft IRQ acknowledgement")
        };
        assert!(matches!(
            recount.kind,
            SdioRequestKind::ReadByte { address, .. } if address.get()
                == device.registers().block_count
        ));
    }
}
