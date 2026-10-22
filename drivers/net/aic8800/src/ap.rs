//! Construction of the 802.11 management-frame responses an access point owes
//! its clients.
//!
//! AIC8800 runs as a fullmac device, but its firmware only forwards client
//! authentication and association requests to the host (mirroring the vendor
//! `cfg80211_rx_mgmt` path); the entity that answers them is the host-side
//! access-point state machine.  StarryOS has no `hostapd`, so the driver plays
//! that role for open networks.  The builders here turn a received request
//! frame into the raw 802.11 response frame that is submitted through the
//! management TX path.  They append no frame check sequence: as in the vendor
//! `rwnx_tx_mgmt` path the hardware generates it.

use alloc::vec::Vec;

/// Capability information advertised by this open access point.  It matches
/// the value placed in the beacon so association responses stay consistent
/// with what clients already saw while scanning.
pub(crate) const AP_CAPABILITY: u16 = 0x0021;

/// Body offsets inside the fixed 24-byte 802.11 management header region that
/// the response builders share: the client address (`addr2` of the request),
/// the address the request was sent to (`addr1`, i.e. this AP) and the
/// sequence-control field.
const ADDR1_OFFSET: usize = 4;
const ADDR2_OFFSET: usize = 10;
const SEQ_CONTROL_OFFSET: usize = 22;
const BODY_OFFSET: usize = 24;

fn frame_control(subtype: u8) -> [u8; 2] {
    // Management frames have `type` 0; only the subtype varies and the
    // protocol-to-data / retry flags are left clear for a fresh response.
    (u16::from(subtype) << 4).to_le_bytes()
}

/// Builds an Open System authentication response (`transaction` 2, status
/// success) for the received authentication request.
///
/// Returns [`None`] when the frame is too short or is not an authentication
/// request (i.e. its transaction sequence is not 1), so retransmissions of the
/// AP's own response can never be echoed back.
pub(crate) fn auth_response(request: &[u8]) -> Option<Vec<u8>> {
    let body = request.get(BODY_OFFSET..BODY_OFFSET + 6)?;
    let algorithm = u16::from_le_bytes([body[0], body[1]]);
    let transaction = u16::from_le_bytes([body[2], body[3]]);
    if transaction != 1 {
        return None;
    }
    let mut frame = Vec::with_capacity(BODY_OFFSET + 6);
    frame.extend_from_slice(&frame_control(11));
    frame.extend_from_slice(&[0, 0]); // duration: patched by the hardware
    frame.extend_from_slice(request.get(ADDR2_OFFSET..ADDR2_OFFSET + 6)?); // addr1 = client
    frame.extend_from_slice(request.get(ADDR1_OFFSET..ADDR1_OFFSET + 6)?); // addr2 = this AP
    frame.extend_from_slice(request.get(ADDR1_OFFSET..ADDR1_OFFSET + 6)?); // addr3 = this AP
    frame.extend_from_slice(request.get(SEQ_CONTROL_OFFSET..SEQ_CONTROL_OFFSET + 2)?);
    frame.extend_from_slice(&algorithm.to_le_bytes());
    frame.extend_from_slice(&2u16.to_le_bytes()); // transaction 2 (response)
    frame.extend_from_slice(&0u16.to_le_bytes()); // status: success
    Some(frame)
}

/// Builds an association response (status success, allocated AID) for the
/// received association request.
///
/// The AID carries the high "association id" bit required by 802.11 and the
/// supported-rate element matching the beacon, which is all an open-network
/// client needs to leave the association state.
pub(crate) fn assoc_response(request: &[u8], aid: u16) -> Option<Vec<u8>> {
    let mut frame = Vec::with_capacity(BODY_OFFSET + 6 + 10);
    frame.extend_from_slice(&frame_control(1)); // subtype 1: association response
    frame.extend_from_slice(&[0, 0]); // duration
    frame.extend_from_slice(request.get(ADDR2_OFFSET..ADDR2_OFFSET + 6)?); // addr1 = client
    frame.extend_from_slice(request.get(ADDR1_OFFSET..ADDR1_OFFSET + 6)?); // addr2 = this AP
    frame.extend_from_slice(request.get(ADDR1_OFFSET..ADDR1_OFFSET + 6)?); // addr3 = this AP
    frame.extend_from_slice(request.get(SEQ_CONTROL_OFFSET..SEQ_CONTROL_OFFSET + 2)?);
    frame.extend_from_slice(&AP_CAPABILITY.to_le_bytes());
    frame.extend_from_slice(&0u16.to_le_bytes()); // status: success
    frame.extend_from_slice(&(aid | 0x8000).to_le_bytes()); // AID with high bit set
    frame.extend_from_slice(&[1, 8, 0x82, 0x84, 0x8b, 0x96, 0x0c, 0x12, 0x18, 0x24]);
    Some(frame)
}

/// Extracts the legacy supported-rate set (information element tag 1) from an
/// association request.  The fixed body of an association request is four
/// bytes (capability information and listen interval), so the element area
/// begins at [`BODY_OFFSET`] + 4.  Returns an empty vector when the request is
/// truncated or advertises no tag-1 element; the caller treats an empty set as
/// "no legacy rates", which the firmware tolerates for a legacy station.
pub(crate) fn assoc_supported_rates(request: &[u8]) -> Vec<u8> {
    let Some(elements) = request.get(BODY_OFFSET + 4..) else {
        return Vec::new();
    };
    let mut offset = 0;
    while offset + 2 <= elements.len() {
        let tag = elements[offset];
        let length = elements[offset + 1] as usize;
        let value = offset + 2..offset + 2 + length;
        let Some(values) = elements.get(value) else {
            break;
        };
        if tag == 1 {
            return values.to_vec();
        }
        offset += 2 + length;
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn request_prefix(dst: [u8; 6], src: [u8; 6]) -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(&[0x00, 0x00]); // frame control (caller fixes subtype)
        frame.extend_from_slice(&[0, 0]); // duration
        frame.extend_from_slice(&dst);
        frame.extend_from_slice(&src);
        frame.extend_from_slice(&dst); // addr3
        frame.extend_from_slice(&[0x10, 0x00]); // sequence control
        frame
    }

    #[test]
    fn auth_response_targets_the_client_with_transaction_two() {
        let ap = [0x50, 0x31, 0x23, 0x23, 0x22, 0xf8];
        let client = [0x7c, 0x50, 0x79, 0x36, 0xf0, 0x15];
        let mut request = request_prefix(ap, client);
        request.extend_from_slice(&0u16.to_le_bytes()); // algorithm: open
        request.extend_from_slice(&1u16.to_le_bytes()); // transaction: 1
        request.extend_from_slice(&0u16.to_le_bytes()); // status: success

        let response = auth_response(&request).expect("transaction 1 is a request");
        assert_eq!(&response[0..2], &[0xb0, 0x00]); // subtype 11
        assert_eq!(&response[4..10], &client); // addr1 = client
        assert_eq!(&response[10..16], &ap); // addr2 = AP
        assert_eq!(&response[24..26], &0u16.to_le_bytes()); // echo algorithm
        assert_eq!(&response[26..28], &2u16.to_le_bytes()); // transaction 2
        assert_eq!(&response[28..30], &0u16.to_le_bytes()); // status success
        assert_eq!(response.len(), 30);
    }

    #[test]
    fn auth_response_ignores_a_response_frame() {
        let mut already_response = request_prefix([1, 0, 0, 0, 0, 2], [2, 0, 0, 0, 0, 1]);
        already_response.extend_from_slice(&0u16.to_le_bytes());
        already_response.extend_from_slice(&2u16.to_le_bytes()); // transaction 2
        already_response.extend_from_slice(&0u16.to_le_bytes());
        assert!(auth_response(&already_response).is_none());
    }

    #[test]
    fn assoc_response_sets_aid_and_capability() {
        let ap = [0x50, 0x31, 0x23, 0x23, 0x22, 0xf8];
        let client = [0x7c, 0x50, 0x79, 0x36, 0xf0, 0x15];
        let mut request = request_prefix(ap, client);
        request.extend_from_slice(&[0x21, 0x00]); // capability
        request.extend_from_slice(&[0, 0, 0, 0]); // listen interval + rates

        let response = assoc_response(&request, 3).expect("valid association request");
        assert_eq!(&response[0..2], &[0x10, 0x00]); // subtype 1
        assert_eq!(&response[4..10], &client); // addr1 = client
        assert_eq!(&response[10..16], &ap); // addr2 = AP
        assert_eq!(&response[24..26], &AP_CAPABILITY.to_le_bytes());
        assert_eq!(&response[26..28], &0u16.to_le_bytes()); // status success
        assert_eq!(&response[28..30], &0x8003u16.to_le_bytes()); // AID 3 with high bit
        assert_eq!(&response[30..32], &[1, 8]); // supported rates element
    }

    #[test]
    fn supported_rates_are_read_from_the_element_area() {
        let mut request = request_prefix([0x50, 0, 0, 0, 0, 2], [0x7c, 0, 0, 0, 0, 1]);
        request.extend_from_slice(&[0x21, 0x00, 0x00, 0x00]); // capability + listen interval
        request.extend_from_slice(&[0x30, 0x06, 0x00, 0x00, 0x0f, 0xac, 0x00, 0x00]); // a preceding RSN element (len 6)
        request.extend_from_slice(&[1, 4, 0x82, 0x84, 0x8b, 0x96]); // supported rates
        assert_eq!(
            assoc_supported_rates(&request),
            vec![0x82, 0x84, 0x8b, 0x96]
        );

        // No tag-1 element yields an empty set rather than a failure.
        let mut bare = request_prefix([0x50, 0, 0, 0, 0, 2], [0x7c, 0, 0, 0, 0, 1]);
        bare.extend_from_slice(&[0x21, 0x00, 0x00, 0x00]);
        assert!(assoc_supported_rates(&bare).is_empty());
    }
}
