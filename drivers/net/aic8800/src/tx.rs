//! Single-owner transmit progression.

use alloc::{collections::VecDeque, vec::Vec};

use crate::{TxToken, protocol::ethernet_tx_frame};

pub(crate) const TX_CAPACITY: usize = 128;

pub(crate) struct PendingTx {
    pub token: TxToken,
    pub frame: Vec<u8>,
}

pub(crate) struct TxState {
    queue: VecDeque<PendingTx>,
}

impl TxState {
    pub(crate) const fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    pub(crate) fn enqueue(&mut self, token: TxToken, frame: Vec<u8>) -> Result<(), Vec<u8>> {
        if self.queue.len() >= TX_CAPACITY {
            return Err(frame);
        }
        self.queue.push_back(PendingTx { token, frame });
        Ok(())
    }

    pub(crate) fn take_wire_frame(
        &mut self,
        interface_index: u8,
        station_index: u8,
        v3: bool,
    ) -> Option<Result<(TxToken, Vec<u8>), TxToken>> {
        let pending = self.queue.pop_front()?;
        Some(
            ethernet_tx_frame(&pending.frame, interface_index, station_index, v3)
                .map(|frame| (pending.token, frame))
                .map_err(|_| pending.token),
        )
    }

    /// Returns the destination MAC of the frame at the head of the queue, or
    /// [`None`] when the queue is empty or its head is too short to carry one.
    /// An access point uses this to resolve the per-destination station index
    /// before it builds the frame, since it has no single associated peer.
    pub(crate) fn front_destination(&self) -> Option<[u8; 6]> {
        let frame = &self.queue.front()?.frame;
        let dest: [u8; 6] = frame.get(..6)?.try_into().expect("six-byte slice converts");
        Some(dest)
    }

    pub(crate) fn drain_tokens(&mut self) -> impl Iterator<Item = TxToken> + '_ {
        self.queue.drain(..).map(|pending| pending.token)
    }
}
