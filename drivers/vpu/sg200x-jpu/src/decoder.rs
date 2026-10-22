//! Synchronous SG200x JPU decode orchestration and DMA ownership.

use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicBool, Ordering},
};

use dma_api::{CpuDmaBuffer, DeviceDma, DmaDirection, InFlightDma};

use super::{
    engine::{
        BBC_STREAM_PAGE_SIZE, GRAM_PREFETCH_PAGES, HardwareDecodeInfo, PollError,
        checked_dma_offset, checked_dma_region, checked_frame_dma_addresses, configure_stream_regs,
        gram_setup, poll_decode_done, start_decode, upload_huff_tables, upload_quant_tables,
    },
    error::{
        JpegHeaderError, JpuBufferError, JpuCreateError, JpuDecodeError, JpuDmaAddressError,
        JpuInspectError,
    },
    header::{JpegHeaderInfo, parse_jpeg_header},
    layout::{FrameLayout, FrameLayoutError, JpuPixelFormat, JpuScale, PlaneLayout},
    regs::{hardware_init_at, wait_bbc_idle_at},
};

const DMA_ALIGNMENT: usize = 16 * 1024;

static JPU_IN_USE: AtomicBool = AtomicBool::new(false);

/// Caller-mapped MMIO bases required by the SG200x JPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JpuMmio {
    /// JPU register block.
    pub jpu_base: usize,
    /// TOP clock/reset register block.
    pub top_base: usize,
    /// Video-codec control register block.
    pub vc_base: usize,
}

impl JpuMmio {
    /// Creates one set of caller-mapped JPU register bases.
    pub const fn new(jpu_base: usize, top_base: usize, vc_base: usize) -> Self {
        Self {
            jpu_base,
            top_base,
            vc_base,
        }
    }
}

struct JpegInspection {
    header: JpegHeaderInfo,
    layout: FrameLayout,
}

fn inspect_jpeg(jpeg_data: &[u8], scale: JpuScale) -> Result<JpegInspection, JpuInspectError> {
    if jpeg_data.is_empty() {
        return Err(JpuInspectError::EmptyStream);
    }
    if !jpeg_data.starts_with(&[0xff, 0xd8]) {
        return Err(JpegHeaderError::MissingSoi.into());
    }

    let header = parse_jpeg_header(jpeg_data)?;
    let entropy = jpeg_data
        .get(header.ecs_offset..)
        .ok_or(JpegHeaderError::EcsOffsetOutOfBounds)?;
    let eoi_offset = entropy.windows(2).position(|marker| marker == [0xff, 0xd9]);
    if !matches!(eoi_offset, Some(offset) if offset > 0) {
        return Err(JpegHeaderError::MissingEntropyDataAndEoi.into());
    }
    let format = JpuPixelFormat::from_raw(header.format)?;
    let layout = FrameLayout::new(header.width, header.height, format, scale)?;
    Ok(JpegInspection { header, layout })
}

/// Inspects a baseline JPEG and calculates its checked JPU output layout.
///
/// This function parses only CPU-visible bytes. It does not acquire, initialize,
/// or access JPU hardware and does not allocate DMA buffers.
pub fn inspect_jpeg_layout(
    jpeg_data: &[u8],
    scale: JpuScale,
) -> Result<FrameLayout, JpuInspectError> {
    Ok(inspect_jpeg(jpeg_data, scale)?.layout)
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Native,
    Yuv420,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PollDisposition {
    Complete,
    /// The engine reached a terminal failure; the decoder resets it before reuse.
    Recover(PollError),
}

const fn poll_disposition(result: Result<(), PollError>) -> PollDisposition {
    match result {
        Ok(()) => PollDisposition::Complete,
        Err(error) => PollDisposition::Recover(error),
    }
}

/// How many recent JPEG sizes are kept for failure diagnostics.
const JPEG_LEN_HISTORY: usize = 4;

/// Sizes of the most recently submitted JPEGs, oldest first.
///
/// A decode that stalls hundreds of frames into a stream reports the stalled
/// frame's length together with this window: a frame much shorter than its
/// neighbours points at truncated input rather than a decoder-side fault.
#[derive(Default)]
struct JpegLenHistory {
    lengths: [u32; JPEG_LEN_HISTORY],
    len: usize,
}

impl JpegLenHistory {
    fn record(&mut self, jpeg_len: usize) {
        let jpeg_len = u32::try_from(jpeg_len).unwrap_or(u32::MAX);
        if self.len < JPEG_LEN_HISTORY {
            self.lengths[self.len] = jpeg_len;
            self.len += 1;
        } else {
            self.lengths.rotate_left(1);
            self.lengths[JPEG_LEN_HISTORY - 1] = jpeg_len;
        }
    }

    fn entries(&self) -> &[u32] {
        &self.lengths[..self.len]
    }
}

/// A borrowed planar frame produced by the JPU.
///
/// The borrow prevents the decoder from starting another operation while the
/// returned DMA buffer is still in use.
///
/// ```compile_fail
/// use sg200x_jpu::JpuDecoder;
///
/// fn cannot_decode_twice(decoder: &mut JpuDecoder, jpeg: &[u8]) {
///     let first = decoder.decode(jpeg).unwrap();
///     let _second = decoder.decode(jpeg).unwrap();
///     let _still_borrowed = first.yuv_data;
/// }
/// ```
#[non_exhaustive]
#[derive(Debug)]
pub struct DecodeResult<'a> {
    /// Meaningful output width after scaling, excluding coded padding.
    pub width: u32,
    /// Meaningful output height after scaling, excluding coded padding.
    pub height: u32,
    /// CPU-visible frame bytes. Plane offsets and strides are in [`Self::layout`].
    pub yuv_data: &'a [u8],
    /// JPU DMA address corresponding to `yuv_data[0]`, for frame metadata.
    ///
    /// The buffer is CPU-owned on return. CPU postprocessing is not flushed
    /// for device access on non-coherent platforms. This address grants neither
    /// a userspace mapping nor permission to submit the buffer to another device;
    /// consumers must copy `yuv_data` into their own DMA-managed buffer.
    pub yuv_dma_addr: u32,
    /// Planar format, scale, extents, offsets, and strides.
    pub layout: FrameLayout,
}

/// Singleton synchronous SG200x JPU decoder.
pub struct JpuDecoder {
    mmio: JpuMmio,
    dma: DeviceDma,
    stream_buffer: Option<CpuDmaBuffer>,
    frame_buffer: Option<CpuDmaBuffer>,
    completed_frame_len: Option<usize>,
    jpeg_len_history: JpegLenHistory,
    poisoned: bool,
}

impl JpuDecoder {
    /// Acquires and initializes the SG200x JPU.
    ///
    /// # Safety
    ///
    /// Every base in `mmio` must be four-byte aligned and remain a valid device
    /// mapping for the decoder's lifetime. The mappings must cover at least the
    /// JPU register range through offset `0x237`, the TOP range through offset
    /// `0x3003`, and four bytes at the VC base. No code outside this decoder may
    /// access or reconfigure the JPU while it is owned. `dma` must allocate
    /// memory visible to this JPU and implement the cache-maintenance contract
    /// required by `dma-api`.
    pub unsafe fn new(mmio: JpuMmio, dma: DeviceDma) -> Result<Self, JpuCreateError> {
        JPU_IN_USE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| JpuCreateError::AlreadyOwned)?;

        if let Err(error) = hardware_init_at(mmio.jpu_base, mmio.top_base, mmio.vc_base) {
            JPU_IN_USE.store(false, Ordering::Release);
            return Err(JpuCreateError::Initialization(error));
        }
        Ok(Self {
            mmio,
            dma,
            stream_buffer: None,
            frame_buffer: None,
            completed_frame_len: None,
            jpeg_len_history: JpegLenHistory::default(),
            poisoned: false,
        })
    }

    /// Decodes at the full coded resolution.
    pub fn decode<'a>(&'a mut self, jpeg_data: &[u8]) -> Result<DecodeResult<'a>, JpuDecodeError> {
        self.decode_scaled(jpeg_data, JpuScale::Full)
    }

    /// Decodes a baseline JPEG using one isotropic hardware downscale mode.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid JPEG data, unsupported layouts, DMA
    /// allocation/address failures, hardware decode errors, and timeouts. A
    /// hardware decode error or timeout resets the JPU and reuses its DMA
    /// buffers when that reset completes; only a failed reset poisons the
    /// decoder, after which subsequent calls return
    /// [`JpuDecodeError::Poisoned`].
    pub fn decode_scaled<'a>(
        &'a mut self,
        jpeg_data: &[u8],
        scale: JpuScale,
    ) -> Result<DecodeResult<'a>, JpuDecodeError> {
        self.decode_scaled_output(jpeg_data, scale, OutputFormat::Native)
    }

    /// Decodes a color JPEG and converts its chroma to planar YUV420 in place.
    ///
    /// Uses a rounded box average over the additional sampling axis or axes,
    /// replicating the last visible sample at odd edges. Luma is unchanged.
    /// Conversion runs only after DMA completion and needs no extra frame buffer.
    /// The result is CPU-visible; see [`DecodeResult::yuv_dma_addr`] for the
    /// address and device-access contract.
    /// Grayscale is rejected before hardware submission. Other errors and
    /// poisoning semantics are the same as [`Self::decode_scaled`].
    pub fn decode_scaled_yuv420<'a>(
        &'a mut self,
        jpeg_data: &[u8],
        scale: JpuScale,
    ) -> Result<DecodeResult<'a>, JpuDecodeError> {
        self.decode_scaled_output(jpeg_data, scale, OutputFormat::Yuv420)
    }

    fn decode_scaled_output<'a>(
        &'a mut self,
        jpeg_data: &[u8],
        scale: JpuScale,
        output: OutputFormat,
    ) -> Result<DecodeResult<'a>, JpuDecodeError> {
        self.validate_decoder_ready()?;

        let JpegInspection { header, layout } = inspect_jpeg(jpeg_data, scale)?;
        let output_layout = match output {
            OutputFormat::Native => layout,
            OutputFormat::Yuv420 => layout.yuv420_layout()?,
        };
        let format = layout.format;
        self.jpeg_len_history.record(jpeg_data.len());
        self.completed_frame_len = None;
        let stream_len = required_stream_capacity(jpeg_data.len(), header.ecs_offset)?;
        let stream_end = declared_stream_end(jpeg_data.len())?;
        validate_dma_allocation_len(stream_len)?;
        validate_dma_allocation_len(layout.total_len)?;
        self.ensure_buffers(stream_len, layout.total_len)?;
        self.write_stream(jpeg_data, stream_len)?;

        let stream_dma = self.stream_dma_region(stream_len)?;
        let frame_dma = self.frame_dma_region(layout.total_len)?;
        let stream_data_end = checked_dma_offset(stream_dma, stream_end, true)?;
        let frame_planes = checked_frame_dma_addresses(frame_dma, &layout)?;
        let hardware = HardwareDecodeInfo::for_format(format);

        configure_stream_regs(
            self.mmio.jpu_base,
            stream_dma,
            stream_data_end,
            stream_end,
            &header,
            &layout,
            hardware,
        );
        upload_huff_tables(self.mmio.jpu_base, &header);
        upload_quant_tables(self.mmio.jpu_base, &header);

        let (stream_in_flight, frame_in_flight) = self.begin_dma()?;
        if let Err(error) = gram_setup(self.mmio.jpu_base, stream_dma, &header) {
            let error = JpuDecodeError::from(error);
            return Err(self.recover_after_failure(stream_in_flight, frame_in_flight, error));
        }
        start_decode(self.mmio.jpu_base, frame_planes, &header, &layout);

        match poll_disposition(poll_decode_done(self.mmio.jpu_base)) {
            PollDisposition::Complete => {
                self.complete_dma(stream_in_flight, frame_in_flight);
            }
            PollDisposition::Recover(error) => {
                log::warn!(
                    "[JPU] decode failure after jpeg_len={} (recent oldest..newest: {:?})",
                    jpeg_data.len(),
                    self.jpeg_len_history.entries()
                );
                let error = JpuDecodeError::from(error);
                return Err(self.recover_after_failure(stream_in_flight, frame_in_flight, error));
            }
        }

        self.clear_frame_padding(&layout)?;
        if matches!(output, OutputFormat::Yuv420) {
            let frame = self
                .frame_buffer
                .as_mut()
                .ok_or(JpuBufferError::MissingCompletedFrameBuffer)?;
            // SAFETY: DONE was acknowledged and complete_dma returned CPU ownership.
            // The exclusive decoder borrow prevents concurrent decode or frame reads.
            let bytes = unsafe { frame.as_mut_slice_cpu() };
            convert_yuv420(bytes, &layout)?;
        }
        let layout = output_layout;
        self.completed_frame_len = Some(layout.total_len);
        let frame = self
            .frame_buffer
            .as_ref()
            .ok_or(JpuBufferError::MissingCompletedFrameBuffer)?;
        let yuv_data = frame
            .as_slice_cpu()
            .get(..layout.total_len)
            .ok_or(JpuBufferError::FrameViewExceedsAllocation)?;

        Ok(DecodeResult {
            width: layout.visible.width,
            height: layout.visible.height,
            yuv_data,
            yuv_dma_addr: frame_dma.start,
            layout,
        })
    }

    /// Copies bytes from the most recently completed frame.
    ///
    /// `frame_len` must come from that decode's [`DecodeResult::layout`]. The
    /// method exists for device ABIs that return frame metadata separately
    /// from a later `read`; it never exposes bytes outside the logical frame.
    pub fn copy_completed_frame(
        &self,
        frame_len: usize,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<usize, JpuDecodeError> {
        self.validate_decoder_ready()?;
        if self.completed_frame_len != Some(frame_len) {
            return Err(JpuBufferError::CompletedFrameMismatch.into());
        }
        let frame = self
            .frame_buffer
            .as_ref()
            .ok_or(JpuBufferError::MissingCompletedFrameBuffer)?;
        Ok(copy_frame_range(
            frame.as_slice_cpu(),
            frame_len,
            offset,
            destination,
        )?)
    }

    fn validate_decoder_ready(&self) -> Result<(), JpuDecodeError> {
        if self.poisoned {
            return Err(JpuDecodeError::Poisoned);
        }
        Ok(())
    }

    fn ensure_buffers(
        &mut self,
        stream_len: usize,
        frame_len: usize,
    ) -> Result<(), JpuDecodeError> {
        let stream_capacity = self
            .stream_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.len().get());
        let frame_capacity = self
            .frame_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.len().get());
        if buffer_plan(stream_capacity, frame_capacity, stream_len, frame_len) == BufferPlan::Reuse
        {
            return Ok(());
        }

        let stream_len =
            NonZeroUsize::new(stream_len).ok_or(JpuBufferError::ZeroSizedStreamBuffer)?;
        let frame_len = NonZeroUsize::new(frame_len).ok_or(JpuBufferError::ZeroSizedFrameBuffer)?;
        let stream =
            CpuDmaBuffer::new_zero(&self.dma, stream_len, DMA_ALIGNMENT, DmaDirection::ToDevice)?;
        let frame = CpuDmaBuffer::new_zero(
            &self.dma,
            frame_len,
            DMA_ALIGNMENT,
            DmaDirection::FromDevice,
        )?;
        self.stream_buffer = Some(stream);
        self.frame_buffer = Some(frame);
        Ok(())
    }

    fn write_stream(&mut self, jpeg_data: &[u8], stream_len: usize) -> Result<(), JpuDecodeError> {
        let stream = self
            .stream_buffer
            .as_mut()
            .ok_or(JpuBufferError::MissingStreamBuffer)?;
        if stream_len > stream.len().get() || jpeg_data.len() > stream_len {
            return Err(JpuBufferError::StreamDataExceedsAllocation.into());
        }

        // SAFETY: the buffer is CPU-owned until begin_dma consumes it. The
        // checked range stays inside this allocation.
        let bytes = unsafe { stream.as_mut_slice_cpu() };
        bytes[..jpeg_data.len()].copy_from_slice(jpeg_data);
        bytes[jpeg_data.len()..stream_len].fill(0);
        Ok(())
    }

    fn stream_dma_region(
        &self,
        stream_len: usize,
    ) -> Result<super::engine::DmaRegion, JpuDecodeError> {
        let stream = self
            .stream_buffer
            .as_ref()
            .ok_or(JpuBufferError::MissingStreamBuffer)?;
        Ok(checked_dma_region(
            stream.dma_addr().as_u64(),
            stream.len().get(),
            stream_len,
        )?)
    }

    fn frame_dma_region(
        &self,
        frame_len: usize,
    ) -> Result<super::engine::DmaRegion, JpuDecodeError> {
        let frame = self
            .frame_buffer
            .as_ref()
            .ok_or(JpuBufferError::MissingFrameBuffer)?;
        Ok(checked_dma_region(
            frame.dma_addr().as_u64(),
            frame.len().get(),
            frame_len,
        )?)
    }

    fn begin_dma(&mut self) -> Result<(InFlightDma, InFlightDma), JpuDecodeError> {
        let stream = self
            .stream_buffer
            .take()
            .ok_or(JpuBufferError::MissingStreamBuffer)?;
        let frame = match self.frame_buffer.take() {
            Some(frame) => frame,
            None => {
                self.stream_buffer = Some(stream);
                return Err(JpuBufferError::MissingFrameBuffer.into());
            }
        };

        let stream = stream.prepare_for_device();
        let frame = frame.prepare_for_device();
        // SAFETY: this state transition occurs immediately before the first
        // register operation that may start stream DMA. Completion is handled
        // only after a terminal status; incomplete operations are quarantined.
        Ok(unsafe { (stream.into_in_flight(), frame.into_in_flight()) })
    }

    fn complete_dma(&mut self, stream: InFlightDma, frame: InFlightDma) {
        // SAFETY: callers invoke this only after the JPU reported DONE and
        // poll_decode_done acknowledged that status.
        let stream = unsafe { stream.complete_after_quiesce() }.into_cpu_buffer();
        // SAFETY: the same DONE status covers the output frame DMA engine.
        let frame = unsafe { frame.complete_after_quiesce() }.into_cpu_buffer();
        self.stream_buffer = Some(stream);
        self.frame_buffer = Some(frame);
    }

    /// Resets the JPU after a terminal decode failure and takes both DMA
    /// buffers back once the reset proves the engines are idle.
    ///
    /// `START_INIT` restarts the JPEG block, which stops the stream and frame
    /// DMA engines; the reset completion plus the BBC idle poll therefore prove
    /// quiescence before the buffers become CPU-owned again. When the reset or
    /// the idle poll fails, quiescence is unknown: both buffers are quarantined
    /// and the decoder stays poisoned. The original error is returned either
    /// way so callers report the frame that actually failed.
    fn recover_after_failure(
        &mut self,
        stream: InFlightDma,
        frame: InFlightDma,
        error: JpuDecodeError,
    ) -> JpuDecodeError {
        let jpu_base = self.mmio.jpu_base;
        let reset = hardware_init_at(jpu_base, self.mmio.top_base, self.mmio.vc_base)
            .and_then(|()| wait_bbc_idle_at(jpu_base));
        match reset {
            Ok(()) => {
                // SAFETY: the soft reset above restarted the JPEG block and the
                // BBC reported idle, so no engine can still access these
                // buffers.
                let stream = unsafe { stream.complete_after_quiesce() }.into_cpu_buffer();
                // SAFETY: the same reset covers the output frame DMA engine.
                let frame = unsafe { frame.complete_after_quiesce() }.into_cpu_buffer();
                self.stream_buffer = Some(stream);
                self.frame_buffer = Some(frame);
                log::warn!(
                    "[JPU] recovered from decode failure ({error}): JPU reset, DMA buffers \
                     reclaimed"
                );
                error
            }
            Err(reset_error) => {
                self.poisoned = true;
                let _stream = stream.quarantine();
                let _frame = frame.quarantine();
                log::warn!(
                    "[JPU] cannot reset after decode failure ({error}): {reset_error}; decoder \
                     poisoned"
                );
                error
            }
        }
    }

    fn clear_frame_padding(&mut self, layout: &FrameLayout) -> Result<(), JpuDecodeError> {
        let frame = self
            .frame_buffer
            .as_mut()
            .ok_or(JpuBufferError::MissingCompletedFrameBuffer)?;
        // SAFETY: poll completion returned ownership to the CPU, and the
        // returned mutable slice does not outlive this exclusive decoder borrow.
        let bytes = unsafe { frame.as_mut_slice_cpu() };
        Ok(clear_frame_padding(bytes, layout)?)
    }
}

impl Drop for JpuDecoder {
    fn drop(&mut self) {
        if !self.poisoned {
            JPU_IN_USE.store(false, Ordering::Release);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BufferPlan {
    Reuse,
    ReplaceBoth,
}

const fn buffer_plan(
    stream_capacity: usize,
    frame_capacity: usize,
    stream_len: usize,
    frame_len: usize,
) -> BufferPlan {
    if stream_capacity >= stream_len && frame_capacity >= frame_len {
        BufferPlan::Reuse
    } else {
        BufferPlan::ReplaceBoth
    }
}

/// End of the stream window declared to the BBC, always one page past the data.
///
/// The bitstream buffer fetches whole 256-byte pages and reports the page count
/// together with `END_FLAG`. A stream whose length is an exact multiple of the
/// page size would end on that boundary with no page left for the parser's
/// final lookahead; the hardware then never reports DONE or ERROR (`status=0x4`)
/// and only the poll timeout ends the decode. Declaring the next page boundary
/// keeps one zero-filled page available, and the parser still stops at the JPEG
/// EOI marker inside it.
fn declared_stream_end(jpeg_len: usize) -> Result<usize, JpegHeaderError> {
    (jpeg_len / BBC_STREAM_PAGE_SIZE)
        .checked_add(1)
        .and_then(|pages| pages.checked_mul(BBC_STREAM_PAGE_SIZE))
        .ok_or(JpegHeaderError::GramPrefetchRangeOverflow)
}

fn required_stream_capacity(jpeg_len: usize, ecs_offset: usize) -> Result<usize, JpegHeaderError> {
    if ecs_offset >= jpeg_len {
        return Err(JpegHeaderError::EntropyDataOutsideStream);
    }
    let prefetch_start = ecs_offset & !(BBC_STREAM_PAGE_SIZE - 1);
    let prefetch_len = BBC_STREAM_PAGE_SIZE * GRAM_PREFETCH_PAGES;
    let prefetch_end = prefetch_start
        .checked_add(prefetch_len)
        .ok_or(JpegHeaderError::GramPrefetchRangeOverflow)?;
    // One guard page past the declared end keeps `bbc_wr_ptr` strictly inside
    // the buffer instead of on `bbc_end_addr`.
    declared_stream_end(jpeg_len)?
        .max(prefetch_end)
        .checked_add(BBC_STREAM_PAGE_SIZE)
        .ok_or(JpegHeaderError::GramPrefetchRangeOverflow)
}

fn validate_dma_allocation_len(len: usize) -> Result<(), JpuDmaAddressError> {
    let len = u64::try_from(len).map_err(|_| JpuDmaAddressError::AllocationLengthDoesNotFitU64)?;
    if len > u32::MAX as u64 {
        return Err(JpuDmaAddressError::AllocationExceedsAddressWindow);
    }
    Ok(())
}

fn clear_frame_padding(buffer: &mut [u8], layout: &FrameLayout) -> Result<(), JpuBufferError> {
    if buffer.len() < layout.total_len {
        return Err(JpuBufferError::FrameLayoutExceedsAllocation);
    }
    let buffer = &mut buffer[..layout.total_len];
    let mut previous_end = 0usize;
    clear_plane_and_gap_padding(buffer, layout.y, &mut previous_end)?;
    if let Some(cb) = layout.cb {
        clear_plane_and_gap_padding(buffer, cb, &mut previous_end)?;
    }
    if let Some(cr) = layout.cr {
        clear_plane_and_gap_padding(buffer, cr, &mut previous_end)?;
    }
    buffer
        .get_mut(previous_end..)
        .ok_or(JpuBufferError::FramePlanesExceedTotalLength)?
        .fill(0);
    Ok(())
}

fn copy_frame_range(
    source: &[u8],
    frame_len: usize,
    offset: usize,
    destination: &mut [u8],
) -> Result<usize, JpuBufferError> {
    let frame = source
        .get(..frame_len)
        .ok_or(JpuBufferError::LogicalFrameExceedsAllocation)?;
    let Some(remaining) = frame.get(offset..) else {
        return Ok(0);
    };
    let copied = remaining.len().min(destination.len());
    destination[..copied].copy_from_slice(&remaining[..copied]);
    Ok(copied)
}

fn clear_plane_and_gap_padding(
    buffer: &mut [u8],
    plane: PlaneLayout,
    previous_end: &mut usize,
) -> Result<(), JpuBufferError> {
    let plane_end = plane
        .offset
        .checked_add(plane.len)
        .ok_or(JpuBufferError::PlaneEndOverflow)?;
    if plane.offset < *previous_end || plane_end > buffer.len() {
        return Err(JpuBufferError::FramePlanesOverlapOrExceedTotalLength);
    }
    buffer[*previous_end..plane.offset].fill(0);

    let stride = usize::try_from(plane.stride).map_err(|_| JpuBufferError::PlaneStrideOverflow)?;
    let row_bytes =
        usize::try_from(plane.storage.width).map_err(|_| JpuBufferError::PlaneWidthOverflow)?;
    let rows =
        usize::try_from(plane.storage.height).map_err(|_| JpuBufferError::PlaneHeightOverflow)?;
    let row_padding = stride
        .checked_sub(row_bytes)
        .ok_or(JpuBufferError::PlaneWidthExceedsStride)?;
    for row in 0..rows {
        let padding_start = row
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(plane.offset))
            .and_then(|offset| offset.checked_add(row_bytes))
            .ok_or(JpuBufferError::RowPaddingOffsetOverflow)?;
        let padding_end = padding_start
            .checked_add(row_padding)
            .ok_or(JpuBufferError::RowPaddingEndOverflow)?;
        buffer
            .get_mut(padding_start..padding_end)
            .ok_or(JpuBufferError::RowPaddingExceedsFrameBuffer)?
            .fill(0);
    }

    *previous_end = plane_end;
    Ok(())
}

// Forward compaction is safe: destination strides and offsets never exceed
// their source counterparts. Finish Cb before moving Cr into the freed space.
fn convert_yuv420(bytes: &mut [u8], native: &FrameLayout) -> Result<(), JpuDecodeError> {
    let output = native.yuv420_layout()?;
    if bytes.len() < native.total_len {
        return Err(JpuBufferError::FrameLayoutExceedsAllocation.into());
    }
    let (step_x, step_y) = match native.format {
        JpuPixelFormat::Yuv420 => return Ok(()),
        JpuPixelFormat::Yuv422Horizontal => (1, 2),
        JpuPixelFormat::Yuv422Vertical => (2, 1),
        JpuPixelFormat::Yuv444 => (2, 2),
        JpuPixelFormat::Grayscale => return Err(FrameLayoutError::UnsupportedPixelFormat.into()),
    };
    let visible = (
        native.visible.width as usize,
        native.visible.height as usize,
    );
    for (source, destination) in [(native.cb, output.cb), (native.cr, output.cr)] {
        let source = source.ok_or(FrameLayoutError::UnsupportedPixelFormat)?;
        let destination = destination.ok_or(FrameLayoutError::UnsupportedPixelFormat)?;
        downsample_chroma(bytes, source, destination, visible, (step_x, step_y));
    }
    clear_frame_padding(bytes, &output)?;
    Ok(())
}

fn downsample_chroma(
    bytes: &mut [u8],
    source: PlaneLayout,
    destination: PlaneLayout,
    (width, height): (usize, usize),
    (step_x, step_y): (usize, usize),
) {
    let source_width = width.div_ceil(2 / step_x);
    let source_height = height.div_ceil(2 / step_y);
    let output_width = width.div_ceil(2);
    let output_height = height.div_ceil(2);
    let count = (step_x * step_y) as u16;
    let src_stride = source.stride as usize;
    let dst_stride = destination.stride as usize;

    // ── 快路径:step=(1,2)(YUV422H→420,最常见)──
    // u64 宽读:8 源像素/迭代,总线事务和指令数降 8 倍。
    // (avg(a,b) = ((a+b)+1)>>1 等价于原 (sum + count/2)/count,count=2)
    if (step_x, step_y) == (1, 2) && output_width >= 8 {
        let chunk = output_width / 8; // 完整 u64 块数
        let _tail = output_width % 8;  // 尾部逐字节
        for y in 0..output_height {
            let r0 = source.offset + (y * 2).min(source_height - 1) * src_stride;
            let r1 = source.offset + (y * 2 + 1).min(source_height - 1) * src_stride;
            let d = destination.offset + y * dst_stride;
            for c in 0..chunk {
                // SAFETY: r0/r1/d + c*8..+8 均在 bytes 范围内
                // (stride ≥ width,offset+height*stride ≤ total_len)。
                let a = unsafe { bytes.as_ptr().add(r0 + c * 8).cast::<u64>().read_unaligned() };
                let b = unsafe { bytes.as_ptr().add(r1 + c * 8).cast::<u64>().read_unaligned() };
                // SWAR 逐字节 round-half-up 平均:
                // floor = (a&b) + (((a^b) & 0xFE..) >> 1)   ← 掩码先清各字节 LSB 再移,防跨字节
                // odd   = (a^b) & 0x01..                     ← a+b 为奇数的字节
                // avg   = floor + odd                        ← (floor ≤ 254, odd ≤ 1,无字节进位)
                let xor = a ^ b;
                let floor = (a & b).wrapping_add((xor & 0xFEFE_FEFE_FEFE_FEFE) >> 1);
                let avg = floor.wrapping_add(xor & 0x0101_0101_0101_0101);
                unsafe { bytes.as_mut_ptr().add(d + c * 8).cast::<u64>().write_unaligned(avg) };
            }
            // 尾部(不足 8 像素)
            for x in (chunk * 8)..output_width {
                let va = bytes[r0 + x] as u16;
                let vb = bytes[r1 + x] as u16;
                bytes[d + x] = (((va + vb) + 1) >> 1) as u8;
            }
        }
    } else {
        // ── 通用慢路径(其它 step 组合,保持原语义)──
        for y in 0..output_height {
            for x in 0..output_width {
                let mut sum = 0u16;
                for dy in 0..step_y {
                    let row = (y * step_y + dy).min(source_height - 1);
                    for dx in 0..step_x {
                        let column = (x * step_x + dx).min(source_width - 1);
                        sum += u16::from(bytes[source.offset + row * src_stride + column]);
                    }
                }
                bytes[destination.offset + y * dst_stride + x] =
                    ((sum + count / 2) / count) as u8;
            }
        }
    }
    // 行尾清零(不变)
    for y in 0..destination.storage.height as usize {
        let row = destination.offset + y * dst_stride;
        let used = if y < output_height { output_width } else { 0 };
        bytes[row + used..row + dst_stride].fill(0);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{
        BufferPlan, JpegHeaderError, JpuInspectError, PollDisposition, buffer_plan,
        clear_frame_padding, copy_frame_range, declared_stream_end, inspect_jpeg_layout,
        poll_disposition, required_stream_capacity, validate_dma_allocation_len,
    };
    use crate::{FrameLayout, JpuPixelFormat, JpuScale, engine::PollError};

    const BASELINE_JPEG: &[u8] = &[
        0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03, 0x01, 0x22, 0x00,
        0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11,
        0x03, 0x11, 0x00, 0x3f, 0x00, 0x00, 0xff, 0xd9,
    ];

    #[test]
    fn color_frames_convert_to_yuv420_with_consistent_readback() {
        // A spatial ramp distinguishes horizontal, vertical, and two-axis
        // averaging. Different component biases detect Cb/Cr overlap.
        for (format, dx, dy, first) in [
            (JpuPixelFormat::Yuv420, 1usize, 1usize, 0),
            (JpuPixelFormat::Yuv422Horizontal, 1, 2, 3),
            (JpuPixelFormat::Yuv422Vertical, 2, 1, 1),
            (JpuPixelFormat::Yuv444, 2, 2, 4),
        ] {
            for (width, height, scale) in [
                (1, 1, JpuScale::Full),
                (16, 16, JpuScale::Full),
                (19, 21, JpuScale::Full),
                (129, 145, JpuScale::Eighth),
            ] {
                let native = FrameLayout::new(width, height, format, scale).unwrap();
                let output = native.yuv420_layout().unwrap();
                let w = native.visible.width as usize;
                let h = native.visible.height as usize;
                let mut bytes = std::vec![0; native.total_len];
                bytes[..native.y.len].fill(42);
                for (plane, bias) in [(native.cb.unwrap(), 0), (native.cr.unwrap(), 50)] {
                    for y in 0..h.div_ceil(2 / dy) {
                        for x in 0..w.div_ceil(2 / dx) {
                            bytes[plane.offset + y * plane.stride as usize + x] =
                                (bias + 2 * x + 6 * y) as u8;
                        }
                    }
                }
                clear_frame_padding(&mut bytes, &native).unwrap();
                let luma = bytes[..native.y.len].to_vec();
                super::convert_yuv420(&mut bytes, &native).unwrap();
                assert_eq!(output.format, JpuPixelFormat::Yuv420);
                assert!(output.total_len <= native.total_len);
                assert_eq!(bytes[..native.y.len], luma);
                for (plane, bias) in [(output.cb.unwrap(), 0), (output.cr.unwrap(), 50)] {
                    assert_eq!(plane.stride, output.y.stride / 2);
                    assert_eq!(plane.storage.height, output.storage.height / 2);
                    for y in 0..plane.storage.height as usize {
                        for x in 0..plane.stride as usize {
                            let expected = if x < w.div_ceil(2) && y < h.div_ceil(2) {
                                // Odd right/bottom edges replicate the last sample,
                                // removing the half-step contribution from that axis.
                                let edge_x = usize::from(x == w / 2) * (dx - 1);
                                let edge_y = usize::from(y == h / 2) * 3 * (dy - 1);
                                bias + 2 * x * dx + 6 * y * dy + first - edge_x - edge_y
                            } else {
                                0
                            };
                            assert_eq!(
                                bytes[plane.offset + y * plane.stride as usize + x],
                                expected as u8,
                                "{format:?} {scale:?} ({x}, {y})"
                            );
                        }
                    }
                }
                let mut readback = std::vec![];
                let mut chunk = [0xa5; 13];
                loop {
                    let n = copy_frame_range(&bytes, output.total_len, readback.len(), &mut chunk)
                        .unwrap();
                    if n == 0 {
                        break;
                    }
                    readback.extend_from_slice(&chunk[..n]);
                }
                assert_eq!(readback, bytes[..output.total_len]);
            }
        }
        let gray = FrameLayout::new(19, 21, JpuPixelFormat::Grayscale, JpuScale::Full).unwrap();
        let mut bytes = std::vec![0x5a; gray.total_len];
        assert!(super::convert_yuv420(&mut bytes, &gray).is_err());
        assert!(bytes.iter().all(|&byte| byte == 0x5a));
    }

    #[test]
    fn buffer_plan_reuses_only_when_both_capacities_fit() {
        assert_eq!(buffer_plan(128, 1024, 127, 1024), BufferPlan::Reuse);
        assert_eq!(buffer_plan(128, 1024, 129, 100), BufferPlan::ReplaceBoth);
        assert_eq!(buffer_plan(128, 1024, 100, 1025), BufferPlan::ReplaceBoth);
    }

    #[test]
    fn completed_frame_copy_honors_logical_length_and_offset() {
        let source = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut destination = [0xff; 4];

        assert_eq!(copy_frame_range(&source, 6, 2, &mut destination), Ok(4));
        assert_eq!(destination, [2, 3, 4, 5]);
        assert_eq!(copy_frame_range(&source, 6, 6, &mut destination), Ok(0));
        assert!(copy_frame_range(&source, 9, 0, &mut destination).is_err());
    }

    #[test]
    fn dma_allocation_length_must_fit_the_32_bit_jpu_window() {
        assert!(validate_dma_allocation_len(u32::MAX as usize).is_ok());
        #[cfg(target_pointer_width = "64")]
        assert!(validate_dma_allocation_len(u32::MAX as usize + 1).is_err());
    }

    #[test]
    fn stream_capacity_covers_two_page_gram_prefetch() {
        assert_eq!(required_stream_capacity(1000, 500), Ok(1280));
        assert_eq!(required_stream_capacity(1000, 900), Ok(1536));
        assert_eq!(required_stream_capacity(16_384, 16_383), Ok(16_896));
        assert!(required_stream_capacity(1000, 1000).is_err());
        assert!(required_stream_capacity(1000, 1001).is_err());
        assert!(required_stream_capacity(usize::MAX, usize::MAX - 1).is_err());
    }

    #[test]
    fn page_aligned_streams_declare_one_extra_zero_page() {
        // 39_936 bytes is exactly 156 pages; the declared window must still
        // cover page 157 so the parser lookahead never runs dry.
        assert_eq!(declared_stream_end(39_936), Ok(40_192));
        assert_eq!(declared_stream_end(39_952), Ok(40_192));
        assert_eq!(required_stream_capacity(39_936, 0), Ok(40_448));
    }

    #[test]
    fn frame_padding_is_cleared_without_touching_plane_samples() {
        let layout = FrameLayout::new(129, 129, JpuPixelFormat::Yuv420, JpuScale::Eighth)
            .expect("valid layout");
        let mut memory = std::vec![0xa5u8; layout.total_len];

        clear_frame_padding(&mut memory, &layout).expect("padding layout is valid");

        for row in 0..18 {
            let start = row * 32;
            assert!(memory[start..start + 18].iter().all(|byte| *byte == 0xa5));
            assert!(memory[start + 18..start + 32].iter().all(|byte| *byte == 0));
        }
        for plane_offset in [576, 720] {
            for row in 0..9 {
                let start = plane_offset + row * 16;
                assert!(memory[start..start + 9].iter().all(|byte| *byte == 0xa5));
                assert!(memory[start + 9..start + 16].iter().all(|byte| *byte == 0));
            }
        }
    }

    #[test]
    fn naturally_packed_frame_preserves_all_samples() {
        let layout = FrameLayout::new(1279, 1706, JpuPixelFormat::Yuv420, JpuScale::Half)
            .expect("valid packed layout");
        let mut memory = std::vec![0xa5u8; layout.total_len];

        clear_frame_padding(&mut memory, &layout).expect("layout is valid");

        assert!(memory.iter().all(|byte| *byte == 0xa5));
    }

    #[test]
    fn inspect_rejects_empty_stream_without_hardware() {
        assert_eq!(
            inspect_jpeg_layout(&[], JpuScale::Full),
            Err(JpuInspectError::EmptyStream)
        );
    }

    #[test]
    fn inspect_returns_layout_for_valid_baseline_jpeg() {
        let layout = inspect_jpeg_layout(BASELINE_JPEG, JpuScale::Full)
            .expect("baseline JPEG is inspectable");

        assert_eq!(layout.format, JpuPixelFormat::Yuv420);
        assert_eq!((layout.source.width, layout.source.height), (16, 16));
        assert_eq!((layout.visible.width, layout.visible.height), (16, 16));
        assert_eq!(layout.total_len, 384);
    }

    #[test]
    fn inspect_rejects_baseline_jpeg_without_eoi() {
        let truncated = &BASELINE_JPEG[..BASELINE_JPEG.len() - 2];

        assert_eq!(
            inspect_jpeg_layout(truncated, JpuScale::Full),
            Err(JpuInspectError::InvalidJpeg(
                JpegHeaderError::MissingEntropyDataAndEoi
            ))
        );
    }

    #[test]
    fn inspect_rejects_baseline_jpeg_without_soi() {
        assert_eq!(
            inspect_jpeg_layout(&BASELINE_JPEG[2..], JpuScale::Full),
            Err(JpuInspectError::InvalidJpeg(JpegHeaderError::MissingSoi))
        );
    }

    #[test]
    fn decode_error_requests_hardware_recovery() {
        assert_eq!(
            poll_disposition(Err(PollError::Decode)),
            PollDisposition::Recover(PollError::Decode)
        );
    }

    #[test]
    fn decode_timeout_requests_hardware_recovery() {
        assert_eq!(
            poll_disposition(Err(PollError::Timeout)),
            PollDisposition::Recover(PollError::Timeout)
        );
    }

    #[test]
    fn jpeg_len_history_keeps_the_newest_entries_in_order() {
        let mut history = super::JpegLenHistory::default();
        for len in [11u32, 22, 33, 44, 55, 66] {
            history.record(len as usize);
        }

        assert_eq!(history.entries(), [33, 44, 55, 66].as_slice());
    }

    #[test]
    fn decode_done_allows_dma_completion() {
        assert_eq!(poll_disposition(Ok(())), PollDisposition::Complete);
    }
}
