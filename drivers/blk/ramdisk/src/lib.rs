//! In-memory block device backed by externally supplied storage.
//!
//! 2026-09-29 重生版:适配上游 IRQ-only 块运行时。无真实硬件中断——
//! 数据搬运在 `submit_batch_owned` 同步完成,完成回报走软件轮询通道
//! (`register_retry_after` → `advance_register_retry`):runtime 在该
//! 通道到期时携带 CompletionSink 调用,队列把暂存的完成交付即可。
//! 空闲时 `register_retry_after` 返回 None,worker 进入长眠。
//!
//! 典型用途:bootloader 把 ramdisk 镜像(GPT+ext4)加载到 fdt
//! `ramdisk` 节点声明的保留 DRAM 区,内核 FDT probe 经线性映射零拷贝
//! 借用该区域当盘面,后续分区探测与 ext4 挂载全走通用路径。

#![no_std]

extern crate alloc;

use alloc::{boxed::Box, collections::VecDeque, vec::Vec};
use core::time::Duration;

use rdif_block::{
    BatchSubmitDisposition, BatchSubmitResult, BlkError, CompletedRequest, CompletionSink,
    ControllerEvent, ControllerState, ControllerUpdate, DeviceInfo, DriverGeneric,
    HardwareQueue, OwnedRequest, OwnedRequestBatch, QueueInfo, QueueLimits, RequestId,
    RequestOp, SubmissionSink,
};

/// 控制器:管理一个 MemQueue 的生命周期(Start 交出,Shutdown 回收)。
pub struct RamDiskCtrl {
    queue: Option<MemQueue>,
}

impl RamDiskCtrl {
    /// 用调用方提供的内存当盘面(零拷贝借用;长度按块大小向下对齐)。
    pub fn with_external_storage(
        name: &'static str,
        block_size: usize,
        storage: &'static mut [u8],
    ) -> Self {
        Self {
            queue: Some(MemQueue::new(name, block_size, storage)),
        }
    }

    /// 自带清零内存(测试用)。
    pub fn new(name: &'static str, block_size: usize, num_blocks: usize) -> Self {
        let mut storage: Vec<u8> = Vec::with_capacity(block_size * num_blocks);
        storage.resize(block_size * num_blocks, 0);
        let storage: &'static mut [u8] = Box::leak(storage.into_boxed_slice());
        Self::with_external_storage(name, block_size, storage)
    }
}

impl DriverGeneric for RamDiskCtrl {
    fn name(&self) -> &str {
        "ramdisk"
    }

    fn raw_any(&self) -> Option<&dyn core::any::Any> {
        Some(self)
    }

    fn raw_any_mut(&mut self) -> Option<&mut dyn core::any::Any> {
        Some(self)
    }
}

impl rdif_block::BlockController for RamDiskCtrl {
    fn device_info(&self) -> DeviceInfo {
        self.queue
            .as_ref()
            .map(|q| q.device_info())
            .unwrap_or_else(|| DeviceInfo::new(0, 512))
    }

    fn max_io_queues(&self) -> usize {
        1
    }

    fn advance(&mut self, event: ControllerEvent) -> Result<ControllerUpdate, BlkError> {
        match event {
            ControllerEvent::Start { .. } => {
                let queue = self.queue.take().ok_or(BlkError::Io)?;
                Ok(ControllerUpdate::with_resources(
                    ControllerState::Ready,
                    Vec::from([Box::new(queue) as Box<dyn HardwareQueue>]),
                    Vec::new(), // 无 IRQ endpoint:完成走 register-retry 软件通道
                ))
            }
            ControllerEvent::Shutdown | ControllerEvent::Watchdog { .. } => {
                Ok(ControllerUpdate::state(ControllerState::Shutdown))
            }
            _ => Ok(ControllerUpdate::state(ControllerState::Ready)),
        }
    }
}

/// 队列:同步 memcpy 数据面 + 暂存完成 + 软件通道交付。
pub struct MemQueue {
    name: &'static str,
    block_size: usize,
    num_blocks: u64,
    storage: &'static mut [u8],
    next_req_id: u64,
    pending: VecDeque<CompletedRequest>,
}

impl MemQueue {
    fn new(name: &'static str, block_size: usize, storage: &'static mut [u8]) -> Self {
        assert!(block_size > 0, "block size must be greater than zero");
        let num_blocks = (storage.len() / block_size) as u64;
        let storage = &mut storage[..num_blocks as usize * block_size];
        Self {
            name,
            block_size,
            num_blocks,
            storage,
            next_req_id: 1,
            pending: VecDeque::new(),
        }
    }

    fn device_info(&self) -> DeviceInfo {
        let mut info = DeviceInfo::new(self.num_blocks, self.block_size);
        info.name = Some(self.name);
        info
    }

    fn limits(&self) -> QueueLimits {
        // 内存盘由 CPU 直接 memcpy,DMA 约束无关紧要;报 Direct/Coherent
        // 让 runtime 跳过不必要的缓冲加工。
        let mut limits = QueueLimits::simple(
            self.block_size,
            dma_api::DmaDeviceInfo::new(
                dma_api::DmaDomainId::Direct,
                dma_api::DmaCoherency::Coherent,
                dma_api::DmaConstraints::new(u64::MAX),
            ),
        );
        limits.max_inflight = 64;
        limits.supports_flush = true;
        limits
    }

    /// 同步执行一条请求的数据搬运;返回 (完成结果, DMA 所有权)。
    fn execute(
        &mut self,
        request: &mut OwnedRequest,
    ) -> (Result<(), BlkError>, Option<dma_api::CompletedDma>) {
        let result = (|| {
            let offset = request
                .lba
                .checked_mul(self.block_size as u64)
                .ok_or(BlkError::InvalidRequest)? as usize;
            let len = request.block_count as usize * self.block_size;
            let end = offset.checked_add(len).ok_or(BlkError::InvalidRequest)?;
            if end > self.storage.len() {
                return Err(BlkError::InvalidRequest);
            }
            match request.op {
                RequestOp::Read => {
                    if let Some(data) = request.data.as_mut() {
                        unsafe {
                            core::ptr::copy_nonoverlapping(
                                self.storage.as_ptr().add(offset),
                                data.cpu_ptr().as_ptr(),
                                len,
                            );
                        }
                    }
                }
                RequestOp::Write => {
                    if let Some(data) = request.data.as_ref() {
                        unsafe {
                            core::ptr::copy_nonoverlapping(
                                data.cpu_ptr().as_ptr(),
                                self.storage.as_mut_ptr().add(offset),
                                len,
                        );
                        }
                    }
                }
                RequestOp::Flush => {}
            }
            Ok(())
        })();
        // 数据已搬完(或已失败):DMA 所有权转出,等待交付。
        let data = request
            .data
            .take()
            .map(dma_api::PreparedDma::complete_without_device);
        (result, data)
    }
}

impl DriverGeneric for MemQueue {
    fn name(&self) -> &str {
        self.name
    }

    fn raw_any(&self) -> Option<&dyn core::any::Any> {
        Some(self)
    }

    fn raw_any_mut(&mut self) -> Option<&mut dyn core::any::Any> {
        Some(self)
    }
}

impl HardwareQueue for MemQueue {
    fn id(&self) -> usize {
        0
    }

    fn info(&self) -> QueueInfo {
        QueueInfo {
            id: 0,
            device: self.device_info(),
            limits: self.limits(),
        }
    }

    fn submit_batch_owned(
        &mut self,
        requests: &mut OwnedRequestBatch,
        sink: &mut dyn SubmissionSink,
    ) -> BatchSubmitResult {
        let mut accepted = 0usize;
        while let Some(mut request) = requests.pop_front() {
            let id = RequestId::new(self.next_req_id.try_into().unwrap());
            self.next_req_id += 1;
            let (result, data) = self.execute(&mut request);
            sink.accepted(id);
            accepted += 1;
            self.pending.push_back(CompletedRequest::new(id, result, data));
        }
        BatchSubmitResult::new(accepted, BatchSubmitDisposition::Continue)
    }

    fn commit_submissions(&mut self) -> Result<(), BlkError> {
        Ok(())
    }

    fn register_retry_after(&self) -> Option<Duration> {
        // 有待交付的完成时报最短延时(runtime 会把 zero 规整为 1µs),
        // 空闲时返回 None 让 worker 长眠。
        self.pending.is_empty().not().then_some(Duration::ZERO)
    }

    fn drain_completions(&mut self, sink: &mut dyn CompletionSink) -> Result<(), BlkError> {
        // 无 IRQ 源;完成统一走 advance_register_retry 软件通道。
        // 此方法按协议只在 IRQ 后被调,对我们而言不应出现——防御性交付。
        while let Some(done) = self.pending.pop_front() {
            sink.complete(done);
        }
        Ok(())
    }

    fn advance_register_retry(&mut self, sink: &mut dyn CompletionSink) -> Result<(), BlkError> {
        while let Some(done) = self.pending.pop_front() {
            sink.complete(done);
        }
        Ok(())
    }

    fn shutdown(&mut self, sink: &mut dyn CompletionSink) -> Result<(), BlkError> {
        while let Some(done) = self.pending.pop_front() {
            sink.complete(done);
        }
        Ok(())
    }
}

/// bool 辅助(no_std 下 Option 映射)。
trait Not {
    fn not(self) -> bool;
}
impl Not for bool {
    fn not(self) -> bool {
        !self
    }
}
