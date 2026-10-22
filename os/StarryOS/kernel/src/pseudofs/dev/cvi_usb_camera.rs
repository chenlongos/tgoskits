use alloc::sync::Arc;
use core::{any::Any, time::Duration};

use ax_memory_addr::{PhysAddr, VirtAddr};
use ax_runtime::hal::{mem::virt_to_phys, time::busy_wait};
use axfs_ng_vfs::{NodeFlags, VfsResult};
use sg200x_bsp::{
    gpio::{Direction, GPIO, GPIO1_BASE},
    pinmux::{FMUX_USB_VBUS_DET, Pinmux},
    soc::{
        CLKGEN_BASE, CV182X_USB2_PHY_BASE, DWC2_BASE, FMUX_BASE, IOBLK_BASE, IOBLK_GRTC_BASE,
        TOP_BASE,
    },
    usb::{
        self,
        class::uvc,
        error::UsbError,
        host::{self, UvcEnumerated, dwc2, dwc2::ep0 as dwc2_ep0},
    },
};
use tock_registers::interfaces::Writeable;

use super::cvi_jpu::CviJpu;
use crate::{
    StarryError, StarryResult,
    mm::{VmMutPtr, vm_write_slice},
    pseudofs::DeviceOps,
    sync::Mutex,
};

const IOBLK_G1_USB_VBUS_DET_OFF: usize = 0x020;

const VBUS_GPIO_PIN: u8 = 6;
const VBUS_GPIO_ACTIVE_HIGH: bool = true;

/// MMIO span of the TOP control block. The PHY ID-pad reset register lives at
/// `TOP_BASE + 0x3000`, so a single 4K page is not enough — map four pages.
const TOP_MMIO_SIZE: usize = 0x4000;
/// MMIO span for the single-page register blocks (CLKGEN, FMUX, IOBLK, GRTC,
/// GPIO, DWC2 controller, USB2 PHY). Each block's registers fit within one 4K
/// page; FMUX/IOBLK share a page so their mappings coincide (idempotent).
const REG_MMIO_SIZE: usize = 0x1000;

/// Map a physical MMIO region into the kernel address space and return its
/// virtual base. Unlike `phys_to_virt`, this works on dynamic platforms where
/// `PHYS_VIRT_OFFSET == 0` and there is no static linear MMIO window — `iomap`
/// installs a real device mapping and is idempotent for already-mapped pages.
fn iomap_usize(paddr: usize, size: usize) -> usize {
    ax_mm::iomap(PhysAddr::from_usize(paddr), size)
        .unwrap_or_else(|err| panic!("failed to iomap MMIO at {paddr:#x}+{size:#x}: {err:?}"))
        .as_usize()
}

const CAMERA_FORMAT_MJPEG: u8 = 1;
const MIN_VALID_JPEG_BYTES: usize = 4096;
const MAX_CAPTURE_TRIES: u32 = 8;
/// 连续多少次抓帧失败后就放弃原地重启视频流,改为丢弃会话让下次 ioctl
/// 重新枚举设备。原地重启只解决控制器/通道状态卡死,救不回真掉线的设备。
const MAX_STREAM_RESTARTS: u32 = 3;
/// Default resolution cap (640×480 = 307200 pixels) guiding UVC frame selection.
const DEFAULT_RESOLUTION: u32 = 640 * 480;

pub const CVI_CAMERA_IOCTL_INIT: u32 = 1;
pub const CVI_CAMERA_IOCTL_GET_INFO: u32 = 2;
pub const CVI_CAMERA_IOCTL_GET_FRAME: u32 = 3;
pub const CVI_CAMERA_IOCTL_GET_YUV_FRAME: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraInfo {
    pub width: u16,
    pub height: u16,
    /// 1 means MJPEG.
    pub format: u8,
    pub connected: u8,
}

struct UsbCameraSession {
    cam: UvcEnumerated,
    sel: uvc::UvcStreamSelection,
}

#[derive(Default)]
struct UsbCameraState {
    session: Option<UsbCameraSession>,
    smallcore: Option<SmallCoreSource>,
    /// 连续抓帧失败次数:成功抓帧或丢弃会话后归零。
    stream_restarts: u32,
}

// ---- 小核帧源(2026-09-27 分核架构)----
// cvirtos 在小核跑 USB 采集:JPEG 乒乓双槽(固定 PA)+ 邮箱 ctx 镜像
// (payload=帧计数,aux=(jpeg_len<<8)|flags,bit2=槽位)。探测到小核心跳
// 在走即切换为「小核帧源」——不枚举本核 USB,直接收双槽。
const SMALLCORE_MBOX_PA: usize = 0x0190_0000;
const SMALLCORE_CTX_OFF: usize = 0x400;
const SMALLCORE_HB_OFF: usize = 0x41C;
const SMALLCORE_SLOT0_PA: usize = 0x8FEC_0000;
const SMALLCORE_SLOT_CAP: usize = 180 * 1024;
const SMALLCORE_FLAG_SLOT: u32 = 1 << 2;
/// 小核 JPU 已解码标志(bit3;与 ipc.rs FLAG_YUV 对齐)。
const SMALLCORE_FLAG_YUV: u32 = 1 << 3;
/// 流控信用单元(与 bare-metal platform.rs FLOW_CREDIT_PA 一致)。
/// 大核在此回写「已完成消费的帧号」,小核据此判断乒乓缓冲能否复用。
/// 落点在 rtos 区 DRAM(不能用 HW 邮箱 ctx[2]——那是跨核日志行锁)。
const SMALLCORE_FLOW_CREDIT_PA: usize = 0x8FEA_0000;

/// I420 staging 乒乓(内核把小核的 I422 转成模型要的 I420 后放这里,
/// TPU DMA 直读)。位于 YUV 外置预留区内(YUV A/B 之后,4MB memreserve 内):
///   YUV A [0x8D000000, 0x8D096000)  YUV B [0x8D096000, 0x8D12C000)
///   I420 A [0x8D140000, 0x8D1B0800) I420 B [0x8D1B1000, 0x8D221800)
const SMALLCORE_I420_A_PA: usize = 0x8D14_0000;
const SMALLCORE_I420_B_PA: usize = 0x8D1B_1000;
const SMALLCORE_I420_SIZE: usize = 640 * 480 * 3 / 2; // 460800
const SMALLCORE_Y_SIZE: usize = 640 * 480; // 307200
/// 小核 YUV 帧最大长度(640×480 YUV422 = 614400;420 为 460800)。
const SMALLCORE_YUV_CAP: usize = 640 * 480 * 2;
/// 小核采集分辨率(固定 640×480;GET_INFO 在小核模式下用)。
const SMALLCORE_FRAME_W: u16 = 640;
const SMALLCORE_FRAME_H: u16 = 480;

/// YUV 乒乓缓冲 A/B(必须与 bare-metal platform.rs YUV_A_PA/YUV_B_PA 一致)。
const SMALLCORE_YUV_A_PA: usize = 0x8D00_0000;
const SMALLCORE_YUV_B_PA: usize = 0x8D09_6000;
/// YUV 乒乓槽索引(bit4;与 ipc.rs FLAG_YUV_SLOT 对齐)。
const SMALLCORE_FLAG_YUV_SLOT: u32 = 1 << 4;
/// live ioctl 9:全内核流水线(小核槽取帧 → JPU 解码 → PA 直通)。
pub const CVI_CAMERA_IOCTL_DECODE_YUV420P_LIVE: u32 = 9;

/// 小核 JPEG 帧的内核弹跳缓冲(uncached 槽 → 这里 → JPU)。
/// 上游内核不用 nightly feature,改用 `AtomicPtr` 风格的裸静态 +
/// 编译期可证明的独占访问(camera state 锁内)。
static mut SMALLCORE_JPEG_BUF: [u8; SMALLCORE_SLOT_CAP] = [0; SMALLCORE_SLOT_CAP];

/// 色度平面垂直 2:1 下采样(I422→I420),u64 SWAR 字节平均。
/// 源是 JPU DMA 写的乒乓缓冲——volatile 读;目标走正常 cache(调用方 Clean)。
///
/// SWAR 平均公式(无进位字节平均,对每个字节独立):
///   floor = (a & b) + ((a^b & 0xFE..) >> 1)
///   avg   = floor + (a^b & 0x01..)
unsafe fn swar_vdown(src: usize, dst: usize, width_bytes: usize, src_rows: usize) {
    let qwords = width_bytes / 8;
    for r in 0..(src_rows / 2) {
        let s0 = src + (2 * r) * width_bytes;
        let s1 = s0 + width_bytes;
        let d = dst + r * width_bytes;
        for i in 0..qwords {
            let (a, b) = unsafe {
                (
                    core::ptr::read_volatile((s0 + i * 8) as *const u64),
                    core::ptr::read_volatile((s1 + i * 8) as *const u64),
                )
            };
            let xor = a ^ b;
            let floor = (a & b).wrapping_add((xor & 0xFEFE_FEFE_FEFE_FEFE) >> 1);
            let avg = floor.wrapping_add(xor & 0x0101_0101_0101_0101);
            unsafe { core::ptr::write_volatile((d + i * 8) as *mut u64, avg) };
        }
    }
}

struct SmallCoreSource {
    mbox: usize,
    slots: usize,
    last_fc: u32,
    /// I420 staging 乒乓翻转(false=A, true=B)。
    staging_flip: bool,
    /// 流控信用单元的虚拟地址(rtos 区 DRAM)。
    flow: usize,
    /// 上一次已**完成消费**(TPU 读完)的帧号——流控信用,回写给小核。
    /// 必须延迟一整轮再回写:本次 ioctl 返回的 PA 正被 TPU 读取,
    /// 此时若宣告该帧已消费,小核会立刻覆盖同一乒乓缓冲。
    last_done: u32,
}

impl SmallCoreSource {
    /// 心跳两读相隔 50ms 在走 = 小核固件在线。
    fn probe() -> Option<Self> {
        let mbox = iomap_usize(SMALLCORE_MBOX_PA, REG_MMIO_SIZE);
        // 槽区在 memreserve 内(小核 2MB 区)——iomap 被拒,改用线性映射。
        // 线性映射始终存在(覆盖全部 DRAM),不需显式映射。
        let slots = axklib::mem::phys_to_virt(
            ax_memory_addr::PhysAddr::from_usize(SMALLCORE_SLOT0_PA),
        )
        .as_usize();
        let flow = axklib::mem::phys_to_virt(
            ax_memory_addr::PhysAddr::from_usize(SMALLCORE_FLOW_CREDIT_PA),
        )
        .as_usize();
        let hb = unsafe { core::ptr::read_volatile((mbox + SMALLCORE_HB_OFF) as *const u32) };
        crate::task::sleep(Duration::from_millis(50));
        let hb2 = unsafe { core::ptr::read_volatile((mbox + SMALLCORE_HB_OFF) as *const u32) };
        if hb2 != hb && hb2 != 0 {
            let last_fc =
                unsafe { core::ptr::read_volatile((mbox + SMALLCORE_CTX_OFF) as *const u32) };
            info!("cvi-camera: detected hb {hb}->{hb2} fc={last_fc:#x}");
            Some(Self { mbox, slots, last_fc, staging_flip: false, flow, last_done: 0 })
        } else {
            None
        }
    }

    /// 流控回写 + 等待下一帧小核 YUV,返回 (yuv_pa, yuv_len)。
    ///
    /// ioctl 9(零拷贝 PA 直通)与 ioctl 4(拷贝到用户态)共用:
    /// - 先宣告「上一帧已消费完毕」(写流控信用 + dcache Clean);
    /// - 轮询硬件邮箱 ctx 直到出现新的 FLAG_YUV 帧;
    /// - 记录本帧为 last_done(下一轮调用开头才宣告,TPU/用户态正在读它)。
    fn wait_next_yuv(&mut self) -> StarryResult<(usize, usize)> {
        if self.last_done != 0 {
            let flow = self.flow;
            unsafe { core::ptr::write_volatile(flow as *mut u32, self.last_done) };
            ax_hal::mem::dcache_range(
                ax_hal::mem::DCacheOp::Clean,
                ax_memory_addr::VirtAddr::from_usize(flow),
                64,
            );
        }
        let deadline = ax_runtime::hal::time::wall_time()
            + core::time::Duration::from_secs(3);
        loop {
            let fc = unsafe {
                core::ptr::read_volatile((self.mbox + SMALLCORE_CTX_OFF) as *const u32)
            };
            if fc != self.last_fc {
                self.last_fc = fc;
                let aux = unsafe {
                    core::ptr::read_volatile(
                        (self.mbox + SMALLCORE_CTX_OFF + 4) as *const u32,
                    )
                };
                let len = (aux >> 8) as usize;
                let flags = aux as u32;
                if (flags & SMALLCORE_FLAG_YUV) != 0 && (2048..=SMALLCORE_YUV_CAP).contains(&len)
                {
                    let yuv_pa = if (flags & SMALLCORE_FLAG_YUV_SLOT) != 0 {
                        SMALLCORE_YUV_B_PA
                    } else {
                        SMALLCORE_YUV_A_PA
                    };
                    self.last_done = fc;
                    return Ok((yuv_pa, len));
                }
                // JPEG-only 帧(小核 JPU 暂未就绪):继续等 YUV 帧
            }
            if ax_runtime::hal::time::wall_time() > deadline {
                warn!("cvi-camera: small-core YUV timeout");
                return Err(StarryError::Io);
            }
            crate::task::sleep(core::time::Duration::from_micros(500));
        }
    }

    /// 等下一帧(帧计数推进),从槽区拷入内核静态缓冲。
    #[allow(dead_code)]
    fn frame(&mut self) -> Option<&'static [u8]> {
        let deadline = ax_runtime::hal::time::wall_time() + Duration::from_secs(3);
        let mut dbg_count = 0u32;
        loop {
            let fc = unsafe {
                core::ptr::read_volatile((self.mbox + SMALLCORE_CTX_OFF) as *const u32)
            };
            dbg_count += 1;
            if dbg_count == 1 {
                info!("FRAME-LOOP fc={:#x} last={:#x}", fc, self.last_fc);
            }
            if dbg_count.is_multiple_of(1_000_000) {
                info!("cvi-camera: frame() spin #{} fc={}", dbg_count, fc);
            }
            if fc != self.last_fc {
                self.last_fc = fc;
                let aux = unsafe {
                    core::ptr::read_volatile((self.mbox + SMALLCORE_CTX_OFF + 4) as *const u32)
                };
                let len = (aux >> 8) as usize;
                let slot = ((aux & SMALLCORE_FLAG_SLOT) != 0) as usize;
                if (4096..=SMALLCORE_SLOT_CAP).contains(&len) {
                    let src = self.slots + slot * SMALLCORE_SLOT_CAP;
                    let dst = core::ptr::addr_of_mut!(SMALLCORE_JPEG_BUF) as *mut u8;
                    ax_hal::mem::dcache_range(
                        ax_hal::mem::DCacheOp::Invalidate,
                        ax_memory_addr::VirtAddr::from_usize(src),
                        len,
                    );
                    unsafe { core::ptr::copy_nonoverlapping(src as *const u8, dst, len) };
                    let (b0, b1) = unsafe { (*dst, *dst.add(1)) };
                    if b0 == 0xff && b1 == 0xd8 {
                        return Some(unsafe { core::slice::from_raw_parts(dst as *const u8, len) });
                    }
                    warn!("cvi-camera: small-core frame bad header len={len}; skip");
                }
            }
            if ax_runtime::hal::time::wall_time() > deadline {
                warn!("cvi-camera: small-core frame timeout");
                return None;
            }
            crate::task::sleep(Duration::from_micros(500));
        }
    }
}

pub struct CviCamera {
    state: Mutex<UsbCameraState>,
    jpu: Arc<CviJpu>,
}

fn ep0_dma_virt_to_phys(p: *const u8) -> u32 {
    virt_to_phys(VirtAddr::from(p as usize)).as_usize() as u32
}

fn root_usb_device_connected() -> bool {
    // SAFETY: camera initialization has installed the DWC2 MMIO base, and the
    // camera state mutex serializes this read with all other host operations.
    let hprt0 = unsafe { dwc2::dwc2_hprt0_read() };
    dwc2::hprt_connsts(hprt0)
}

unsafe fn enable_usb_clocks_cv181x() {
    let b = iomap_usize(CLKGEN_BASE, REG_MMIO_SIZE);
    let en1 = (b + 0x004) as *mut u32;
    let en2 = (b + 0x008) as *mut u32;
    let byp0 = (b + 0x030) as *mut u32;
    unsafe {
        let v1_pre = core::ptr::read_volatile(en1);
        let v2_pre = core::ptr::read_volatile(en2);
        let byp_pre = core::ptr::read_volatile(byp0);
        core::ptr::write_volatile(en1, v1_pre | (0xFu32 << 28));
        core::ptr::write_volatile(en2, v2_pre | 1u32);
        core::ptr::write_volatile(byp0, byp_pre & !((1u32 << 17) | (1u32 << 18)));
    }
}

/// PHY ID pad toggle workaround: switch to device mode first, then host mode.
unsafe fn cvitek_usb_top_host_bringup() {
    let top = iomap_usize(TOP_BASE, TOP_MMIO_SIZE);
    let rst = (top + 0x3000) as *mut u32;
    unsafe {
        let v = core::ptr::read_volatile(rst);
        core::ptr::write_volatile(rst, v & !(1 << 11));
        busy_wait(Duration::from_micros(50));
        core::ptr::write_volatile(rst, v | (1 << 11));
        busy_wait(Duration::from_micros(50));

        let usb_pin = (top + 0x48) as *mut u32;
        let x = core::ptr::read_volatile(usb_pin);
        let dev_mode = (x & !0xC0u32) | 0xC0u32 | 0x01u32;
        core::ptr::write_volatile(usb_pin, dev_mode);
        busy_wait(Duration::from_micros(1000));
        let host_mode = (x & !0xC0u32) | 0x40u32 | 0x01u32;
        core::ptr::write_volatile(usb_pin, host_mode);
        busy_wait(Duration::from_micros(1000));

        let eco = (top + 0xB4) as *mut u32;
        core::ptr::write_volatile(eco, core::ptr::read_volatile(eco) | 0x80);
    }
}

fn pinmux_usb_vbus_det_gpio_output_prep() {
    let fmux_vaddr = iomap_usize(FMUX_BASE, REG_MMIO_SIZE);
    let ioblk_vaddr = iomap_usize(IOBLK_BASE, REG_MMIO_SIZE);
    let ioblk_grtc_vaddr = iomap_usize(IOBLK_GRTC_BASE, REG_MMIO_SIZE);
    let pinmux = unsafe { Pinmux::new(fmux_vaddr, ioblk_vaddr, ioblk_grtc_vaddr) };
    pinmux
        .fmux()
        .usb_vbus_det
        .write(FMUX_USB_VBUS_DET::FSEL::XGPIOB_6);
    let r = (ioblk_vaddr + IOBLK_G1_USB_VBUS_DET_OFF) as *mut u32;
    unsafe {
        let v = core::ptr::read_volatile(r);
        core::ptr::write_volatile(r, v | (7 << 5));
    }
}

fn enable_usb_vbus_gpio() {
    let gpio = unsafe { GPIO::new(iomap_usize(GPIO1_BASE, REG_MMIO_SIZE)) };
    gpio.pin(VBUS_GPIO_PIN).set_direction(Direction::Output);
    gpio.pin(VBUS_GPIO_PIN).set(VBUS_GPIO_ACTIVE_HIGH);
}

fn usb_init_failure_context(e: UsbError) -> &'static str {
    match e {
        UsbError::NotImplemented => "no VS bulk/isoch video endpoint found",
        _ => "failed to parse UVC stream parameters",
    }
}

fn init_usb_camera() -> StarryResult<UsbCameraSession> {
    unsafe {
        enable_usb_clocks_cv181x();
        cvitek_usb_top_host_bringup();
    }
    pinmux_usb_vbus_det_gpio_output_prep();
    enable_usb_vbus_gpio();
    crate::task::sleep(Duration::from_micros(2_000_000));

    usb::set_dwc2_base_virt(iomap_usize(DWC2_BASE, REG_MMIO_SIZE));
    usb::set_cv182x_phy_base_virt(iomap_usize(CV182X_USB2_PHY_BASE, REG_MMIO_SIZE));
    usb::set_usb_dma_to_phys_fn(Some(ep0_dma_virt_to_phys));

    unsafe {
        dwc2::dwc2_probe().map_err(|error| {
            warn!("cvi-camera: DWC2 probe failed: {error:?}");
            StarryError::Io
        })?;
    }

    let mut last_err = None;
    let extras = (0..4)
        .find_map(|attempt| {
            if attempt > 0 {
                busy_wait(Duration::from_micros(1_500_000 * attempt as u64));
            }
            match host::enumerate_topology_only() {
                Ok(extras) => Some(extras),
                Err(e) => {
                    warn!("cvi-camera: USB enumerate failed #{}: {:?}", attempt + 1, e);
                    last_err = Some(e);
                    None
                }
            }
        })
        .ok_or_else(|| {
            warn!(
                "cvi-camera: USB enumerate retries exhausted: {:?}",
                last_err
            );
            if root_usb_device_connected() {
                StarryError::Io
            } else {
                StarryError::NoSuchDevice
            }
        })?;

    let cam = extras.uvc.ok_or(StarryError::NoSuchDevice)?;
    info!(
        "cvi-camera: UVC addr={} VID={:04x} PID={:04x} ep0_mps={}",
        cam.addr, cam.vid, cam.pid, cam.ep0_mps
    );

    let dev = u32::from(cam.addr);
    let ep0 = cam.ep0_mps;
    let cfg_buf = uvc::read_configuration_descriptor(dev, ep0, 1).map_err(|error| {
        warn!("cvi-camera: read configuration descriptor failed: {error:?}");
        StarryError::Io
    })?;
    let cfg_total = u16::from_le_bytes([cfg_buf[2], cfg_buf[3]]) as usize;
    let cfg = &cfg_buf[..cfg_total.min(cfg_buf.len())];
    uvc::set_preferred_max_pixels(DEFAULT_RESOLUTION);
    let mut sel = uvc::parse_uvc_video_stream(cfg, cfg_total).map_err(|error| {
        warn!(
            "cvi-camera: parse UVC video stream failed: {error:?} ({})",
            usb_init_failure_context(error)
        );
        StarryError::Io
    })?;

    if let Some(entities) = uvc::parse_uvc_control_entities(cfg, cfg_total) {
        let tune = uvc::UvcImageTuning {
            brightness: Some(96),
            ..uvc::UvcImageTuning::default()
        };
        let _ = uvc::uvc_init_camera_controls(dev, ep0, &entities, &tune);
    }

    uvc::uvc_start_video_stream(dev, ep0, &mut sel).map_err(|error| {
        warn!("cvi-camera: start UVC stream failed: {error:?}");
        StarryError::Io
    })?;
    info!(
        "cvi-camera: stream ready {}x{} payload={} frame_size={}",
        sel.frame_w, sel.frame_h, sel.negotiated_payload_size, sel.negotiated_frame_size
    );

    // Warm-up frame: discard the first capture after stream start so the
    // isochronous pipeline and DMA buffer are ready for real reads.
    let _ = uvc::uvc_capture_one_frame(dev, ep0, &sel);
    Ok(UsbCameraSession { cam, sel })
}

fn capture_frame(session: &UsbCameraSession) -> StarryResult<&'static [u8]> {
    let dev = u32::from(session.cam.addr);
    let ep0 = session.cam.ep0_mps;
    let mut last_n = 0;
    let mut last_msg = None;
    for attempt in 0..MAX_CAPTURE_TRIES {
        let n = uvc::uvc_capture_one_frame(dev, ep0, &session.sel).map_err(|error| {
            warn!("cvi-camera: capture failed: {error:?}");
            StarryError::Io
        })?;
        last_n = n;
        let frame =
            dwc2_ep0::dma_rx_slice(uvc::UVC_ASSEMBLED_JPEG_DMA_OFF, n).ok_or(StarryError::Io)?;
        let starts_jpeg = n >= 2 && frame[0] == 0xff && frame[1] == 0xd8;
        let ends_jpeg = n >= 2 && frame[n - 2] == 0xff && frame[n - 1] == 0xd9;
        if starts_jpeg && ends_jpeg && n >= MIN_VALID_JPEG_BYTES {
            return Ok(frame);
        }
        last_msg = Some(if !starts_jpeg {
            "first bytes are not ff d8"
        } else if !ends_jpeg {
            "last bytes are not ff d9 (truncated)"
        } else {
            "frame too small"
        });
        warn!(
            "cvi-camera: invalid frame (try #{}/{}, size={}, {}), reset FID",
            attempt + 1,
            MAX_CAPTURE_TRIES,
            n,
            last_msg.unwrap_or("?")
        );
        uvc::reset_frame_continuity();
    }
    warn!(
        "cvi-camera: no complete JPEG after {} retries, size={} {}",
        MAX_CAPTURE_TRIES,
        last_n,
        last_msg.unwrap_or("?")
    );
    Err(StarryError::Io)
}

/// 抓帧失败后的恢复:先分辨是设备掉线还是控制器/通道卡死。
///
/// 返回 `true` = 会话已原地重启、可以继续缓存;`false` = 调用方必须丢弃会话,
/// 由下一次 ioctl 的 `ensure_initialized()` 重新枚举设备。
fn recover_capture_failure(session: &mut UsbCameraSession, restarts: &mut u32) -> bool {
    let dev = u32::from(session.cam.addr);
    let ep0 = session.cam.ep0_mps;
    let vs_if = session.sel.vs_interface;

    if !root_usb_device_connected() {
        warn!(
            "cvi-camera: capture failed with USB device gone; dropping session, next ioctl \
             re-enumerates"
        );
        return false;
    }

    *restarts += 1;
    if *restarts > MAX_STREAM_RESTARTS {
        warn!(
            "cvi-camera: {} consecutive capture failures with USB device present; dropping \
             session, next ioctl re-enumerates",
            *restarts
        );
        return false;
    }

    warn!(
        "cvi-camera: restarting UVC stream in place after capture failure ({}/{})",
        *restarts, MAX_STREAM_RESTARTS
    );
    if let Err(error) = uvc::uvc_stop_streaming(dev, ep0, vs_if) {
        warn!(
            "cvi-camera: uvc_stop_streaming failed: {error:?}; dropping session, next ioctl \
             re-enumerates"
        );
        return false;
    }
    // SET_INTERFACE(alt=0) 不会清抓帧连续性,必须显式复位 FID 跟踪,
    // 否则重启后第一帧会被当成上一帧的尾巴继续拼。
    uvc::reset_frame_continuity();
    match uvc::uvc_start_video_stream(dev, ep0, &mut session.sel) {
        Ok(()) => {
            info!(
                "cvi-camera: UVC stream restarted {}x{} payload={} frame_size={}",
                session.sel.frame_w,
                session.sel.frame_h,
                session.sel.negotiated_payload_size,
                session.sel.negotiated_frame_size
            );
            true
        }
        Err(error) => {
            warn!(
                "cvi-camera: uvc_start_video_stream failed: {error:?}; dropping session, next \
                 ioctl re-enumerates"
            );
            false
        }
    }
}

impl UsbCameraState {
    fn ensure_initialized(&mut self) -> StarryResult<()> {
        if self.session.is_none() {
            self.session = Some(init_usb_camera()?);
        }
        Ok(())
    }

    fn info(&mut self) -> StarryResult<CameraInfo> {
        // 小核在线时:摄像头由小核独占,大核的 DWC2 从未 bring-up,
        // ensure_initialized() 必然失败(ENODEV)。此时直接报固定参数——
        // 真正取帧走 ioctl 9 的小核直通路径,不需要大核 USB 会话。
        if self.smallcore.is_none() {
            self.smallcore = SmallCoreSource::probe();
        }
        if self.smallcore.is_some() {
            return Ok(CameraInfo {
                width: SMALLCORE_FRAME_W,
                height: SMALLCORE_FRAME_H,
                format: CAMERA_FORMAT_MJPEG,
                connected: 1,
            });
        }
        self.ensure_initialized()?;
        let session = self.session.as_ref().ok_or(StarryError::NoSuchDevice)?;
        Ok(CameraInfo {
            width: session.sel.frame_w,
            height: session.sel.frame_h,
            format: CAMERA_FORMAT_MJPEG,
            connected: 1,
        })
    }

    fn frame(&mut self) -> StarryResult<&'static [u8]> {
        self.ensure_initialized()?;
        let session = self.session.as_mut().ok_or(StarryError::NoSuchDevice)?;
        match capture_frame(session) {
            Ok(jpeg) => {
                self.stream_restarts = 0;
                Ok(jpeg)
            }
            Err(error) => {
                // 报错已经由 capture_frame 打过;这里只负责别让缓存的会话
                // 永久卡死:救得回来就留,救不回来就清掉触发重新枚举。
                if !recover_capture_failure(session, &mut self.stream_restarts) {
                    self.session = None;
                    self.stream_restarts = 0;
                }
                Err(error)
            }
        }
    }
}

impl CviCamera {
    pub fn new(jpu: Arc<CviJpu>) -> Self {
        Self {
            state: Mutex::new(UsbCameraState::default()),
            jpu,
        }
    }

    fn write_yuv_frame(
        &self,
        current: &crate::task::UserTaskRef,
        destination: *mut u8,
    ) -> StarryResult<usize> {
        // Hold the camera lock until decode has consumed the static USB DMA
        // slice; another capture must not overwrite it concurrently.
        let mut state = self.state.lock();
        let jpeg = state.frame()?;
        self.jpu.decode_camera_to_user(current, jpeg, destination)
    }

    /// ioctl 9:live 全内核流水线(小核槽取帧 → JPU 解码 YUV420 → 返回
    /// TPU 可 DMA 直读的 PA)。返回值 = YUV 帧的 DMA 地址(0 = 内核内部失败
    /// 或小核不在线——用户态应回退旧路径)。
    fn decode_live(&self, _stage: usize) -> StarryResult<usize> {
        // ── 小核 JPU 直通:小核已在硬件上完成 JPEG→YUV,大核零 JPU 开销。
        let mut state = self.state.lock();
        if state.smallcore.is_none() {
            state.smallcore = SmallCoreSource::probe();
        }
        let sc = match state.smallcore.as_mut() {
            Some(s) => s,
            None => return Ok(0),
        };
        let (yuv_pa, yuv_len) = sc.wait_next_yuv()?;
        // 相机 JPEG 是 422 子采样 → 小核 JPU 输出 I422(614400B),但模型
        // 输入是 I420(460800B)。直通会把 I422 当 I420 喂 TPU——色度平面
        // 错位,检测色彩损坏(live 只测时间不查精度,曾长期未暴露)。
        // 此处做 Y 直拷 + U/V 2:1 SWAR 下采样到 staging 乒乓,TPU DMA 直读。
        if yuv_len != SMALLCORE_I420_SIZE {
            let src_va = axklib::mem::phys_to_virt(
                ax_memory_addr::PhysAddr::from_usize(yuv_pa),
            )
            .as_usize();
            // 源是 JPU DMA 写入:读前失效本核 cache(缓冲隔帧复用)
            ax_hal::mem::dcache_range(
                ax_hal::mem::DCacheOp::Invalidate,
                ax_memory_addr::VirtAddr::from_usize(src_va),
                yuv_len,
            );
            let dst_pa = if sc.staging_flip {
                SMALLCORE_I420_B_PA
            } else {
                SMALLCORE_I420_A_PA
            };
            sc.staging_flip = !sc.staging_flip;
            let dst_va = axklib::mem::phys_to_virt(
                ax_memory_addr::PhysAddr::from_usize(dst_pa),
            )
            .as_usize();
            unsafe {
                // Y 平面直拷(307200B)
                core::ptr::copy_nonoverlapping(
                    src_va as *const u8,
                    dst_va as *mut u8,
                    SMALLCORE_Y_SIZE,
                );
                // U 平面:153600B(320×480) → 76800B(320×240)
                swar_vdown(
                    src_va + SMALLCORE_Y_SIZE,
                    dst_va + SMALLCORE_Y_SIZE,
                    320,
                    480,
                );
                // V 平面:I422 布局 Y(307200) + U(153600) = V 起点 460800。
                // 踩坑:曾写成 2*SMALLCORE_Y_SIZE(614400)——越界读到下一块
                // 乒乓缓冲的 Y 平面,TPU 的红色度 = 错帧亮度 → 黄球色彩特征
                // 消失,置信度从 ~0.8 崩到 0.3(检测看似"失明")。
                swar_vdown(
                    src_va + SMALLCORE_Y_SIZE + 153_600,
                    dst_va + SMALLCORE_Y_SIZE + 76_800,
                    320,
                    480,
                );
            }
            // 本核经 cache 写入 staging:TPU DMA 读物理内存,必须写回
            ax_hal::mem::dcache_range(
                ax_hal::mem::DCacheOp::Clean,
                ax_memory_addr::VirtAddr::from_usize(dst_va),
                SMALLCORE_I420_SIZE,
            );
            drop(state);
            return Ok(dst_pa);
        }
        drop(state);
        Ok(yuv_pa)
    }

    
    fn ioctl_inner(
        &self,
        current: &crate::task::UserTaskRef,
        cmd: u32,
        arg: usize,
    ) -> StarryResult<usize> {
        match cmd {
            CVI_CAMERA_IOCTL_INIT => {
                // 小核在线时跳过 USB 初始化(大核 USB 会干扰小核的 USB 会话)。
                let mut state = self.state.lock();
                if state.smallcore.is_none() {
                    state.smallcore = SmallCoreSource::probe();
                }
                if state.smallcore.is_some() {
                    Ok(0) // 小核模式:无需本核 USB
                } else {
                    state.ensure_initialized()?;
                    Ok(0)
                }
            }
            CVI_CAMERA_IOCTL_GET_INFO => {
                let info = self.state.lock().info()?;
                (arg as *mut CameraInfo).vm_write(current, info)?;
                Ok(0)
            }
            CVI_CAMERA_IOCTL_GET_FRAME => {
                // 小核在线:等 YUV 帧后拷贝到用户态缓冲(与 ioctl 9 共用
                // 等帧+流控逻辑)。返回实际字节数——460800=I420,
                // 614400=I422,由用户态按尺寸自适应格式。
                {
                    let mut state = self.state.lock();
                    if state.smallcore.is_none() {
                        state.smallcore = SmallCoreSource::probe();
                    }
                    if let Some(sc) = state.smallcore.as_mut() {
                        let (yuv_pa, yuv_len) = sc.wait_next_yuv()?;
                        // yuv_pa 是物理地址——必须转虚拟地址才能当指针读
                        // (0x8D000000 直读 = EFAULT,踩过)。
                        let src_va = axklib::mem::phys_to_virt(
                            ax_memory_addr::PhysAddr::from_usize(yuv_pa),
                        )
                        .as_usize();
                        // JPU DMA 写过乒乓缓冲:读前失效本核 cache 行,
                        // 避免读到两帧前的陈旧行(缓冲隔帧复用)。
                        ax_hal::mem::dcache_range(
                            ax_hal::mem::DCacheOp::Invalidate,
                            ax_memory_addr::VirtAddr::from_usize(src_va),
                            yuv_len,
                        );
                        let src: &[u8] = unsafe {
                            core::slice::from_raw_parts(src_va as *const u8, yuv_len)
                        };
                        vm_write_slice(current, arg as *mut u8, src)?;
                        return Ok(yuv_len);
                    }
                }
                let frame = self.state.lock().frame()?;
                vm_write_slice(current, arg as *mut u8, frame)?;
                Ok(frame.len())
            }
            CVI_CAMERA_IOCTL_GET_YUV_FRAME => {
                // 小核在线:等 YUV 帧 + 拷贝到用户态(与 ioctl 9 共用等帧+流控)。
                // 返回实际字节数——460800=I420(小核 JPU),614400=I422(大核)。
                {
                    let mut state = self.state.lock();
                    if state.smallcore.is_none() {
                        state.smallcore = SmallCoreSource::probe();
                    }
                    if let Some(sc) = state.smallcore.as_mut() {
                        let (yuv_pa, yuv_len) = sc.wait_next_yuv()?;
                        // yuv_pa 是物理地址——先转虚拟地址(直读 = EFAULT)。
                        let src_va = axklib::mem::phys_to_virt(
                            ax_memory_addr::PhysAddr::from_usize(yuv_pa),
                        )
                        .as_usize();
                        // JPU DMA 写过乒乓缓冲:读前失效(缓冲隔帧复用)。
                        ax_hal::mem::dcache_range(
                            ax_hal::mem::DCacheOp::Invalidate,
                            ax_memory_addr::VirtAddr::from_usize(src_va),
                            yuv_len,
                        );
                        let src: &[u8] = unsafe {
                            core::slice::from_raw_parts(src_va as *const u8, yuv_len)
                        };
                        vm_write_slice(current, arg as *mut u8, src)?;
                        return Ok(yuv_len);
                    }
                }
                self.write_yuv_frame(current, arg as *mut u8)
            }
            CVI_CAMERA_IOCTL_DECODE_YUV420P_LIVE => self.decode_live(arg),
            _ => Err(StarryError::NotATty),
        }
    }
}

impl DeviceOps for CviCamera {
    fn read_at(&self, _buf: &mut [u8], _offset: u64) -> VfsResult<usize> {
        Ok(0)
    }

    fn write_at(&self, buf: &[u8], _offset: u64) -> VfsResult<usize> {
        Ok(buf.len())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn flags(&self) -> NodeFlags {
        NodeFlags::NON_CACHEABLE | NodeFlags::STREAM
    }

    fn ioctl(&self, current: &crate::task::UserTaskRef, cmd: u32, arg: usize) -> VfsResult<usize> {
        self.ioctl_inner(current, cmd, arg).map_err(Into::into)
    }
}
