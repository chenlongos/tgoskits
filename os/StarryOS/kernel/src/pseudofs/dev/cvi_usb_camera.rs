use core::{any::Any, time::Duration};

use ax_errno::AxError;
use ax_memory_addr::{PhysAddr, VirtAddr};
use ax_runtime::hal::{mem::virt_to_phys, time::busy_wait};
use ax_sync::Mutex;
use axfs_ng_vfs::{NodeFlags, VfsResult};
use sg200x_bsp::{
    gpio::{Direction, GPIO, GPIO1_BASE},
    jpu::{
        JpuDecoder,
        regs::{JPU_REG_BASE, VC_REG_BASE},
    },
    pinmux::{FMUX_USB_VBUS_DET, Pinmux},
    soc::{
        CLKGEN_BASE, CV182X_USB2_PHY_BASE, DWC2_BASE, FMUX_BASE, IOBLK_BASE, IOBLK_GRTC_BASE,
        TOP_BASE,
    },
    usb::{
        self,
        class::{uvc, uvc_session::UvcSession},
        host::dwc2,
    },
};
use starry_vm::{VmMutPtr, vm_write_slice};
use tock_registers::interfaces::Writeable;

use crate::pseudofs::DeviceOps;

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
/// Default resolution cap (640×480 = 307200 pixels) guiding UVC frame selection.
/// Also the JPU 1 MiB DMA pool's hard decode ceiling — larger frames make
/// `jpu_alloc` fail.
const DEFAULT_RESOLUTION: u32 = 640 * 480;

pub const CVI_CAMERA_IOCTL_INIT: u32 = 1;
pub const CVI_CAMERA_IOCTL_GET_INFO: u32 = 2;
pub const CVI_CAMERA_IOCTL_GET_FRAME: u32 = 3;
pub const CVI_CAMERA_IOCTL_GET_YUV_FRAME: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraInfo {
    pub width: u16,
    pub height: u16,
    /// 1 means MJPEG.
    pub format: u8,
    pub connected: u8,
}

#[derive(Default)]
struct UsbCameraState {
    /// 当前 UVC 会话。`None` = 未初始化，或上次因拔插/错误拆除后待重新建立。
    session: Option<UvcSession>,
    jpu: Option<JpuDecoder>,
    /// 板级平台初始化（时钟/PHY/VBUS/MMIO 基址/DWC2 probe）只做一次；热拔插重 open
    /// 时不重复——控制器由 `UvcSession::open` 内部 `dwc2_host_init` 重新 bring-up。
    platform_inited: bool,
}

fn jpu_dma_to_phys(v: usize) -> usize {
    virt_to_phys(VirtAddr::from(v)).as_usize()
}

pub struct CviCamera {
    state: Mutex<UsbCameraState>,
}

fn ep0_dma_virt_to_phys(p: *const u8) -> u32 {
    virt_to_phys(VirtAddr::from(p as usize)).as_usize() as u32
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

impl UsbCameraState {
    /// 板级平台初始化（幂等，仅首次执行）。
    fn platform_init_once(&mut self) -> VfsResult<()> {
        if self.platform_inited {
            return Ok(());
        }
        unsafe {
            enable_usb_clocks_cv181x();
            cvitek_usb_top_host_bringup();
        }
        pinmux_usb_vbus_det_gpio_output_prep();
        enable_usb_vbus_gpio();
        ax_task::sleep(Duration::from_micros(2_000_000));

        usb::set_dwc2_base_virt(iomap_usize(DWC2_BASE, REG_MMIO_SIZE));
        usb::set_cv182x_phy_base_virt(iomap_usize(CV182X_USB2_PHY_BASE, REG_MMIO_SIZE));
        usb::set_usb_dma_to_phys_fn(Some(ep0_dma_virt_to_phys));

        unsafe {
            dwc2::dwc2_probe().map_err(|e| {
                warn!("cvi-camera: DWC2 probe failed: {e:?}");
                AxError::Io
            })?;
        }
        self.platform_inited = true;
        Ok(())
    }

    /// 确保会话就绪：平台初始化 + `UvcSession::open`（枚举/协商/warmup）。
    /// 已有会话则直接返回；会话为 None（未初始化或上次 teardown 后）则重新建立。
    fn ensure_initialized(&mut self) -> VfsResult<()> {
        if self.session.is_some() {
            return Ok(());
        }
        self.platform_init_once()?;

        // 帧选择偏好：只设像素数上限（同时也是 JPU 1 MiB DMA pool 的硬解码上限），
        // 让驱动自动挑 "≤ 上限里面积最大" 的那档，而不是钉死精确尺寸：
        //   - 相机 A(0c45:64ab) 有 640×480（正好等于上限）→ 仍命中 640×480，行为不变。
        //   - 相机 B(b349:b182) 无 640×480。若用 set_preferred_frame_size(640,480) 钉死
        //     精确尺寸，rank() 在命不中精确尺寸时会退化成 `pref_area - area`（面积越小
        //     分越高），把 B 推到最小的 320×240；改走 max-pixels 分支后，B 落在 ≤640×480
        //     里最大的 640×360，仍在 JPU pool 内，也更接近 akars 期望的几何。
        // 精确尺寸偏好仅在相机确有该档时才安全，故这里显式关闭 (0, 0)。
        uvc::set_preferred_max_pixels(DEFAULT_RESOLUTION);
        uvc::set_preferred_frame_size(0, 0);
        uvc::set_preferred_frame_interval(333_333);

        let tune = uvc::UvcImageTuning {
            brightness: Some(96),
            ..uvc::UvcImageTuning::default()
        };
        match UvcSession::open(&tune) {
            Ok(s) => {
                let sel = s.selection();
                info!(
                    "cvi-camera: UVC session ready addr={} {}x{} fmt={} iv={} alt={}",
                    s.dev(),
                    sel.frame_w,
                    sel.frame_h,
                    if sel.is_mjpeg { "MJPG" } else { "YUYV" },
                    sel.frame_interval,
                    sel.alt_setting,
                );
                self.session = Some(s);
                Ok(())
            }
            Err(e) => {
                warn!("cvi-camera: UvcSession::open failed: {e:?}");
                Err(AxError::Io)
            }
        }
    }

    fn info(&mut self) -> VfsResult<CameraInfo> {
        self.ensure_initialized()?;
        let session = self.session.as_ref().ok_or(AxError::BadState)?;
        Ok(CameraInfo {
            width: session.selection().frame_w,
            height: session.selection().frame_h,
            format: CAMERA_FORMAT_MJPEG,
            connected: u8::from(session.connected()),
        })
    }

    /// 抓 1 帧 MJPEG，带错误恢复 + 热拔插透明处理。
    ///
    /// `UvcSession::capture_recovering` 内部按 sdmmc 风格重试（中止通道 + 重协商）；
    /// 返回 `Disconnected`/`NeedsReenum` 时拆除会话并重新 `open` 再试一次（即热插回）。
    /// 恢复耗尽时返回 `AxError::Io`。
    fn frame(&mut self) -> VfsResult<&'static [u8]> {
        for attempt in 0..2u32 {
            self.ensure_initialized()?;
            let res = self
                .session
                .as_mut()
                .expect("session set by ensure_initialized")
                .capture_recovering();
            match res {
                Ok(jpeg) => {
                    let n = jpeg.len();
                    let bad_header = n < 2 || jpeg[0] != 0xff || jpeg[1] != 0xd8;
                    let bad_footer = n < 2 || jpeg[n - 2] != 0xff || jpeg[n - 1] != 0xd9;
                    if n < MIN_VALID_JPEG_BYTES || bad_header || bad_footer {
                        warn!(
                            "cvi-camera: invalid frame size={n} (attempt {}); retry",
                            attempt + 1
                        );
                        continue;
                    }
                    return Ok(jpeg);
                }
                Err(e) => {
                    warn!(
                        "cvi-camera: capture_recovering failed (attempt {}): {e:?}; teardown + \
                         re-init",
                        attempt + 1
                    );
                    if let Some(mut s) = self.session.take() {
                        s.teardown();
                    }
                }
            }
        }
        warn!("cvi-camera: frame recovery exhausted (camera unplugged?)");
        Err(AxError::Io)
    }

    fn ensure_jpu(&mut self) -> VfsResult<&mut JpuDecoder> {
        if self.jpu.is_none() {
            let jpu_v = iomap_usize(JPU_REG_BASE, REG_MMIO_SIZE);
            let top_v = iomap_usize(TOP_BASE, TOP_MMIO_SIZE);
            let vc_v = iomap_usize(VC_REG_BASE, REG_MMIO_SIZE);
            let decoder = unsafe {
                JpuDecoder::new_at(jpu_v, top_v, vc_v, jpu_dma_to_phys).map_err(|e| {
                    warn!("cvi-camera: JPU init failed: {e}");
                    AxError::Io
                })?
            };
            self.jpu = Some(decoder);
        }
        Ok(self.jpu.as_mut().unwrap())
    }

    fn yuv_frame(&mut self) -> VfsResult<&'static [u8]> {
        let jpeg = self.frame()?;
        let jpu = self.ensure_jpu()?;
        let result = match jpu.decode(jpeg) {
            Ok(r) => r,
            Err(e) => {
                // 解卡：sg200x-bsp 的 `start_decode` 在 `START_PIC` 前不做 `START_INIT` 软
                // 复位，一旦某帧（典型为长度是 256 整数倍、BBC 页边界缺前瞻）把引擎挂死，
                // 之后每帧都会超时。丢弃 decoder，下一帧 `ensure_jpu` 重新 `new_at` 会走
                // `hardware_init_at -> wait_sw_reset_done`（START_INIT）复位并回收 buffer。
                warn!("cvi-camera: JPU decode failed: {e}; reset JPU");
                self.jpu = None;
                return Err(AxError::Io);
            }
        };
        info!(
            "cvi-camera: JPU decode OK {}x{} yuv={} bytes",
            result.width,
            result.height,
            result.yuv_data.len()
        );
        Ok(result.yuv_data)
    }
}

impl CviCamera {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(UsbCameraState::default()),
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

    fn ioctl(&self, cmd: u32, arg: usize) -> VfsResult<usize> {
        match cmd {
            CVI_CAMERA_IOCTL_INIT => {
                self.state.lock().ensure_initialized()?;
                Ok(0)
            }
            CVI_CAMERA_IOCTL_GET_INFO => {
                let info = self.state.lock().info()?;
                (arg as *mut CameraInfo).vm_write(info)?;
                Ok(0)
            }
            CVI_CAMERA_IOCTL_GET_FRAME => {
                let frame = self.state.lock().frame()?;
                vm_write_slice(arg as *mut u8, frame)?;
                Ok(frame.len())
            }
            CVI_CAMERA_IOCTL_GET_YUV_FRAME => {
                let yuv = self.state.lock().yuv_frame()?;
                vm_write_slice(arg as *mut u8, yuv)?;
                Ok(yuv.len())
            }
            _ => Err(AxError::InvalidInput),
        }
    }
}
