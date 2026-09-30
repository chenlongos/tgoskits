//! FDT ramdisk 块设备:把 bootloader 预加载到保留 DRAM 的磁盘镜像直接
//! 注册为块设备(零拷贝)。
//!
//! 设备树约定(见 `licheerv-rd.dtb`):
//! ```dts
//! /memreserve/ 0x8df00000 0x1f00000;      // 镜像区不被页分配器触碰
//! ramdisk@8df00000 {
//!     compatible = "generic,ramdisk";
//!     reg = <0x0 0x8df00000 0x0 0x1f00000>;
//! };
//! ```
//! U-Boot 把(GPT+ext4 的)rootfs 镜像 tftp 到 `reg` 区域后 `bootm`,
//! 本 probe 经线性映射借用该区域作内存盘盘面;后续分区探测与
//! `root=PARTLABEL=...` 选择走 axfs-ng 通用路径。

use rdrive::probe::OnProbeError;
use rdrive::register::ProbeFdt;

use crate::block::ProbeFdtBlock;

pub const BLOCK_SIZE: usize = 512;

pub const DEVICE_NAME: &str = "ramdisk";

crate::model_register!(
    name: "FDT RAMDISK",
    level: ProbeLevel::PostKernel,
    priority: ProbePriority::DEFAULT,
    probe_kinds: &[ProbeKind::Fdt {
        compatibles: &["generic,ramdisk"],
        on_probe: probe_fdt,
    }],
);

fn probe_fdt(probe: ProbeFdt<'_>) -> Result<(), OnProbeError> {
    let info = probe.info();
    let reg = info
        .node
        .regs()
        .into_iter()
        .next()
        .ok_or_else(|| OnProbeError::other(alloc::format!("[{}] has no reg", info.node.name())))?;
    let size = reg.size.ok_or_else(|| {
        OnProbeError::other(alloc::format!("[{}] reg has no size", info.node.name()))
    })? as usize;
    let pa = reg.address as usize;

    // 普通保留 DRAM,非 MMIO:走线性映射直接访问(cached),不做 ioremap。
    let va = axklib::mem::phys_to_virt(ax_memory_addr::PhysAddr::from_usize(pa));
    let len = size & !(BLOCK_SIZE - 1);
    // SAFETY: fdt /memreserve/ 保证该物理区不被内核分配器复用;镜像由
    // bootloader 在内核启动前写入且本 probe 只借用一次(唯一 &mut)。
    let storage: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(va.as_usize() as *mut u8, len) };

    log::info!(
        "ramdisk probe: node={} pa={:#x} len={} blocks={}",
        info.node.name(),
        pa,
        len,
        len / BLOCK_SIZE
    );

    let ctrl = ramdisk::RamDiskCtrl::with_external_storage(DEVICE_NAME, BLOCK_SIZE, storage);
    let idx = probe
        .register_block(ctrl)
        .map_err(|e| OnProbeError::other(alloc::format!("register_block failed: {e:?}")))?;
    log::info!("ramdisk block controller registered: idx={:?}", idx);
    Ok(())
}
