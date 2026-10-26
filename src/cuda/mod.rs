//! Optional resident FP32 CUDA training for NVIDIA GPUs.
//!
//! Uses only the CUDA driver API through `libcuda.so.1` (loaded with dlopen) plus kernels
//! compiled ahead of time by `build.rs`, so the crate gains no new runtime dependency and
//! non-CUDA builds never link or probe the driver. Parameters, activations, gradients and
//! Adam state stay resident on the device; each step uploads one batched crop payload and
//! reads back four statistics values. There is no automatic CPU fallback: any driver,
//! launch or numerical failure is returned as an error.
use crate::{backend::Step, model::Model, parameterization::FlowConfig, runtime::Adam};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    ffi::{c_char, c_int, c_uint, c_void, CStr, CString},
    sync::Arc,
    time::Instant,
};

const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/feral.cubin"));
const PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/feral.ptx"));
const NVCC: &str = env!("FERAL_CUDA_NVCC");
const BUILD_ARCH: &str = env!("FERAL_CUDA_ARCH");
// Minimal dlopen binding so the backend needs no extra crate and non-CUDA builds stay clean.
#[allow(non_camel_case_types)]
type c_void_p = *mut c_void;
const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> c_void_p;
    fn dlsym(handle: c_void_p, symbol: *const c_char) -> c_void_p;
    fn dlerror() -> *const c_char;
}

const TILE: usize = 16; // thread block
const TILE_OUT: usize = 32; // register-blocked output tile
const BATCH_THREADS: usize = 256;
/// Resident device allocation ceiling; the owner-approved working-memory ceiling is 3 GiB
/// for the whole process, and this leaves room for the host-side checkpoint and dataset.
/// Default resident ceiling. The desktop path may raise it with `FERAL_CUDA_BUDGET_MB`
/// (256..=9216 MiB); the value actually in force is reported in `backend_info`.
const DEFAULT_DEVICE_BYTES: usize = 1024 * 1024 * 1024;
fn device_budget_bytes() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("FERAL_CUDA_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map(|mb| mb.clamp(256, 9216) * 1024 * 1024)
            .unwrap_or(DEFAULT_DEVICE_BYTES)
    })
}

type CuResult = c_int;
type CuDevice = c_int;
type CuDeviceptr = u64;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

type FnInit = unsafe extern "C" fn(c_uint) -> CuResult;
type FnGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
type FnGetDevice = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
type FnGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;
type FnGetAttr = unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult;
type FnTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult;
type FnCtxCreate = unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;
type FnCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type FnCtxDestroy = unsafe extern "C" fn(CuContext) -> CuResult;
type FnModuleLoad = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
type FnGetFunction = unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type FnMemAlloc = unsafe extern "C" fn(*mut CuDeviceptr, usize) -> CuResult;
type FnMemcpy = unsafe extern "C" fn(CuDeviceptr, *const c_void, usize) -> CuResult;
type FnMemcpyOut = unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize) -> CuResult;
type FnLaunch = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type FnErrString = unsafe extern "C" fn(c_int, *mut *const c_char) -> CuResult;
type FnDriverVersion = unsafe extern "C" fn(*mut c_int) -> CuResult;
type FnMemInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult;
type FnMemcpyAsync = unsafe extern "C" fn(CuDeviceptr, *const c_void, usize, CuStream) -> CuResult;
type FnEventCreate = unsafe extern "C" fn(*mut CuEvent, c_uint) -> CuResult;
type FnEventRecord = unsafe extern "C" fn(CuEvent, CuStream) -> CuResult;
type FnEventElapsed = unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult;
type FnEventSync = unsafe extern "C" fn(CuEvent) -> CuResult;
type FnMemcpyDtoHAsync =
    unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize, CuStream) -> CuResult;
type FnHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CuResult;
type CuEvent = *mut c_void;
type CuStream = *mut c_void;

struct Driver {
    init: FnInit,
    device_count: FnGetCount,
    get_device: FnGetDevice,
    get_name: FnGetName,
    get_attr: FnGetAttr,
    total_mem: FnTotalMem,
    ctx_create: FnCtxCreate,
    // Kept for completeness of the FFI surface; context teardown uses this pointer.
    _ctx_destroy: FnCtxDestroy,
    ctx_set_current: FnCtxSetCurrent,
    module_load: FnModuleLoad,
    get_function: FnGetFunction,
    mem_alloc: FnMemAlloc,
    ht_o: FnMemcpy,
    dt_h: FnMemcpyOut,
    launch: FnLaunch,
    err_string: FnErrString,
    driver_version: FnDriverVersion,
    mem_info: FnMemInfo,
    ht_o_async: FnMemcpyAsync,
    ev_create: FnEventCreate,
    ev_record: FnEventRecord,
    ev_elapsed: FnEventElapsed,
    ev_sync: FnEventSync,
    dt_h_async: FnMemcpyDtoHAsync,
    host_alloc: FnHostAlloc,
}

unsafe fn symbol(handle: *mut c_void, name: &str) -> Result<*mut c_void> {
    let cname = CString::new(name)?;
    let p = unsafe { dlsym(handle, cname.as_ptr()) };
    if p.is_null() {
        let detail = unsafe {
            let e = dlerror();
            if e.is_null() {
                "not exported".to_string()
            } else {
                CStr::from_ptr(e).to_string_lossy().into_owned()
            }
        };
        bail!("libcuda.so.1 does not export {name}: {detail}");
    }
    Ok(p)
}

macro_rules! resolve {
    ($h:expr, $name:expr, $ty:ty) => {{
        let raw = unsafe { symbol($h, $name).with_context(|| format!("resolve {}", $name))? };
        // SAFETY: the driver ABI for these entry points is stable; the symbol was found.
        unsafe { std::mem::transmute::<*mut c_void, $ty>(raw) }
    }};
}

impl Driver {
    fn load() -> Result<Self> {
        let name = CString::new("libcuda.so.1")?;
        // SAFETY: plain dlopen; the handle lives for the process lifetime (never dlclose'd).
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            let detail = unsafe {
                let e = dlerror();
                if e.is_null() {
                    "unknown dlopen failure".to_string()
                } else {
                    CStr::from_ptr(e).to_string_lossy().into_owned()
                }
            };
            bail!("no CUDA driver library (libcuda.so.1): {detail}; install the NVIDIA driver");
        }
        let driver = Self {
            init: resolve!(handle, "cuInit", FnInit),
            device_count: resolve!(handle, "cuDeviceGetCount", FnGetCount),
            get_device: resolve!(handle, "cuDeviceGet", FnGetDevice),
            get_name: resolve!(handle, "cuDeviceGetName", FnGetName),
            get_attr: resolve!(handle, "cuDeviceGetAttribute", FnGetAttr),
            total_mem: resolve!(handle, "cuDeviceTotalMem_v2", FnTotalMem),
            ctx_create: resolve!(handle, "cuCtxCreate_v2", FnCtxCreate),
            _ctx_destroy: resolve!(handle, "cuCtxDestroy_v2", FnCtxDestroy),
            ctx_set_current: resolve!(handle, "cuCtxSetCurrent", FnCtxSetCurrent),
            module_load: resolve!(handle, "cuModuleLoadData", FnModuleLoad),
            get_function: resolve!(handle, "cuModuleGetFunction", FnGetFunction),
            mem_alloc: resolve!(handle, "cuMemAlloc_v2", FnMemAlloc),
            ht_o: resolve!(handle, "cuMemcpyHtoD_v2", FnMemcpy),
            dt_h: resolve!(handle, "cuMemcpyDtoH_v2", FnMemcpyOut),
            launch: resolve!(handle, "cuLaunchKernel", FnLaunch),
            err_string: resolve!(handle, "cuGetErrorString", FnErrString),
            driver_version: resolve!(handle, "cuDriverGetVersion", FnDriverVersion),
            mem_info: resolve!(handle, "cuMemGetInfo_v2", FnMemInfo),
            ht_o_async: resolve!(handle, "cuMemcpyHtoDAsync_v2", FnMemcpyAsync),
            ev_create: resolve!(handle, "cuEventCreate", FnEventCreate),
            ev_record: resolve!(handle, "cuEventRecord", FnEventRecord),
            ev_elapsed: resolve!(handle, "cuEventElapsedTime", FnEventElapsed),
            ev_sync: resolve!(handle, "cuEventSynchronize", FnEventSync),
            dt_h_async: resolve!(handle, "cuMemcpyDtoHAsync_v2", FnMemcpyDtoHAsync),
            host_alloc: resolve!(handle, "cuMemHostAlloc", FnHostAlloc),
        };
        Ok(driver)
    }
    fn check(&self, code: CuResult, what: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let mut text: *const c_char = std::ptr::null();
        // SAFETY: CUDA writes a static string pointer for a known result code.
        let detail = match unsafe { (self.err_string)(code, &mut text) } {
            0 if !text.is_null() => unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned(),
            _ => format!("CUDA error {code}"),
        };
        bail!("{what} failed: {detail} ({code})")
    }
}

struct ContextGuard {
    destroy: FnCtxDestroy,
    ctx: CuContext,
}
impl Drop for ContextGuard {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: destroys the context that owns every buffer and module created below.
            unsafe {
                (self.destroy)(self.ctx);
            }
        }
    }
}

#[derive(Clone)]
struct Spec {
    ci: u32,
    co: u32,
    k: u32,
    d: u32,
    s: u32,
    w: u32,
    b: u32,
}

struct Kernels {
    features: CuFunction,
    pool: CuFunction,
    forward: CuFunction,
    loss: CuFunction,
    dz: CuFunction,
    dw: CuFunction,
    dx: CuFunction,
    stats: CuFunction,
    adam: CuFunction,
}

#[derive(Clone, Copy)]
struct Buf {
    ptr: CuDeviceptr,
    len: usize,
}

macro_rules! kargs {
    ($($x:ident),* $(,)?) => {{
        let mut v: Vec<*mut c_void> = Vec::new();
        $( v.push(std::ptr::addr_of_mut!($x) as *mut c_void); )*
        v
    }};
}

pub struct CudaTrainer {
    driver: Arc<Driver>,
    ctx: ContextGuard,
    k: Kernels,
    weights: Buf,
    m: Buf,
    v: Buf,
    grad: Buf,
    data: Buf,
    stats: Buf,
    loss_parts: Buf,
    acts: Vec<Buf>,
    pooled: Vec<Buf>,
    z: Vec<Buf>,
    dh: Vec<Buf>,
    dz: Vec<Buf>,
    specs: Vec<Spec>,
    n: usize,
    items: usize,
    frames: u32,
    block: u32,
    count: usize,
    flow: FlowConfig,
    pub step: usize,
    lr: f32,
    info: Value,
    allocated: usize,
    pub total_wall_ms: f64,
    unknown_buf: Buf,
    trig: Vec<f32>,
    ev_start: CuEvent,
    ev_end: CuEvent,
    pub last_gpu_ms: f64,
    pub last_host_ms: f64,
    // One-step-deferred statistics readback: the device never drains at a step boundary.
    pinned: *mut c_void,
    pin_event: CuEvent,
    pin_pending: bool,
    prev_stats: [f32; 4],
    reported_steps: usize,
    prev_wall_ms: f64,
}

impl CudaTrainer {
    pub fn new(
        model: &Model,
        adam: &Adam,
        n: usize,
        items: usize,
        flow: FlowConfig,
    ) -> Result<Self> {
        model.validate()?;
        flow.validate()?;
        ensure!(
            n > 0
                && n <= crate::model::MAX_WINDOW_SAMPLES
                && n.is_multiple_of(model.architecture.block),
            "invalid CUDA crop size"
        );
        ensure!((1..=64).contains(&items), "CUDA batch size must be 1..64");
        ensure!(
            adam.m.len() == model.count() && adam.v.len() == model.count(),
            "CUDA optimizer shape mismatch"
        );
        ensure!(
            adam.m.iter().chain(&adam.v).all(|v| v.is_finite())
                && adam.v.iter().all(|v| *v >= 0.)
                && adam.lr.is_finite()
                && adam.lr > 0.,
            "invalid CUDA optimizer state"
        );
        let driver = Arc::new(Driver::load()?);
        // SAFETY: cuInit is idempotent and must precede every other driver call.
        driver.check(unsafe { (driver.init)(0) }, "cuInit")?;
        let mut count = 0;
        driver.check(
            unsafe { (driver.device_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        ensure!(
            count > 0,
            "no CUDA device found; the CUDA backend never falls back to CPU"
        );
        let mut dev: CuDevice = 0;
        driver.check(unsafe { (driver.get_device)(&mut dev, 0) }, "cuDeviceGet")?;
        let mut name = [0 as c_char; 256];
        driver.check(
            unsafe { (driver.get_name)(name.as_mut_ptr(), 256, dev) },
            "cuDeviceGetName",
        )?;
        // SAFETY: the driver null-terminates the name within the provided buffer.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut major = 0;
        let mut minor = 0;
        let mut sms = 0;
        let mut clock_khz = 0;
        driver.check(
            unsafe { (driver.get_attr)(&mut major, 75, dev) },
            "compute capability major",
        )?;
        driver.check(
            unsafe { (driver.get_attr)(&mut minor, 76, dev) },
            "compute capability minor",
        )?;
        driver.check(
            unsafe { (driver.get_attr)(&mut sms, 16, dev) },
            "multiprocessor count",
        )?;
        driver.check(
            unsafe { (driver.get_attr)(&mut clock_khz, 13, dev) },
            "clock rate",
        )?;
        let mut total_mem = 0usize;
        driver.check(
            unsafe { (driver.total_mem)(&mut total_mem, dev) },
            "cuDeviceTotalMem",
        )?;
        let mut version = 0;
        driver.check(
            unsafe { (driver.driver_version)(&mut version) },
            "cuDriverGetVersion",
        )?;
        let mut ctx: CuContext = std::ptr::null_mut();
        // SAFETY: creates a fresh context bound to this thread and the selected device.
        driver.check(
            unsafe { (driver.ctx_create)(&mut ctx, 0, dev) },
            "cuCtxCreate",
        )?;
        let ctx = ContextGuard {
            destroy: driver._ctx_destroy,
            ctx,
        };
        // Prefer the precompiled cubin; fall back to PTX so a newer driver or a different
        // GPU can still JIT the same kernels instead of silently changing backend.
        let mut module: CuModule = std::ptr::null_mut();
        let cubin = driver.check(
            unsafe { (driver.module_load)(&mut module, CUBIN.as_ptr() as *const c_void) },
            "cuModuleLoadData(cubin)",
        );
        let loader = match cubin {
            Ok(()) => "cubin".to_string(),
            Err(cubin_err) => {
                let mut ptx = PTX.to_vec();
                ptx.push(0);
                driver.check(
                    unsafe { (driver.module_load)(&mut module, ptx.as_ptr() as *const c_void) },
                    &format!("cuModuleLoadData(cubin failed: {cubin_err:#}; ptx)"),
                )?;
                "ptx-jit".to_string()
            }
        };
        let load = |name: &str| -> Result<CuFunction> {
            let cname = CString::new(name)?;
            let mut f: CuFunction = std::ptr::null_mut();
            driver.check(
                unsafe { (driver.get_function)(&mut f, module, cname.as_ptr()) },
                &format!("cuModuleGetFunction({name})"),
            )?;
            Ok(f)
        };
        let k = Kernels {
            features: load("feral_features")?,
            pool: load("feral_pool")?,
            forward: load("feral_conv_forward")?,
            loss: load("feral_loss_grad")?,
            dz: load("feral_activation_grad")?,
            dw: load("feral_weight_grad")?,
            dx: load("feral_input_grad")?,
            stats: load("feral_statistics")?,
            adam: load("feral_adam")?,
        };
        let frames = n / model.architecture.block;
        let mut specs = Vec::with_capacity(model.layers.len());
        let mut offset = 0;
        for (j, l) in model.layers.iter().enumerate() {
            let s = if j > 0 && j + 1 < model.layers.len() {
                model.architecture.scale(j - 1)
            } else {
                1
            };
            specs.push(Spec {
                ci: l.input as u32,
                co: l.output as u32,
                k: l.kernel as u32,
                d: l.dilation as u32,
                s: s as u32,
                w: offset as u32,
                b: (offset + l.w.len()) as u32,
            });
            offset += l.w.len() + l.b.len();
        }
        let mut allocated = 0usize;
        let mut alloc = |len: usize| -> Result<Buf> {
            let bytes = len
                .max(1)
                .checked_mul(4)
                .context("CUDA allocation overflow")?;
            allocated += bytes;
            ensure!(
                allocated <= device_budget_bytes(),
                "CUDA allocation would exceed the 1 GiB resident budget; reduce crop, batch or width"
            );
            let mut ptr: CuDeviceptr = 0;
            // SAFETY: allocate bytes validated above; freed with the owning context.
            driver.check(unsafe { (driver.mem_alloc)(&mut ptr, bytes) }, "cuMemAlloc")?;
            Ok(Buf { ptr, len })
        };
        let weights = alloc(offset)?;
        let m = alloc(offset)?;
        let v = alloc(offset)?;
        let grad = alloc(offset)?;
        let data = alloc(3 * items * n + 4 * items)?;
        let stats = alloc(4)?;
        let parts = (items * n).div_ceil(BATCH_THREADS);
        let loss_parts = alloc(parts)?;
        let mut acts = vec![alloc(items * frames * model.layers[0].input)?];
        let mut dh = vec![alloc(1)?];
        let mut pooled = Vec::with_capacity(specs.len());
        let mut z = Vec::with_capacity(specs.len());
        let mut dz = Vec::with_capacity(specs.len());
        for s in &specs {
            let cf = frames.div_ceil(s.s as usize);
            acts.push(alloc(items * frames * s.co as usize)?);
            dh.push(alloc(items * frames * s.co as usize)?);
            pooled.push(alloc(if s.s > 1 {
                items * cf * s.ci as usize
            } else {
                1
            })?);
            z.push(alloc(items * cf * s.co as usize)?);
            dz.push(alloc(items * cf * s.co as usize)?);
        }
        let p = model.parameters();
        Self::upload(&driver, weights, &p, "parameters")?;
        Self::upload(&driver, m, &adam.m, "adam m")?;
        Self::upload(&driver, v, &adam.v, "adam v")?;
        let unknown_buf = alloc(items.max(1))?;
        drop(alloc);
        let mut pinned: *mut c_void = std::ptr::null_mut();
        driver.check(
            unsafe { (driver.host_alloc)(&mut pinned, 64, 0) },
            "cuMemHostAlloc",
        )?;
        let ev_start = Self::new_event(&driver)?;
        let ev_end = Self::new_event(&driver)?;
        let pin_event = Self::new_event(&driver)?;
        let mut free = 0usize;
        let mut total_free = 0usize;
        if unsafe { (driver.mem_info)(&mut free, &mut total_free) } == 0 {
            // Values are advisory only; allocation is the authoritative budget check.
        }
        let info = json!({
            "backend": "cuda-f32-resident",
            "precision": "fp32",
            "device": name,
            "compute_capability": format!("{major}.{minor}"),
            "multiprocessors": sms,
            "clock_mhz": clock_khz / 1000,
            "driver_version": format!("{}.{}", version / 1000, (version % 1000) / 10),
            "device_total_memory_bytes": total_mem,
            "device_free_memory_bytes": free,
            "kernel_build": loader,
            "kernel_arch": format!("sm_{BUILD_ARCH}"),
            "nvcc": NVCC,
            "tile": TILE_OUT,
            "batch_threads": BATCH_THREADS,
            "resident_budget_bytes": device_budget_bytes(),
            "allocated_bytes": allocated,
            "batched": true,
            "note": "driver API via dlopen; kernels AOT-compiled by build.rs; fmad disabled",
        });
        Ok(Self {
            driver,
            ctx,
            k,
            weights,
            m,
            v,
            grad,
            data,
            stats,
            loss_parts,
            acts,
            pooled,
            z,
            dh,
            dz,
            specs,
            n,
            items,
            frames: frames as u32,
            block: model.architecture.block as u32,
            count: offset,
            flow,
            step: adam.step,
            lr: adam.lr,
            info,
            allocated,
            total_wall_ms: 0.,
            unknown_buf,
            trig: vec![0.; 4 * items.max(1)],
            ev_start,
            ev_end,
            last_gpu_ms: 0.,
            last_host_ms: 0.,
            pinned,
            pin_event,
            pin_pending: false,
            prev_stats: [0.; 4],
            reported_steps: 0,
            prev_wall_ms: 0.,
        })
    }
    fn new_event(driver: &Driver) -> Result<CuEvent> {
        let mut e: CuEvent = std::ptr::null_mut();
        driver.check(unsafe { (driver.ev_create)(&mut e, 0) }, "cuEventCreate")?;
        Ok(e)
    }
    fn upload(driver: &Driver, buf: Buf, host: &[f32], what: &str) -> Result<()> {
        ensure!(host.len() <= buf.len, "CUDA upload length mismatch: {what}");
        // SAFETY: host slice length and byte count are checked against the allocation.
        driver.check(
            unsafe { (driver.ht_o)(buf.ptr, host.as_ptr() as *const c_void, host.len() * 4) },
            &format!("cuMemcpyHtoD({what})"),
        )
    }
    fn download(driver: &Driver, buf: Buf, host: &mut [f32], what: &str) -> Result<()> {
        ensure!(
            host.len() <= buf.len,
            "CUDA download length mismatch: {what}"
        );
        // SAFETY: destination slice matches the checked allocation size.
        driver.check(
            unsafe { (driver.dt_h)(host.as_mut_ptr() as *mut c_void, buf.ptr, host.len() * 4) },
            &format!("cuMemcpyDtoH({what})"),
        )
    }
    /// Drivers keep one current context per thread, so every entry point re-binds this trainer's
    /// context. Without it a second live trainer silently invalidates the first one's handles.
    fn activate(&self) -> Result<()> {
        self.driver.check(
            unsafe { (self.driver.ctx_set_current)(self.ctx.ctx) },
            "cuCtxSetCurrent",
        )
    }
    fn launch(
        &self,
        what: &str,
        f: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &mut [*mut c_void],
    ) -> Result<()> {
        // SAFETY: arguments were built to match the compiled kernel signature; the array
        // outlives the call, and cuLaunchKernel copies parameter values synchronously.
        self.driver.check(
            unsafe {
                (self.driver.launch)(
                    f,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            &format!("cuLaunchKernel({what})"),
        )
    }
    fn flush_pending(&self) -> Result<()> {
        if self.pin_pending {
            self.driver.check(
                unsafe { (self.driver.ev_sync)(self.pin_event) },
                "cuEventSynchronize",
            )?;
        }
        Ok(())
    }
    pub fn info(&self) -> Value {
        let mut info = self.info.clone();
        info["allocated_bytes"] = self.allocated.into();
        info["batch"] = self.items.into();
        info["total_step_wall_ms"] = self.total_wall_ms.into();
        info["reported_loss_lag_steps"] = 1.into();
        info["last_step_gpu_ms"] = self.last_gpu_ms.into();
        info["last_step_host_ms"] = self.last_host_ms.into();
        info
    }
    /// Forward-only field evaluation for sampling: payload upload and network forward, then the
    /// prediction comes back. No loss reduction, no backward pass, no optimizer.
    ///
    /// A pending statistics readback from an earlier training step is deliberately left in place:
    /// it belongs to that step, and resolving it here would change the reported loss sequence.
    pub fn predict(&mut self, x: &[f32], mask: &[f32], t: f32) -> Result<Vec<f32>> {
        self.activate()?;
        self.forward_pass(x, mask, x, &[t], false)?;
        self.read_prediction()
    }
    pub fn gradients(
        &mut self,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        t: &[f32],
    ) -> Result<Step> {
        self.execute(clean, mask, noise, t, false)
    }
    pub fn train_step(
        &mut self,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        t: &[f32],
    ) -> Result<Step> {
        self.execute(clean, mask, noise, t, true)
    }
    /// Uploads the payload and runs the network forward, leaving the prediction in the last
    /// activation buffer. Returns the upload milliseconds and the instant the kernels were
    /// enqueued, which `execute` folds into its device and wall timings. `predict` uses this alone:
    /// evaluating a field needs no loss reduction, backward pass or optimizer.
    fn forward_pass(
        &mut self,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        t: &[f32],
        timed: bool,
    ) -> Result<(f64, Instant)> {
        let total_samples = self.items * self.n;
        ensure!(
            t.len() == self.items
                && clean.len() == total_samples
                && mask.len() == total_samples
                && noise.len() == total_samples,
            "CUDA input shape mismatch"
        );
        ensure!(
            t.iter().all(|t| t.is_finite() && (0.0..=1.).contains(t))
                && clean.iter().chain(noise).all(|v| v.is_finite())
                && mask.iter().all(|&v| v == 0. || v == 1.),
            "invalid CUDA input values"
        );
        let start = Instant::now();
        self.trig.clear();
        for &ti in t {
            self.trig.extend([
                ti,
                1. - ti,
                (ti * std::f32::consts::TAU).sin(),
                (ti * std::f32::consts::TAU).cos(),
            ]);
        }
        let mut unknown = vec![0f32; self.items];
        for (i, slot) in unknown.iter_mut().enumerate() {
            let row = &mask[i * self.n..(i + 1) * self.n];
            *slot = row.iter().filter(|&&v| v == 0.).count().max(1) as f32;
        }
        Self::upload(&self.driver, self.unknown_buf, &unknown, "unknown counts")?;
        // Three straight uploads instead of one host-side concatenation: the payload build used to
        // copy every crop twice (host memcpy, then PCIe) while the device sat idle.
        Self::upload_at(&self.driver, self.data, 0, clean, "clean")?;
        Self::upload_at(&self.driver, self.data, total_samples, noise, "noise")?;
        Self::upload_at(&self.driver, self.data, 2 * total_samples, mask, "mask")?;
        Self::upload_at(
            &self.driver,
            self.data,
            3 * total_samples,
            &self.trig,
            "trig",
        )?;
        let upload_ms = start.elapsed().as_secs_f64() * 1000.;
        if timed {
            self.driver.check(
                unsafe { (self.driver.ev_record)(self.ev_start, std::ptr::null_mut()) },
                "cuEventRecord",
            )?;
        }
        let n = total_samples as u32;
        let items = self.items as u32;
        let frames = self.frames;
        let block = self.block;
        let cin = self.flow.coefficients(t[0]).2;
        let (pre, scale) = match self.flow {
            FlowConfig::Velocity => (0i32, 1f32),
            FlowConfig::Preconditioned { data_scale, .. } => (1i32, data_scale),
        };
        let launch_start = Instant::now();
        // features
        let mut a_data = self.data.ptr;
        let mut a_x = self.acts[0].ptr;
        let mut a_n = n;
        let mut a_block = block;
        let mut a_pre = pre;
        let mut a_cin = cin;
        let mut a_scale = scale;
        let mut a_frames = frames;
        let feature_elems =
            (self.frames as usize * self.items * (3 * self.block as usize + 4)) as u32;
        self.launch(
            "features",
            self.k.features,
            (feature_elems.div_ceil(BATCH_THREADS as u32).max(1), 1, 1),
            (BATCH_THREADS as u32, 1, 1),
            &mut kargs![a_data, a_x, a_n, a_block, a_pre, a_cin, a_scale, a_frames],
        )?;
        for (j, s) in self.specs.iter().enumerate() {
            let cf = (self.frames as usize).div_ceil(s.s as usize) as u32;
            if s.s > 1 {
                let mut b_x = self.acts[j].ptr;
                let mut b_y = self.pooled[j].ptr;
                let mut b_frames = frames;
                let mut b_ci = s.ci;
                let mut b_s = s.s;
                let mut b_items = items;
                let groups = cf as usize * self.items;
                self.launch(
                    "pool",
                    self.k.pool,
                    (
                        (groups * s.ci as usize).div_ceil(BATCH_THREADS) as u32,
                        1,
                        1,
                    ),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![b_x, b_y, b_frames, b_ci, b_s, b_items],
                )?;
            }
            let x = if s.s > 1 {
                self.pooled[j]
            } else {
                self.acts[j]
            };
            let activation = if j + 1 < self.specs.len() { 1i32 } else { 0 };
            let residual = if j > 0 && j + 1 < self.specs.len() {
                1i32
            } else {
                0
            };
            let mut c_x = x.ptr;
            let mut c_w = self.weights.ptr;
            let mut c_skip = self.acts[j].ptr;
            let mut c_z = self.z[j].ptr;
            let mut c_y = self.acts[j + 1].ptr;
            let mut c_frames = frames;
            let mut c_ci = s.ci;
            let mut c_co = s.co;
            let mut c_k = s.k;
            let mut c_d = s.d;
            let mut c_wo = s.w;
            let mut c_bo = s.b;
            let mut c_scale = s.s;
            let mut c_act = activation;
            let mut c_res = residual;
            let mut c_items = items;
            let grid_y = (self.items * cf as usize).div_ceil(TILE_OUT) as u32;
            self.launch(
                "conv",
                self.k.forward,
                (s.co.div_ceil(TILE_OUT as u32), grid_y, 1),
                (TILE as u32, TILE as u32, 1),
                &mut kargs![
                    c_x, c_w, c_skip, c_z, c_y, c_frames, c_ci, c_co, c_k, c_d, c_wo, c_bo,
                    c_scale, c_act, c_res, c_items
                ],
            )?;
        }
        Ok((upload_ms, launch_start))
    }
    fn execute(
        &mut self,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        t: &[f32],
        update: bool,
    ) -> Result<Step> {
        self.activate()?;
        // Collect the previous step's statistics without draining the device for this step's work.
        if self.pin_pending {
            self.driver.check(
                unsafe { (self.driver.ev_sync)(self.pin_event) },
                "cuEventSynchronize",
            )?;
            // SAFETY: the pinned block is 64 bytes and holds exactly these four floats.
            unsafe {
                let p = self.pinned as *const f32;
                for (i, slot) in self.prev_stats.iter_mut().enumerate() {
                    *slot = *p.add(i);
                }
            }
            self.pin_pending = false;
            let mut g = 0f32;
            self.driver.check(
                unsafe { (self.driver.ev_elapsed)(&mut g, self.ev_start, self.ev_end) },
                "cuEventElapsedTime",
            )?;
            self.last_gpu_ms = g as f64;
            self.last_host_ms = (self.prev_wall_ms - self.last_gpu_ms).max(0.);
        }
        let start = Instant::now();
        let (upload_ms, launch_start) = self.forward_pass(clean, mask, noise, t, true)?;
        let total_samples = self.items * self.n;
        let items = self.items as u32;
        let frames = self.frames;
        let (cskip, cout, _) = self.flow.coefficients(t[0]);
        let pre = match self.flow {
            FlowConfig::Velocity => 0i32,
            FlowConfig::Preconditioned { .. } => 1i32,
        };
        let last = self.specs.len();
        let parts = (total_samples).div_ceil(BATCH_THREADS) as u32;
        let mut d_pred = self.acts[last].ptr;
        let mut d_data = self.data.ptr;
        let mut d_unknown = self.unknown_buf.ptr;
        let mut d_dy = self.dh[last].ptr;
        let mut d_parts = self.loss_parts.ptr;
        let mut d_n = self.n as u32;
        let mut d_pre = pre;
        let mut d_cskip = cskip;
        let mut d_cout = cout;
        let mut d_items = items;
        self.launch(
            "pool+conv",
            self.k.loss,
            (parts, 1, 1),
            (BATCH_THREADS as u32, 1, 1),
            &mut kargs![
                d_pred, d_data, d_unknown, d_dy, d_parts, d_n, d_pre, d_cskip, d_cout, d_items
            ],
        )?;
        for j in (0..last).rev() {
            let s = &self.specs[j];
            let cf = (self.frames as usize).div_ceil(s.s as usize) as u32;
            let residual = if j > 0 && j + 1 < last { 1i32 } else { 0 };
            if j + 1 < last {
                let mut e_dh = self.dh[j + 1].ptr;
                let mut e_z = self.z[j].ptr;
                let mut e_dz = self.dz[j].ptr;
                let mut e_frames = frames;
                let mut e_co = s.co;
                let mut e_scale = s.s;
                let mut e_res = residual;
                let mut e_items = items;
                let elems = self.items * cf as usize * s.co as usize;
                self.launch(
                    "loss",
                    self.k.dz,
                    (elems.div_ceil(BATCH_THREADS) as u32, 1, 1),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![e_dh, e_z, e_dz, e_frames, e_co, e_scale, e_res, e_items],
                )?;
            }
            let dz = if j + 1 < last {
                self.dz[j]
            } else {
                self.dh[j + 1]
            };
            let x = if s.s > 1 {
                self.pooled[j]
            } else {
                self.acts[j]
            };
            let mut f_x = x.ptr;
            let mut f_dz = dz.ptr;
            let mut f_grad = self.grad.ptr;
            let mut f_frames = cf;
            let mut f_ci = s.ci;
            let mut f_co = s.co;
            let mut f_k = s.k;
            let mut f_d = s.d;
            let mut f_wo = s.w;
            let mut f_bo = s.b;
            let mut f_inv = 1.0f32 / self.items as f32;
            let mut f_items = items;
            self.launch(
                "dsoftmax",
                self.k.dw,
                (
                    (s.ci * s.k).div_ceil(TILE_OUT as u32),
                    s.co.div_ceil(TILE_OUT as u32),
                    1,
                ),
                (TILE as u32, TILE as u32, 1),
                &mut kargs![
                    f_x, f_dz, f_grad, f_frames, f_ci, f_co, f_k, f_d, f_wo, f_bo, f_inv, f_items
                ],
            )?;
            if j > 0 {
                let mut g_dz = dz.ptr;
                let mut g_w = self.weights.ptr;
                let mut g_skip = self.dh[j + 1].ptr;
                let mut g_dx = self.dh[j].ptr;
                let mut g_frames = frames;
                let mut g_ci = s.ci;
                let mut g_co = s.co;
                let mut g_k = s.k;
                let mut g_d = s.d;
                let mut g_wo = s.w;
                let mut g_scale = s.s;
                let mut g_res = residual;
                let mut g_items = items;
                self.launch(
                    "conv-backward",
                    self.k.dx,
                    (
                        s.ci.div_ceil(TILE_OUT as u32),
                        (self.items * cf as usize).div_ceil(TILE_OUT) as u32,
                        1,
                    ),
                    (TILE as u32, TILE as u32, 1),
                    &mut kargs![
                        g_dz, g_w, g_skip, g_dx, g_frames, g_ci, g_co, g_k, g_d, g_wo, g_scale,
                        g_res, g_items
                    ],
                )?;
            }
        }
        let mut h_grad = self.grad.ptr;
        let mut h_parts = self.loss_parts.ptr;
        let mut h_stats = self.stats.ptr;
        let mut h_count = self.count as u32;
        let mut h_nparts = parts;
        let mut h_inv = 1.0f32 / self.items as f32;
        self.launch(
            "param-grad",
            self.k.stats,
            (1, 1, 1),
            (BATCH_THREADS as u32, 1, 1),
            &mut kargs![h_grad, h_parts, h_stats, h_count, h_nparts, h_inv],
        )?;
        self.driver.check(
            unsafe { (self.driver.ev_record)(self.ev_end, std::ptr::null_mut()) },
            "cuEventRecord",
        )?;
        let neural_ms = launch_start.elapsed().as_secs_f64() * 1000. + upload_ms;
        let optimizer_start = Instant::now();
        if update {
            let step = self.step + 1;
            let c1 = 1. - 0.9f32.powf(step as f32);
            let c2 = 1. - 0.999f32.powf(step as f32);
            let mut i_w = self.weights.ptr;
            let mut i_m = self.m.ptr;
            let mut i_v = self.v.ptr;
            let mut i_grad = self.grad.ptr;
            let mut i_stats = self.stats.ptr;
            let mut i_count = self.count as u32;
            let mut i_lr = self.lr;
            let mut i_c1 = c1;
            let mut i_c2 = c2;
            self.launch(
                "adam",
                self.k.adam,
                ((self.count as u32).div_ceil(64), 1, 1),
                (64, 1, 1),
                &mut kargs![i_w, i_m, i_v, i_grad, i_stats, i_count, i_lr, i_c1, i_c2],
            )?;
            // No explicit barrier: the blocking statistics download below drains the stream, so the
            // optimizer is complete before the next step's uploads are enqueued.
        }
        let optimizer_ms = optimizer_start.elapsed().as_secs_f64() * 1000.;
        // Enqueue the readback for this step and (first step only) wait for it, so the reported
        // loss lags by one step instead of forcing a full drain every step.
        self.driver.check(
            unsafe {
                (self.driver.dt_h_async)(self.pinned, self.stats.ptr, 16, std::ptr::null_mut())
            },
            "cuMemcpyDtoHAsync(statistics)",
        )?;
        self.driver.check(
            unsafe { (self.driver.ev_record)(self.pin_event, std::ptr::null_mut()) },
            "cuEventRecord",
        )?;
        self.pin_pending = true;
        let mut stats = self.prev_stats;
        if self.reported_steps == 0 {
            self.driver.check(
                unsafe { (self.driver.ev_sync)(self.pin_event) },
                "cuEventSynchronize",
            )?;
            // SAFETY: pinned block holds the four statistics floats written above.
            unsafe {
                let p = self.pinned as *const f32;
                for (i, slot) in stats.iter_mut().enumerate() {
                    *slot = *p.add(i);
                }
            }
            // The next step reports `prev_stats`; seed it here or the second step of every run
            // reports a zero loss and gradient norm.
            self.prev_stats = stats;
            self.pin_pending = false;
            let mut g = 0f32;
            if unsafe { (self.driver.ev_elapsed)(&mut g, self.ev_start, self.ev_end) } == 0 {
                self.last_gpu_ms = g as f64;
                self.last_host_ms = (self.prev_wall_ms - self.last_gpu_ms).max(0.);
            }
        }
        self.reported_steps += 1;
        ensure!(
            stats[3].to_bits() == 0 && stats[0].is_finite() && stats[1].is_finite(),
            "CUDA detected nonfinite loss, gradients or optimizer state; no checkpoint written"
        );
        if update {
            self.step += 1;
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.;
        self.prev_wall_ms = wall_ms;
        self.total_wall_ms += wall_ms;
        Ok(Step {
            loss: stats[0],
            norm: stats[1],
            neural_ms,
            optimizer_ms,
        })
    }
    /// Small transient buffers (per-item unknown counts) allocate and free per step. They are
    /// tiny (batch * 4 bytes) and freed immediately, so the resident budget stays exact.
    fn upload_at(driver: &Driver, buf: Buf, offset: usize, host: &[f32], what: &str) -> Result<()> {
        ensure!(
            offset + host.len() <= buf.len,
            "CUDA upload offset out of range: {what}"
        );
        ensure!(!host.is_empty(), "CUDA upload of an empty slice: {what}");
        // SAFETY: the destination range is inside the checked allocation and the source outlives
        // the asynchronous copy, which is ordered on the default stream ahead of the kernels.
        driver.check(
            unsafe {
                (driver.ht_o_async)(
                    buf.ptr + (offset * 4) as u64,
                    host.as_ptr() as *const c_void,
                    host.len() * 4,
                    std::ptr::null_mut(),
                )
            },
            &format!("cuMemcpyHtoDAsync({what})"),
        )
    }
    pub fn read_gradients(&self) -> Result<Vec<f32>> {
        self.activate()?;
        let mut g = vec![0.; self.count];
        Self::download(&self.driver, self.grad, &mut g, "gradients")?;
        Ok(g)
    }
    pub fn read_prediction(&self) -> Result<Vec<f32>> {
        self.activate()?;
        let mut y = vec![0.; self.items * self.n];
        let last = self.acts.last().context("no activation buffer")?;
        Self::download(&self.driver, *last, &mut y, "prediction")?;
        Ok(y)
    }
    pub fn sync(&self, model: &mut Model, adam: &mut Adam) -> Result<()> {
        self.activate()?;
        ensure!(
            model.count() == self.count,
            "CUDA checkpoint model shape mismatch"
        );
        let mut p = vec![0.; self.count];
        Self::download(&self.driver, self.weights, &mut p, "weights")?;
        Self::download(&self.driver, self.m, &mut adam.m, "adam m")?;
        Self::download(&self.driver, self.v, &mut adam.v, "adam v")?;
        ensure!(
            p.iter()
                .chain(&adam.m)
                .chain(&adam.v)
                .all(|v| v.is_finite()),
            "nonfinite CUDA checkpoint state"
        );
        model.set_parameters(&p);
        adam.step = self.step;
        Ok(())
    }
    /// Bytes currently reserved on the device by this trainer.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated
    }
    /// Context handle kept only so teardown order is explicit in this module.
    pub fn has_context(&self) -> bool {
        !self.ctx.ctx.is_null()
    }
}

impl crate::backend::Resident for CudaTrainer {
    fn step(
        &mut self,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        t: &[f32],
        update: bool,
    ) -> Result<Step> {
        if update {
            self.train_step(clean, mask, noise, t)
        } else {
            self.gradients(clean, mask, noise, t)
        }
    }
    fn read_gradients(&self) -> Result<Vec<f32>> {
        CudaTrainer::read_gradients(self)
    }
    fn read_prediction(&self) -> Result<Vec<f32>> {
        CudaTrainer::read_prediction(self)
    }
    fn predict(&mut self, x: &[f32], mask: &[f32], t: f32) -> Result<Vec<f32>> {
        CudaTrainer::predict(self, x, mask, t)
    }
    fn sync(&self, model: &mut Model, adam: &mut Adam) -> Result<()> {
        CudaTrainer::sync(self, model, adam)
    }
    fn flush(&self) -> Result<()> {
        self.flush_pending()
    }
    fn info(&self) -> Value {
        CudaTrainer::info(self)
    }
    fn completed_steps(&self) -> usize {
        self.step
    }
}
