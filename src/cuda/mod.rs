//! Optional resident FP32 CUDA training for NVIDIA GPUs.
//!
//! Uses only the CUDA driver API through `libcuda.so.1` (loaded with dlopen) plus kernels
//! compiled ahead of time by `build.rs`, so the crate gains no new runtime dependency and
//! non-CUDA builds never link or probe the driver. Parameters, activations, gradients and
//! Adam state stay resident on the device; each step uploads one batched crop payload and
//! reads back four statistics values. There is no automatic CPU fallback: any driver,
//! launch or numerical failure is returned as an error.
use crate::{
    backend::Step,
    model::{BlockStyle, Model, SAMPLE_CONV_CHANNELS},
    parameterization::FlowConfig,
    runtime::Adam,
};
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
/// Resident device allocation ceiling. The owner-approved working-memory ceiling is 8 GiB on
/// the desktop and 3 GiB on mobile (AGENTS.md); the default below leaves room for the
/// host-side checkpoint and dataset under either.
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
    ema: CuFunction,
    unpack_stem_in: CuFunction,
    apply_stem_residual: CuFunction,
    extract_stem_grad: CuFunction,
    scale_half: CuFunction,
    sample_conv_fwd: CuFunction,
    sample_conv_dx: CuFunction,
    sample_conv_dw: CuFunction,
    time_embed_proj_fwd: CuFunction,
    rmsnorm_film_fwd: CuFunction,
    gated_residual_fwd: CuFunction,
    gated_activation_grad: CuFunction,
    rmsnorm_film_dx: CuFunction,
    rmsnorm_film_dfilm: CuFunction,
    time_proj_dw: CuFunction,
    zero_buf: CuFunction,
    stft_frame: CuFunction,
    stft_accum: CuFunction,
    add_loss_sum: CuFunction,
}

#[derive(Clone, Copy)]
struct Buf {
    ptr: CuDeviceptr,
    len: usize,
}
impl Buf {
    const EMPTY: Self = Self { ptr: 0, len: 0 };
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
    time_proj_specs: Vec<Spec>,
    stem_specs: Vec<Spec>,
    head_specs: Vec<Spec>,
    has_sample_conv: bool,
    has_gated_film: bool,
    width: u32,
    norm_acts: Vec<Buf>,
    inv_rms: Vec<Buf>,
    films: Vec<Buf>,
    d_film: Buf,
    d_norm: Buf,
    stem_in: Buf,
    stem_a0: Buf,
    stem_a1: Buf,
    head_a0: Buf,
    head_out: Buf,
    dy_head: Buf,
    dh_scratch: Buf,
    dz_sample: Buf,
    n: usize,
    items: usize,
    frames: u32,
    block: u32,
    count: usize,
    flow: FlowConfig,
    /// Detail-term weight applied by the loss kernel; 0 reproduces the plain objective.
    detail: f32,
    /// Multiresolution STFT log-magnitude L1 endpoint loss weight; 0 disables it.
    spectral_weight: f32,
    spec_overlap_buf: Buf,
    spec_loss_buf: Buf,
    ema_decay: f32,
    ema_buf: Buf,
    pub step: usize,
    lr: f32,
    info: Value,
    allocated: usize,
    pub total_wall_ms: f64,
    unknown_buf: Buf,
    trig: Vec<f32>,
    unknown_host: Vec<f32>,
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
            ema: load("feral_ema")?,
            unpack_stem_in: load("feral_unpack_stem_in")?,
            apply_stem_residual: load("feral_apply_stem_residual")?,
            extract_stem_grad: load("feral_extract_stem_grad")?,
            scale_half: load("feral_scale_half")?,
            sample_conv_fwd: load("feral_sample_conv_fwd")?,
            sample_conv_dx: load("feral_sample_conv_dx")?,
            sample_conv_dw: load("feral_sample_conv_dw")?,
            time_embed_proj_fwd: load("feral_time_embed_proj_fwd")?,
            rmsnorm_film_fwd: load("feral_rmsnorm_film_fwd")?,
            gated_residual_fwd: load("feral_gated_residual_fwd")?,
            gated_activation_grad: load("feral_gated_activation_grad")?,
            rmsnorm_film_dx: load("feral_rmsnorm_film_dx")?,
            rmsnorm_film_dfilm: load("feral_rmsnorm_film_dfilm")?,
            time_proj_dw: load("feral_time_proj_dw")?,
            zero_buf: load("feral_zero_buf")?,
            stft_frame: load("feral_stft_frame")?,
            stft_accum: load("feral_stft_accum")?,
            add_loss_sum: load("feral_add_loss_sum")?,
        };
        let frames = n / model.architecture.block;
        let has_gated_film = model.architecture.block_style == BlockStyle::GatedFilm;
        let width = model.architecture.width;
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
        let mut time_proj_specs = Vec::with_capacity(model.time_proj.len());
        for l in &model.time_proj {
            time_proj_specs.push(Spec {
                ci: l.input as u32,
                co: l.output as u32,
                k: l.kernel as u32,
                d: l.dilation as u32,
                s: 1,
                w: offset as u32,
                b: (offset + l.w.len()) as u32,
            });
            offset += l.w.len() + l.b.len();
        }
        let mut stem_specs = Vec::with_capacity(model.stem.len());
        for l in &model.stem {
            stem_specs.push(Spec {
                ci: l.input as u32,
                co: l.output as u32,
                k: l.kernel as u32,
                d: l.dilation as u32,
                s: 1,
                w: offset as u32,
                b: (offset + l.w.len()) as u32,
            });
            offset += l.w.len() + l.b.len();
        }
        let mut head_specs = Vec::with_capacity(model.head.len());
        for l in &model.head {
            head_specs.push(Spec {
                ci: l.input as u32,
                co: l.output as u32,
                k: l.kernel as u32,
                d: l.dilation as u32,
                s: 1,
                w: offset as u32,
                b: (offset + l.w.len()) as u32,
            });
            offset += l.w.len() + l.b.len();
        }
        let has_sample_conv = !stem_specs.is_empty();
        let mut allocated = 0usize;
        let mut alloc = |len: usize| -> Result<Buf> {
            let bytes = len
                .max(1)
                .checked_mul(4)
                .context("CUDA allocation overflow")?;
            allocated += bytes;
            let budget = device_budget_bytes();
            ensure!(
                allocated <= budget,
                "CUDA allocation would exceed the {} MiB resident budget; reduce crop, batch or width",
                budget >> 20
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
        let mut dh = vec![alloc(if has_sample_conv {
            items * frames * model.layers[0].input
        } else {
            1
        })?];
        let mut pooled = Vec::with_capacity(specs.len());
        let mut z = Vec::with_capacity(specs.len());
        let mut dz = Vec::with_capacity(specs.len());
        for (j, s) in specs.iter().enumerate() {
            let cf = frames.div_ceil(s.s as usize);
            let is_gated_res = has_gated_film && j > 0 && j + 1 < specs.len();
            let act_ch = if is_gated_res { width } else { s.co as usize };
            acts.push(alloc(items * frames * act_ch)?);
            dh.push(alloc(items * frames * act_ch)?);
            pooled.push(alloc(if s.s > 1 {
                items * cf * s.ci as usize
            } else {
                1
            })?);
            z.push(alloc(items * cf * s.co as usize)?);
            dz.push(alloc(items * cf * s.co as usize)?);
        }
        let mut norm_acts = Vec::with_capacity(time_proj_specs.len());
        let mut inv_rms = Vec::with_capacity(time_proj_specs.len());
        let mut films = Vec::with_capacity(time_proj_specs.len());
        let (d_film, d_norm) = if has_gated_film {
            for _ in 0..time_proj_specs.len() {
                norm_acts.push(alloc(items * frames * width)?);
                inv_rms.push(alloc(items * frames)?);
                films.push(alloc(items * 2 * width)?);
            }
            (alloc(items * 2 * width)?, alloc(items * frames * width)?)
        } else {
            (Buf::EMPTY, Buf::EMPTY)
        };
        let (stem_in, stem_a0, stem_a1, head_a0, head_out, dy_head, dh_scratch, dz_sample) =
            if has_sample_conv {
                let total = items * n;
                (
                    alloc(total * 3)?,
                    alloc(total * SAMPLE_CONV_CHANNELS)?,
                    alloc(total)?,
                    alloc(total * SAMPLE_CONV_CHANNELS)?,
                    alloc(total)?,
                    alloc(total)?,
                    alloc(total)?,
                    alloc(total * SAMPLE_CONV_CHANNELS)?,
                )
            } else {
                (
                    Buf::EMPTY,
                    Buf::EMPTY,
                    Buf::EMPTY,
                    Buf::EMPTY,
                    Buf::EMPTY,
                    Buf::EMPTY,
                    Buf::EMPTY,
                    Buf::EMPTY,
                )
            };
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
            time_proj_specs,
            stem_specs,
            head_specs,
            has_sample_conv,
            has_gated_film,
            width: width as u32,
            norm_acts,
            inv_rms,
            films,
            d_film,
            d_norm,
            stem_in,
            stem_a0,
            stem_a1,
            head_a0,
            head_out,
            dy_head,
            dh_scratch,
            dz_sample,
            n,
            items,
            frames: frames as u32,
            block: model.architecture.block as u32,
            count: offset,
            flow,
            detail: 0.,
            spectral_weight: 0.,
            spec_overlap_buf: Buf::EMPTY,
            spec_loss_buf: Buf::EMPTY,
            ema_decay: 0.,
            ema_buf: Buf::EMPTY,
            step: adam.step,
            lr: adam.lr,
            info,
            allocated,
            total_wall_ms: 0.,
            unknown_buf,
            trig: vec![0.; 4 * items.max(1)],
            unknown_host: vec![0.; items.max(1)],
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
    fn alloc_buf(&mut self, len: usize) -> Result<Buf> {
        let bytes = len
            .max(1)
            .checked_mul(4)
            .context("CUDA allocation overflow")?;
        let next = self.allocated + bytes;
        let budget = device_budget_bytes();
        ensure!(
            next <= budget,
            "CUDA allocation would exceed the {} MiB resident budget; reduce crop, batch or width",
            budget >> 20
        );
        let mut ptr: CuDeviceptr = 0;
        // SAFETY: allocate bytes validated above; freed with the owning context.
        self.driver.check(
            unsafe { (self.driver.mem_alloc)(&mut ptr, bytes) },
            "cuMemAlloc",
        )?;
        self.allocated = next;
        Ok(Buf { ptr, len })
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
            let mut stats = [0f32; 4];
            // SAFETY: pin_event completed above; the 64-byte pinned block holds these four floats.
            unsafe {
                let p = self.pinned as *const f32;
                for (i, slot) in stats.iter_mut().enumerate() {
                    *slot = *p.add(i);
                }
            }
            ensure!(
                stats[3].to_bits() == 0 && stats[0].is_finite() && stats[1].is_finite(),
                "CUDA detected nonfinite loss, gradients or optimizer state; no checkpoint written"
            );
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
    /// Weight of the detail term in the loss. 0 reproduces the plain objective exactly.
    pub fn set_detail(&mut self, detail: f32) {
        self.detail = detail.max(0.);
    }
    /// Weight of the multiresolution STFT log-magnitude L1 endpoint loss. 0 disables it.
    pub fn set_spectral(&mut self, weight: f32) {
        self.spectral_weight = weight.max(0.);
    }
    /// Set the learning rate for subsequent Adam updates (supports cosine/warmup schedules).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
    /// Enable resident EMA weight tracking on the device and upload initial EMA parameters.
    pub fn configure_ema(&mut self, decay: f32, initial: &[f32]) -> Result<()> {
        ensure!(
            decay.is_finite() && (0. ..1.).contains(&decay),
            "invalid CUDA EMA decay ({decay})"
        );
        ensure!(initial.len() == self.count, "CUDA EMA shape mismatch");
        self.activate()?;
        if self.ema_buf.len < self.count {
            self.ema_buf = self.alloc_buf(self.count)?;
        }
        Self::upload(&self.driver, self.ema_buf, initial, "ema weights")?;
        self.ema_decay = decay;
        Ok(())
    }
    /// Download resident EMA weights from the device.
    pub fn sync_ema(&self, ema_out: &mut [f32]) -> Result<()> {
        self.activate()?;
        self.flush_pending()?;
        ensure!(
            ema_out.len() == self.count && self.ema_buf.len >= self.count,
            "CUDA EMA not configured or shape mismatch"
        );
        Self::download(&self.driver, self.ema_buf, ema_out, "ema weights")?;
        ensure!(
            ema_out.iter().all(|v| v.is_finite()),
            "nonfinite CUDA EMA state"
        );
        Ok(())
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
        for (i, slot) in self.unknown_host.iter_mut().enumerate() {
            let row = &mask[i * self.n..(i + 1) * self.n];
            *slot = row.iter().filter(|&&v| v == 0.).count().max(1) as f32;
        }
        Self::upload_at(
            &self.driver,
            self.unknown_buf,
            0,
            &self.unknown_host,
            "unknown counts",
        )?;
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
        let (pre, data_scale, prior_std) = match self.flow {
            FlowConfig::Velocity => (0i32, 1f32, 1f32),
            FlowConfig::Preconditioned {
                data_scale,
                prior_std,
            } => (1i32, data_scale, prior_std),
        };
        let launch_start = Instant::now();
        // features: 2D grid keeps total_frames on blockIdx.x (2^31-1 max) and channels on blockIdx.y.
        let total_frames = self.frames * items;
        let channels = 3u32 * self.block + 4u32;
        let mut a_data = self.data.ptr;
        let mut a_x = self.acts[0].ptr;
        let mut a_n = n;
        let mut a_block = block;
        let mut a_pre = pre;
        let mut a_data_scale = data_scale;
        let mut a_prior_std = prior_std;
        let mut a_frames = frames;
        let mut a_total_frames = total_frames;
        self.launch(
            "features",
            self.k.features,
            (
                total_frames.div_ceil(8).max(1),
                channels.div_ceil(32).max(1),
                1,
            ),
            (32, 8, 1),
            &mut kargs![
                a_data,
                a_x,
                a_n,
                a_block,
                a_pre,
                a_data_scale,
                a_prior_std,
                a_frames,
                a_total_frames
            ],
        )?;
        if self.has_sample_conv {
            let n_samples = self.n as u32;
            let sample_grid = total_samples.div_ceil(BATCH_THREADS) as u32;
            let mut u_x = self.acts[0].ptr;
            let mut u_in = self.stem_in.ptr;
            let mut u_frames = frames;
            let mut u_block = block;
            let mut u_items = items;
            self.launch(
                "unpack_stem_in",
                self.k.unpack_stem_in,
                (sample_grid, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![u_x, u_in, u_frames, u_block, u_items],
            )?;
            let s0 = &self.stem_specs[0];
            let mut p_x = self.stem_in.ptr;
            let mut p_w = self.weights.ptr;
            let mut p_y = self.stem_a0.ptr;
            let mut p_n = n_samples;
            let mut p_ci = s0.ci;
            let mut p_co = s0.co;
            let mut p_k = s0.k;
            let mut p_wo = s0.w;
            let mut p_bo = s0.b;
            let mut p_act = 1i32;
            let mut p_skip: CuDeviceptr = 0;
            let mut p_ss = 0.0f32;
            let mut p_items = items;
            self.launch(
                "stem_conv0",
                self.k.sample_conv_fwd,
                (
                    (total_samples * s0.co as usize).div_ceil(BATCH_THREADS) as u32,
                    1,
                    1,
                ),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    p_x, p_w, p_y, p_n, p_ci, p_co, p_k, p_wo, p_bo, p_act, p_skip, p_ss, p_items
                ],
            )?;
            let s1 = &self.stem_specs[1];
            let mut q_x = self.stem_a0.ptr;
            let mut q_w = self.weights.ptr;
            let mut q_y = self.stem_a1.ptr;
            let mut q_n = n_samples;
            let mut q_ci = s1.ci;
            let mut q_co = s1.co;
            let mut q_k = s1.k;
            let mut q_wo = s1.w;
            let mut q_bo = s1.b;
            let mut q_act = 1i32;
            let mut q_skip: CuDeviceptr = 0;
            let mut q_ss = 0.0f32;
            let mut q_items = items;
            self.launch(
                "stem_conv1",
                self.k.sample_conv_fwd,
                (sample_grid, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    q_x, q_w, q_y, q_n, q_ci, q_co, q_k, q_wo, q_bo, q_act, q_skip, q_ss, q_items
                ],
            )?;
            let mut r_x = self.acts[0].ptr;
            let mut r_s1 = self.stem_a1.ptr;
            let mut r_frames = frames;
            let mut r_block = block;
            let mut r_items = items;
            self.launch(
                "apply_stem_residual",
                self.k.apply_stem_residual,
                (sample_grid, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![r_x, r_s1, r_frames, r_block, r_items],
            )?;
        }
        for (j, s) in self.specs.iter().enumerate() {
            let cf = (self.frames as usize).div_ceil(s.s as usize) as u32;
            let residual = if j > 0 && j + 1 < self.specs.len() {
                1i32
            } else {
                0
            };
            if residual != 0 && self.has_gated_film {
                let block_idx = j - 1;
                let tp = &self.time_proj_specs[block_idx];
                let mut t_data = self.data.ptr;
                let mut t_w = self.weights.ptr;
                let mut t_film = self.films[block_idx].ptr;
                let mut t_total = n;
                let mut t_width = self.width;
                let mut t_wo = tp.w;
                let mut t_bo = tp.b;
                let mut t_items = items;
                self.launch(
                    "time_embed_proj_fwd",
                    self.k.time_embed_proj_fwd,
                    ((2 * self.width).div_ceil(64), items, 1),
                    (64, 1, 1),
                    &mut kargs![t_data, t_w, t_film, t_total, t_width, t_wo, t_bo, t_items],
                )?;
                let mut rn_h = self.acts[j].ptr;
                let mut rn_film = self.films[block_idx].ptr;
                let mut rn_out = self.norm_acts[block_idx].ptr;
                let mut rn_inv = self.inv_rms[block_idx].ptr;
                let mut rn_frames = frames;
                let mut rn_width = self.width;
                let mut rn_items = items;
                self.launch(
                    "rmsnorm_film_fwd",
                    self.k.rmsnorm_film_fwd,
                    (items * frames, 1, 1),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![rn_h, rn_film, rn_out, rn_inv, rn_frames, rn_width, rn_items],
                )?;
                if s.s > 1 {
                    let mut b_x = self.norm_acts[block_idx].ptr;
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
                    self.norm_acts[block_idx]
                };
                // Run conv linearly with frames=cf, scale=1, y=z[j] so output stays within z[j]
                // even when scale > 1, then fuse tanh*sigmoid + repeat + residual add below.
                let mut c_x = x.ptr;
                let mut c_w = self.weights.ptr;
                let mut c_skip = self.z[j].ptr;
                let mut c_z = self.z[j].ptr;
                let mut c_y = self.z[j].ptr;
                let mut c_frames = cf;
                let mut c_ci = s.ci;
                let mut c_co = s.co;
                let mut c_k = s.k;
                let mut c_d = s.d;
                let mut c_wo = s.w;
                let mut c_bo = s.b;
                let mut c_scale = 1u32;
                let mut c_act = 0i32;
                let mut c_res = 0i32;
                let mut c_items = items;
                let grid_y = (self.items * cf as usize).div_ceil(TILE_OUT) as u32;
                self.launch(
                    "gated_conv",
                    self.k.forward,
                    (s.co.div_ceil(TILE_OUT as u32), grid_y, 1),
                    (TILE as u32, TILE as u32, 1),
                    &mut kargs![
                        c_x, c_w, c_skip, c_z, c_y, c_frames, c_ci, c_co, c_k, c_d, c_wo, c_bo,
                        c_scale, c_act, c_res, c_items
                    ],
                )?;
                let mut g_z = self.z[j].ptr;
                let mut g_skip = self.acts[j].ptr;
                let mut g_y = self.acts[j + 1].ptr;
                let mut g_frames = frames;
                let mut g_width = self.width;
                let mut g_scale = s.s;
                let mut g_items = items;
                let elems = self.items * cf as usize * self.width as usize;
                self.launch(
                    "gated_residual_fwd",
                    self.k.gated_residual_fwd,
                    (elems.div_ceil(BATCH_THREADS) as u32, 1, 1),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![g_z, g_skip, g_y, g_frames, g_width, g_scale, g_items],
                )?;
                continue;
            }
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
        if self.has_sample_conv {
            let last = self.specs.len();
            let n_samples = self.n as u32;
            let h0 = &self.head_specs[0];
            let mut p_x = self.acts[last].ptr;
            let mut p_w = self.weights.ptr;
            let mut p_y = self.head_a0.ptr;
            let mut p_n = n_samples;
            let mut p_ci = h0.ci;
            let mut p_co = h0.co;
            let mut p_k = h0.k;
            let mut p_wo = h0.w;
            let mut p_bo = h0.b;
            let mut p_act = 1i32;
            let mut p_skip: CuDeviceptr = 0;
            let mut p_ss = 0.0f32;
            let mut p_items = items;
            self.launch(
                "head_conv0",
                self.k.sample_conv_fwd,
                (
                    (total_samples * h0.co as usize).div_ceil(BATCH_THREADS) as u32,
                    1,
                    1,
                ),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    p_x, p_w, p_y, p_n, p_ci, p_co, p_k, p_wo, p_bo, p_act, p_skip, p_ss, p_items
                ],
            )?;
            let h1 = &self.head_specs[1];
            let mut q_x = self.head_a0.ptr;
            let mut q_w = self.weights.ptr;
            let mut q_y = self.head_out.ptr;
            let mut q_n = n_samples;
            let mut q_ci = h1.ci;
            let mut q_co = h1.co;
            let mut q_k = h1.k;
            let mut q_wo = h1.w;
            let mut q_bo = h1.b;
            let mut q_act = 0i32;
            let mut q_skip = self.acts[last].ptr;
            let mut q_ss = 0.5f32;
            let mut q_items = items;
            self.launch(
                "head_conv1",
                self.k.sample_conv_fwd,
                (total_samples.div_ceil(BATCH_THREADS) as u32, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    q_x, q_w, q_y, q_n, q_ci, q_co, q_k, q_wo, q_bo, q_act, q_skip, q_ss, q_items
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
        let (pre, data_scale, prior_std) = match self.flow {
            FlowConfig::Velocity => (0i32, 1f32, 1f32),
            FlowConfig::Preconditioned {
                data_scale,
                prior_std,
            } => (1i32, data_scale, prior_std),
        };
        let last = self.specs.len();
        let parts = (total_samples).div_ceil(BATCH_THREADS) as u32;
        let pred_buf = if self.has_sample_conv {
            self.head_out
        } else {
            self.acts[last]
        };
        let loss_dy_buf = if self.has_sample_conv {
            self.dy_head
        } else {
            self.dh[last]
        };
        let mut d_pred = pred_buf.ptr;
        let mut d_data = self.data.ptr;
        let mut d_unknown = self.unknown_buf.ptr;
        let mut d_dy = loss_dy_buf.ptr;
        let mut d_parts = self.loss_parts.ptr;
        let mut d_n = self.n as u32;
        let mut d_pre = pre;
        let mut d_data_scale = data_scale;
        let mut d_prior_std = prior_std;
        let mut d_detail = self.detail;
        let mut d_items = items;
        self.launch(
            "loss",
            self.k.loss,
            (parts, 1, 1),
            (BATCH_THREADS as u32, 1, 1),
            &mut kargs![
                d_pred,
                d_data,
                d_unknown,
                d_dy,
                d_parts,
                d_n,
                d_pre,
                d_data_scale,
                d_prior_std,
                d_detail,
                d_items
            ],
        )?;
        if self.spectral_weight > 0. {
            // Resident GPU multiresolution STFT loss & gradient (zero PCIe sync, zero atomics).
            let max_frames = self.n.div_ceil(crate::spectral::SIZES[0] / 4);
            if self.spec_overlap_buf.len < 4 * total_samples {
                self.spec_overlap_buf = self.alloc_buf(4 * total_samples)?;
            }
            if self.spec_loss_buf.len < self.items * max_frames {
                self.spec_loss_buf = self.alloc_buf(self.items * max_frames)?;
            }
            let num_res = crate::spectral::SIZES.len() as f32;
            let overlap_elems = (4 * total_samples) as u32;
            for &size in &crate::spectral::SIZES {
                let hop = size / 4;
                let log2_n = size.trailing_zeros();
                let frame_count = self.n.div_ceil(hop);
                let mut z_dst = self.spec_overlap_buf.ptr;
                let mut z_total = overlap_elems;
                self.launch(
                    "zero_overlap",
                    self.k.zero_buf,
                    (overlap_elems.div_ceil(BATCH_THREADS as u32), 1, 1),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![z_dst, z_total],
                )?;
                let mut f_pred = pred_buf.ptr;
                let mut f_data = self.data.ptr;
                let mut f_ov = self.spec_overlap_buf.ptr;
                let mut f_fl = self.spec_loss_buf.ptr;
                let mut f_n = self.n as u32;
                let mut f_items = items;
                let mut f_size = size as u32;
                let mut f_log2 = log2_n;
                let mut f_hop = hop as u32;
                let mut f_fc = frame_count as u32;
                let mut f_pre = pre;
                let mut f_ds = data_scale;
                let mut f_ps = prior_std;
                let mut f_sw = self.spectral_weight;
                let mut f_nr = num_res;
                let threads = if size == 128 { 128u32 } else { 256u32 };
                self.launch(
                    "stft_frame",
                    self.k.stft_frame,
                    (frame_count as u32, items, 1),
                    (threads, 1, 1),
                    &mut kargs![
                        f_pred, f_data, f_ov, f_fl, f_n, f_items, f_size, f_log2, f_hop, f_fc,
                        f_pre, f_ds, f_ps, f_sw, f_nr
                    ],
                )?;
                let mut a_dy = loss_dy_buf.ptr;
                let mut a_ov = self.spec_overlap_buf.ptr;
                let mut a_total = total_samples as u32;
                self.launch(
                    "stft_accum",
                    self.k.stft_accum,
                    (parts, 1, 1),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![a_dy, a_ov, a_total],
                )?;
                let mut l_parts = self.loss_parts.ptr;
                let mut l_fl = self.spec_loss_buf.ptr;
                let mut l_count = (self.items * frame_count) as u32;
                self.launch(
                    "add_loss_sum",
                    self.k.add_loss_sum,
                    (1, 1, 1),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![l_parts, l_fl, l_count],
                )?;
            }
        }
        let inv_batch = 1.0f32 / self.items as f32;
        if self.has_sample_conv {
            let n_samples = self.n as u32;
            let mut sh_src = self.dy_head.ptr;
            let mut sh_dst = self.dh_scratch.ptr;
            let mut sh_total = total_samples as u32;
            self.launch(
                "scale_half",
                self.k.scale_half,
                (parts, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![sh_src, sh_dst, sh_total],
            )?;
            let h1 = &self.head_specs[1];
            let mut dw_x = self.head_a0.ptr;
            let mut dw_dz = self.dh_scratch.ptr;
            let mut dw_grad = self.grad.ptr;
            let mut dw_n = n_samples;
            let mut dw_ci = h1.ci;
            let mut dw_co = h1.co;
            let mut dw_k = h1.k;
            let mut dw_wo = h1.w;
            let mut dw_bo = h1.b;
            let mut dw_inv = inv_batch;
            let mut dw_items = items;
            self.launch(
                "head_conv1_dw",
                self.k.sample_conv_dw,
                (h1.co * h1.k * h1.ci + h1.co, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    dw_x, dw_dz, dw_grad, dw_n, dw_ci, dw_co, dw_k, dw_wo, dw_bo, dw_inv, dw_items
                ],
            )?;
            let mut dx_dz = self.dh_scratch.ptr;
            let mut dx_w = self.weights.ptr;
            let mut dx_a = self.head_a0.ptr;
            let mut dx_skip: CuDeviceptr = 0;
            let mut dx_out = self.dz_sample.ptr;
            let mut dx_n = n_samples;
            let mut dx_ci = h1.ci;
            let mut dx_co = h1.co;
            let mut dx_k = h1.k;
            let mut dx_wo = h1.w;
            let mut dx_tanh = 1i32;
            let mut dx_items = items;
            self.launch(
                "head_conv1_dx",
                self.k.sample_conv_dx,
                (
                    (total_samples * h1.ci as usize).div_ceil(BATCH_THREADS) as u32,
                    1,
                    1,
                ),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    dx_dz, dx_w, dx_a, dx_skip, dx_out, dx_n, dx_ci, dx_co, dx_k, dx_wo, dx_tanh,
                    dx_items
                ],
            )?;
            let h0 = &self.head_specs[0];
            let mut dw0_x = self.acts[last].ptr;
            let mut dw0_dz = self.dz_sample.ptr;
            let mut dw0_grad = self.grad.ptr;
            let mut dw0_n = n_samples;
            let mut dw0_ci = h0.ci;
            let mut dw0_co = h0.co;
            let mut dw0_k = h0.k;
            let mut dw0_wo = h0.w;
            let mut dw0_bo = h0.b;
            let mut dw0_inv = inv_batch;
            let mut dw0_items = items;
            self.launch(
                "head_conv0_dw",
                self.k.sample_conv_dw,
                (h0.co * h0.k * h0.ci + h0.co, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    dw0_x, dw0_dz, dw0_grad, dw0_n, dw0_ci, dw0_co, dw0_k, dw0_wo, dw0_bo, dw0_inv,
                    dw0_items
                ],
            )?;
            let mut dx0_dz = self.dz_sample.ptr;
            let mut dx0_w = self.weights.ptr;
            let mut dx0_a: CuDeviceptr = 0;
            let mut dx0_skip = self.dy_head.ptr;
            let mut dx0_out = self.dh[last].ptr;
            let mut dx0_n = n_samples;
            let mut dx0_ci = h0.ci;
            let mut dx0_co = h0.co;
            let mut dx0_k = h0.k;
            let mut dx0_wo = h0.w;
            let mut dx0_tanh = 0i32;
            let mut dx0_items = items;
            self.launch(
                "head_conv0_dx",
                self.k.sample_conv_dx,
                (parts, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    dx0_dz, dx0_w, dx0_a, dx0_skip, dx0_out, dx0_n, dx0_ci, dx0_co, dx0_k, dx0_wo,
                    dx0_tanh, dx0_items
                ],
            )?;
        }
        for j in (0..last).rev() {
            let s = &self.specs[j];
            let cf = (self.frames as usize).div_ceil(s.s as usize) as u32;
            let residual = if j > 0 && j + 1 < last { 1i32 } else { 0 };
            if residual != 0 && self.has_gated_film {
                let block_idx = j - 1;
                let mut e_dh = self.dh[j + 1].ptr;
                let mut e_z = self.z[j].ptr;
                let mut e_dz = self.dz[j].ptr;
                let mut e_frames = frames;
                let mut e_width = self.width;
                let mut e_scale = s.s;
                let mut e_items = items;
                let elems = self.items * cf as usize * self.width as usize;
                self.launch(
                    "gated_activation_grad",
                    self.k.gated_activation_grad,
                    (elems.div_ceil(BATCH_THREADS) as u32, 1, 1),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![e_dh, e_z, e_dz, e_frames, e_width, e_scale, e_items],
                )?;
                let x = if s.s > 1 {
                    self.pooled[j]
                } else {
                    self.norm_acts[block_idx]
                };
                let mut f_x = x.ptr;
                let mut f_dz = self.dz[j].ptr;
                let mut f_grad = self.grad.ptr;
                let mut f_frames = cf;
                let mut f_ci = s.ci;
                let mut f_co = s.co;
                let mut f_k = s.k;
                let mut f_d = s.d;
                let mut f_wo = s.w;
                let mut f_bo = s.b;
                let mut f_inv = inv_batch;
                let mut f_items = items;
                self.launch(
                    "gated_weight_grad",
                    self.k.dw,
                    (
                        (s.ci * s.k).div_ceil(TILE_OUT as u32),
                        s.co.div_ceil(TILE_OUT as u32),
                        1,
                    ),
                    (TILE as u32, TILE as u32, 1),
                    &mut kargs![
                        f_x, f_dz, f_grad, f_frames, f_ci, f_co, f_k, f_d, f_wo, f_bo, f_inv,
                        f_items
                    ],
                )?;
                let mut g_dz = self.dz[j].ptr;
                let mut g_w = self.weights.ptr;
                let mut g_skip = self.d_norm.ptr;
                let mut g_dx = self.d_norm.ptr;
                let mut g_frames = frames;
                let mut g_ci = s.ci;
                let mut g_co = s.co;
                let mut g_k = s.k;
                let mut g_d = s.d;
                let mut g_wo = s.w;
                let mut g_scale = s.s;
                let mut g_res = 0i32;
                let mut g_items = items;
                self.launch(
                    "gated_input_grad",
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
                let mut rx_du = self.d_norm.ptr;
                let mut rx_h = self.acts[j].ptr;
                let mut rx_inv = self.inv_rms[block_idx].ptr;
                let mut rx_film = self.films[block_idx].ptr;
                let mut rx_skip = self.dh[j + 1].ptr;
                let mut rx_out = self.dh[j].ptr;
                let mut rx_frames = frames;
                let mut rx_width = self.width;
                let mut rx_items = items;
                self.launch(
                    "rmsnorm_film_dx",
                    self.k.rmsnorm_film_dx,
                    (items * frames, 1, 1),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![
                        rx_du, rx_h, rx_inv, rx_film, rx_skip, rx_out, rx_frames, rx_width,
                        rx_items
                    ],
                )?;
                let mut rf_du = self.d_norm.ptr;
                let mut rf_h = self.acts[j].ptr;
                let mut rf_inv = self.inv_rms[block_idx].ptr;
                let mut rf_df = self.d_film.ptr;
                let mut rf_frames = frames;
                let mut rf_width = self.width;
                let mut rf_items = items;
                self.launch(
                    "rmsnorm_film_dfilm",
                    self.k.rmsnorm_film_dfilm,
                    (2 * self.width, items, 1),
                    (BATCH_THREADS as u32, 1, 1),
                    &mut kargs![rf_du, rf_h, rf_inv, rf_df, rf_frames, rf_width, rf_items],
                )?;
                let tp = &self.time_proj_specs[block_idx];
                let mut tp_data = self.data.ptr;
                let mut tp_df = self.d_film.ptr;
                let mut tp_grad = self.grad.ptr;
                let mut tp_total = total_samples as u32;
                let mut tp_width = self.width;
                let mut tp_wo = tp.w;
                let mut tp_bo = tp.b;
                let mut tp_inv = inv_batch;
                let mut tp_items = items;
                self.launch(
                    "time_proj_dw",
                    self.k.time_proj_dw,
                    (2 * self.width, 1, 1),
                    (32, 1, 1),
                    &mut kargs![
                        tp_data, tp_df, tp_grad, tp_total, tp_width, tp_wo, tp_bo, tp_inv, tp_items
                    ],
                )?;
                continue;
            }
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
                    "activation_grad",
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
            let mut f_inv = inv_batch;
            let mut f_items = items;
            self.launch(
                "weight_grad",
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
            if j > 0 || self.has_sample_conv {
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
        if self.has_sample_conv {
            let n_samples = self.n as u32;
            let mut ex_dh0 = self.dh[0].ptr;
            let mut ex_a1 = self.stem_a1.ptr;
            let mut ex_dz = self.dh_scratch.ptr;
            let mut ex_frames = frames;
            let mut ex_block = self.block;
            let mut ex_items = items;
            self.launch(
                "extract_stem_grad",
                self.k.extract_stem_grad,
                (parts, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![ex_dh0, ex_a1, ex_dz, ex_frames, ex_block, ex_items],
            )?;
            let s1 = &self.stem_specs[1];
            let mut dw1_x = self.stem_a0.ptr;
            let mut dw1_dz = self.dh_scratch.ptr;
            let mut dw1_grad = self.grad.ptr;
            let mut dw1_n = n_samples;
            let mut dw1_ci = s1.ci;
            let mut dw1_co = s1.co;
            let mut dw1_k = s1.k;
            let mut dw1_wo = s1.w;
            let mut dw1_bo = s1.b;
            let mut dw1_inv = inv_batch;
            let mut dw1_items = items;
            self.launch(
                "stem_conv1_dw",
                self.k.sample_conv_dw,
                (s1.co * s1.k * s1.ci + s1.co, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    dw1_x, dw1_dz, dw1_grad, dw1_n, dw1_ci, dw1_co, dw1_k, dw1_wo, dw1_bo, dw1_inv,
                    dw1_items
                ],
            )?;
            let mut dx1_dz = self.dh_scratch.ptr;
            let mut dx1_w = self.weights.ptr;
            let mut dx1_a = self.stem_a0.ptr;
            let mut dx1_skip: CuDeviceptr = 0;
            let mut dx1_out = self.dz_sample.ptr;
            let mut dx1_n = n_samples;
            let mut dx1_ci = s1.ci;
            let mut dx1_co = s1.co;
            let mut dx1_k = s1.k;
            let mut dx1_wo = s1.w;
            let mut dx1_tanh = 1i32;
            let mut dx1_items = items;
            self.launch(
                "stem_conv1_dx",
                self.k.sample_conv_dx,
                (
                    (total_samples * s1.ci as usize).div_ceil(BATCH_THREADS) as u32,
                    1,
                    1,
                ),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    dx1_dz, dx1_w, dx1_a, dx1_skip, dx1_out, dx1_n, dx1_ci, dx1_co, dx1_k, dx1_wo,
                    dx1_tanh, dx1_items
                ],
            )?;
            let s0 = &self.stem_specs[0];
            let mut dw0_x = self.stem_in.ptr;
            let mut dw0_dz = self.dz_sample.ptr;
            let mut dw0_grad = self.grad.ptr;
            let mut dw0_n = n_samples;
            let mut dw0_ci = s0.ci;
            let mut dw0_co = s0.co;
            let mut dw0_k = s0.k;
            let mut dw0_wo = s0.w;
            let mut dw0_bo = s0.b;
            let mut dw0_inv = inv_batch;
            let mut dw0_items = items;
            self.launch(
                "stem_conv0_dw",
                self.k.sample_conv_dw,
                (s0.co * s0.k * s0.ci + s0.co, 1, 1),
                (BATCH_THREADS as u32, 1, 1),
                &mut kargs![
                    dw0_x, dw0_dz, dw0_grad, dw0_n, dw0_ci, dw0_co, dw0_k, dw0_wo, dw0_bo, dw0_inv,
                    dw0_items
                ],
            )?;
        }
        let mut h_grad = self.grad.ptr;
        let mut h_parts = self.loss_parts.ptr;
        let mut h_stats = self.stats.ptr;
        let mut h_count = self.count as u32;
        let mut h_nparts = parts;
        let mut h_inv = inv_batch;
        self.launch(
            "statistics",
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
            if self.ema_decay > 0. && self.ema_buf.len >= self.count {
                let mut e_ema = self.ema_buf.ptr;
                let mut e_w = self.weights.ptr;
                let mut e_stats = self.stats.ptr;
                let mut e_count = self.count as u32;
                let mut e_decay = self.ema_decay;
                self.launch(
                    "ema",
                    self.k.ema,
                    ((self.count as u32).div_ceil(64), 1, 1),
                    (64, 1, 1),
                    &mut kargs![e_ema, e_w, e_stats, e_count, e_decay],
                )?;
            }
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
    /// Asynchronous host-to-device slice upload on the default stream.
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
        let pred_buf = if self.has_sample_conv {
            self.head_out
        } else {
            *self.acts.last().context("no activation buffer")?
        };
        Self::download(&self.driver, pred_buf, &mut y, "prediction")?;
        Ok(y)
    }
    /// Upload updated parameters onto this resident trainer without recreating its context or
    /// allocations. Used by cached preview and validation fields.
    pub fn load_weights(&self, model: &Model) -> Result<()> {
        self.activate()?;
        ensure!(
            model.count() == self.count,
            "CUDA checkpoint model shape mismatch"
        );
        Self::upload(
            &self.driver,
            self.weights,
            &model.parameters(),
            "parameters",
        )
    }
    pub fn sync(&self, model: &mut Model, adam: &mut Adam) -> Result<()> {
        self.activate()?;
        self.flush_pending()?;
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
    fn set_detail(&mut self, detail: f32) {
        CudaTrainer::set_detail(self, detail)
    }
    fn set_spectral(&mut self, weight: f32) {
        CudaTrainer::set_spectral(self, weight)
    }
    fn set_lr(&mut self, lr: f32) {
        CudaTrainer::set_lr(self, lr)
    }
    fn sync(&self, model: &mut Model, adam: &mut Adam) -> Result<()> {
        CudaTrainer::sync(self, model, adam)
    }
    fn flush(&self) -> Result<()> {
        self.flush_pending()
    }
    fn load_weights(&self, model: &Model) -> Result<()> {
        CudaTrainer::load_weights(self, model)
    }
    fn info(&self) -> Value {
        CudaTrainer::info(self)
    }
    fn completed_steps(&self) -> usize {
        self.step
    }
}
