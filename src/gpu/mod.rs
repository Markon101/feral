//! Optional resident FP32 OpenCL training. CPU retains the portable checkpoint schema.
use crate::{model::Model, parameterization::FlowConfig, runtime::Adam};
use anyhow::{ensure, Context, Result};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context as ClContext,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    event::Event,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_WRITE},
    program::Program,
    types::CL_BLOCKING,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, ptr, time::Instant};
const SOURCE: &str = include_str!("kernels.cl");
const OPTIONS: &str = "-cl-std=CL1.2";
fn aligned(n: usize, t: usize) -> usize {
    n.div_ceil(t) * t
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
    features: Kernel,
    pool: Kernel,
    forward: Kernel,
    loss: Kernel,
    dz: Kernel,
    dw: Kernel,
    dx: Kernel,
    stats: Kernel,
    adam: Kernel,
}
pub struct Step {
    pub loss: f32,
    pub norm: f32,
    pub neural_ms: f64,
    pub optimizer_ms: f64,
    pub transfer_ms: f64,
    pub wall_ms: f64,
}
pub struct OpenClTrainer {
    queue: CommandQueue,
    _context: ClContext,
    _program: Program,
    k: Kernels,
    weights: Buffer<f32>,
    m: Buffer<f32>,
    v: Buffer<f32>,
    grad: Buffer<f32>,
    data: Buffer<f32>,
    stats: Buffer<f32>,
    loss_parts: Buffer<f32>,
    acts: Vec<Buffer<f32>>,
    pooled: Vec<Buffer<f32>>,
    z: Vec<Buffer<f32>>,
    dh: Vec<Buffer<f32>>,
    dz: Vec<Buffer<f32>>,
    specs: Vec<Spec>,
    n: usize,
    frames: u32,
    block: u32,
    count: usize,
    flow: FlowConfig,
    pub step: usize,
    lr: f32,
    info: Value,
    pub last_kernel_times: Vec<(String, f64)>,
    pub total_wall_ms: f64,
    pub total_transfer_ms: f64,
}
impl OpenClTrainer {
    pub fn new(model: &Model, adam: &Adam, n: usize, flow: FlowConfig) -> Result<Self> {
        model.validate()?;
        flow.validate()?;
        ensure!(
            n > 0 && n <= 160000 && n.is_multiple_of(model.architecture.block),
            "invalid GPU crop size"
        );
        ensure!(
            adam.m.len() == model.count() && adam.v.len() == model.count(),
            "GPU optimizer shape mismatch"
        );
        let id=*get_all_devices(CL_DEVICE_TYPE_GPU).context("OpenCL discovery failed; on Android use scripts/feral-opencl.sh")?.first().context("no OpenCL GPU found; on Android use scripts/feral-opencl.sh; CPU fallback is not automatic")?;
        let device = Device::new(id);
        ensure!(
            adam.m.iter().chain(&adam.v).all(|v| v.is_finite())
                && adam.v.iter().all(|v| *v >= 0.)
                && adam.lr.is_finite()
                && adam.lr > 0.,
            "invalid GPU optimizer state"
        );
        ensure!(
            device.max_work_group_size()? >= 256 && device.local_mem_size()? >= 4096,
            "device cannot run FERAL tiled kernels"
        );
        let info = json!({"name":device.name()?,"vendor":device.vendor()?,"version":device.version()?,"driver":device.driver_version()?,"opencl_c":device.opencl_c_version()?,"compute_units":device.max_compute_units()?,"max_workgroup":device.max_work_group_size()?,"global_memory_bytes":device.global_mem_size()?,"max_allocation_bytes":device.max_mem_alloc_size()?,"local_memory_bytes":device.local_mem_size()?,"precision":"fp32","options":OPTIONS,"tile":16});
        let context = ClContext::from_device(&device)?;
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;
        let key = format!("{:x}", Sha256::digest(format!("{SOURCE}{OPTIONS}{info}")));
        let cache = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("cache/opencl")
            .join(format!("{key}.bin"));
        let cached = fs::metadata(&cache)
            .ok()
            .filter(|m| m.len() <= 32 * 1024 * 1024)
            .and_then(|_| fs::read(&cache).ok());
        let program = if let Some(bytes) = cached {
            if bytes.len() <= 32 * 1024 * 1024 {
                Program::create_and_build_from_binary(&context, &[&bytes], OPTIONS).ok()
            } else {
                None
            }
        } else {
            None
        };
        let cache_hit = program.is_some();
        let program = match program {
            Some(p) => p,
            None => Program::create_and_build_from_source(&context, SOURCE, OPTIONS)
                .map_err(|e| anyhow::anyhow!("FERAL OpenCL compilation: {e}"))?,
        };
        if !cache_hit {
            if let (Some(parent), Ok(binaries)) = (cache.parent(), program.get_binaries()) {
                fs::create_dir_all(parent)?;
                if let Some(bytes) = binaries.first() {
                    if let Ok(mut f) = fs::OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(&cache)
                    {
                        f.write_all(bytes)?;
                    }
                }
            }
        }
        let kernels = Kernels {
            features: Kernel::create(&program, "features")?,
            pool: Kernel::create(&program, "pool_frames")?,
            forward: Kernel::create(&program, "conv_forward")?,
            loss: Kernel::create(&program, "loss_grad")?,
            dz: Kernel::create(&program, "activation_grad")?,
            dw: Kernel::create(&program, "weight_grad")?,
            dx: Kernel::create(&program, "input_grad")?,
            stats: Kernel::create(&program, "statistics")?,
            adam: Kernel::create(&program, "adam_update")?,
        };
        let frames = n / model.architecture.block;
        let mut specs = vec![];
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
        let mut alloc =
            |len: usize| -> Result<Buffer<f32>> {
                let bytes = len
                    .max(1)
                    .checked_mul(4)
                    .context("GPU allocation overflow")?;
                allocated += bytes;
                ensure!(
                allocated <= 256 * 1024 * 1024 && bytes as u64 <= device.max_mem_alloc_size()?,
                "FERAL GPU allocation exceeds 256 MiB budget or device limit; reduce crop/width"
            );
                // SAFETY: context owns device; element counts bounded and lifetime retained here.
                Ok(unsafe {
                    Buffer::create(&context, CL_MEM_READ_WRITE, len.max(1), ptr::null_mut())?
                })
            };
        let mut weights = alloc(offset)?;
        let mut m = alloc(offset)?;
        let mut v = alloc(offset)?;
        let grad = alloc(offset)?;
        let data = alloc(3 * n + 4)?;
        let stats = alloc(4)?;
        let loss_parts = alloc(n.div_ceil(256))?;
        let mut acts = vec![alloc(frames * model.layers[0].input)?];
        let mut dh = vec![alloc(1)?];
        let mut pooled = vec![];
        let mut z = vec![];
        let mut dz = vec![];
        for s in &specs {
            let cf = frames.div_ceil(s.s as usize);
            acts.push(alloc(frames * s.co as usize)?);
            dh.push(alloc(frames * s.co as usize)?);
            pooled.push(alloc(if s.s > 1 { cf * s.ci as usize } else { 1 })?);
            z.push(alloc(cf * s.co as usize)?);
            dz.push(alloc(cf * s.co as usize)?);
        }
        // SAFETY: blocking uploads keep host buffers alive; matching counts were checked.
        unsafe {
            queue.enqueue_write_buffer(&mut weights, CL_BLOCKING, 0, &model.parameters(), &[])?;
            queue.enqueue_write_buffer(&mut m, CL_BLOCKING, 0, &adam.m, &[])?;
            queue.enqueue_write_buffer(&mut v, CL_BLOCKING, 0, &adam.v, &[])?;
        }
        let mut info = info;
        info["allocated_bytes"] = allocated.into();
        info["program_cache_hit"] = cache_hit.into();
        info["program_key"] = key.into();
        Ok(Self {
            queue,
            _context: context,
            _program: program,
            k: kernels,
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
            frames: frames as u32,
            block: model.architecture.block as u32,
            count: offset,
            flow,
            step: adam.step,
            lr: adam.lr,
            info,
            last_kernel_times: vec![],
            total_wall_ms: 0.,
            total_transfer_ms: 0.,
        })
    }
    pub fn info(&self) -> Value {
        let mut info = self.info.clone();
        info["total_step_wall_ms"] = self.total_wall_ms.into();
        info["total_transfer_event_ms"] = self.total_transfer_ms.into();
        info["last_kernel_times_ms"] = json!(self.last_kernel_times);
        info
    }
    pub fn gradients(
        &mut self,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        t: f32,
    ) -> Result<Step> {
        self.execute(clean, mask, noise, t, false)
    }
    pub fn train_step(
        &mut self,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        t: f32,
    ) -> Result<Step> {
        self.execute(clean, mask, noise, t, true)
    }
    fn execute(
        &mut self,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        t: f32,
        update: bool,
    ) -> Result<Step> {
        ensure!(
            clean.len() == self.n && mask.len() == self.n && noise.len() == self.n,
            "GPU input shape mismatch"
        );
        ensure!(
            t.is_finite()
                && (0.0..=1.).contains(&t)
                && clean.iter().chain(noise).all(|v| v.is_finite())
                && mask.iter().all(|&v| v == 0. || v == 1.),
            "invalid GPU input values"
        );
        let start = Instant::now();
        let mut payload = Vec::with_capacity(3 * self.n + 4);
        payload.extend_from_slice(clean);
        payload.extend_from_slice(noise);
        payload.extend_from_slice(mask);
        payload.extend([
            t,
            1. - t,
            (t * std::f32::consts::TAU).sin(),
            (t * std::f32::consts::TAU).cos(),
        ]);
        let unknown = mask.iter().filter(|&&v| v == 0.).count().max(1) as f32;
        let n = self.n as u32;
        let (cskip, cout, cin) = self.flow.coefficients(t);
        let (pre, scale) = match self.flow {
            FlowConfig::Velocity => (0i32, 1f32),
            FlowConfig::Preconditioned { data_scale, .. } => (1i32, data_scale),
        };
        let mut events: Vec<(&str, Event)> = vec![];
        // SAFETY: all kernels and bounded buffers belong to one context/in-order queue.
        // The input upload is blocking, including on later enqueue failures; every launch is
        // bounds checked in the kernel, with full 16x16 tiles for barrier convergence.
        unsafe {
            let upload =
                self.queue
                    .enqueue_write_buffer(&mut self.data, CL_BLOCKING, 0, &payload, &[])?;
            events.push(("upload", upload));
            events.push((
                "features",
                ExecuteKernel::new(&self.k.features)
                    .set_arg(&self.data)
                    .set_arg(&self.acts[0])
                    .set_arg(&n)
                    .set_arg(&self.block)
                    .set_arg(&pre)
                    .set_arg(&cin)
                    .set_arg(&scale)
                    .set_global_work_size(aligned(
                        self.frames as usize * (3 * self.block as usize + 4),
                        64,
                    ))
                    .set_local_work_size(64)
                    .enqueue_nd_range(&self.queue)?,
            ));
            for (j, s) in self.specs.iter().enumerate() {
                let cf = (self.frames as usize).div_ceil(s.s as usize) as u32;
                if s.s > 1 {
                    events.push((
                        "pool",
                        ExecuteKernel::new(&self.k.pool)
                            .set_arg(&self.acts[j])
                            .set_arg(&self.pooled[j])
                            .set_arg(&self.frames)
                            .set_arg(&s.ci)
                            .set_arg(&s.s)
                            .set_global_work_size(aligned(cf as usize * s.ci as usize, 64))
                            .set_local_work_size(64)
                            .enqueue_nd_range(&self.queue)?,
                    ));
                }
                let x = if s.s > 1 {
                    &self.pooled[j]
                } else {
                    &self.acts[j]
                };
                let activation = if j + 1 < self.specs.len() { 1i32 } else { 0 };
                let residual = if j > 0 && j + 1 < self.specs.len() {
                    1i32
                } else {
                    0
                };
                events.push((
                    "forward",
                    ExecuteKernel::new(&self.k.forward)
                        .set_arg(x)
                        .set_arg(&self.weights)
                        .set_arg(&self.acts[j])
                        .set_arg(&self.z[j])
                        .set_arg(&self.acts[j + 1])
                        .set_arg(&cf)
                        .set_arg(&s.ci)
                        .set_arg(&s.co)
                        .set_arg(&s.k)
                        .set_arg(&s.d)
                        .set_arg(&s.w)
                        .set_arg(&s.b)
                        .set_arg(&self.frames)
                        .set_arg(&s.s)
                        .set_arg(&activation)
                        .set_arg(&residual)
                        .set_global_work_sizes(&[
                            aligned(s.co as usize, 16),
                            aligned(cf as usize, 16),
                        ])
                        .set_local_work_sizes(&[16, 16])
                        .enqueue_nd_range(&self.queue)?,
                ));
            }
            let last = self.specs.len();
            events.push((
                "loss",
                ExecuteKernel::new(&self.k.loss)
                    .set_arg(&self.acts[last])
                    .set_arg(&self.data)
                    .set_arg(&self.dh[last])
                    .set_arg(&self.loss_parts)
                    .set_arg(&n)
                    .set_arg(&unknown)
                    .set_arg(&pre)
                    .set_arg(&cskip)
                    .set_arg(&cout)
                    .set_global_work_size(aligned(self.n, 256))
                    .set_local_work_size(256)
                    .enqueue_nd_range(&self.queue)?,
            ));
            for j in (0..last).rev() {
                let s = &self.specs[j];
                let cf = (self.frames as usize).div_ceil(s.s as usize) as u32;
                let residual = if j > 0 && j + 1 < last { 1i32 } else { 0 };
                if j + 1 < last {
                    events.push((
                        "activation_backward",
                        ExecuteKernel::new(&self.k.dz)
                            .set_arg(&self.dh[j + 1])
                            .set_arg(&self.z[j])
                            .set_arg(&self.dz[j])
                            .set_arg(&self.frames)
                            .set_arg(&s.co)
                            .set_arg(&s.s)
                            .set_arg(&residual)
                            .set_global_work_size(aligned(cf as usize * s.co as usize, 64))
                            .set_local_work_size(64)
                            .enqueue_nd_range(&self.queue)?,
                    ));
                }
                let dz = if j + 1 < last {
                    &self.dz[j]
                } else {
                    &self.dh[j + 1]
                };
                let x = if s.s > 1 {
                    &self.pooled[j]
                } else {
                    &self.acts[j]
                };
                events.push((
                    "weight_backward",
                    ExecuteKernel::new(&self.k.dw)
                        .set_arg(x)
                        .set_arg(dz)
                        .set_arg(&self.grad)
                        .set_arg(&cf)
                        .set_arg(&s.ci)
                        .set_arg(&s.co)
                        .set_arg(&s.k)
                        .set_arg(&s.d)
                        .set_arg(&s.w)
                        .set_arg(&s.b)
                        .set_global_work_sizes(&[
                            aligned((s.ci * s.k) as usize, 16),
                            aligned(s.co as usize, 16),
                        ])
                        .set_local_work_sizes(&[16, 16])
                        .enqueue_nd_range(&self.queue)?,
                ));
                if j > 0 {
                    events.push((
                        "input_backward",
                        ExecuteKernel::new(&self.k.dx)
                            .set_arg(dz)
                            .set_arg(&self.weights)
                            .set_arg(&self.dh[j + 1])
                            .set_arg(&self.dh[j])
                            .set_arg(&cf)
                            .set_arg(&s.ci)
                            .set_arg(&s.co)
                            .set_arg(&s.k)
                            .set_arg(&s.d)
                            .set_arg(&s.w)
                            .set_arg(&self.frames)
                            .set_arg(&s.s)
                            .set_arg(&residual)
                            .set_global_work_sizes(&[
                                aligned(s.ci as usize, 16),
                                aligned(cf as usize, 16),
                            ])
                            .set_local_work_sizes(&[16, 16])
                            .enqueue_nd_range(&self.queue)?,
                    ));
                }
            }
            let count = self.count as u32;
            let parts = self.n.div_ceil(256) as u32;
            events.push((
                "norm_reduce",
                ExecuteKernel::new(&self.k.stats)
                    .set_arg(&self.grad)
                    .set_arg(&self.loss_parts)
                    .set_arg(&self.stats)
                    .set_arg(&count)
                    .set_arg(&parts)
                    .set_global_work_size(256)
                    .set_local_work_size(256)
                    .enqueue_nd_range(&self.queue)?,
            ));
            if update {
                let step = self.step + 1;
                let c1 = 1. - 0.9f32.powf(step as f32);
                let c2 = 1. - 0.999f32.powf(step as f32);
                events.push((
                    "adam",
                    ExecuteKernel::new(&self.k.adam)
                        .set_arg(&self.weights)
                        .set_arg(&self.m)
                        .set_arg(&self.v)
                        .set_arg(&self.grad)
                        .set_arg(&self.stats)
                        .set_arg(&count)
                        .set_arg(&self.lr)
                        .set_arg(&c1)
                        .set_arg(&c2)
                        .set_global_work_size(aligned(self.count, 64))
                        .set_local_work_size(64)
                        .enqueue_nd_range(&self.queue)?,
                ));
            }
            let mut stats = [0f32; 4];
            let read =
                self.queue
                    .enqueue_read_buffer(&self.stats, CL_BLOCKING, 0, &mut stats, &[])?;
            events.push(("read_stats", read));
            ensure!(stats[3].to_bits()==0&&stats[0].is_finite()&&stats[1].is_finite(),"OpenCL detected nonfinite loss, gradients or optimizer state; no checkpoint written");
            let mut neural_ms = 0.;
            let mut optimizer_ms = 0.;
            let mut transfer_ms = 0.;
            self.last_kernel_times.clear();
            for (name, event) in events {
                let ms = (event.profiling_command_end()? - event.profiling_command_start()?) as f64
                    / 1e6;
                self.last_kernel_times.push((name.into(), ms));
                match name {
                    "upload" | "read_stats" => transfer_ms += ms,
                    "adam" => optimizer_ms += ms,
                    _ => neural_ms += ms,
                }
            }
            if update {
                self.step += 1;
            }
            let wall_ms = start.elapsed().as_secs_f64() * 1000.;
            self.total_wall_ms += wall_ms;
            self.total_transfer_ms += transfer_ms;
            Ok(Step {
                loss: stats[0],
                norm: stats[1],
                neural_ms,
                optimizer_ms,
                transfer_ms,
                wall_ms,
            })
        }
    }
    pub fn read_gradients(&self) -> Result<Vec<f32>> {
        let mut g = vec![0.; self.count];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.grad, CL_BLOCKING, 0, &mut g, &[])?;
        }
        Ok(g)
    }
    pub fn read_prediction(&self) -> Result<Vec<f32>> {
        let mut y = vec![0.; self.n];
        unsafe {
            self.queue.enqueue_read_buffer(
                self.acts.last().unwrap(),
                CL_BLOCKING,
                0,
                &mut y,
                &[],
            )?;
        }
        Ok(y)
    }
    pub fn sync(&self, model: &mut Model, adam: &mut Adam) -> Result<()> {
        ensure!(
            model.count() == self.count,
            "GPU checkpoint model shape mismatch"
        );
        let mut p = vec![0.; self.count];
        // SAFETY: buffers and output slices have the same validated element counts.
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.weights, CL_BLOCKING, 0, &mut p, &[])?;
            self.queue
                .enqueue_read_buffer(&self.m, CL_BLOCKING, 0, &mut adam.m, &[])?;
            self.queue
                .enqueue_read_buffer(&self.v, CL_BLOCKING, 0, &mut adam.v, &[])?;
        }
        ensure!(
            p.iter()
                .chain(&adam.m)
                .chain(&adam.v)
                .all(|v| v.is_finite()),
            "nonfinite GPU checkpoint state"
        );
        model.set_parameters(&p);
        adam.step = self.step;
        Ok(())
    }
}

mod audit;
pub use audit::{audit, temperatures};
