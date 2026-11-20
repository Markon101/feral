//! Explicit training backend selection. There is no automatic or silent CPU fallback.
use crate::{
    model::Model,
    parameterization::{self, FlowConfig},
    runtime::Adam,
};
use anyhow::{bail, ensure, Result};
use serde_json::{json, Value};
use std::time::Instant;
/// One optimizer step's worth of crops. `t` holds one flow time per item, so `t.len()` is the
/// batch size and every other slice holds `items * n` samples in per-item order.
pub struct Batch<'a> {
    pub clean: &'a [f32],
    pub mask: &'a [f32],
    pub noise: &'a [f32],
    pub t: &'a [f32],
    /// Weight of the detail term in the loss; 0 is the plain objective.
    pub detail: f32,
}
impl Batch<'_> {
    pub fn items(&self) -> usize {
        self.t.len()
    }
}
pub struct Step {
    pub loss: f32,
    pub norm: f32,
    pub neural_ms: f64,
    pub optimizer_ms: f64,
}
/// Resident-device operations shared by the accelerator audits. Each accelerator keeps its
/// parameters on the device and exposes read-back only for verification.
pub trait Resident {
    fn step(
        &mut self,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        t: &[f32],
        update: bool,
    ) -> Result<Step>;
    /// Weight of the detail term in the loss. Backends without it keep the plain objective; a
    /// caller that asks for a non-zero weight on such a backend gets an error from `step`.
    fn set_detail(&mut self, _detail: f32) {}
    /// Forward-only field evaluation. A backend without a cheaper path stays correct by running a
    /// full step without an update; the CUDA resident skips the backward pass instead.
    fn predict(&mut self, x: &[f32], mask: &[f32], t: f32) -> Result<Vec<f32>> {
        self.step(x, mask, x, &[t], false)?;
        self.read_prediction()
    }
    fn read_gradients(&self) -> Result<Vec<f32>>;
    fn read_prediction(&self) -> Result<Vec<f32>>;
    fn sync(&self, model: &mut Model, adam: &mut Adam) -> Result<()>;
    /// Wait for every queued device operation. Timing harnesses must call this: a trainer is free
    /// to pipeline steps, so the time around `step()` alone is enqueue time, not device time.
    fn flush(&self) -> Result<()>;
    /// Upload updated parameters onto an existing resident device trainer without recreating its
    /// context or allocations.
    fn load_weights(&self, _model: &Model) -> Result<()> {
        bail!("resident weight reload is not implemented for this backend")
    }
    fn info(&self) -> Value;
    fn completed_steps(&self) -> usize;
}
pub enum Trainer {
    Cpu,
    #[cfg(feature = "opencl")]
    OpenCl(Box<crate::gpu::OpenClTrainer>),
    #[cfg(feature = "cuda")]
    Cuda(Box<crate::cuda::CudaTrainer>),
}
impl Trainer {
    pub fn new(
        name: &str,
        model: &Model,
        adam: &Adam,
        n: usize,
        flow: &FlowConfig,
        batch: usize,
    ) -> Result<Self> {
        ensure!((1..=64).contains(&batch), "batch size must be 1..64");
        match name {
            "cpu" => Ok(Self::Cpu),
            "opencl" => {
                #[cfg(feature = "opencl")]
                {
                    ensure!(
                        batch == 1,
                        "the OpenCL backend is single-crop; use --backend cuda or --batch 1"
                    );
                    Ok(Self::OpenCl(Box::new(crate::gpu::OpenClTrainer::new(
                        model,
                        adam,
                        n,
                        flow.clone(),
                    )?)))
                }
                #[cfg(not(feature = "opencl"))]
                {
                    let _ = (model, adam, n, flow, batch);
                    bail!("OpenCL was requested but not compiled; cargo build --locked --offline --release --features opencl -j 8")
                }
            }
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Self::Cuda(Box::new(crate::cuda::CudaTrainer::new(
                        model,
                        adam,
                        n,
                        batch,
                        flow.clone(),
                    )?)))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (model, adam, n, flow, batch);
                    bail!("CUDA was requested but not compiled; cargo build --locked --offline --release --features cuda -j 8")
                }
            }
            _ => bail!("backend must be cpu, opencl or cuda"),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu-f32",
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => "opencl-f32-resident",
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda-f32-resident",
        }
    }
    pub fn info(&self) -> Value {
        match self {
            Self::Cpu => {
                json!({"backend":"cpu-f32","precision":"fp32","training":"CPU reference"})
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(g) => g.info(),
            #[cfg(feature = "cuda")]
            Self::Cuda(g) => g.info(),
        }
    }
    pub fn step(
        &mut self,
        model: &mut Model,
        adam: &mut Adam,
        flow: &FlowConfig,
        batch: Batch<'_>,
    ) -> Result<Step> {
        match self {
            Self::Cpu => {
                let items = batch.items();
                ensure!(
                    items > 0 && batch.clean.len().is_multiple_of(items),
                    "invalid CPU batch shape"
                );
                let n = batch.clean.len() / items;
                let mut loss = 0f32;
                let mut accumulated = vec![0f32; model.count()];
                let neural = Instant::now();
                for i in 0..items {
                    ensure!(
                        batch.mask.len() == batch.clean.len()
                            && batch.noise.len() == batch.clean.len(),
                        "invalid CPU batch shape"
                    );
                    let row = i * n..(i + 1) * n;
                    let (item_loss, g) = parameterization::loss_gradient_detailed(
                        model,
                        flow,
                        &batch.clean[row.clone()],
                        &batch.mask[row.clone()],
                        &batch.noise[row.clone()],
                        batch.t[i],
                        batch.detail,
                    );
                    loss += item_loss;
                    for (dst, src) in accumulated.iter_mut().zip(&g) {
                        *dst += src;
                    }
                }
                for g in &mut accumulated {
                    *g /= items as f32;
                }
                let neural_ms = neural.elapsed().as_secs_f64() * 1000.;
                let optimizer = Instant::now();
                let norm = adam.update(model, &accumulated)?;
                Ok(Step {
                    loss: loss / items as f32,
                    norm,
                    neural_ms,
                    optimizer_ms: optimizer.elapsed().as_secs_f64() * 1000.,
                })
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(g) => {
                ensure!(
                    batch.items() == 1,
                    "the OpenCL backend is single-crop; use --backend cuda or --batch 1"
                );
                ensure!(
                    batch.detail == 0.,
                    "detail weighting is implemented on the CPU and CUDA backends only"
                );
                let s = g.train_step(
                    batch.clean,
                    batch.mask,
                    batch.noise,
                    batch.t.first().copied().unwrap_or(0.),
                )?;
                adam.step = g.step;
                Ok(Step {
                    loss: s.loss,
                    norm: s.norm,
                    neural_ms: s.neural_ms,
                    optimizer_ms: s.optimizer_ms,
                })
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(g) => {
                g.set_detail(batch.detail);
                let s = g.train_step(batch.clean, batch.mask, batch.noise, batch.t)?;
                adam.step = g.step;
                Ok(Step {
                    loss: s.loss,
                    norm: s.norm,
                    neural_ms: s.neural_ms,
                    optimizer_ms: s.optimizer_ms,
                })
            }
        }
    }
    pub fn sync(&self, model: &mut Model, adam: &mut Adam) -> Result<()> {
        match self {
            Self::Cpu => {
                let _ = (model, adam);
                Ok(())
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(g) => g.sync(model, adam),
            #[cfg(feature = "cuda")]
            Self::Cuda(g) => g.sync(model, adam),
        }
    }
}
