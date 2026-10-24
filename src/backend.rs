//! Explicit training backend selection. There is no automatic or silent CPU fallback.
use crate::{
    model::Model,
    parameterization::{self, FlowConfig},
    runtime::Adam,
};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::time::Instant;
pub struct Batch<'a> {
    pub clean: &'a [f32],
    pub mask: &'a [f32],
    pub noise: &'a [f32],
    pub t: f32,
}
pub struct Step {
    pub loss: f32,
    pub norm: f32,
    pub neural_ms: f64,
    pub optimizer_ms: f64,
}
pub enum Trainer {
    Cpu,
    #[cfg(feature = "opencl")]
    OpenCl(Box<crate::gpu::OpenClTrainer>),
}
impl Trainer {
    pub fn new(
        name: &str,
        model: &Model,
        adam: &Adam,
        n: usize,
        flow: &FlowConfig,
    ) -> Result<Self> {
        match name {
            "cpu" => Ok(Self::Cpu),
            "opencl" => {
                #[cfg(feature = "opencl")]
                {
                    Ok(Self::OpenCl(Box::new(crate::gpu::OpenClTrainer::new(
                        model,
                        adam,
                        n,
                        flow.clone(),
                    )?)))
                }
                #[cfg(not(feature = "opencl"))]
                {
                    let _ = (model, adam, n, flow);
                    bail!("OpenCL was requested but not compiled; cargo build --locked --offline --release --features opencl -j 8")
                }
            }
            _ => bail!("backend must be cpu or opencl"),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu-f32",
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => "opencl-f32-resident",
        }
    }
    pub fn info(&self) -> Value {
        match self {
            Self::Cpu => json!({"backend":"cpu-f32","precision":"fp32","training":"CPU reference"}),
            #[cfg(feature = "opencl")]
            Self::OpenCl(g) => g.info(),
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
                let start = Instant::now();
                let (loss, g) = parameterization::loss_gradient(
                    model,
                    flow,
                    batch.clean,
                    batch.mask,
                    batch.noise,
                    batch.t,
                );
                let neural_ms = start.elapsed().as_secs_f64() * 1000.;
                let start = Instant::now();
                let norm = adam.update(model, &g)?;
                Ok(Step {
                    loss,
                    norm,
                    neural_ms,
                    optimizer_ms: start.elapsed().as_secs_f64() * 1000.,
                })
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(g) => {
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
        }
    }
}
