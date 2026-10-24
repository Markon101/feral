//! Checkpointed velocity parameterizations. The Gaussian reference is a control,
//! not a replacement for the learned acoustic correction.
use crate::{
    flow,
    model::{Model, Velocity},
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum FlowConfig {
    #[default]
    Velocity,
    Preconditioned {
        data_scale: f32,
    },
}
impl FlowConfig {
    pub fn is_legacy(&self) -> bool {
        matches!(self, Self::Velocity)
    }
    pub fn validate(&self) -> Result<()> {
        if let Self::Preconditioned { data_scale } = self {
            ensure!(
                data_scale.is_finite() && (0.01..=2.).contains(data_scale),
                "data scale must be finite and 0.01..2"
            );
        }
        Ok(())
    }
    /// For y variance A and unit noise variance, D=t²A+(1-t)².
    /// v = skip*x + out*f(x/sqrt(D)); out=sqrt(A/D).
    pub fn coefficients(&self, t: f32) -> (f32, f32, f32) {
        match self {
            Self::Velocity => (0., 1., 1.),
            Self::Preconditioned { data_scale } => {
                let a = data_scale * data_scale;
                let d = t * t * a + (1. - t) * (1. - t);
                ((t * a - (1. - t)) / d, data_scale / d.sqrt(), 1. / d.sqrt())
            }
        }
    }
    fn inputs(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> (Vec<f32>, Vec<f32>) {
        let Self::Preconditioned { data_scale } = self else {
            return (x.to_vec(), known.to_vec());
        };
        let (_, _, input_scale) = self.coefficients(t);
        let k: Vec<f32> = known.iter().map(|v| v / data_scale).collect();
        let xx = x
            .iter()
            .zip(&k)
            .zip(mask)
            .map(|((&x, &known), &m)| if m == 1. { known } else { x * input_scale })
            .collect();
        (xx, k)
    }
}
pub struct ConfiguredVelocity<'a> {
    pub model: &'a Model,
    pub config: &'a FlowConfig,
}
impl Velocity for ConfiguredVelocity<'_> {
    fn block(&self) -> usize {
        self.model.block()
    }
    fn velocity(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> Vec<f32> {
        if self.config.is_legacy() {
            return self.model.velocity(x, known, mask, t);
        }
        let (xx, k) = self.config.inputs(x, known, mask, t);
        let f = self.model.velocity(&xx, &k, mask, t);
        let (skip, out, _) = self.config.coefficients(t);
        f.iter().zip(x).map(|(&f, &x)| skip * x + out * f).collect()
    }
}
pub fn loss_gradient(
    model: &Model,
    config: &FlowConfig,
    clean: &[f32],
    mask: &[f32],
    noise: &[f32],
    t: f32,
) -> (f32, Vec<f32>) {
    if config.is_legacy() {
        return flow::loss_gradient(model, clean, mask, noise, t);
    }
    let x: Vec<f32> = clean
        .iter()
        .zip(noise)
        .zip(mask)
        .map(|((&y, &z), &m)| if m == 1. { y } else { (1. - t) * z + t * y })
        .collect();
    let (xx, k) = config.inputs(&x, clean, mask, t);
    let (f, tape) = model.forward(&xx, &k, mask, t);
    let (skip, out, _) = config.coefficients(t);
    let count = mask.iter().filter(|&&m| m == 0.).count().max(1) as f32;
    let mut loss = 0.;
    let dy: Vec<f32> = f
        .iter()
        .zip(&x)
        .zip(clean)
        .zip(noise)
        .zip(mask)
        .map(|((((&f, &x), &y), &z), &m)| {
            let target = (y - z - skip * x) / out;
            let d = (f - target) * (1. - m);
            loss += d * d / count;
            2. * d / count
        })
        .collect();
    (loss, model.backward(&tape, &dy))
}
/// Linear Gaussian reference with zero learned correction, for scientific controls.
pub struct GaussianReference {
    pub block: usize,
    pub data_scale: f32,
}
impl Velocity for GaussianReference {
    fn block(&self) -> usize {
        self.block
    }
    fn velocity(&self, x: &[f32], _: &[f32], _: &[f32], t: f32) -> Vec<f32> {
        let (skip, _, _) = FlowConfig::Preconditioned {
            data_scale: self.data_scale,
        }
        .coefficients(t);
        x.iter().map(|v| skip * v).collect()
    }
}
