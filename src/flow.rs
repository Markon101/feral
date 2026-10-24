use crate::{
    conditioning,
    model::{Model, Velocity},
    rng::Rng,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sampling {
    pub seed: u64,
    pub steps: usize,
    pub solver: String,
    pub noise_scale: f32,
    pub perturb: f32,
    pub collar: usize,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            seed: 42,
            steps: 32,
            solver: "heun".into(),
            noise_scale: 1.,
            perturb: 0.,
            collar: 128,
        }
    }
}
impl Sampling {
    pub fn validate(&self) -> Result<()> {
        ensure!((1..=512).contains(&self.steps), "flow steps must be 1..512");
        ensure!(
            ["euler", "heun"].contains(&self.solver.as_str()),
            "solver must be euler or heun"
        );
        ensure!(
            self.noise_scale.is_finite() && (0.0..=4.).contains(&self.noise_scale),
            "noise scale must be 0..4"
        );
        ensure!(
            self.perturb.is_finite() && (0.0..=0.5).contains(&self.perturb),
            "perturb must be 0..0.5"
        );
        Ok(())
    }
}
/// Conditional rectified flow: unknown x_t=(1-t)z+t*y; known x_t=y.
/// The loss is mean squared velocity error on unknown samples only.
pub fn loss_gradient(
    model: &Model,
    clean: &[f32],
    mask: &[f32],
    noise: &[f32],
    t: f32,
) -> (f32, Vec<f32>) {
    let x: Vec<f32> = clean
        .iter()
        .zip(noise)
        .zip(mask)
        .map(|((&y, &z), &m)| if m == 1. { y } else { (1. - t) * z + t * y })
        .collect();
    let (pred, tape) = model.forward(&x, clean, mask, t);
    let count = mask.iter().filter(|&&m| m == 0.).count().max(1) as f32;
    let mut loss = 0.;
    let dy: Vec<f32> = pred
        .iter()
        .zip(clean)
        .zip(noise)
        .zip(mask)
        .map(|(((&v, &y), &z), &m)| {
            let d = (v - (y - z)) * (1. - m);
            loss += d * d / count;
            2. * d / count
        })
        .collect();
    (loss, model.backward(&tape, &dy))
}
fn field<M: Velocity + ?Sized>(
    model: &M,
    x: &[f32],
    known: &[f32],
    mask: &[f32],
    t: f32,
    c: &Sampling,
    texture: &[f32],
) -> Result<Vec<f32>> {
    let mut v = model.velocity(x, known, mask, t);
    ensure!(v.len() == x.len(), "backend output shape mismatch");
    for (i, y) in v.iter_mut().enumerate() {
        *y = (*y + c.perturb * (std::f32::consts::PI * t).sin() * texture[i].tanh())
            * (1. - mask[i]);
    }
    ensure!(
        v.iter().all(|v| v.is_finite()),
        "nonfinite velocity; reduce perturbation or reject checkpoint"
    );
    Ok(v)
}
pub fn sample<M: Velocity + ?Sized>(
    model: &M,
    known: &[f32],
    mask: &[f32],
    c: &Sampling,
) -> Result<Vec<f32>> {
    c.validate()?;
    ensure!(
        !known.is_empty()
            && known.len() == mask.len()
            && known.len().is_multiple_of(model.block())
            && known.len() <= 160000,
        "invalid sample shape or >160000 samples"
    );
    ensure!(
        known.iter().all(|v| v.is_finite()) && mask.iter().all(|&m| m == 0. || m == 1.),
        "invalid conditioning"
    );
    let mut rng = Rng::new(c.seed);
    let mut x: Vec<f32> = known.iter().map(|_| rng.normal() * c.noise_scale).collect();
    let texture: Vec<f32> = known.iter().map(|_| rng.normal()).collect();
    conditioning::project(&mut x, known, mask);
    let dt = 1. / c.steps as f32;
    for step in 0..c.steps {
        let t = step as f32 * dt;
        let v = field(model, &x, known, mask, t, c, &texture)?;
        if c.solver == "heun" {
            let mut next: Vec<f32> = x.iter().zip(&v).map(|(&x, &v)| x + dt * v).collect();
            conditioning::project(&mut next, known, mask);
            let vv = field(model, &next, known, mask, t + dt, c, &texture)?;
            for i in 0..x.len() {
                x[i] += 0.5 * dt * (v[i] + vv[i]);
            }
        } else {
            for i in 0..x.len() {
                x[i] += dt * v[i];
            }
        }
        conditioning::project(&mut x, known, mask);
        ensure!(
            x.iter().all(|v| v.is_finite() && v.abs() < 100.),
            "unstable trajectory at step {step}; output not written"
        );
    }
    // Hard safety limiter is explicit and only applies to generated samples.
    for (v, &m) in x.iter_mut().zip(mask) {
        if m == 0. {
            *v = v.clamp(-0.98, 0.98);
        }
    }
    conditioning::continuity(&mut x, known, mask, c.collar);
    for (v, &m) in x.iter_mut().zip(mask) {
        if m == 0. {
            *v = v.clamp(-0.98, 0.98);
        }
    }
    Ok(x)
}
