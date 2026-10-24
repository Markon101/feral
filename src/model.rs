//! A specialized differentiable residual convolution network, not a tensor engine.
//! Time-major contiguous [frame, channel] arrays. All training math is f32.
use crate::rng::Rng;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Architecture {
    pub block: usize,
    pub width: usize,
    pub dilations: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scales: Vec<usize>,
}
impl Default for Architecture {
    fn default() -> Self {
        Self {
            block: 32,
            width: 64,
            dilations: vec![1, 2, 4, 8],
            scales: vec![],
        }
    }
}
impl Architecture {
    pub fn pyramid() -> Self {
        Self {
            scales: vec![1, 4, 16, 64],
            dilations: vec![1, 1, 1, 1],
            ..Self::default()
        }
    }
    pub fn scale(&self, layer: usize) -> usize {
        self.scales.get(layer).copied().unwrap_or(1)
    }
    /// Structural interior support, including aligned average-pooling groups.
    pub fn context_samples(&self) -> usize {
        let period = self.scales.iter().copied().max().unwrap_or(1);
        (0..period)
            .map(|p| {
                let (mut lo, mut hi) = (p as isize, p as isize);
                for j in (0..self.dilations.len()).rev() {
                    let s = self.scale(j) as isize;
                    let d = self.dilations[j] as isize;
                    lo = lo.div_euclid(s) * s - d * s;
                    hi = hi.div_euclid(s) * s + d * s + s - 1;
                }
                (hi - lo + 1) as usize * self.block
            })
            .max()
            .unwrap_or(self.block)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=64).contains(&self.block) && (1..=128).contains(&self.width),
            "invalid model size"
        );
        ensure!(
            !self.dilations.is_empty()
                && self.dilations.len() <= 8
                && self.dilations.iter().all(|d| (1..=64).contains(d)),
            "invalid dilations"
        );
        ensure!(
            self.scales.is_empty()
                || (self.scales.len() == self.dilations.len()
                    && self
                        .scales
                        .iter()
                        .all(|s| [1, 2, 4, 8, 16, 32, 64].contains(s))),
            "invalid residual scales"
        );
        Ok(())
    }
}
// Average-pooling and its transpose use the actual final-group length, never padded zeros.
fn pool(x: &[f32], frames: usize, channels: usize, scale: usize) -> Vec<f32> {
    let mut y = vec![0.; frames.div_ceil(scale) * channels];
    for p in 0..frames {
        let count = scale.min(frames - (p / scale) * scale) as f32;
        for c in 0..channels {
            y[(p / scale) * channels + c] += x[p * channels + c] / count;
        }
    }
    y
}
fn pool_transpose(x: &[f32], frames: usize, channels: usize, scale: usize) -> Vec<f32> {
    let mut y = vec![0.; frames * channels];
    for p in 0..frames {
        let count = scale.min(frames - (p / scale) * scale) as f32;
        for c in 0..channels {
            y[p * channels + c] = x[(p / scale) * channels + c] / count;
        }
    }
    y
}
fn repeat(x: &[f32], frames: usize, channels: usize, scale: usize) -> Vec<f32> {
    (0..frames * channels)
        .map(|i| x[(i / channels / scale) * channels + i % channels])
        .collect()
}
fn repeat_transpose(x: &[f32], frames: usize, channels: usize, scale: usize) -> Vec<f32> {
    let mut y = vec![0.; frames.div_ceil(scale) * channels];
    for p in 0..frames {
        for c in 0..channels {
            y[(p / scale) * channels + c] += x[p * channels + c];
        }
    }
    y
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub input: usize,
    pub output: usize,
    pub kernel: usize,
    pub dilation: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub architecture: Architecture,
    pub layers: Vec<Layer>,
}
/// CPU-independent boundary for flow and conditioning; another backend can own buffers.
pub trait Velocity {
    fn prior_std(&self) -> f32 {
        1.
    }
    fn block(&self) -> usize;
    fn velocity(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> Vec<f32>;
}
pub struct Tape {
    features: Vec<f32>,
    acts: Vec<Vec<f32>>,
    nonlinear: Vec<Vec<f32>>,
}
impl Layer {
    fn new(input: usize, output: usize, kernel: usize, dilation: usize, rng: &mut Rng) -> Self {
        let scale = (1.0 / (input * kernel) as f32).sqrt();
        Self {
            input,
            output,
            kernel,
            dilation,
            w: (0..input * output * kernel)
                .map(|_| rng.normal() * scale)
                .collect(),
            b: vec![0.; output],
        }
    }
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let mut y = vec![0.; frames * self.output];
        for p in 0..frames {
            for o in 0..self.output {
                let mut sum = self.b[o];
                for k in 0..self.kernel {
                    let q = p as isize
                        + (k as isize - (self.kernel / 2) as isize) * self.dilation as isize;
                    if q < 0 || q >= frames as isize {
                        continue;
                    }
                    let row = &self.w[(o * self.kernel + k) * self.input
                        ..(o * self.kernel + k + 1) * self.input];
                    let xx = &x[q as usize * self.input..(q as usize + 1) * self.input];
                    for i in 0..self.input {
                        sum += row[i] * xx[i];
                    }
                }
                y[p * self.output + o] = sum;
            }
        }
        y
    }
    fn backward(&self, x: &[f32], dy: &[f32], frames: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut dx = vec![0.; x.len()];
        let mut dw = vec![0.; self.w.len()];
        let mut db = vec![0.; self.output];
        for p in 0..frames {
            for o in 0..self.output {
                let g = dy[p * self.output + o];
                db[o] += g;
                for k in 0..self.kernel {
                    let q = p as isize
                        + (k as isize - (self.kernel / 2) as isize) * self.dilation as isize;
                    if q < 0 || q >= frames as isize {
                        continue;
                    }
                    let offset = (o * self.kernel + k) * self.input;
                    let xx = &x[q as usize * self.input..(q as usize + 1) * self.input];
                    let dd = &mut dx[q as usize * self.input..(q as usize + 1) * self.input];
                    for i in 0..self.input {
                        dw[offset + i] += g * xx[i];
                        dd[i] += g * self.w[offset + i];
                    }
                }
            }
        }
        (dx, dw, db)
    }
}
impl Model {
    pub fn new(a: Architecture, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut layers = vec![Layer::new(3 * a.block + 4, a.width, 1, 1, &mut rng)];
        for &d in &a.dilations {
            layers.push(Layer::new(a.width, a.width, 3, d, &mut rng));
        }
        layers.push(Layer::new(a.width, a.block, 1, 1, &mut rng));
        Self {
            architecture: a,
            layers,
        }
    }
    pub fn count(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }
    pub fn parameters(&self) -> Vec<f32> {
        self.layers
            .iter()
            .flat_map(|l| l.w.iter().chain(&l.b).copied())
            .collect()
    }
    pub fn set_parameters(&mut self, p: &[f32]) {
        let mut offset = 0;
        for l in &mut self.layers {
            for v in l.w.iter_mut().chain(&mut l.b) {
                *v = p[offset];
                offset += 1;
            }
        }
        assert_eq!(offset, p.len());
    }
    pub fn forward(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> (Vec<f32>, Tape) {
        let b = self.architecture.block;
        assert_eq!(x.len() % b, 0);
        assert_eq!(x.len(), known.len());
        assert_eq!(x.len(), mask.len());
        let frames = x.len() / b;
        let mut features = Vec::with_capacity(frames * (3 * b + 4));
        for p in 0..frames {
            features.extend_from_slice(&x[p * b..(p + 1) * b]);
            features.extend((p * b..(p + 1) * b).map(|i| known[i] * mask[i]));
            features.extend_from_slice(&mask[p * b..(p + 1) * b]);
            features.extend([
                t,
                1. - t,
                (t * std::f32::consts::TAU).sin(),
                (t * std::f32::consts::TAU).cos(),
            ]);
        }
        let mut acts = vec![];
        let mut nonlinear = vec![];
        let mut h = features.clone();
        for (j, l) in self.layers.iter().enumerate() {
            let residual = j > 0 && j + 1 < self.layers.len();
            let scale = if residual {
                self.architecture.scale(j - 1)
            } else {
                1
            };
            let pooled = if scale > 1 {
                pool(&h, frames, l.input, scale)
            } else {
                h.clone()
            };
            let mut z = l.forward(&pooled, frames.div_ceil(scale));
            if j + 1 < self.layers.len() {
                for v in &mut z {
                    *v = v.tanh();
                }
                nonlinear.push(z.clone());
                if scale > 1 {
                    z = repeat(&z, frames, l.output, scale);
                }
                if residual {
                    for (v, old) in z.iter_mut().zip(&h) {
                        *v = old + 0.5 * *v;
                    }
                }
            }
            acts.push(z.clone());
            h = z;
        }
        (
            h,
            Tape {
                features,
                acts,
                nonlinear,
            },
        )
    }
    /// Exact reverse-mode derivatives for the specialized convolution/tanh/residual graph.
    pub fn backward(&self, tape: &Tape, dy: &[f32]) -> Vec<f32> {
        let frames = dy.len() / self.architecture.block;
        let mut dh = dy.to_vec();
        let mut grads = vec![Vec::new(); self.layers.len()];
        for j in (0..self.layers.len()).rev() {
            let residual = j > 0 && j + 1 < self.layers.len();
            let layer = &self.layers[j];
            let scale = if residual {
                self.architecture.scale(j - 1)
            } else {
                1
            };
            let mut dz = if scale > 1 {
                repeat_transpose(&dh, frames, layer.output, scale)
            } else {
                dh.clone()
            };
            if j + 1 < self.layers.len() {
                for (g, &a) in dz.iter_mut().zip(&tape.nonlinear[j]) {
                    *g *= (1. - a * a) * if residual { 0.5 } else { 1. };
                }
            }
            let input = if j == 0 {
                &tape.features
            } else {
                &tape.acts[j - 1]
            };
            let pooled = if scale > 1 {
                pool(input, frames, layer.input, scale)
            } else {
                input.clone()
            };
            let (dx, dw, db) = layer.backward(&pooled, &dz, frames.div_ceil(scale));
            let mut dx = if scale > 1 {
                pool_transpose(&dx, frames, layer.input, scale)
            } else {
                dx
            };
            if residual {
                for (g, old) in dx.iter_mut().zip(&dh) {
                    *g += old;
                }
            }
            grads[j] = dw.into_iter().chain(db).collect();
            dh = dx;
        }
        grads.into_iter().flatten().collect()
    }
    pub fn validate(&self) -> Result<()> {
        self.architecture.validate()?;
        let reference = Self::new(self.architecture.clone(), 0);
        ensure!(
            self.layers.len() == reference.layers.len(),
            "checkpoint layer count mismatch"
        );
        for (a, b) in self.layers.iter().zip(reference.layers) {
            ensure!(
                (
                    a.input,
                    a.output,
                    a.kernel,
                    a.dilation,
                    a.w.len(),
                    a.b.len()
                ) == (
                    b.input,
                    b.output,
                    b.kernel,
                    b.dilation,
                    b.w.len(),
                    b.b.len()
                ),
                "checkpoint layer shape mismatch"
            );
        }
        ensure!(
            self.parameters().iter().all(|v| v.is_finite()),
            "nonfinite checkpoint"
        );
        Ok(())
    }
}
impl Velocity for Model {
    fn block(&self) -> usize {
        self.architecture.block
    }
    fn velocity(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> Vec<f32> {
        self.forward(x, known, mask, t).0
    }
}
