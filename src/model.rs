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
}
impl Default for Architecture {
    fn default() -> Self {
        Self {
            block: 32,
            width: 64,
            dilations: vec![1, 2, 4, 8],
        }
    }
}
impl Architecture {
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
        Ok(())
    }
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
            let mut z = l.forward(&h, frames);
            if j + 1 < self.layers.len() {
                for v in &mut z {
                    *v = v.tanh();
                }
                nonlinear.push(z.clone());
                if j > 0 {
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
            let mut dz = dh.clone();
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
            let (mut dx, dw, db) = self.layers[j].backward(input, &dz, frames);
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
