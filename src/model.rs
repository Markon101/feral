//! A specialized differentiable residual convolution network, not a tensor engine.
//! Time-major contiguous [frame, channel] arrays. All training math is f32.
use crate::rng::Rng;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// Largest window any backend may be asked to hold: 32 s at 16 kHz. Crops, render windows and
/// previews all refer to this, so a checkpoint cannot describe a window that training rejects.
pub const MAX_WINDOW_SAMPLES: usize = 512_000;
/// Hidden channels in the optional sample-level residual stem and head convolutions.
pub const SAMPLE_CONV_CHANNELS: usize = 16;

fn is_zero_usize(v: &usize) -> bool {
    *v == 0
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Architecture {
    pub block: usize,
    pub width: usize,
    pub dilations: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scales: Vec<usize>,
    /// Optional odd sample-level 1D conv kernel size (0 disables; 3..=15 odd enables residual
    /// sample-domain stem and head convolutions around the frame stack).
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub sample_conv: usize,
}
impl Default for Architecture {
    fn default() -> Self {
        Self {
            block: 32,
            width: 64,
            dilations: vec![1, 2, 4, 8],
            scales: vec![],
            sample_conv: 0,
        }
    }
}
impl Architecture {
    /// Select a bounded training width without changing the baseline topology.
    /// Small custom architectures remain valid for numerical-reference tests.
    pub fn with_width(mut self, width: usize) -> Result<Self> {
        ensure!(
            (32..=512).contains(&width) && width.is_multiple_of(16),
            "training width must be a multiple of 16 between 32 and 512"
        );
        self.width = width;
        self.validate()?;
        Ok(self)
    }
    /// Samples per frame. The stride trades time resolution for context: convolution cost scales
    /// with frames, so a larger block is cheaper per sample and multiplies the receptive field by
    /// the same factor.
    pub fn with_frame_samples(mut self, block: usize) -> Result<Self> {
        ensure!(
            [16, 32, 64, 128].contains(&block),
            "frame samples must be 16, 32, 64 or 128"
        );
        self.block = block;
        self.validate()?;
        Ok(self)
    }
    /// Rebuild the baseline residual stack with `blocks` power-of-two dilations. The default
    /// four blocks are the v1 baseline; five at width 256 is the ~1.02M-parameter configuration.
    pub fn with_blocks(mut self, blocks: usize) -> Result<Self> {
        ensure!(
            (1..=10).contains(&blocks),
            "block count must be 1..10 (dilations up to 512)"
        );
        ensure!(
            self.dilations.len() == Self::default().dilations.len() && self.scales.is_empty(),
            "with_blocks only applies to the unscaled baseline residual stack"
        );
        self.dilations = (0..blocks).map(|i| 1 << i).collect();
        self.validate()?;
        Ok(self)
    }
    /// Enable (odd 3..=15) or disable (0) residual sample-domain 1D convolutions around the
    /// frame stack to bridge frame boundaries with shift-equivariant sample context.
    pub fn with_sample_conv(mut self, kernel: usize) -> Result<Self> {
        ensure!(
            kernel == 0 || ((3..=15).contains(&kernel) && kernel % 2 == 1),
            "sample_conv must be 0 (disabled) or an odd kernel size in 3..=15"
        );
        self.sample_conv = kernel;
        self.validate()?;
        Ok(self)
    }
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
    /// Structural interior support, including aligned average-pooling groups and any sample convs.
    pub fn context_samples(&self) -> usize {
        let period = self.scales.iter().copied().max().unwrap_or(1);
        let frame_support = (0..period)
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
            .unwrap_or(self.block);
        let sample_extra = if self.sample_conv > 0 {
            4 * (self.sample_conv - 1)
        } else {
            0
        };
        frame_support + sample_extra
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=128).contains(&self.block) && (1..=512).contains(&self.width),
            "invalid model size"
        );
        ensure!(
            !self.dilations.is_empty()
                && self.dilations.len() <= 10
                && self.dilations.iter().all(|d| (1..=512).contains(d)),
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
        ensure!(
            self.sample_conv == 0
                || ((3..=15).contains(&self.sample_conv) && self.sample_conv % 2 == 1),
            "invalid sample_conv kernel size (must be 0 or odd 3..=15)"
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
    /// Optional sample-domain residual stem `[Layer(3->16, k=K), Layer(16->1, k=K)]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stem: Vec<Layer>,
    /// Optional sample-domain residual head `[Layer(1->16, k=K), Layer(16->1, k=K)]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub head: Vec<Layer>,
}
/// CPU-independent boundary for flow and conditioning; another backend can own buffers.
pub trait Velocity {
    fn prior_std(&self) -> f32 {
        1.
    }
    fn block(&self) -> usize;
    fn velocity(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> Vec<f32>;
    /// Fallible field evaluation. The CPU reference cannot fail, but an accelerator can (driver
    /// error, window mismatch); a sampler must report that instead of panicking or continuing.
    fn try_velocity(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> Result<Vec<f32>> {
        Ok(self.velocity(x, known, mask, t))
    }
}
pub struct Tape {
    features: Vec<f32>,
    acts: Vec<Vec<f32>>,
    nonlinear: Vec<Vec<f32>>,
    stem_in: Vec<f32>,
    stem_a0: Vec<f32>,
    stem_a1: Vec<f32>,
    head_a0: Vec<f32>,
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
        let (stem, head) = if a.sample_conv > 0 {
            let k = a.sample_conv;
            (
                vec![
                    Layer::new(3, SAMPLE_CONV_CHANNELS, k, 1, &mut rng),
                    Layer::new(SAMPLE_CONV_CHANNELS, 1, k, 1, &mut rng),
                ],
                vec![
                    Layer::new(1, SAMPLE_CONV_CHANNELS, k, 1, &mut rng),
                    Layer::new(SAMPLE_CONV_CHANNELS, 1, k, 1, &mut rng),
                ],
            )
        } else {
            (vec![], vec![])
        };
        Self {
            architecture: a,
            layers,
            stem,
            head,
        }
    }
    pub fn count(&self) -> usize {
        self.layers
            .iter()
            .chain(&self.stem)
            .chain(&self.head)
            .map(|l| l.w.len() + l.b.len())
            .sum()
    }
    pub fn parameters(&self) -> Vec<f32> {
        self.layers
            .iter()
            .chain(&self.stem)
            .chain(&self.head)
            .flat_map(|l| l.w.iter().chain(&l.b).copied())
            .collect()
    }
    pub fn set_parameters(&mut self, p: &[f32]) {
        let mut offset = 0;
        for l in self
            .layers
            .iter_mut()
            .chain(&mut self.stem)
            .chain(&mut self.head)
        {
            for v in l.w.iter_mut().chain(&mut l.b) {
                *v = p[offset];
                offset += 1;
            }
        }
        assert_eq!(offset, p.len());
    }
    pub fn forward(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> (Vec<f32>, Tape) {
        let b = self.architecture.block;
        let n = x.len();
        assert_eq!(n % b, 0);
        assert_eq!(n, known.len());
        assert_eq!(n, mask.len());
        let frames = n / b;
        let (x_eff, stem_in, stem_a0, stem_a1) = if !self.stem.is_empty() {
            let mut stem_in = Vec::with_capacity(3 * n);
            for i in 0..n {
                stem_in.push(x[i]);
                stem_in.push(known[i] * mask[i]);
                stem_in.push(mask[i]);
            }
            let mut s0 = self.stem[0].forward(&stem_in, n);
            for v in &mut s0 {
                *v = v.tanh();
            }
            let mut s1 = self.stem[1].forward(&s0, n);
            for v in &mut s1 {
                *v = v.tanh();
            }
            let x_eff: Vec<f32> = x.iter().zip(&s1).map(|(&xi, &si)| xi + 0.5 * si).collect();
            (x_eff, stem_in, s0, s1)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        let x_src = if x_eff.is_empty() { x } else { &x_eff };
        let mut features = Vec::with_capacity(frames * (3 * b + 4));
        for p in 0..frames {
            features.extend_from_slice(&x_src[p * b..(p + 1) * b]);
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
        let (out, head_a0) = if !self.head.is_empty() {
            let mut h0 = self.head[0].forward(&h, n);
            for v in &mut h0 {
                *v = v.tanh();
            }
            let h1 = self.head[1].forward(&h0, n);
            let out: Vec<f32> = h.iter().zip(&h1).map(|(&hi, &ri)| hi + 0.5 * ri).collect();
            (out, h0)
        } else {
            (h, Vec::new())
        };
        (
            out,
            Tape {
                features,
                acts,
                nonlinear,
                stem_in,
                stem_a0,
                stem_a1,
                head_a0,
            },
        )
    }
    /// Exact reverse-mode derivatives for the specialized convolution/tanh/residual graph.
    pub fn backward(&self, tape: &Tape, dy: &[f32]) -> Vec<f32> {
        let b = self.architecture.block;
        let n = dy.len();
        let frames = n / b;
        let (mut dh, head_grads) = if !self.head.is_empty() {
            let dh1: Vec<f32> = dy.iter().map(|&g| 0.5 * g).collect();
            let (d_h0, dw_h1, db_h1) = self.head[1].backward(&tape.head_a0, &dh1, n);
            let dz_h0: Vec<f32> = d_h0
                .iter()
                .zip(&tape.head_a0)
                .map(|(&g, &a)| g * (1. - a * a))
                .collect();
            let h_raw = tape.acts.last().expect("frame activations");
            let (d_hraw, dw_h0, db_h0) = self.head[0].backward(h_raw, &dz_h0, n);
            let dh_frame: Vec<f32> = dy.iter().zip(&d_hraw).map(|(&a, &b)| a + b).collect();
            let hg: Vec<f32> = dw_h0
                .into_iter()
                .chain(db_h0)
                .chain(dw_h1)
                .chain(db_h1)
                .collect();
            (dh_frame, hg)
        } else {
            (dy.to_vec(), Vec::new())
        };
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
        let mut out: Vec<f32> = grads.into_iter().flatten().collect();
        if !self.stem.is_empty() {
            let stride = 3 * b + 4;
            let mut dz_s1 = Vec::with_capacity(n);
            for p in 0..frames {
                for c in 0..b {
                    let i = p * b + c;
                    let a = tape.stem_a1[i];
                    dz_s1.push(dh[p * stride + c] * 0.5 * (1. - a * a));
                }
            }
            let (d_s0, dw_s1, db_s1) = self.stem[1].backward(&tape.stem_a0, &dz_s1, n);
            let dz_s0: Vec<f32> = d_s0
                .iter()
                .zip(&tape.stem_a0)
                .map(|(&g, &a)| g * (1. - a * a))
                .collect();
            let (_, dw_s0, db_s0) = self.stem[0].backward(&tape.stem_in, &dz_s0, n);
            out.extend(dw_s0);
            out.extend(db_s0);
            out.extend(dw_s1);
            out.extend(db_s1);
            out.extend(head_grads);
        }
        out
    }
    pub fn validate(&self) -> Result<()> {
        self.architecture.validate()?;
        let reference = Self::new(self.architecture.clone(), 0);
        let check = |actual: &[Layer], expected: &[Layer], kind: &str| -> Result<()> {
            ensure!(
                actual.len() == expected.len(),
                "checkpoint {kind} layer count mismatch"
            );
            for (a, b) in actual.iter().zip(expected) {
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
                    "checkpoint {kind} layer shape mismatch"
                );
            }
            Ok(())
        };
        check(&self.layers, &reference.layers, "frame")?;
        check(&self.stem, &reference.stem, "stem")?;
        check(&self.head, &reference.head, "head")?;
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
