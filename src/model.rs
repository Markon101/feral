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
/// Dimension of the deterministic multi-harmonic sinusoidal time embedding for `GatedFilm` blocks.
pub const TIME_EMBED_DIM: usize = 32;
/// Epsilon inside pre-RMSNorm for `GatedFilm` residual blocks.
pub const RMS_NORM_EPS: f32 = 1e-5;

/// Residual block nonlinearity and conditioning style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BlockStyle {
    /// Legacy v1-v3 residual block: `h + 0.5 * tanh(Conv1d(h))`.
    #[default]
    Tanh,
    /// Pre-RMSNorm + per-block FiLM time conditioning + WaveNet gated activation
    /// `h + 0.5 * (tanh(a) * sigmoid(b))`.
    GatedFilm,
}
impl BlockStyle {
    pub fn is_tanh(&self) -> bool {
        matches!(self, Self::Tanh)
    }
}

/// Deterministic 32-dim log-spaced sinusoidal time embedding (16 frequencies from 1 to ~181 cycles).
pub fn time_embedding(t: f32) -> [f32; TIME_EMBED_DIM] {
    let mut out = [0f32; TIME_EMBED_DIM];
    for k in 0..TIME_EMBED_DIM / 2 {
        let freq = std::f32::consts::TAU * 2.0f32.powf(k as f32 * 0.5);
        let ang = t * freq;
        out[2 * k] = ang.sin();
        out[2 * k + 1] = ang.cos();
    }
    out
}

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
    /// Residual block style (`tanh` legacy default, or `gated-film` with pre-RMSNorm + FiLM + gate).
    #[serde(default, skip_serializing_if = "BlockStyle::is_tanh")]
    pub block_style: BlockStyle,
}
impl Default for Architecture {
    fn default() -> Self {
        Self {
            block: 32,
            width: 64,
            dilations: vec![1, 2, 4, 8],
            scales: vec![],
            sample_conv: 0,
            block_style: BlockStyle::Tanh,
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
    pub fn with_blocks(self, blocks: usize) -> Result<Self> {
        self.with_dilation_pattern(blocks, "ascending")
    }
    /// Build power-of-two dilations using `ascending` (`1..2^(b-1)`), `symmetric`
    /// (`1..2^(b-1)..1`, U-Net coarse-to-fine refinement), or `cycles` (two full `1..2^(b-1)` passes).
    pub fn with_dilation_pattern(mut self, blocks: usize, pattern: &str) -> Result<Self> {
        ensure!(
            (1..=10).contains(&blocks),
            "block count must be 1..10 (dilations up to 512)"
        );
        ensure!(
            self.dilations.len() == Self::default().dilations.len() && self.scales.is_empty(),
            "dilation patterns only apply to the unscaled baseline residual stack"
        );
        let up: Vec<usize> = (0..blocks).map(|i| 1 << i).collect();
        self.dilations = match pattern {
            "ascending" => up,
            "symmetric" => {
                ensure!(
                    blocks >= 2,
                    "symmetric dilation pattern requires blocks >= 2"
                );
                let mut d = up;
                for i in (0..blocks - 1).rev() {
                    d.push(1 << i);
                }
                d
            }
            "cycles" => {
                let mut d = up.clone();
                d.extend(up);
                d
            }
            other => anyhow::bail!(
                "dilation pattern must be ascending, symmetric or cycles, got {other}"
            ),
        };
        self.validate()?;
        Ok(self)
    }
    /// Select residual block style (`Tanh` legacy default or `GatedFilm`).
    pub fn with_block_style(mut self, style: BlockStyle) -> Result<Self> {
        self.block_style = style;
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
                && self.dilations.len() <= 24
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
    /// Per-block time-conditioning FiLM projections `32 -> 2*width` when `block_style == GatedFilm`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub time_proj: Vec<Layer>,
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
    time_embed: [f32; TIME_EMBED_DIM],
    films: Vec<Vec<f32>>,
    inv_rms: Vec<Vec<f32>>,
    normed: Vec<Vec<f32>>,
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
        let gated = a.block_style == BlockStyle::GatedFilm;
        let res_out = if gated { 2 * a.width } else { a.width };
        let mut layers = vec![Layer::new(3 * a.block + 4, a.width, 1, 1, &mut rng)];
        for &d in &a.dilations {
            layers.push(Layer::new(a.width, res_out, 3, d, &mut rng));
        }
        layers.push(Layer::new(a.width, a.block, 1, 1, &mut rng));
        let time_proj = if gated {
            let scale = 0.1 / (TIME_EMBED_DIM as f32).sqrt();
            (0..a.dilations.len())
                .map(|_| Layer {
                    input: TIME_EMBED_DIM,
                    output: 2 * a.width,
                    kernel: 1,
                    dilation: 1,
                    w: (0..TIME_EMBED_DIM * 2 * a.width)
                        .map(|_| rng.normal() * scale)
                        .collect(),
                    b: vec![0.; 2 * a.width],
                })
                .collect()
        } else {
            vec![]
        };
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
            time_proj,
            stem,
            head,
        }
    }
    pub fn count(&self) -> usize {
        self.layers
            .iter()
            .chain(&self.time_proj)
            .chain(&self.stem)
            .chain(&self.head)
            .map(|l| l.w.len() + l.b.len())
            .sum()
    }
    pub fn parameters(&self) -> Vec<f32> {
        self.layers
            .iter()
            .chain(&self.time_proj)
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
            .chain(&mut self.time_proj)
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
        let gated = self.architecture.block_style == BlockStyle::GatedFilm;
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
        let time_embed = if gated {
            time_embedding(t)
        } else {
            [0.; TIME_EMBED_DIM]
        };
        let mut acts = vec![];
        let mut nonlinear = vec![];
        let mut films = vec![];
        let mut inv_rms_all = vec![];
        let mut normed_all = vec![];
        let mut h = features.clone();
        let w = self.architecture.width;
        for (j, l) in self.layers.iter().enumerate() {
            let residual = j > 0 && j + 1 < self.layers.len();
            let scale = if residual {
                self.architecture.scale(j - 1)
            } else {
                1
            };
            if residual && gated {
                let block_idx = j - 1;
                let film = self.time_proj[block_idx].forward(&time_embed, 1);
                let mut inv_rms = Vec::with_capacity(frames);
                let mut normed = vec![0f32; frames * w];
                for p in 0..frames {
                    let row = &h[p * w..(p + 1) * w];
                    let sq: f32 = row.iter().map(|&v| v * v).sum();
                    let r = 1. / (sq / w as f32 + RMS_NORM_EPS).sqrt();
                    inv_rms.push(r);
                    for c in 0..w {
                        normed[p * w + c] = (1. + film[c]) * (row[c] * r) + film[w + c];
                    }
                }
                let conv_in = if scale > 1 {
                    pool(&normed, frames, w, scale)
                } else {
                    normed.clone()
                };
                let cf = frames.div_ceil(scale);
                let pre = l.forward(&conv_in, cf);
                let mut nl = vec![0f32; cf * 2 * w];
                let mut gated_act = vec![0f32; cf * w];
                for q in 0..cf {
                    for c in 0..w {
                        let a = pre[q * 2 * w + c].tanh();
                        let s = 1. / (1. + (-pre[q * 2 * w + w + c]).exp());
                        nl[q * 2 * w + c] = a;
                        nl[q * 2 * w + w + c] = s;
                        gated_act[q * w + c] = a * s;
                    }
                }
                nonlinear.push(nl);
                let rep = if scale > 1 {
                    repeat(&gated_act, frames, w, scale)
                } else {
                    gated_act
                };
                let mut next = vec![0f32; frames * w];
                for i in 0..frames * w {
                    next[i] = h[i] + 0.5 * rep[i];
                }
                films.push(film);
                inv_rms_all.push(inv_rms);
                normed_all.push(normed);
                acts.push(next.clone());
                h = next;
                continue;
            }
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
                time_embed,
                films,
                inv_rms: inv_rms_all,
                normed: normed_all,
                stem_in,
                stem_a0,
                stem_a1,
                head_a0,
            },
        )
    }
    /// Exact reverse-mode derivatives for the specialized convolution/tanh/gated-FiLM/residual graph.
    pub fn backward(&self, tape: &Tape, dy: &[f32]) -> Vec<f32> {
        let b = self.architecture.block;
        let w = self.architecture.width;
        let n = dy.len();
        let frames = n / b;
        let gated = self.architecture.block_style == BlockStyle::GatedFilm;
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
        let mut time_proj_grads = vec![Vec::new(); self.time_proj.len()];
        for j in (0..self.layers.len()).rev() {
            let residual = j > 0 && j + 1 < self.layers.len();
            let layer = &self.layers[j];
            let scale = if residual {
                self.architecture.scale(j - 1)
            } else {
                1
            };
            if residual && gated {
                let block_idx = j - 1;
                let cf = frames.div_ceil(scale);
                let g_rep = if scale > 1 {
                    repeat_transpose(&dh, frames, w, scale)
                } else {
                    dh.clone()
                };
                let nl = &tape.nonlinear[j];
                let mut dz = vec![0f32; cf * 2 * w];
                for q in 0..cf {
                    for c in 0..w {
                        let g = 0.5 * g_rep[q * w + c];
                        let ta = nl[q * 2 * w + c];
                        let sb = nl[q * 2 * w + w + c];
                        dz[q * 2 * w + c] = g * sb * (1. - ta * ta);
                        dz[q * 2 * w + w + c] = g * ta * sb * (1. - sb);
                    }
                }
                let conv_in = if scale > 1 {
                    pool(&tape.normed[block_idx], frames, w, scale)
                } else {
                    tape.normed[block_idx].clone()
                };
                let (d_conv_in, dw, db) = layer.backward(&conv_in, &dz, cf);
                let du = if scale > 1 {
                    pool_transpose(&d_conv_in, frames, w, scale)
                } else {
                    d_conv_in
                };
                let h_in = &tape.acts[j - 1];
                let inv_rms = &tape.inv_rms[block_idx];
                let film = &tape.films[block_idx];
                let mut d_film = vec![0f32; 2 * w];
                let mut dx = vec![0f32; frames * w];
                for p in 0..frames {
                    let r = inv_rms[p];
                    let mut dot = 0f32;
                    for c in 0..w {
                        let xh = h_in[p * w + c] * r;
                        let duv = du[p * w + c];
                        let gt = (1. + film[c]) * duv;
                        dot += gt * xh;
                        d_film[c] += duv * xh;
                        d_film[w + c] += duv;
                    }
                    dot /= w as f32;
                    for c in 0..w {
                        let xh = h_in[p * w + c] * r;
                        let gt = (1. + film[c]) * du[p * w + c];
                        dx[p * w + c] = r * (gt - xh * dot) + dh[p * w + c];
                    }
                }
                let (_, dw_tp, db_tp) =
                    self.time_proj[block_idx].backward(&tape.time_embed, &d_film, 1);
                time_proj_grads[block_idx] = dw_tp.into_iter().chain(db_tp).collect();
                grads[j] = dw.into_iter().chain(db).collect();
                dh = dx;
                continue;
            }
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
        let mut out: Vec<f32> = grads.into_iter().chain(time_proj_grads).flatten().collect();
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
        check(&self.time_proj, &reference.time_proj, "time_proj")?;
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
