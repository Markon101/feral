use crate::{
    audio::{self, Preprocess},
    backend::{Batch, Trainer},
    conditioning,
    data::{self, Dataset, DatasetIdentity},
    flow::{self, Sampling},
    model::{Architecture, Model},
    parameterization::{
        self, ConfiguredVelocity, FlowConfig, HorizonSchedule, LrSchedule, TimeSampling,
    },
    rng::Rng,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {} (will not overwrite)", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}
/// Checkpoints are large numeric payloads that nobody reads by hand; pretty-printing them costs
/// about 39 % more bytes and the matching write time on every save. Artifacts keep the readable
/// writer. Both forms parse identically, so older checkpoints stay loadable.
pub fn write_json_compact<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {} (will not overwrite)", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}
fn git(args: &[&str]) -> String {
    Command::new("git")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "uncommitted".into())
}
pub fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Adam {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub step: usize,
    pub lr: f32,
}
impl Adam {
    pub fn new(n: usize, lr: f32) -> Self {
        Self {
            m: vec![0.; n],
            v: vec![0.; n],
            step: 0,
            lr,
        }
    }
    pub fn update(&mut self, model: &mut Model, grad: &[f32]) -> Result<f32> {
        let lr = self.lr;
        self.update_with_lr(model, grad, lr)
    }
    pub fn update_with_lr(&mut self, model: &mut Model, grad: &[f32], lr: f32) -> Result<f32> {
        self.update_with_lr_wd(model, grad, lr, 0., None)
    }
    pub fn update_with_lr_wd(
        &mut self,
        model: &mut Model,
        grad: &[f32],
        lr: f32,
        weight_decay: f32,
        decay_mask: Option<&[f32]>,
    ) -> Result<f32> {
        ensure!(
            grad.len() == self.m.len() && grad.iter().all(|g| g.is_finite()),
            "nonfinite or invalid gradients"
        );
        ensure!(lr.is_finite() && lr >= 0., "invalid step learning rate");
        ensure!(
            weight_decay.is_finite() && weight_decay >= 0.,
            "invalid weight decay"
        );
        if let Some(mask) = decay_mask {
            ensure!(mask.len() == self.m.len(), "invalid weight decay mask");
        }
        let norm = grad.iter().map(|g| (*g as f64).powi(2)).sum::<f64>().sqrt() as f32;
        let scale = 1. / norm.max(1.);
        self.step += 1;
        let c1 = 1. - 0.9f32.powf(self.step as f32);
        let c2 = 1. - 0.999f32.powf(self.step as f32);
        let mut p = model.parameters();
        for i in 0..p.len() {
            let g = grad[i] * scale;
            self.m[i] = 0.9 * self.m[i] + 0.1 * g;
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g;
            if weight_decay > 0. {
                p[i] *= 1. - lr * weight_decay * decay_mask.map_or(1., |mask| mask[i]);
            }
            p[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
        ensure!(
            p.iter().all(|v| v.is_finite()),
            "optimizer produced nonfinite parameters"
        );
        model.set_parameters(&p);
        Ok(norm)
    }
}
fn is_zero_f32(v: &f32) -> bool {
    *v == 0.
}
fn is_zero_usize(v: &usize) -> bool {
    *v == 0
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub format_version: u32,
    #[serde(default, skip_serializing_if = "FlowConfig::is_legacy")]
    pub flow: FlowConfig,
    /// Weight of the detail term the model was trained with. Checkpointed, because it is part of
    /// the objective the weights belong to; older checkpoints default to 0 (the plain objective).
    #[serde(default)]
    pub detail: f32,
    /// Weight of the multiresolution STFT log-magnitude L1 endpoint loss; 0 disables it.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub spectral_weight: f32,
    /// Flow-time distribution used during training; older checkpoints default to Uniform.
    #[serde(default, skip_serializing_if = "TimeSampling::is_uniform")]
    pub time_sampling: TimeSampling,
    /// Optimizer learning-rate schedule; older checkpoints default to Constant.
    #[serde(default, skip_serializing_if = "LrSchedule::is_constant")]
    pub lr_schedule: LrSchedule,
    /// Exponential moving average decay for inference weights; 0 disables EMA.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub ema_decay: f32,
    /// Decoupled AdamW weight decay on 2D convolution/projection weights; 0 disables it.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub weight_decay: f32,
    /// Multi-scale crop lengths in samples; empty means fixed legacy `crop_samples`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub horizon_samples: Vec<usize>,
    /// Multi-horizon crop schedule; older checkpoints default to Fixed.
    #[serde(default, skip_serializing_if = "HorizonSchedule::is_fixed")]
    pub horizon_schedule: HorizonSchedule,
    /// Total schedule steps for ordered multi-horizon schedules; 0 means N/A.
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub horizon_total_steps: usize,
    /// Checkpointed EMA parameter shadow (empty when `ema_decay == 0`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ema_weights: Vec<f32>,
    pub model: Model,
    pub adam: Adam,
    pub rng: Rng,
    pub completed_steps: usize,
    pub sample_rate: u32,
    pub crop_samples: usize,
    pub dataset: DatasetIdentity,
    pub initial_seed: u64,
    pub git_commit: String,
    pub git_dirty: bool,
    pub source_hash: String,
    pub objective: String,
    pub representation: String,
    pub model_hash: String,
    pub unix_seconds: u64,
}
pub fn source_hash() -> Result<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths = vec![PathBuf::from("Cargo.toml"), PathBuf::from("Cargo.lock")];
    fn sources(root: &Path, dir: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let p = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                sources(root, &p, paths)?;
            } else if kind.is_file() {
                paths.push(p.strip_prefix(root)?.to_path_buf());
            }
        }
        Ok(())
    }
    sources(root, &root.join("src"), &mut paths)?;
    paths.sort();
    let mut h = Sha256::new();
    for p in paths {
        h.update(p.to_string_lossy().as_bytes());
        h.update(fs::read(root.join(p))?);
    }
    Ok(format!("{:x}", h.finalize()))
}
/// Guard against absurd or untrusted files rather than a model-size limit.
///
/// Legacy checkpoints <= 4M parameters write JSON (~13.5 bytes per float); larger checkpoints
/// automatically use single-file `FERALCK1` binary format (~16 bytes per parameter for
/// weights + m + v + EMA, so 24.4M parameters is ~390 MB).
pub const MAX_CHECKPOINT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const BINARY_CHECKPOINT_MAGIC: &[u8; 8] = b"FERALCK1";
pub const BINARY_CHECKPOINT_THRESHOLD: usize = 4_000_000;
const MAX_BINARY_HEADER_BYTES: usize = 16 * 1024 * 1024;
const MAX_BINARY_FLOATS: usize = 256_000_000;

#[derive(Serialize, Deserialize)]
struct BinaryCheckpointHeader {
    metadata: Checkpoint,
    param_count: usize,
    has_ema: bool,
    tensors_sha256: String,
}

fn binary_model_hash(architecture: &Architecture, params: &[f32]) -> Result<String> {
    let mut h = Sha256::new();
    h.update(BINARY_CHECKPOINT_MAGIC);
    h.update(serde_json::to_vec(architecture)?);
    for &v in params {
        h.update(v.to_le_bytes());
    }
    Ok(format!("{:x}", h.finalize()))
}

fn hash_tensor_bytes(h: &mut Sha256, data: &[f32]) {
    for &v in data {
        h.update(v.to_le_bytes());
    }
}

fn tensors_hash(params: &[f32], m: &[f32], v: &[f32], ema: &[f32]) -> String {
    let mut h = Sha256::new();
    hash_tensor_bytes(&mut h, params);
    hash_tensor_bytes(&mut h, m);
    hash_tensor_bytes(&mut h, v);
    hash_tensor_bytes(&mut h, ema);
    format!("{:x}", h.finalize())
}

fn take_layer_vecs(model: &mut Model) -> Vec<(Vec<f32>, Vec<f32>)> {
    model
        .layers
        .iter_mut()
        .chain(&mut model.time_proj)
        .chain(&mut model.stem)
        .chain(&mut model.head)
        .map(|l| (std::mem::take(&mut l.w), std::mem::take(&mut l.b)))
        .collect()
}

fn restore_layer_vecs(model: &mut Model, vecs: Vec<(Vec<f32>, Vec<f32>)>) {
    for (l, (w, b)) in model
        .layers
        .iter_mut()
        .chain(&mut model.time_proj)
        .chain(&mut model.stem)
        .chain(&mut model.head)
        .zip(vecs)
    {
        l.w = w;
        l.b = b;
    }
}

fn write_f32_slice<W: Write>(writer: &mut W, data: &[f32]) -> Result<()> {
    let mut buf = [0u8; 4096];
    for chunk in data.chunks(1024) {
        let bytes = chunk.len() * 4;
        let (slots, _) = buf.as_chunks_mut::<4>();
        for (&val, slot) in chunk.iter().zip(slots) {
            *slot = val.to_le_bytes();
        }
        writer.write_all(&buf[..bytes])?;
    }
    Ok(())
}

fn read_f32_vec<R: Read>(reader: &mut R, len: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; len];
    let mut buf = [0u8; 4096];
    let mut pos = 0;
    while pos < len {
        let take = (len - pos).min(1024);
        reader.read_exact(&mut buf[..take * 4])?;
        let (chunks, _) = buf[..take * 4].as_chunks::<4>();
        for (dst, chunk) in out[pos..pos + take].iter_mut().zip(chunks) {
            *dst = f32::from_le_bytes(*chunk);
        }
        pos += take;
    }
    Ok(out)
}

impl Checkpoint {
    pub fn velocity(&self) -> ConfiguredVelocity<'_> {
        ConfiguredVelocity {
            model: &self.model,
            config: &self.flow,
        }
    }
    pub fn channels(&self) -> usize {
        self.model.architecture.channels
    }
    pub fn has_ema(&self) -> bool {
        !self.ema_weights.is_empty()
    }
    /// Returns the model to use for inference, previews, and validation: EMA weights when
    /// `use_ema` is true and EMA weights are present, otherwise raw optimizer weights.
    pub fn sampling_model(&self, use_ema: bool) -> Model {
        let mut m = self.model.clone();
        if use_ema && !self.ema_weights.is_empty() {
            m.set_parameters(&self.ema_weights);
        }
        m
    }
    pub fn save(&mut self, path: &Path) -> Result<()> {
        if self.model.count() > BINARY_CHECKPOINT_THRESHOLD {
            return self.save_binary(path);
        }
        self.model_hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&self.model)?));
        write_json_compact(path, self)
    }
    /// Save checkpoint in single-file `FERALCK1` binary format (compact JSON header + raw LE f32
    /// payloads for weights, Adam `m`, Adam `v`, and optional EMA weights).
    pub fn save_binary(&mut self, path: &Path) -> Result<()> {
        let params = self.model.parameters();
        let param_count = params.len();
        ensure!(
            self.adam.m.len() == param_count && self.adam.v.len() == param_count,
            "optimizer state length mismatch while saving binary checkpoint"
        );
        self.model_hash = binary_model_hash(&self.model.architecture, &params)?;
        let tensors_sha256 = tensors_hash(&params, &self.adam.m, &self.adam.v, &self.ema_weights);
        let has_ema = !self.ema_weights.is_empty();

        // Strip large f32 vectors into locals before cloning metadata so serialization uses KB
        // instead of duplicating multi-hundred-MB weight/optimizer arrays in JSON.
        let saved_layers = take_layer_vecs(&mut self.model);
        let saved_m = std::mem::take(&mut self.adam.m);
        let saved_v = std::mem::take(&mut self.adam.v);
        let saved_ema = std::mem::take(&mut self.ema_weights);
        let metadata = self.clone();
        restore_layer_vecs(&mut self.model, saved_layers);
        self.adam.m = saved_m;
        self.adam.v = saved_v;
        self.ema_weights = saved_ema;

        let header = BinaryCheckpointHeader {
            metadata,
            param_count,
            has_ema,
            tensors_sha256,
        };
        let header_json = serde_json::to_vec(&header)?;
        ensure!(
            header_json.len() <= MAX_BINARY_HEADER_BYTES,
            "binary checkpoint header exceeds {} MB",
            MAX_BINARY_HEADER_BYTES >> 20
        );
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create {} (will not overwrite)", path.display()))?;
        let mut writer = BufWriter::new(file);
        writer.write_all(BINARY_CHECKPOINT_MAGIC)?;
        writer.write_all(&(header_json.len() as u64).to_le_bytes())?;
        writer.write_all(&header_json)?;
        write_f32_slice(&mut writer, &params)?;
        write_f32_slice(&mut writer, &self.adam.m)?;
        write_f32_slice(&mut writer, &self.adam.v)?;
        if has_ema {
            write_f32_slice(&mut writer, &self.ema_weights)?;
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        Ok(())
    }
    fn load_binary<R: Read>(mut reader: R) -> Result<Self> {
        let mut len_buf = [0u8; 8];
        reader.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        ensure!(
            (2..=MAX_BINARY_HEADER_BYTES).contains(&header_len),
            "invalid binary checkpoint header size"
        );
        let mut header_bytes = vec![0u8; header_len];
        reader.read_exact(&mut header_bytes)?;
        let header: BinaryCheckpointHeader = serde_json::from_slice(&header_bytes)?;
        ensure!(
            header.param_count > 0 && header.param_count <= MAX_BINARY_FLOATS,
            "invalid binary checkpoint parameter count"
        );
        let mut c = header.metadata;
        c.model.architecture.validate()?;
        let mut expected = 0usize;
        for l in c
            .model
            .layers
            .iter_mut()
            .chain(&mut c.model.time_proj)
            .chain(&mut c.model.stem)
            .chain(&mut c.model.head)
        {
            let w_len = l
                .input
                .checked_mul(l.output)
                .and_then(|v| v.checked_mul(l.kernel))
                .context("binary layer dimension overflow")?;
            let layer_total = w_len
                .checked_add(l.output)
                .context("binary layer count overflow")?;
            expected = expected
                .checked_add(layer_total)
                .context("binary model count overflow")?;
            ensure!(
                expected <= MAX_BINARY_FLOATS,
                "binary layer dimensions exceed limit"
            );
            l.w = vec![0f32; w_len];
            l.b = vec![0f32; l.output];
        }
        ensure!(
            expected == header.param_count,
            "binary checkpoint parameter count mismatch"
        );
        let params = read_f32_vec(&mut reader, expected)?;
        let m = read_f32_vec(&mut reader, expected)?;
        let v = read_f32_vec(&mut reader, expected)?;
        let ema = if header.has_ema {
            read_f32_vec(&mut reader, expected)?
        } else {
            vec![]
        };
        let mut extra = [0u8; 1];
        ensure!(
            reader.read(&mut extra)? == 0,
            "unexpected trailing bytes in binary checkpoint"
        );
        ensure!(
            tensors_hash(&params, &m, &v, &ema) == header.tensors_sha256,
            "binary checkpoint payload checksum mismatch"
        );
        ensure!(
            binary_model_hash(&c.model.architecture, &params)? == c.model_hash,
            "binary checkpoint model hash mismatch"
        );
        c.model.set_parameters(&params);
        c.adam.m = m;
        c.adam.v = v;
        c.ema_weights = ema;
        c.validate(false)?;
        Ok(c)
    }
    pub fn load(path: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(path)?.len() <= MAX_CHECKPOINT_BYTES,
            "checkpoint exceeds {} MB",
            MAX_CHECKPOINT_BYTES >> 20
        );
        let mut file = fs::File::open(path)?;
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_ok() && &magic == BINARY_CHECKPOINT_MAGIC {
            return Self::load_binary(BufReader::new(file));
        }
        file.rewind()?;
        let c: Self = serde_json::from_reader(BufReader::new(file))?;
        c.validate(true)?;
        Ok(c)
    }
    fn validate(&self, verify_json_hash: bool) -> Result<()> {
        let c = self;
        ensure!(
            [1, 2, 3].contains(&c.format_version),
            "unsupported checkpoint version"
        );
        c.flow.validate()?;
        ensure!(
            c.detail.is_finite() && (0. ..=100.).contains(&c.detail),
            "invalid detail weight"
        );
        ensure!(
            c.spectral_weight.is_finite() && (0. ..=100.).contains(&c.spectral_weight),
            "invalid spectral weight"
        );
        ensure!(
            c.weight_decay.is_finite() && (0. ..=10.).contains(&c.weight_decay),
            "invalid weight decay"
        );
        c.time_sampling.validate()?;
        c.lr_schedule.validate(c.adam.lr)?;
        ensure!(
            c.ema_decay.is_finite() && c.ema_decay >= 0. && c.ema_decay < 1.,
            "invalid EMA decay"
        );
        ensure!(
            c.format_version == c.flow.format_version(),
            "checkpoint format/flow configuration mismatch"
        );
        c.model.validate()?;
        let n = c.model.count();
        ensure!(
            c.adam.m.len() == n
                && c.adam.v.len() == n
                && c.adam.m.iter().chain(&c.adam.v).all(|v| v.is_finite())
                && c.adam.v.iter().all(|&v| v >= 0.),
            "invalid optimizer state"
        );
        ensure!(
            if c.ema_decay > 0. {
                c.ema_weights.len() == n && c.ema_weights.iter().all(|v| v.is_finite())
            } else {
                c.ema_weights.is_empty()
                    || (c.ema_weights.len() == n && c.ema_weights.iter().all(|v| v.is_finite()))
            },
            "invalid EMA weights in checkpoint"
        );
        ensure!(
            c.adam.lr.is_finite()
                && c.adam.lr > 0.
                && c.adam.lr <= 0.01
                && c.adam.step == c.completed_steps,
            "invalid optimizer step/lr"
        );
        ensure!(
            (8000..=48000).contains(&c.sample_rate)
                && c.crop_samples >= 32
                && c.crop_samples <= crate::model::MAX_WINDOW_SAMPLES
                && c.crop_samples.is_multiple_of(c.model.architecture.block),
            "invalid checkpoint audio configuration"
        );
        ensure!(
            c.horizon_samples.len() <= 16
                && c.horizon_samples.iter().all(|&s| {
                    (32..=crate::model::MAX_WINDOW_SAMPLES).contains(&s)
                        && s.is_multiple_of(c.model.architecture.block)
                }),
            "invalid checkpoint crop horizon samples"
        );
        ensure!(
            c.horizon_schedule.is_fixed()
                || (!c.horizon_samples.is_empty() && c.horizon_total_steps > 0),
            "invalid checkpoint horizon schedule configuration"
        );
        if verify_json_hash {
            ensure!(
                format!("{:x}", Sha256::digest(serde_json::to_vec(&c.model)?)) == c.model_hash,
                "checkpoint model hash mismatch"
            );
        }
        Ok(())
    }
}
#[derive(Clone)]
pub struct TrainConfig {
    pub input: PathBuf,
    pub run: PathBuf,
    pub backend: String,
    pub steps: usize,
    pub seed: u64,
    pub crop_seconds: f32,
    pub preprocess: Preprocess,
    pub lr: f32,
    pub preview_every: usize,
    pub checkpoint_every: usize,
    pub resume: Option<PathBuf>,
    pub weights_only: bool,
    /// Allow a full resume at a different crop length. The network is fully convolutional, so the
    /// weights transfer; the data stream and the RNG trajectory do not continue bit-exactly.
    pub grow_crop: bool,
    pub architecture: Architecture,
    pub flow: FlowConfig,
    /// Weight of the detail term, `<r,r> + detail * <dr,dr>`, in the objective.
    pub detail: f32,
    /// Weight of the multiresolution STFT log-magnitude L1 endpoint loss; 0 disables it.
    pub spectral_weight: f32,
    /// Training flow-time distribution (`Uniform` or `LogitNormal`).
    pub time_sampling: TimeSampling,
    /// Optimizer learning-rate schedule (`Constant` or `Cosine`).
    pub lr_schedule: LrSchedule,
    /// Exponential moving average decay for inference weights; 0 disables EMA.
    pub ema_decay: f32,
    /// Decoupled AdamW weight decay on 2D weights / conv kernels (`0.` disables).
    pub weight_decay: f32,
    /// Multi-scale crop horizons in seconds; empty preserves fixed `crop_seconds`.
    pub crop_horizons: Vec<f32>,
    /// Multi-scale crop horizon scheduling across training steps.
    pub horizon_schedule: HorizonSchedule,
    /// Total schedule steps for ordered multi-horizon schedules; `None` uses `steps`.
    pub horizon_steps: Option<usize>,
    /// Crops per optimizer step. The loss and gradients are averaged over the batch.
    pub batch: usize,
    /// Optional held-out corpus for periodic validation. Unset means train-set probes only.
    pub validation: Option<PathBuf>,
    /// Validation interval in steps; every interval also writes a validation.jsonl row.
    pub validation_every: usize,
}
fn evaluation(
    field: &dyn crate::model::Velocity,
    flow: &FlowConfig,
    dataset: &Dataset,
    n: usize,
) -> f32 {
    let ch = field.channels();
    let mut rng = Rng::new(901);
    let mut sum = 0.;
    for _ in 0..4 {
        let clean = dataset.crop(n, &mut rng);
        if clean.is_empty() {
            return f32::NAN;
        }
        let (mask, _) = conditioning::training_mask_channels(n, ch, &mut rng);
        let noise: Vec<f32> = (0..clean.len())
            .map(|_| rng.normal() * flow.prior_std())
            .collect();
        sum += parameterization::loss(field, flow, &clean, &mask, &noise, 0.5);
    }
    sum / 4.
}
fn make_cuda_velocity(
    ck: &Checkpoint,
    n: usize,
) -> Result<crate::sampler::DeviceVelocity<'static>> {
    let model = ck.sampling_model(true);
    crate::sampler::DeviceVelocity::from_cuda(&model, ck.adam.lr, &ck.flow, n)
}
/// Evaluates the fixed probes against the checkpoint's *current* sampling weights (EMA when
/// enabled), reusing a cached CUDA device field when supplied so periodic validation does not
/// rebuild the context each interval.
fn evaluate_checkpoint(
    ck: &Checkpoint,
    backend: &str,
    dataset: &Dataset,
    n: usize,
    cached: Option<&crate::sampler::DeviceVelocity<'_>>,
) -> Result<f32> {
    let model = ck.sampling_model(true);
    match backend {
        "cuda" => {
            let field = cached.context("missing cached CUDA evaluation field")?;
            field.load_weights(&model)?;
            Ok(evaluation(field, &ck.flow, dataset, n))
        }
        _ => {
            let field = ConfiguredVelocity {
                model: &model,
                config: &ck.flow,
            };
            Ok(evaluation(&field, &ck.flow, dataset, n))
        }
    }
}
/// Previews and the final take use the same backend as training, reusing a cached CUDA field so
/// only updated sampling weights (EMA when enabled) upload per preview.
fn preview_sample(
    ck: &Checkpoint,
    c: &TrainConfig,
    sampling: &Sampling,
    n: usize,
    cached: Option<&crate::sampler::DeviceVelocity<'_>>,
) -> Result<Vec<f32>> {
    let zeros = vec![0.; n * ck.channels()];
    let model = ck.sampling_model(true);
    match c.backend.as_str() {
        "cuda" => {
            let field = cached.context("missing cached CUDA preview field")?;
            field.load_weights(&model)?;
            flow::sample(field, &zeros, &zeros, sampling)
        }
        // cpu, and opencl which has no sampled field of its own: the reference path.
        _ => {
            let field = ConfiguredVelocity {
                model: &model,
                config: &ck.flow,
            };
            flow::sample(&field, &zeros, &zeros, sampling)
        }
    }
}
/// Fixed probe and held-out validation window, in seconds of audio.
///
/// Pinned by duration rather than by sample count so `validation_loss` means the same window at
/// every sample rate: 64,000 samples is 4 s at 16 kHz but 1.33 s at 48 kHz, which made runs at
/// different rates incomparable. The CPU reference costs parameters x window, so 4 s stays the
/// cheap end for probes; for a 6.5 M model that is still minutes per pass on the CPU.
pub const PROBE_WINDOW_SECONDS: f32 = 4.0;

/// The probe/validation window for a checkpoint at `sample_rate`, clamped to the training crop.
///
/// The field needs whole frames, so the duration is rounded down to a multiple of `block` — by
/// less than one frame, and never above 4 s. The CLI's frame sizes (16, 32, 64, 128) divide 4 s at
/// 16, 24 and 48 kHz exactly, and at 16 kHz the result is the old `min(crop, 64_000)`, so existing
/// runs keep their numbers bit for bit.
pub fn probe_window_samples(sample_rate: u32, crop_samples: usize, block: usize) -> usize {
    let window = (PROBE_WINDOW_SECONDS as f64 * sample_rate as f64).round() as usize;
    crop_samples.min(window - window % block)
}

pub fn train(c: TrainConfig) -> Result<PathBuf> {
    ensure!(
        (1..=10_000_000).contains(&c.steps),
        "training steps must be 1..10000000"
    );
    // Crops beyond 1 s were measured to cost the same per sample on the resident backends, so the
    // bound is a memory/context guard rather than a speed limit: 4 s at 16 kHz is 64k samples per
    // crop, which still fits the phone and desktop resident budgets at the documented batch sizes.
    ensure!(
        c.crop_seconds.is_finite()
            && (0.032..=32.).contains(&c.crop_seconds)
            && (c.crop_seconds * c.preprocess.sample_rate as f32) as usize
                <= crate::model::MAX_WINDOW_SAMPLES,
        "crop seconds must be 0.032..32 and fit the maximum window"
    );
    ensure!(
        c.lr.is_finite() && c.lr > 0. && c.lr <= 0.01,
        "learning rate must be >0 and <=0.01"
    );
    ensure!(
        c.detail.is_finite() && (0. ..=100.).contains(&c.detail),
        "detail weight must be finite and 0..100"
    );
    ensure!(
        c.spectral_weight.is_finite() && (0. ..=100.).contains(&c.spectral_weight),
        "spectral weight must be finite and 0..100"
    );
    c.time_sampling.validate()?;
    c.lr_schedule.validate(c.lr)?;
    ensure!(
        c.ema_decay.is_finite() && c.ema_decay >= 0. && c.ema_decay < 1.,
        "EMA decay must be finite and in [0, 1)"
    );
    ensure!(
        c.weight_decay.is_finite() && (0. ..=10.).contains(&c.weight_decay),
        "weight decay must be finite and 0..10"
    );
    ensure!(
        c.horizon_schedule.is_fixed() || !c.crop_horizons.is_empty(),
        "multi-horizon schedule requires --crop-horizons"
    );
    ensure!(
        !(c.horizon_schedule.is_fixed() && c.crop_horizons.len() > 1),
        "multiple --crop-horizons require --horizon-schedule random|short-to-long|long-to-short"
    );
    ensure!(
        c.crop_horizons.len() <= 16,
        "at most 16 crop horizons are supported"
    );
    c.architecture.validate()?;
    c.flow.validate()?;
    c.preprocess.validate()?;
    ensure!(
        c.architecture.channels == c.preprocess.channels(),
        "architecture channel count ({}) does not match preprocessing channel mode ({})",
        c.architecture.channels,
        c.preprocess.channel
    );
    ensure!(
        c.preview_every > 0 && c.checkpoint_every > 0 && c.validation_every > 0,
        "intervals must be positive"
    );
    ensure!((1..=64).contains(&c.batch), "batch size must be 1..64");
    let b = c.architecture.block;
    let round_crop = |secs: f32| -> Result<usize> {
        ensure!(
            secs.is_finite()
                && (0.032..=32.).contains(&secs)
                && (secs * c.preprocess.sample_rate as f32) as usize
                    <= crate::model::MAX_WINDOW_SAMPLES,
            "crop seconds ({secs}) must be 0.032..32 and fit the maximum window"
        );
        let rounded = ((secs * c.preprocess.sample_rate as f32).round() as usize).div_ceil(b) * b;
        ensure!(
            rounded <= crate::model::MAX_WINDOW_SAMPLES,
            "rounded crop ({rounded} samples) exceeds maximum window ({})",
            crate::model::MAX_WINDOW_SAMPLES
        );
        Ok(rounded)
    };
    let n = round_crop(c.crop_seconds)?;
    let mut horizon_samples = Vec::with_capacity(c.crop_horizons.len());
    for &secs in &c.crop_horizons {
        horizon_samples.push(round_crop(secs)?);
    }
    horizon_samples.sort_unstable();
    horizon_samples.dedup();
    let horizon_total_steps = if c.horizon_schedule.is_fixed() {
        0
    } else {
        let hs = c.horizon_steps.unwrap_or(c.steps);
        ensure!(hs > 0, "horizon steps must be positive");
        hs
    };
    let samples_budget = n * c.batch;
    let horizon_pairs: Vec<(usize, usize)> = if horizon_samples.is_empty() {
        vec![(n, c.batch)]
    } else {
        horizon_samples
            .iter()
            .map(|&hn| (hn, (samples_budget / hn).clamp(1, 64)))
            .collect()
    };
    let max_n = horizon_pairs.iter().map(|&(hn, _)| hn).max().unwrap_or(n);
    let pre_start = Instant::now();
    let dataset = Dataset::load(&c.input, c.preprocess.clone(), max_n)?;
    // Fixed probes and validation always use at most a 4 s window, whatever the training crop is.
    // Two reasons: the CPU reference evaluation costs parameters x window, which at a 16 s crop is
    // minutes per probe, and a constant window keeps `validation_loss` comparable across runs that
    // train at different crop lengths. It is a duration rather than a sample count so the window
    // covers the same span of audio at every rate; 64,000 samples was 4 s only at 16 kHz.
    let eval_n = probe_window_samples(c.preprocess.sample_rate, max_n, b);
    let validation = match &c.validation {
        Some(path) => Some(Dataset::load(path, c.preprocess.clone(), eval_n)?),
        None => None,
    };
    let preprocessing_ms = pre_start.elapsed().as_secs_f64() * 1000.;
    let mut ck = if let Some(path) = &c.resume {
        let mut ck = Checkpoint::load(path)?;
        ensure!(
            ck.flow == c.flow,
            "resume parameterization mismatch; train a fresh model for a different flow"
        );
        ensure!(
            ck.sample_rate == c.preprocess.sample_rate,
            "resume sample-rate mismatch"
        );
        if c.grow_crop {
            // Curriculum step: the same weights continue at a longer (or shorter) crop. The
            // optimizer, step counter and RNG state carry over, but the data stream does not
            // continue bit-exactly because crop offsets change with the window length.
            ck.crop_samples = n;
        } else {
            ensure!(
                ck.crop_samples == n,
                "resume crop mismatch ({} vs {} samples); pass --grow-crop to continue at a new crop length",
                ck.crop_samples,
                n
            );
        }
        ensure!(
            serde_json::to_string(&ck.model.architecture)?
                == serde_json::to_string(&c.architecture)?,
            "resume architecture mismatch"
        );
        if c.weights_only {
            ck.adam = Adam::new(ck.model.count(), c.lr);
            ck.rng = Rng::new(c.seed);
            ck.completed_steps = 0;
            ck.dataset = dataset.identity.clone();
            ck.initial_seed = c.seed;
            ck.detail = c.detail;
            ck.spectral_weight = c.spectral_weight;
            ck.time_sampling = c.time_sampling;
            ck.lr_schedule = c.lr_schedule;
            ck.ema_decay = c.ema_decay;
            ck.weight_decay = c.weight_decay;
            ck.horizon_samples = horizon_samples.clone();
            ck.horizon_schedule = c.horizon_schedule;
            ck.horizon_total_steps = horizon_total_steps;
            ck.ema_weights = if c.ema_decay > 0. {
                ck.model.parameters()
            } else {
                vec![]
            };
        } else {
            ensure!(
                ck.dataset.manifest_hash == dataset.identity.manifest_hash,
                "full resume requires identical dataset/preprocessing"
            );
            ensure!(
                ck.adam.lr == c.lr,
                "full resume requires same learning rate"
            );
            ensure!(
                ck.detail == c.detail,
                "full resume requires the same detail weight ({} vs {}); it is part of the objective",
                ck.detail,
                c.detail
            );
            ensure!(
                ck.spectral_weight == c.spectral_weight,
                "full resume requires the same spectral weight ({} vs {}); it is part of the objective",
                ck.spectral_weight,
                c.spectral_weight
            );
            ensure!(
                ck.time_sampling == c.time_sampling,
                "full resume requires the same time sampling configuration"
            );
            ensure!(
                ck.lr_schedule == c.lr_schedule,
                "full resume requires the same lr schedule configuration"
            );
            ensure!(
                ck.ema_decay == c.ema_decay,
                "full resume requires the same EMA decay ({} vs {})",
                ck.ema_decay,
                c.ema_decay
            );
            ensure!(
                ck.weight_decay == c.weight_decay,
                "full resume requires the same weight decay ({} vs {})",
                ck.weight_decay,
                c.weight_decay
            );
            ensure!(
                ck.horizon_samples == horizon_samples
                    && ck.horizon_schedule == c.horizon_schedule
                    && ck.horizon_total_steps == horizon_total_steps,
                "full resume requires the same crop horizon schedule"
            );
        }
        ck
    } else {
        let model = Model::new(c.architecture.clone(), c.seed);
        let adam = Adam::new(model.count(), c.lr);
        let ema_weights = if c.ema_decay > 0. {
            model.parameters()
        } else {
            vec![]
        };
        Checkpoint {
            format_version: c.flow.format_version(),
            flow: c.flow.clone(),
            detail: c.detail,
            spectral_weight: c.spectral_weight,
            time_sampling: c.time_sampling,
            lr_schedule: c.lr_schedule,
            ema_decay: c.ema_decay,
            weight_decay: c.weight_decay,
            horizon_samples: horizon_samples.clone(),
            horizon_schedule: c.horizon_schedule,
            horizon_total_steps,
            ema_weights,
            model,
            adam,
            rng: Rng::new(c.seed ^ 0x464552414c),
            completed_steps: 0,
            sample_rate: c.preprocess.sample_rate,
            crop_samples: n,
            dataset: dataset.identity.clone(),
            initial_seed: c.seed,
            git_commit: git(&["rev-parse", "HEAD"]),
            git_dirty: !git(&["status", "--porcelain"]).is_empty(),
            source_hash: source_hash()?,
            objective: {
                let mut desc = if c.flow.is_legacy() {
                    "conditional rectified flow, unknown-sample MSE, N(0,1) noise".to_string()
                } else {
                    "conditional rectified flow, Gaussian residual preconditioning, residual MSE (=velocity MSE/out^2), checkpointed Gaussian prior".to_string()
                };
                match c.time_sampling {
                    TimeSampling::Uniform => desc.push_str(", uniform t"),
                    TimeSampling::LogitNormal { mean, std } => {
                        desc.push_str(&format!(", logit-normal t(mean={mean}, std={std})"));
                    }
                }
                if c.detail > 0. {
                    desc.push_str(&format!(
                        " + {} x mean squared first difference of the residual",
                        c.detail
                    ));
                }
                if c.spectral_weight > 0. {
                    desc.push_str(&format!(
                        " + {} x t-weighted multiresolution STFT log-magnitude L1 on y_hat",
                        c.spectral_weight
                    ));
                }
                desc
            },
            representation: if c.architecture.channels == 2 {
                "f32 interleaved stereo waveform [L,R]; lossless time-major blocks".into()
            } else {
                "f32 mono waveform; lossless time-major blocks".into()
            },
            model_hash: String::new(),
            unix_seconds: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        }
    };
    let mut trainer =
        Trainer::new_with_horizons(&c.backend, &ck.model, &ck.adam, &horizon_pairs, &ck.flow)?;
    trainer.configure_ema(ck.ema_decay, &ck.ema_weights)?;
    if let Some(parent) = c.run.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&c.run).with_context(|| {
        format!(
            "new run directory {} must not exist; create its parent first",
            c.run.display()
        )
    })?;
    let current_source = source_hash()?;
    let ch = ck.channels();
    let mut run_json = serde_json::json!({"git_commit":git(&["rev-parse","HEAD"]),"git_dirty":!git(&["status","--porcelain"]).is_empty(),"source_hash":current_source,"dataset":dataset.identity,"parameters":ck.model.count(),"architecture":ck.model.architecture,"channels":ch,"crop_samples":n,"batch":c.batch,"samples_per_step":n*c.batch,"validation_input":c.validation,"sample_rate":ck.sample_rate,"additional_steps":c.steps,"seed":c.seed,"lr":ck.adam.lr,"lr_schedule":ck.lr_schedule,"time_sampling":ck.time_sampling,"ema_decay":ck.ema_decay,"spectral_weight":ck.spectral_weight,"optimizer":"AdamW beta1=.9 beta2=.999 eps=1e-8 global gradient clip=1","resume":c.resume,"weights_only":c.weights_only,"backend":trainer.name(),"backend_info":trainer.info(),"objective":ck.objective,"representation":ck.representation,"flow":ck.flow,"detail":ck.detail});
    if let Some(obj) = run_json.as_object_mut() {
        obj.insert("weight_decay".into(), serde_json::json!(ck.weight_decay));
        obj.insert(
            "horizon_samples".into(),
            serde_json::json!(ck.horizon_samples),
        );
        obj.insert(
            "horizon_schedule".into(),
            serde_json::json!(ck.horizon_schedule),
        );
        obj.insert(
            "horizon_total_steps".into(),
            serde_json::json!(ck.horizon_total_steps),
        );
    }
    write_json(&c.run.join("run.json"), &run_json)?;
    ck.save(&c.run.join("initial.json"))?;
    let cached_eval = if c.backend == "cuda" {
        Some(make_cuda_velocity(&ck, eval_n)?)
    } else {
        None
    };
    let cached_preview = if c.backend == "cuda" && n != eval_n {
        Some(make_cuda_velocity(&ck, n)?)
    } else {
        None
    };
    let preview_field = cached_preview.as_ref().or(cached_eval.as_ref());
    let sample_config = Sampling {
        steps: 16,
        ..Default::default()
    };
    let mut sampling_ms = 0.;
    let start = Instant::now();
    let initial_audio = preview_sample(&ck, &c, &sample_config, n, preview_field)?;
    sampling_ms += start.elapsed().as_secs_f64() * 1000.;
    audio::write_wav_channels(
        &c.run.join("untrained-or-resumed-initial.wav"),
        &initial_audio,
        ck.sample_rate,
        ch,
    )?;
    write_json(&c.run.join("preview-config.json"), &sample_config)?;
    let initial_eval =
        evaluate_checkpoint(&ck, &c.backend, &dataset, eval_n, cached_eval.as_ref())?;
    ensure!(initial_eval.is_finite(), "invalid initial evaluation");
    // Validation uses the held-out corpus when supplied; the probe seed and t are fixed so
    // successive readings are comparable and are not drawn from the training crop stream.
    let val_loss = |ck: &Checkpoint| -> Result<f32> {
        match &validation {
            Some(v) => evaluate_checkpoint(ck, &c.backend, v, eval_n, cached_eval.as_ref()),
            None => evaluate_checkpoint(ck, &c.backend, &dataset, eval_n, cached_eval.as_ref()),
        }
    };
    let initial_val = val_loss(&ck)?;
    ensure!(initial_val.is_finite(), "invalid initial validation loss");
    let mut log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(c.run.join("losses.jsonl"))?;
    let mut val_log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(c.run.join("validation.jsonl"))?;
    let mut val_curve = vec![serde_json::json!({
        "step": ck.completed_steps,
        "split": if validation.is_some() { "validation" } else { "train-probe" },
        "validation_loss": initial_val,
        "train_probe_loss": initial_eval,
    })];
    writeln!(val_log, "{}", val_curve[0])?;
    val_log.flush()?;
    let mut neural_ms = 0.;
    let mut optimizer_ms = 0.;
    let mut step_ms = Vec::new();
    let mut task_counts = std::collections::BTreeMap::new();
    let mut source_counts = std::collections::BTreeMap::new();
    let mut silence_retries = 0usize;
    let training_start = Instant::now();
    eprintln!(
        "FERAL training: {} parameters, {}ch, {} frames/crop, {} files, preprocessing {:.1} ms",
        ck.model.count(),
        ch,
        n,
        dataset.audio.len(),
        preprocessing_ms
    );
    let max_items = horizon_pairs
        .iter()
        .map(|&(_, hi)| hi)
        .max()
        .unwrap_or(c.batch);
    let samples_cap = horizon_pairs
        .iter()
        .map(|&(hn, hi)| hn * hi * ch)
        .max()
        .unwrap_or(n * c.batch * ch);
    let debug_timing = std::env::var_os("FERAL_DEBUG_TIMING").is_some();
    let mut clean_batch: Vec<f32> = Vec::with_capacity(samples_cap);
    let mut mask_batch: Vec<f32> = Vec::with_capacity(samples_cap);
    let mut noise_batch: Vec<f32> = Vec::with_capacity(samples_cap);
    let mut t_batch: Vec<f32> = Vec::with_capacity(max_items);
    let mut batch_tasks: Vec<&'static str> = Vec::with_capacity(max_items);
    let mut batch_sources: Vec<usize> = Vec::with_capacity(max_items);
    let mut batch_starts: Vec<usize> = Vec::with_capacity(max_items);
    for local in 1..=c.steps {
        let start = Instant::now();
        let horizon_idx = if horizon_pairs.len() == 1 || ck.horizon_schedule.is_fixed() {
            0
        } else {
            ck.horizon_schedule.select_index(
                horizon_pairs.len(),
                ck.completed_steps + 1,
                ck.horizon_total_steps,
                &mut ck.rng,
            )
        };
        let (step_n, step_items) = horizon_pairs[horizon_idx];
        let step_scalars = step_n * ch;
        trainer.set_shape(step_n, step_items)?;
        clean_batch.clear();
        mask_batch.clear();
        noise_batch.clear();
        t_batch.clear();
        batch_tasks.clear();
        batch_sources.clear();
        batch_starts.clear();
        let mut t_crop = 0f64;
        let mut t_mask = 0f64;
        let mut t_noise = 0f64;
        for _ in 0..step_items {
            let mark = Instant::now();
            let crop = dataset
                .crop_traced_into(step_n, &mut ck.rng, &mut clean_batch)
                .context(
                    "16 crop attempts were silent; reduce --silence-rms or select useful audio",
                )?;
            *source_counts.entry(crop.source_index).or_insert(0usize) += 1;
            silence_retries += crop.silence_rejections;
            t_crop += mark.elapsed().as_secs_f64() * 1000.;
            let mark = Instant::now();
            let mask_start = mask_batch.len();
            mask_batch.resize(mask_start + step_scalars, 0.);
            let task = conditioning::training_mask_channels_into(
                &mut mask_batch[mask_start..mask_start + step_scalars],
                ch,
                &mut ck.rng,
            );
            *task_counts.entry(task).or_insert(0usize) += 1;
            t_mask += mark.elapsed().as_secs_f64() * 1000.;
            let mark = Instant::now();
            noise_batch.extend((0..step_scalars).map(|_| ck.rng.normal() * ck.flow.prior_std()));
            t_noise += mark.elapsed().as_secs_f64() * 1000.;
            t_batch.push(ck.time_sampling.sample(&mut ck.rng));
            batch_tasks.push(task);
            batch_sources.push(crop.source_index);
            batch_starts.push(crop.start_sample);
        }
        let prep_ms = start.elapsed().as_secs_f64() * 1000.;
        if debug_timing {
            eprintln!(
                "timing step {}: prep {prep_ms:.1} ms (crop {t_crop:.1} mask {t_mask:.1} noise {t_noise:.1})",
                ck.completed_steps + 1
            );
        }
        let task = batch_tasks[0];
        let t = t_batch[0];
        let step_lr = ck.lr_schedule.lr_at(ck.adam.lr, ck.completed_steps + 1);
        let result = trainer.step(
            &mut ck.model,
            &mut ck.adam,
            &ck.flow,
            Batch {
                clean: &clean_batch,
                mask: &mask_batch,
                noise: &noise_batch,
                t: &t_batch,
                detail: ck.detail,
                spectral: ck.spectral_weight,
                lr: step_lr,
                weight_decay: ck.weight_decay,
            },
        )?;
        if c.backend == "cpu" && ck.ema_decay > 0. {
            let decay = ck.ema_decay;
            let one_minus = 1. - decay;
            let params = ck.model.parameters();
            for (e, &w) in ck.ema_weights.iter_mut().zip(&params) {
                *e = decay * *e + one_minus * w;
            }
        }
        let loss = result.loss;
        let norm = result.norm;
        neural_ms += result.neural_ms;
        optimizer_ms += result.optimizer_ms;
        ensure!(loss.is_finite(), "nonfinite loss");
        ck.completed_steps += 1;
        let ms = start.elapsed().as_secs_f64() * 1000.;
        step_ms.push(ms);
        writeln!(
            log,
            "{}",
            serde_json::json!({"step":ck.completed_steps,"loss":loss,"gradient_norm":norm,"lr":step_lr,"task":task,"t":t,"milliseconds":ms,"rng_state":ck.rng.state,"crop_samples":step_n,"batch_items":step_items,"batch_sources":batch_sources,"batch_start_samples":batch_starts,"batch_tasks":batch_tasks,"batch_t":t_batch})
        )?;
        if local == 1 || local % 10 == 0 || local == c.steps {
            eprintln!(
                "step {} | loss {:.6} | grad {:.4} | lr {:.2e} | {:.2} ms",
                ck.completed_steps, loss, norm, step_lr, ms
            );
            log.flush()?;
        }
        let synced = if local % c.validation_every == 0 {
            trainer.sync(&mut ck.model, &mut ck.adam)?;
            trainer.sync_ema(&mut ck.ema_weights)?;
            let v = val_loss(&ck)?;
            ensure!(v.is_finite(), "nonfinite validation loss");
            let entry = serde_json::json!({
                "step": ck.completed_steps,
                "split": if validation.is_some() { "validation" } else { "train-probe" },
                "validation_loss": v,
                "train_probe_loss": serde_json::Value::Null,
            });
            writeln!(val_log, "{entry}")?;
            val_log.flush()?;
            val_curve.push(entry);
            if local % (c.validation_every * 5) == 0 || local == c.steps {
                eprintln!("  validation @ step {} | {:.6}", ck.completed_steps, v);
            }
            true
        } else {
            false
        };
        if !synced && (local % c.preview_every == 0 || local % c.checkpoint_every == 0) {
            trainer.sync(&mut ck.model, &mut ck.adam)?;
            trainer.sync_ema(&mut ck.ema_weights)?;
        }
        if local % c.preview_every == 0 {
            let now = Instant::now();
            let x = preview_sample(&ck, &c, &sample_config, n, preview_field)?;
            sampling_ms += now.elapsed().as_secs_f64() * 1000.;
            audio::write_wav_channels(
                &c.run.join(format!("preview-{:06}.wav", ck.completed_steps)),
                &x,
                ck.sample_rate,
                ch,
            )?;
        }
        if local % c.checkpoint_every == 0 && local != c.steps {
            ck.save(
                &c.run
                    .join(format!("checkpoint-{:06}.json", ck.completed_steps)),
            )?;
        }
    }
    let training_wall_ms = training_start.elapsed().as_secs_f64() * 1000.;
    trainer.sync(&mut ck.model, &mut ck.adam)?;
    trainer.sync_ema(&mut ck.ema_weights)?;
    let final_eval = evaluate_checkpoint(&ck, &c.backend, &dataset, eval_n, cached_eval.as_ref())?;
    let final_val = val_loss(&ck)?;
    ensure!(final_val.is_finite(), "nonfinite final validation loss");
    let final_entry = serde_json::json!({
        "step": ck.completed_steps,
        "split": if validation.is_some() { "validation" } else { "train-probe" },
        "validation_loss": final_val,
        "train_probe_loss": final_eval,
    });
    if val_curve
        .last()
        .map(|e| e["step"] != final_entry["step"])
        .unwrap_or(true)
    {
        writeln!(val_log, "{final_entry}")?;
        val_log.flush()?;
        val_curve.push(final_entry);
    }
    let now = Instant::now();
    let final_audio = preview_sample(&ck, &c, &sample_config, n, preview_field)?;
    let final_sampling_ms = now.elapsed().as_secs_f64() * 1000.;
    audio::write_wav_channels(
        &c.run.join("generated.wav"),
        &final_audio,
        ck.sample_rate,
        ch,
    )?;
    let delta = initial_audio
        .iter()
        .zip(&final_audio)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / (n * ch) as f32;
    let out = c.run.join("checkpoint.json");
    ck.save(&out)?;
    step_ms.sort_by(|a, b| a.total_cmp(b));
    let mut metrics = serde_json::json!({"backend":trainer.name(),"backend_info":trainer.info(),"timing_scope":match c.backend.as_str(){"cpu"=>"CPU wall operations","cuda"=>"host wall time around synchronized CUDA kernel groups; includes payload upload",_=>"GPU event operation totals; step wall includes transfers, enqueue and sync"},"flow":ck.flow,"detail":ck.detail,"loss_units":if ck.flow.is_legacy(){"waveform velocity MSE"}else{"normalized Gaussian-residual MSE"},"parameters":ck.model.count(),"steps":c.steps,"total_steps":ck.completed_steps,"preprocessing_ms":preprocessing_ms,"neural_forward_backward_ms":neural_ms,"optimizer_ms":optimizer_ms,"training_wall_ms_with_previews":training_wall_ms,"step_mean_ms":step_ms.iter().sum::<f64>()/step_ms.len() as f64,"step_median_ms":step_ms[step_ms.len()/2],"step_max_ms":step_ms.last(),"final_sampling_ms":final_sampling_ms,"other_sampling_ms":sampling_ms,"sampling":sample_config,"sampling_backend":c.backend,"batch":c.batch,"samples_per_step":n*c.batch,"steps_per_second":c.steps as f64/(training_wall_ms/1000.).max(1e-9),"validation_input":c.validation,"validation_initial_loss":initial_val,"validation_final_loss":final_val,"validation_curve":val_curve,"peak_rss_kib":peak_rss_kib(),"fixed_probe_initial_loss":initial_eval,"fixed_probe_final_loss":final_eval,"same_noise_output_mse_change":delta,"generated_rms":audio::rms(&final_audio),"task_counts":task_counts,"source_counts":source_counts,"silence_retries":silence_retries,"crop_sampling":"uniform file then uniform start; retry silent crops","quality_claim":"pipeline verification only; no perceptual quality or generalization established"});
    if let Some(obj) = metrics.as_object_mut() {
        obj.insert(
            "spectral_weight".into(),
            serde_json::json!(ck.spectral_weight),
        );
        obj.insert("time_sampling".into(), serde_json::json!(ck.time_sampling));
        obj.insert("lr_schedule".into(), serde_json::json!(ck.lr_schedule));
        obj.insert("ema_decay".into(), serde_json::json!(ck.ema_decay));
        obj.insert("using_ema".into(), serde_json::json!(ck.has_ema()));
        obj.insert("weight_decay".into(), serde_json::json!(ck.weight_decay));
        obj.insert(
            "horizon_samples".into(),
            serde_json::json!(ck.horizon_samples),
        );
        obj.insert(
            "horizon_schedule".into(),
            serde_json::json!(ck.horizon_schedule),
        );
        obj.insert(
            "horizon_total_steps".into(),
            serde_json::json!(ck.horizon_total_steps),
        );
        obj.insert("channels".into(), serde_json::json!(ch));
    }
    write_json(&c.run.join("metrics.json"), &metrics)?;
    eprintln!(
        "checkpoint {} | probe {:.6} -> {:.6} | output delta MSE {:.6}",
        out.display(),
        initial_eval,
        final_eval,
        delta
    );
    Ok(out)
}
#[derive(Clone, Copy)]
pub enum Mode {
    Generate,
    Variation,
    Continue,
    Inpaint,
}
pub struct RenderConfig {
    pub max_input_seconds: f32,
    pub checkpoint: PathBuf,
    pub input: Option<PathBuf>,
    pub output: PathBuf,
    pub seconds: f32,
    pub start: f32,
    pub end: f32,
    pub sampling: Sampling,
    pub mode: Mode,
    /// Velocity-field backend for sampling: `cpu` is the reference, `cuda` evaluates the same
    /// field on the device. Never a silent fallback - an unavailable backend is a hard error.
    pub backend: String,
    /// Use EMA shadow weights when present in the checkpoint (default true).
    pub use_ema: bool,
}
/// Enforces the protected-sample contract for a conditioned render.
///
/// Every position with `mask != 0` is a known sample — a continue suffix, an inpaint collar or a
/// variation anchor — and must come back bit-identical (README, AGENTS.md, reports/VALIDATION.md).
/// This used to be reported as `protected_samples_exact` in the take's JSON without being
/// enforced; a take that differs by one bit is broken, so the assertion names the mode and the
/// first offending index. The JSON field stays, because readers and reports key on it.
pub fn ensure_protected(output: &[f32], known: &[f32], mask: &[f32], mode: &str) -> Result<()> {
    ensure!(
        output.len() == known.len() && output.len() == mask.len(),
        "protected-sample check shape mismatch"
    );
    if let Some(i) = output
        .iter()
        .zip(known)
        .zip(mask)
        .position(|((&a, &b), &m)| m != 0. && a.to_bits() != b.to_bits())
    {
        anyhow::bail!(
            "{mode} render changed protected sample {i} ({} -> {}); known samples must be retained bit-exactly",
            known[i],
            output[i]
        );
    }
    Ok(())
}
pub fn render(c: RenderConfig) -> Result<()> {
    let ck = Checkpoint::load(&c.checkpoint)?;
    let ch = ck.channels();
    c.sampling.validate()?;
    ensure!(
        ["cpu", "cuda"].contains(&c.backend.as_str()),
        "sampling backend must be cpu or cuda, got {}",
        c.backend
    );
    ensure!(
        !c.output.exists() && !c.output.with_extension("json").exists(),
        "output or metadata already exists; choose a new name"
    );
    ensure!(
        c.seconds.is_finite() && (0.01..=16.).contains(&c.seconds),
        "seconds must be 0.01..16"
    );
    let start_time = Instant::now();
    let sr = ck.sample_rate;
    let block = ck.model.architecture.block;
    let p = Preprocess {
        sample_rate: sr,
        max_seconds: c.max_input_seconds,
        normalize: false,
        silence_rms: 0.,
        channel: if ch == 2 {
            "stereo".into()
        } else {
            "mix".into()
        },
    };
    p.validate()?;
    ensure!(
        !matches!(c.mode, Mode::Generate) || c.input.is_none(),
        "generate does not use an input; use continue, inpaint or variation"
    );
    let input = if let Some(path) = &c.input {
        let duration = audio::duration(path)?;
        ensure!(duration <= c.max_input_seconds as f64, "source is {duration:.3}s, beyond --max-input-seconds {}; explicitly decode a crop or raise the limit (max 600s)", c.max_input_seconds);
        Some(audio::decode(path, &p)?)
    } else {
        None
    };
    let input_hash = c.input.as_ref().map(|p| data::hash_file(p)).transpose()?;
    let additional = (c.seconds * sr as f32).round() as usize;
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    let (mut known, mut mask, output_len, offset) = match c.mode {
        Mode::Generate => (
            vec![0.; additional * ch],
            vec![0.; additional * ch],
            additional,
            0,
        ),
        Mode::Variation => {
            let x = input.context("variation needs --input")?;
            let n = x.len() / ch;
            ensure!(
                n <= crate::model::MAX_WINDOW_SAMPLES,
                "variation limited to the maximum window; crop source first"
            );
            (x, vec![0.; n * ch], n, 0)
        }
        Mode::Continue => {
            let x = input.context("continue needs --input")?;
            let x_frames = x.len() / ch;
            let context = (ck.crop_samples / 2).min(x_frames);
            let offset = x_frames - context;
            prefix = x[..offset * ch].to_vec();
            let mut known = x[offset * ch..].to_vec();
            let mut mask = vec![1.; context * ch];
            known.resize((context + additional) * ch, 0.);
            mask.resize(known.len(), 0.);
            let n = context + additional;
            (known, mask, n, offset)
        }
        Mode::Inpaint => {
            let x = input.context("inpaint needs --input")?;
            let x_frames = x.len() / ch;
            ensure!(
                c.start.is_finite()
                    && c.end.is_finite()
                    && c.start >= 0.
                    && c.end > c.start
                    && c.end <= x_frames as f32 / sr as f32,
                "invalid inpaint interval"
            );
            let a = (c.start * sr as f32).round() as usize;
            let b = (c.end * sr as f32).round() as usize;
            ensure!(
                b > a && b <= x_frames,
                "empty/out of range inpaint interval"
            );
            let collar = ck.crop_samples / 4;
            let left = a.saturating_sub(collar);
            let right = (b + collar).min(x_frames);
            prefix = x[..left * ch].to_vec();
            suffix = x[right * ch..].to_vec();
            let known = x[left * ch..right * ch].to_vec();
            let mut mask = vec![1.; known.len()];
            mask[(a - left) * ch..(b - left) * ch].fill(0.);
            let n = right - left;
            (known, mask, n, left)
        }
    };
    let preprocess_ms = start_time.elapsed().as_secs_f64() * 1000.;
    let n = output_len.div_ceil(block) * block;
    ensure!(
        n <= crate::model::MAX_WINDOW_SAMPLES,
        "conditioned window exceeds the maximum window; request a shorter interval"
    );
    known.resize(n * ch, 0.);
    mask.resize(n * ch, 0.);
    // Variation uses explicit protected anchors and the same masked flow path.
    let now = Instant::now();
    if matches!(c.mode, Mode::Variation) {
        // Experimental disconnected anchors; training uses contiguous mask patterns.
        let stride = (sr as usize / 32).max(block);
        for f in 0..output_len {
            if (f / stride) % 4 == 0 {
                mask[f * ch..(f + 1) * ch].fill(1.);
            }
        }
    }
    let active_model = ck.sampling_model(c.use_ema);
    let using_ema = c.use_ema && ck.has_ema();
    let mut output = match c.backend.as_str() {
        "cpu" => {
            let field = ConfiguredVelocity {
                model: &active_model,
                config: &ck.flow,
            };
            flow::sample(&field, &known, &mask, &c.sampling)?
        }
        "cuda" => {
            // One resident trainer for the whole trajectory: the weights upload once, and each
            // field evaluation is a forward pass on the device.
            let field =
                crate::sampler::DeviceVelocity::from_cuda(&active_model, ck.adam.lr, &ck.flow, n)?;
            flow::sample(&field, &known, &mask, &c.sampling)?
        }
        other => anyhow::bail!("sampling backend must be cpu or cuda, got {other}"),
    };
    let sampling_ms = now.elapsed().as_secs_f64() * 1000.;
    let jumps = conditioning::boundary_jumps_channels(
        &output[..output_len * ch],
        &mask[..output_len * ch],
        ch,
    );
    let mode_label = match c.mode {
        Mode::Generate => "generate",
        Mode::Continue => "continue",
        Mode::Inpaint => "inpaint",
        Mode::Variation => "variation",
    };
    ensure_protected(&output, &known, &mask, mode_label)?;
    // The check above refuses any changed protected sample, so a written take is always exact.
    // The field stays in the JSON because readers and existing reports key on it.
    let preserved = true;
    output.truncate(output_len * ch);
    let mut all = prefix;
    all.extend(output);
    all.extend(suffix);
    audio::write_wav_channels(&c.output, &all, sr, ch)?;
    write_json(
        &c.output.with_extension("json"),
        &serde_json::json!({"project":"FERAL","mode":match c.mode{Mode::Generate=>"generate",Mode::Continue=>"continue",Mode::Inpaint=>"inpaint",Mode::Variation=>"variation-masked-anchors"},"checkpoint":c.checkpoint,"checkpoint_sha256":data::hash_file(&c.checkpoint)?,"model_hash":ck.model_hash,"flow":ck.flow,"training_steps":ck.completed_steps,"git_commit":git(&["rev-parse","HEAD"]),"input":c.input,"input_sha256":input_hash,"preprocess":p,"sampling":c.sampling,"sampling_backend":c.backend,"using_ema":using_ema,"sample_rate":sr,"channels":ch,"output_samples":all.len()/ch,"window_offset_samples":offset,"inpaint_start":c.start,"inpaint_end":c.end,"preprocessing_ms":preprocess_ms,"sampling_ms":sampling_ms,"peak_rss_kib":peak_rss_kib(),"protected_samples_exact":preserved,"boundary_jumps":jumps,"rms":audio::rms(&all),"output_sha256":data::hash_file(&c.output)?}),
    )?;
    eprintln!(
        "wrote {} | {:.1} ms sampling | protected exact={}",
        c.output.display(),
        sampling_ms,
        preserved
    );
    Ok(())
}
