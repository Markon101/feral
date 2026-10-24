use crate::{
    audio::{self, Preprocess},
    conditioning,
    data::{self, Dataset, DatasetIdentity},
    flow::{self, Sampling},
    model::{Architecture, Model},
    parameterization::{self, ConfiguredVelocity, FlowConfig},
    rng::Rng,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
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
        ensure!(
            grad.len() == self.m.len() && grad.iter().all(|g| g.is_finite()),
            "nonfinite or invalid gradients"
        );
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
            p[i] -= self.lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
        ensure!(
            p.iter().all(|v| v.is_finite()),
            "optimizer produced nonfinite parameters"
        );
        model.set_parameters(&p);
        Ok(norm)
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub format_version: u32,
    #[serde(default, skip_serializing_if = "FlowConfig::is_legacy")]
    pub flow: FlowConfig,
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
    if root.join("src").exists() {
        for e in fs::read_dir(root.join("src"))? {
            let p = e?.path();
            if p.is_file() {
                paths.push(p.strip_prefix(root)?.to_path_buf());
            }
        }
    }
    paths.sort();
    let mut h = Sha256::new();
    for p in paths {
        h.update(p.to_string_lossy().as_bytes());
        h.update(fs::read(root.join(p))?);
    }
    Ok(format!("{:x}", h.finalize()))
}
impl Checkpoint {
    pub fn velocity(&self) -> ConfiguredVelocity<'_> {
        ConfiguredVelocity {
            model: &self.model,
            config: &self.flow,
        }
    }
    pub fn save(&mut self, path: &Path) -> Result<()> {
        self.model_hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&self.model)?));
        write_json(path, self)
    }
    pub fn load(path: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(path)?.len() <= 64 * 1024 * 1024,
            "checkpoint exceeds 64 MB"
        );
        let c: Self = serde_json::from_reader(std::io::BufReader::new(fs::File::open(path)?))?;
        ensure!(
            [1, 2, 3].contains(&c.format_version),
            "unsupported checkpoint version"
        );
        c.flow.validate()?;
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
            c.adam.lr.is_finite()
                && c.adam.lr > 0.
                && c.adam.lr <= 0.01
                && c.adam.step == c.completed_steps,
            "invalid optimizer step/lr"
        );
        ensure!(
            (8000..=48000).contains(&c.sample_rate)
                && c.crop_samples >= 32
                && c.crop_samples <= 160000
                && c.crop_samples.is_multiple_of(c.model.architecture.block),
            "invalid checkpoint audio configuration"
        );
        ensure!(
            format!("{:x}", Sha256::digest(serde_json::to_vec(&c.model)?)) == c.model_hash,
            "checkpoint model hash mismatch"
        );
        Ok(c)
    }
}
#[derive(Clone)]
pub struct TrainConfig {
    pub input: PathBuf,
    pub run: PathBuf,
    pub steps: usize,
    pub seed: u64,
    pub crop_seconds: f32,
    pub preprocess: Preprocess,
    pub lr: f32,
    pub preview_every: usize,
    pub checkpoint_every: usize,
    pub resume: Option<PathBuf>,
    pub weights_only: bool,
    pub architecture: Architecture,
    pub flow: FlowConfig,
}
fn evaluation(model: &Model, flow: &FlowConfig, dataset: &Dataset, n: usize) -> f32 {
    let mut rng = Rng::new(901);
    let mut sum = 0.;
    for _ in 0..4 {
        let clean = dataset.crop(n, &mut rng);
        if clean.is_empty() {
            return f32::NAN;
        }
        let (mask, _) = conditioning::training_mask(n, &mut rng);
        let noise: Vec<f32> = (0..n).map(|_| rng.normal() * flow.prior_std()).collect();
        sum += parameterization::loss_gradient(model, flow, &clean, &mask, &noise, 0.5).0;
    }
    sum / 4.
}
pub fn train(c: TrainConfig) -> Result<PathBuf> {
    ensure!(
        (1..=100000).contains(&c.steps),
        "training steps must be 1..100000"
    );
    ensure!(
        c.crop_seconds.is_finite() && (0.032..=1.).contains(&c.crop_seconds),
        "crop seconds must be 0.032..1"
    );
    ensure!(
        c.lr.is_finite() && c.lr > 0. && c.lr <= 0.01,
        "learning rate must be >0 and <=0.01"
    );
    c.architecture.validate()?;
    c.flow.validate()?;
    c.preprocess.validate()?;
    ensure!(
        c.preview_every > 0 && c.checkpoint_every > 0,
        "intervals must be positive"
    );
    let b = c.architecture.block;
    let n = ((c.crop_seconds * c.preprocess.sample_rate as f32).round() as usize).div_ceil(b) * b;
    let pre_start = Instant::now();
    let dataset = Dataset::load(&c.input, c.preprocess.clone(), n)?;
    let preprocessing_ms = pre_start.elapsed().as_secs_f64() * 1000.;
    let mut ck = if let Some(path) = &c.resume {
        let mut ck = Checkpoint::load(path)?;
        ensure!(
            ck.flow == c.flow,
            "resume parameterization mismatch; train a fresh model for a different flow"
        );
        ensure!(
            ck.crop_samples == n && ck.sample_rate == c.preprocess.sample_rate,
            "resume crop/sample-rate mismatch"
        );
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
        } else {
            ensure!(
                ck.dataset.manifest_hash == dataset.identity.manifest_hash,
                "full resume requires identical dataset/preprocessing"
            );
            ensure!(
                ck.adam.lr == c.lr,
                "full resume requires same learning rate"
            );
        }
        ck
    } else {
        let model = Model::new(c.architecture.clone(), c.seed);
        let adam = Adam::new(model.count(), c.lr);
        Checkpoint {
            format_version: c.flow.format_version(),
            flow: c.flow.clone(),
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
            objective: if c.flow.is_legacy() {
                "conditional rectified flow, unknown-sample MSE, uniform t, N(0,1) noise".into()
            } else {
                "conditional rectified flow, Gaussian residual preconditioning, residual MSE (=velocity MSE/out^2), uniform t, checkpointed Gaussian prior".into()
            },
            representation: "f32 mono waveform; lossless time-major blocks".into(),
            model_hash: String::new(),
            unix_seconds: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        }
    };
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
    write_json(
        &c.run.join("run.json"),
        &serde_json::json!({"git_commit":git(&["rev-parse","HEAD"]),"git_dirty":!git(&["status","--porcelain"]).is_empty(),"source_hash":current_source,"dataset":dataset.identity,"parameters":ck.model.count(),"architecture":ck.model.architecture,"crop_samples":n,"sample_rate":ck.sample_rate,"additional_steps":c.steps,"seed":c.seed,"lr":ck.adam.lr,"optimizer":"Adam beta1=.9 beta2=.999 eps=1e-8 global gradient clip=1","resume":c.resume,"weights_only":c.weights_only,"backend":"cpu-f32","objective":ck.objective,"representation":ck.representation,"flow":ck.flow}),
    )?;
    ck.save(&c.run.join("initial.json"))?;
    let sample_config = Sampling {
        steps: 16,
        ..Default::default()
    };
    let zeros = vec![0.; n];
    let mut sampling_ms = 0.;
    let start = Instant::now();
    let initial_audio = flow::sample(&ck.velocity(), &zeros, &zeros, &sample_config)?;
    sampling_ms += start.elapsed().as_secs_f64() * 1000.;
    audio::write_wav(
        &c.run.join("untrained-or-resumed-initial.wav"),
        &initial_audio,
        ck.sample_rate,
    )?;
    write_json(&c.run.join("preview-config.json"), &sample_config)?;
    let initial_eval = evaluation(&ck.model, &ck.flow, &dataset, n);
    ensure!(initial_eval.is_finite(), "invalid initial evaluation");
    let mut log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(c.run.join("losses.jsonl"))?;
    let mut neural_ms = 0.;
    let mut optimizer_ms = 0.;
    let mut step_ms = Vec::new();
    let mut task_counts = std::collections::BTreeMap::new();
    let training_start = Instant::now();
    eprintln!(
        "FERAL CPU: {} parameters, {} samples/crop, {} files, preprocessing {:.1} ms",
        ck.model.count(),
        n,
        dataset.audio.len(),
        preprocessing_ms
    );
    for local in 1..=c.steps {
        let start = Instant::now();
        let clean = dataset.crop(n, &mut ck.rng);
        ensure!(
            !clean.is_empty(),
            "16 crop attempts were silent; reduce --silence-rms or select useful audio"
        );
        let (mask, task) = conditioning::training_mask(n, &mut ck.rng);
        *task_counts.entry(task).or_insert(0usize) += 1;
        let noise: Vec<f32> = (0..n)
            .map(|_| ck.rng.normal() * ck.flow.prior_std())
            .collect();
        let t = ck.rng.uniform();
        let op = Instant::now();
        let (loss, grad) =
            parameterization::loss_gradient(&ck.model, &ck.flow, &clean, &mask, &noise, t);
        neural_ms += op.elapsed().as_secs_f64() * 1000.;
        ensure!(loss.is_finite(), "nonfinite loss");
        let op = Instant::now();
        let norm = ck.adam.update(&mut ck.model, &grad)?;
        optimizer_ms += op.elapsed().as_secs_f64() * 1000.;
        ck.completed_steps += 1;
        let ms = start.elapsed().as_secs_f64() * 1000.;
        step_ms.push(ms);
        writeln!(
            log,
            "{}",
            serde_json::json!({"step":ck.completed_steps,"loss":loss,"gradient_norm":norm,"task":task,"t":t,"milliseconds":ms,"rng_state":ck.rng.state})
        )?;
        if local == 1 || local % 10 == 0 || local == c.steps {
            eprintln!(
                "step {} | loss {:.6} | grad {:.4} | {:.2} ms",
                ck.completed_steps, loss, norm, ms
            );
            log.flush()?;
        }
        if local % c.preview_every == 0 {
            let now = Instant::now();
            let x = flow::sample(&ck.velocity(), &zeros, &zeros, &sample_config)?;
            sampling_ms += now.elapsed().as_secs_f64() * 1000.;
            audio::write_wav(
                &c.run.join(format!("preview-{:06}.wav", ck.completed_steps)),
                &x,
                ck.sample_rate,
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
    let final_eval = evaluation(&ck.model, &ck.flow, &dataset, n);
    let now = Instant::now();
    let final_audio = flow::sample(&ck.velocity(), &zeros, &zeros, &sample_config)?;
    let final_sampling_ms = now.elapsed().as_secs_f64() * 1000.;
    audio::write_wav(&c.run.join("generated.wav"), &final_audio, ck.sample_rate)?;
    let delta = initial_audio
        .iter()
        .zip(&final_audio)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / n as f32;
    let out = c.run.join("checkpoint.json");
    ck.save(&out)?;
    step_ms.sort_by(|a, b| a.total_cmp(b));
    write_json(
        &c.run.join("metrics.json"),
        &serde_json::json!({"flow":ck.flow,"loss_units":if ck.flow.is_legacy(){"waveform velocity MSE"}else{"normalized Gaussian-residual MSE"},"parameters":ck.model.count(),"steps":c.steps,"total_steps":ck.completed_steps,"preprocessing_ms":preprocessing_ms,"neural_forward_backward_ms":neural_ms,"optimizer_ms":optimizer_ms,"training_wall_ms_with_previews":training_wall_ms,"step_mean_ms":step_ms.iter().sum::<f64>()/step_ms.len() as f64,"step_median_ms":step_ms[step_ms.len()/2],"step_max_ms":step_ms.last(),"final_sampling_ms":final_sampling_ms,"other_sampling_ms":sampling_ms,"sampling":sample_config,"peak_rss_kib":peak_rss_kib(),"fixed_probe_initial_loss":initial_eval,"fixed_probe_final_loss":final_eval,"same_noise_output_mse_change":delta,"generated_rms":audio::rms(&final_audio),"task_counts":task_counts,"quality_claim":"pipeline verification only; no perceptual quality or generalization established"}),
    )?;
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
}
pub fn render(c: RenderConfig) -> Result<()> {
    let ck = Checkpoint::load(&c.checkpoint)?;
    c.sampling.validate()?;
    ensure!(
        !c.output.exists() && !c.output.with_extension("json").exists(),
        "output or metadata already exists; choose a new name"
    );
    ensure!(
        c.seconds.is_finite() && (0.01..=5.).contains(&c.seconds),
        "seconds must be 0.01..5"
    );
    let start_time = Instant::now();
    let sr = ck.sample_rate;
    let block = ck.model.architecture.block;
    let p = Preprocess {
        sample_rate: sr,
        max_seconds: c.max_input_seconds,
        normalize: false,
        silence_rms: 0.,
        channel: "mix".into(),
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
        Mode::Generate => (vec![0.; additional], vec![0.; additional], additional, 0),
        Mode::Variation => {
            let x = input.context("variation needs --input")?;
            ensure!(
                x.len() <= 160000,
                "variation limited to 160000 samples; crop source first"
            );
            let n = x.len();
            (x, vec![0.; n], n, 0)
        }
        Mode::Continue => {
            let x = input.context("continue needs --input")?;
            let context = (ck.crop_samples / 2).min(x.len());
            let offset = x.len() - context;
            prefix = x[..offset].to_vec();
            let mut known = x[offset..].to_vec();
            let mut mask = vec![1.; context];
            known.resize(context + additional, 0.);
            mask.resize(known.len(), 0.);
            let n = known.len();
            (known, mask, n, offset)
        }
        Mode::Inpaint => {
            let x = input.context("inpaint needs --input")?;
            ensure!(
                c.start.is_finite()
                    && c.end.is_finite()
                    && c.start >= 0.
                    && c.end > c.start
                    && c.end <= x.len() as f32 / sr as f32,
                "invalid inpaint interval"
            );
            let a = (c.start * sr as f32).round() as usize;
            let b = (c.end * sr as f32).round() as usize;
            ensure!(b > a && b <= x.len(), "empty/out of range inpaint interval");
            let collar = ck.crop_samples / 4;
            let left = a.saturating_sub(collar);
            let right = (b + collar).min(x.len());
            prefix = x[..left].to_vec();
            suffix = x[right..].to_vec();
            let known = x[left..right].to_vec();
            let mut mask = vec![1.; known.len()];
            mask[a - left..b - left].fill(0.);
            let n = known.len();
            (known, mask, n, left)
        }
    };
    let preprocess_ms = start_time.elapsed().as_secs_f64() * 1000.;
    let n = known.len().div_ceil(block) * block;
    ensure!(
        n <= 160000,
        "conditioned window exceeds 160000 samples; request a shorter interval"
    );
    known.resize(n, 0.);
    mask.resize(n, 0.);
    // Variation uses explicit protected anchors and the same masked flow path.
    let now = Instant::now();
    let mut output = if matches!(c.mode, Mode::Variation) {
        // Experimental disconnected anchors; training uses contiguous mask patterns.
        let stride = (sr as usize / 32).max(block);
        for (i, m) in mask.iter_mut().take(output_len).enumerate() {
            if (i / stride) % 4 == 0 {
                *m = 1.;
            }
        }
        flow::sample(&ck.velocity(), &known, &mask, &c.sampling)?
    } else {
        flow::sample(&ck.velocity(), &known, &mask, &c.sampling)?
    };
    let sampling_ms = now.elapsed().as_secs_f64() * 1000.;
    let jumps = conditioning::boundary_jumps(&output[..output_len], &mask[..output_len]);
    let preserved = output
        .iter()
        .zip(&known)
        .zip(&mask)
        .all(|((&a, &b), &m)| m == 0. || a.to_bits() == b.to_bits());
    output.truncate(output_len);
    let mut all = prefix;
    all.extend(output);
    all.extend(suffix);
    audio::write_wav(&c.output, &all, sr)?;
    write_json(
        &c.output.with_extension("json"),
        &serde_json::json!({"project":"FERAL","mode":match c.mode{Mode::Generate=>"generate",Mode::Continue=>"continue",Mode::Inpaint=>"inpaint",Mode::Variation=>"variation-masked-anchors"},"checkpoint":c.checkpoint,"checkpoint_sha256":data::hash_file(&c.checkpoint)?,"model_hash":ck.model_hash,"flow":ck.flow,"training_steps":ck.completed_steps,"git_commit":git(&["rev-parse","HEAD"]),"input":c.input,"input_sha256":input_hash,"preprocess":p,"sampling":c.sampling,"sample_rate":sr,"output_samples":all.len(),"window_offset_samples":offset,"inpaint_start":c.start,"inpaint_end":c.end,"preprocessing_ms":preprocess_ms,"sampling_ms":sampling_ms,"peak_rss_kib":peak_rss_kib(),"protected_samples_exact":preserved,"boundary_jumps":jumps,"rms":audio::rms(&all),"output_sha256":data::hash_file(&c.output)?}),
    )?;
    eprintln!(
        "wrote {} | {:.1} ms sampling | protected exact={}",
        c.output.display(),
        sampling_ms,
        preserved
    );
    Ok(())
}
