use anyhow::{ensure, Result};
use clap::{Args, Parser, Subcommand};
use feral::{
    audio::{self, Preprocess},
    flow::Sampling,
    model::{Architecture, BlockStyle},
    parameterization::{FlowConfig, LrSchedule, TimeSampling},
    runtime::{self, Mode, RenderConfig, TrainConfig},
};
use std::{fs, path::PathBuf};
#[derive(Parser)]
#[command(
    name = "feral",
    version,
    about = "FERAL — Freeform Evolutionary Resonant Audio Laboratory"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    /// Inspect CPU, memory, decoder and optional OpenCL discovery.
    Doctor,
    /// Create four deterministic diagnostic patterns (not a music dataset).
    Corpus {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 16000)]
        sample_rate: u32,
    },
    /// Decode into canonical mono f32 WAV without changing the original.
    Decode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[command(flatten)]
        audio: AudioArgs,
    },
    /// Train unconditional and masked conditional rectified flow on unlabeled audio.
    Train(Box<TrainArgs>),
    /// Generate a new waveform from seeded noise and learned weights.
    Generate(RenderArgs),
    /// Append new audio using a protected prefix; original canonical samples are retained.
    Continue(RenderArgs),
    /// Regenerate an interval while preserving every sample outside it.
    Inpaint(RenderArgs),
    /// Regenerate between short protected anchors (experimental).
    Variation(RenderArgs),
    /// Fixed paired held-out flow probes, acoustic metrics and zero/noisy controls.
    Evaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 4)]
        crops: usize,
        #[arg(long, default_value_t = 7001)]
        seed: u64,
        /// Velocity-field backend for probes: cpu (reference) or cuda (device).
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Evaluate raw optimizer weights instead of EMA weights when EMA is present.
        #[arg(long)]
        no_ema: bool,
    },
    /// Signal descriptors and bounded copying screen with shuffled/noise/copy controls.
    Analyze {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// CPU/GPU forward, gradient, Adam and resume checks plus a short benchmark.
    GpuAudit {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 20)]
        steps: usize,
        /// Backend to audit; must match the build features and the machine.
        #[arg(long, default_value="opencl", value_parser=["opencl","cuda"])]
        backend: String,
    },
    /// Tiny actual training + sampling on deterministic diagnostic audio.
    Smoke {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value_t = 20)]
        steps: usize,
    },
}
#[derive(Args, Clone)]
struct AudioArgs {
    #[arg(long, default_value_t = 16000)]
    sample_rate: u32,
    #[arg(long, default_value_t = 30.)]
    max_file_seconds: f32,
    /// mix down stereo, or select left/right; neural output is mono.
    #[arg(long, default_value = "mix")]
    channel: String,
    #[arg(long)]
    no_normalize: bool,
    #[arg(long, default_value_t = 0.0001)]
    silence_rms: f32,
}
impl AudioArgs {
    fn config(&self) -> Preprocess {
        Preprocess {
            sample_rate: self.sample_rate,
            max_seconds: self.max_file_seconds,
            normalize: !self.no_normalize,
            channel: self.channel.clone(),
            silence_rms: self.silence_rms,
        }
    }
}
#[derive(Args)]
struct TrainArgs {
    /// Explicit backend; opencl/cuda require the matching --features build and hardware.
    #[arg(long, default_value="cpu", value_parser=["cpu","opencl","cuda"])]
    backend: String,
    /// Crops per optimizer step; loss and gradients are averaged over the batch.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(usize))]
    batch: usize,
    /// Residual branch count for --network baseline (dilations 1,2,4,...; max 10).
    #[arg(long, default_value_t = 4, value_parser = parse_blocks)]
    blocks: usize,
    /// Dilation schedule for --network baseline: ascending (legacy), symmetric (U-Net coarse-to-fine),
    /// or cycles (two full ascending dilation passes).
    #[arg(long, default_value = "ascending", value_parser = ["ascending", "symmetric", "cycles"])]
    dilation_pattern: String,
    /// Residual block style: tanh (legacy) or gated-film (pre-RMSNorm + per-block FiLM time
    /// conditioning + WaveNet gated activation; CPU and CUDA backends only).
    #[arg(long, default_value = "tanh", value_parser = ["tanh", "gated-film"])]
    block_style: String,
    /// Samples per frame before the first convolution (16/32/64/128). Larger frames are cheaper
    /// per sample and multiply the receptive field by the same factor.
    #[arg(long, default_value_t = 32, value_parser = parse_frame_samples)]
    block: usize,
    /// Optional held-out corpus for periodic validation (never overlapping train crops).
    #[arg(long)]
    validation: Option<PathBuf>,
    /// Validation interval in optimizer steps.
    #[arg(long, default_value_t = 100)]
    validation_every: usize,
    /// Checkpointed velocity parameterization; changing it requires fresh training.
    #[arg(long, default_value="velocity", value_parser=["velocity", "preconditioned"])]
    parameterization: String,
    /// Fixed dataset amplitude scale for the preconditioned network.
    #[arg(long, default_value_t = 0.1)]
    data_scale: f32,
    /// Training AND sampling prior standard deviation; checkpointed, not an OOD knob.
    #[arg(long, default_value_t = 1.)]
    prior_std: f32,
    /// Weight of the detail term in the objective: `<r,r> + weight * <dr,dr>` over unknown samples.
    /// 0 is the plain waveform residual MSE. Checkpointed; CPU and CUDA backends only.
    #[arg(long, default_value_t = 0.)]
    detail_weight: f32,
    /// Weight of the t-weighted multiresolution STFT log-magnitude L1 loss on the predicted clean
    /// endpoint `y_hat`. 0 disables it. Checkpointed; CPU and CUDA backends only.
    #[arg(long, default_value_t = 0.)]
    spectral_weight: f32,
    /// Flow-time distribution during training: uniform (legacy) or logit-normal.
    #[arg(long, default_value = "uniform", value_parser = ["uniform", "logit-normal"])]
    time_sampling: String,
    /// Logit-normal mean when `--time-sampling logit-normal` is selected (-4..4).
    #[arg(long, default_value_t = 0., allow_hyphen_values = true)]
    time_mean: f32,
    /// Logit-normal std when `--time-sampling logit-normal` is selected (0.1..4).
    #[arg(long, default_value_t = 1.)]
    time_std: f32,
    /// Optimizer learning-rate schedule: constant (legacy) or cosine.
    #[arg(long, default_value = "constant", value_parser = ["constant", "cosine"])]
    lr_schedule: String,
    /// Linear warmup steps before cosine decay when `--lr-schedule cosine` is selected.
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    /// Final floor learning rate when `--lr-schedule cosine` is selected.
    #[arg(long, default_value_t = 1e-5)]
    min_lr: f32,
    /// Total schedule horizon in steps for `--lr-schedule cosine` (defaults to `--steps`).
    #[arg(long)]
    schedule_steps: Option<usize>,
    /// Exponential moving average decay for inference weights (0 disables; e.g. 0.999).
    /// Checkpointed; CPU and CUDA backends only.
    #[arg(long, default_value_t = 0.)]
    ema_decay: f32,
    /// Optional odd sample-level 1D conv kernel size around the frame stack (0 disables; odd 3..=15).
    /// Bridges frame boundaries with shift-equivariant sample context; CPU and CUDA backends only.
    #[arg(long, default_value_t = 0, value_parser = parse_sample_conv)]
    sample_conv: usize,
    /// baseline preserves v1; pyramid pools residual branches at scales 1/4/16/64.
    #[arg(long, default_value="baseline", value_parser=["baseline", "pyramid"])]
    network: String,
    /// Hidden channels; 32..512 in multiples of 16. Width changes require fresh weights.
    #[arg(long, default_value_t = 64, value_parser = parse_width)]
    width: usize,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    run: PathBuf,
    #[arg(long, default_value = "phone")]
    profile: String,
    #[arg(long, default_value_t = 100)]
    steps: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0.512)]
    crop_seconds: f32,
    #[arg(long, default_value_t = 0.0003)]
    learning_rate: f32,
    #[arg(long, default_value_t = 50)]
    preview_every: usize,
    #[arg(long, default_value_t = 100)]
    checkpoint_every: usize,
    /// Full model + Adam + RNG resume into a fresh run directory.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// With --resume, reset Adam and RNG (explicit weights-only warm start).
    #[arg(long, requires = "resume")]
    weights_only: bool,
    /// With --resume, continue at a different --crop-seconds (curriculum). The weights, optimizer
    /// and RNG carry over; the data stream does not continue bit-exactly.
    #[arg(long, requires = "resume")]
    grow_crop: bool,
    #[command(flatten)]
    audio: AudioArgs,
}
#[derive(Args)]
struct RenderArgs {
    /// Reject longer conditioning inputs rather than truncate them; max 600 seconds.
    #[arg(long, default_value_t = 600.)]
    max_input_seconds: f32,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    input: Option<PathBuf>,
    /// Newly generated duration; ignored by inpaint and variation.
    #[arg(long, default_value_t = 0.512)]
    seconds: f32,
    #[arg(long, default_value_t = 0.)]
    start: f32,
    #[arg(long, default_value_t = 0.)]
    end: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 32)]
    flow_steps: usize,
    #[arg(long, default_value = "heun")]
    solver: String,
    #[arg(long, default_value_t = 1.)]
    noise_scale: f32,
    /// Bounded, seeded additive velocity perturbation; 0 disables it.
    #[arg(long, default_value_t = 0.)]
    perturb: f32,
    /// Unknown-side smoothing in samples; 0 disables it.
    #[arg(long, default_value_t = 128)]
    collar: usize,
    /// Velocity-field backend for sampling: cpu (reference) or cuda (same field on device).
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// Render with raw optimizer weights instead of EMA weights when EMA is present.
    #[arg(long)]
    no_ema: bool,
}
impl RenderArgs {
    fn config(self, mode: Mode) -> RenderConfig {
        RenderConfig {
            max_input_seconds: self.max_input_seconds,
            checkpoint: self.checkpoint,
            input: self.input,
            output: self.output,
            seconds: self.seconds,
            start: self.start,
            end: self.end,
            backend: self.backend,
            use_ema: !self.no_ema,
            sampling: Sampling {
                seed: self.seed,
                steps: self.flow_steps,
                solver: self.solver,
                noise_scale: self.noise_scale,
                perturb: self.perturb,
                collar: self.collar,
            },
            mode,
        }
    }
}
fn parse_sample_conv(value: &str) -> std::result::Result<usize, String> {
    let k = value.parse::<usize>().map_err(|e| e.to_string())?;
    if k == 0 || ((3..=15).contains(&k) && k % 2 == 1) {
        Ok(k)
    } else {
        Err("sample-conv must be 0 (disabled) or an odd kernel size in 3..=15".into())
    }
}
fn parse_blocks(value: &str) -> std::result::Result<usize, String> {
    let blocks = value.parse::<usize>().map_err(|e| e.to_string())?;
    if (1..=10).contains(&blocks) {
        Ok(blocks)
    } else {
        Err("blocks must be 1..10 (dilations 1..512)".into())
    }
}
fn parse_frame_samples(value: &str) -> std::result::Result<usize, String> {
    let block = value.parse::<usize>().map_err(|e| e.to_string())?;
    if [16, 32, 64, 128].contains(&block) {
        Ok(block)
    } else {
        Err("block must be 16, 32, 64 or 128 samples per frame".into())
    }
}

fn parse_width(value: &str) -> std::result::Result<usize, String> {
    let width = value.parse::<usize>().map_err(|e| e.to_string())?;
    Architecture::default()
        .with_width(width)
        .map_err(|e| e.to_string())?;
    Ok(width)
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Doctor => {
            println!(
                "FERAL CPU reference backend | {} {} | logical CPUs {}",
                std::env::consts::OS,
                std::env::consts::ARCH,
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            );
            if let Ok(s) = fs::read_to_string("/proc/meminfo") {
                for l in s.lines().take(3) {
                    println!("{l}");
                }
            }
            for tool in ["rustc", "cargo", "ffmpeg", "ffprobe"] {
                let out = if tool.starts_with("ff") {
                    audio::decoder_command(tool).arg("-version").output()
                } else {
                    std::process::Command::new(tool).arg("--version").output()
                };
                match out {
                    Ok(o) if o.status.success() => println!(
                        "{}",
                        String::from_utf8_lossy(&o.stdout)
                            .lines()
                            .next()
                            .unwrap_or("")
                    ),
                    Ok(o) => println!("{tool}: FAILED {}", String::from_utf8_lossy(&o.stderr)),
                    Err(e) => println!("{tool}: unavailable: {e}"),
                }
            }
            match std::process::Command::new("clinfo").arg("--list").output() {
                Ok(o) => println!(
                    "OpenCL discovery (not a FERAL backend): status={} stdout={} stderr={}",
                    o.status,
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                ),
                Err(e) => println!("OpenCL discovery unavailable: {e}"),
            }
            println!(
                "CPU reference available. OpenCL training is optional (--features opencl, --backend opencl). No silent GPU fallback or pretrained models."
            );
        }
        Commands::Corpus {
            output,
            sample_rate,
        } => {
            ensure!((8000..=48000).contains(&sample_rate), "invalid sample rate");
            audio::synthetic(&output, sample_rate)?;
        }
        Commands::Decode {
            input,
            output,
            audio: args,
        } => {
            let p = args.config();
            let x = audio::decode(&input, &p)?;
            audio::write_wav(&output, &x, p.sample_rate)?;
            runtime::write_json(
                &output.with_extension("json"),
                &serde_json::json!({"input":fs::canonicalize(&input)?,"input_sha256":feral::data::hash_file(&input)?,"preprocess":p,"samples":x.len()}),
            )?;
        }
        Commands::Train(a) => {
            ensure!(a.profile == "phone", "only --profile phone is implemented");
            ensure!(
                a.parameterization != "velocity" || a.prior_std == 1.,
                "velocity reference fixes prior std=1; use preconditioned for a calibrated prior"
            );
            let time_sampling = if a.time_sampling == "logit-normal" {
                TimeSampling::LogitNormal {
                    mean: a.time_mean,
                    std: a.time_std,
                }
            } else {
                TimeSampling::Uniform
            };
            let lr_schedule = if a.lr_schedule == "cosine" {
                LrSchedule::Cosine {
                    warmup_steps: a.warmup_steps,
                    total_steps: a.schedule_steps.unwrap_or(a.steps),
                    min_lr: a.min_lr,
                }
            } else {
                LrSchedule::Constant
            };
            runtime::train(TrainConfig {
                flow: if a.parameterization == "preconditioned" {
                    FlowConfig::Preconditioned {
                        data_scale: a.data_scale,
                        prior_std: a.prior_std,
                    }
                } else {
                    FlowConfig::Velocity
                },
                detail: a.detail_weight,
                spectral_weight: a.spectral_weight,
                time_sampling,
                lr_schedule,
                ema_decay: a.ema_decay,
                backend: a.backend,
                input: a.input,
                run: a.run,
                steps: a.steps,
                seed: a.seed,
                crop_seconds: a.crop_seconds,
                preprocess: a.audio.config(),
                lr: a.learning_rate,
                preview_every: a.preview_every,
                checkpoint_every: a.checkpoint_every,
                resume: a.resume,
                weights_only: a.weights_only,
                grow_crop: a.grow_crop,
                batch: a.batch,
                validation: a.validation,
                validation_every: a.validation_every,
                architecture: (if a.network == "pyramid" {
                    ensure!(
                        a.blocks == 4 && a.dilation_pattern == "ascending",
                        "--blocks and --dilation-pattern do not apply to --network pyramid"
                    );
                    Architecture::pyramid()
                } else {
                    Architecture::default().with_dilation_pattern(a.blocks, &a.dilation_pattern)?
                })
                .with_width(a.width)?
                .with_frame_samples(a.block)?
                .with_sample_conv(a.sample_conv)?
                .with_block_style(if a.block_style == "gated-film" {
                    BlockStyle::GatedFilm
                } else {
                    BlockStyle::Tanh
                })?,
            })?;
        }
        Commands::Generate(a) => runtime::render(a.config(Mode::Generate))?,
        Commands::Continue(a) => runtime::render(a.config(Mode::Continue))?,
        Commands::Inpaint(a) => runtime::render(a.config(Mode::Inpaint))?,
        Commands::Variation(a) => runtime::render(a.config(Mode::Variation))?,
        Commands::Evaluate {
            checkpoint,
            input,
            output,
            crops,
            seed,
            backend,
            no_ema,
        } => {
            feral::evaluate::evaluate_with_options(
                &checkpoint,
                &input,
                &output,
                crops,
                seed,
                &backend,
                !no_ema,
            )?;
        }
        Commands::Analyze {
            input,
            reference,
            output,
        } => {
            feral::evaluate::analyze(&input, &reference, &output)?;
        }
        Commands::GpuAudit {
            checkpoint,
            output,
            steps,
            backend: audit_backend,
        } => {
            let make = |model: &feral::model::Model,
                        adam: &feral::runtime::Adam,
                        n: usize,
                        flow: &FlowConfig|
             -> anyhow::Result<Box<dyn feral::backend::Resident>> {
                match audit_backend.as_str() {
                    "cuda" => {
                        #[cfg(feature = "cuda")]
                        {
                            Ok(Box::new(feral::cuda::CudaTrainer::new(
                                model,
                                adam,
                                n,
                                1,
                                flow.clone(),
                            )?))
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            let _ = (model, adam, n, flow);
                            anyhow::bail!("gpu-audit --backend cuda requires --features cuda")
                        }
                    }
                    "opencl" => {
                        #[cfg(feature = "opencl")]
                        {
                            Ok(Box::new(feral::gpu::OpenClTrainer::new(
                                model,
                                adam,
                                n,
                                flow.clone(),
                            )?))
                        }
                        #[cfg(not(feature = "opencl"))]
                        {
                            let _ = (model, adam, n, flow);
                            anyhow::bail!("gpu-audit --backend opencl requires --features opencl")
                        }
                    }
                    other => anyhow::bail!("gpu-audit backend must be opencl or cuda, got {other}"),
                }
            };
            #[cfg(any(feature = "opencl", feature = "cuda"))]
            feral::audit::audit(&checkpoint, &output, steps, &make)?;
            #[cfg(not(any(feature = "opencl", feature = "cuda")))]
            {
                let _ = (checkpoint, output, steps, make);
                anyhow::bail!("gpu-audit requires --features opencl or --features cuda");
            }
        }
        Commands::Smoke { run, steps } => {
            if let Some(parent) = run.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::create_dir(&run)?;
            let corpus = run.join("corpus");
            audio::synthetic(&corpus, 16000)?;
            runtime::train(TrainConfig {
                flow: FlowConfig::Velocity,
                detail: 0.,
                spectral_weight: 0.,
                time_sampling: TimeSampling::Uniform,
                lr_schedule: LrSchedule::Constant,
                ema_decay: 0.,
                backend: "cpu".into(),
                input: corpus,
                run: run.join("training"),
                steps,
                seed: 42,
                crop_seconds: 0.128,
                preprocess: Preprocess::default(),
                lr: 0.0003,
                preview_every: 10,
                checkpoint_every: 10,
                resume: None,
                weights_only: false,
                grow_crop: false,
                batch: 1,
                validation: None,
                validation_every: 10,
                architecture: Architecture::default(),
            })?;
        }
    }
    Ok(())
}
