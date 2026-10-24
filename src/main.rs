use anyhow::{ensure, Result};
use clap::{Args, Parser, Subcommand};
use feral::{
    audio::{self, Preprocess},
    flow::Sampling,
    model::Architecture,
    parameterization::FlowConfig,
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
    Train(TrainArgs),
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
    /// Checkpointed velocity parameterization; changing it requires fresh training.
    #[arg(long, default_value="velocity", value_parser=["velocity", "preconditioned"])]
    parameterization: String,
    /// Fixed dataset amplitude scale for the preconditioned network.
    #[arg(long, default_value_t = 0.1)]
    data_scale: f32,
    /// baseline preserves v1; pyramid pools residual branches at scales 1/4/16/64.
    #[arg(long, default_value="baseline", value_parser=["baseline", "pyramid"])]
    network: String,
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
                "Training backend: CPU f32 only. No silent GPU fallback. No pretrained models."
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
            runtime::train(TrainConfig {
                flow: if a.parameterization == "preconditioned" {
                    FlowConfig::Preconditioned {
                        data_scale: a.data_scale,
                    }
                } else {
                    FlowConfig::Velocity
                },
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
                architecture: if a.network == "pyramid" {
                    Architecture::pyramid()
                } else {
                    Architecture::default()
                },
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
        } => {
            feral::evaluate::evaluate(&checkpoint, &input, &output, crops, seed)?;
        }
        Commands::Analyze {
            input,
            reference,
            output,
        } => {
            feral::evaluate::analyze(&input, &reference, &output)?;
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
                architecture: Architecture::default(),
            })?;
        }
    }
    Ok(())
}
