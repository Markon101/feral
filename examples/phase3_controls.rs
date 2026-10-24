//! Explicit non-neural hostile controls, never the FERAL generator.
use anyhow::{ensure, Result};
use clap::Parser;
use feral::{
    audio::{self, Preprocess},
    data, evaluate,
    flow::{self, Sampling},
    parameterization::GaussianReference,
    rng::Rng,
    runtime,
};
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 8192)]
    samples: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.samples >= 32 && args.samples <= 160000 && args.samples.is_multiple_of(32),
        "samples must be a multiple of 32 in 32..160000"
    );
    let p = Preprocess {
        normalize: false,
        max_seconds: 30.,
        ..Default::default()
    };
    let reference = audio::decode(&args.reference, &p)?;
    let rms = audio::rms(&reference);
    let power = reference
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .max(1e-20);
    let rho = ((1..reference.len())
        .map(|i| reference[i] as f64 * reference[i - 1] as f64)
        .sum::<f64>()
        / power)
        .clamp(-0.98, 0.98) as f32;
    fs::create_dir(&args.output)?;
    let zeros = vec![0.; args.samples];
    let sampling = Sampling {
        seed: args.seed,
        steps: 64,
        collar: 0,
        ..Default::default()
    };
    let gaussian = flow::sample(
        &GaussianReference {
            block: 32,
            data_scale: 0.1,
        },
        &zeros,
        &zeros,
        &sampling,
    )?;
    audio::write_wav(&args.output.join("gaussian.wav"), &gaussian, 16000)?;
    let mut rng = Rng::new(args.seed);
    let mut state = 0.;
    let mut ar1 = Vec::with_capacity(args.samples);
    for i in 0..args.samples + 2048 {
        state = rho * state + rms * (1. - rho * rho).sqrt() * rng.normal();
        if i >= 2048 {
            ar1.push(state);
        }
    }
    audio::write_wav(&args.output.join("ar1.wav"), &ar1, 16000)?;
    runtime::write_json(
        &args.output.join("controls.json"),
        &serde_json::json!({"reference":args.reference,"reference_sha256":data::hash_file(&args.reference)?,"preprocess":p,"sampling":sampling,"ar1_rho":rho,"ar1_rms":rms,"ar1_burnin_samples":2048,"gaussian":evaluate::describe(&gaussian,16000),"ar1":evaluate::describe(&ar1,16000),"note":"non-neural controls only; AR1 fits two training moments, Gaussian is zero learned residual; neither is presented as FERAL generation"}),
    )?;
    Ok(())
}
