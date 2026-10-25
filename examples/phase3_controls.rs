//! Explicit non-neural hostile controls, never the FERAL generator.
use anyhow::{ensure, Result};
use clap::Parser;
use feral::{
    audio::{self, Preprocess},
    data::{self, Dataset},
    evaluate,
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
    let manifest = args.reference.extension().and_then(|s| s.to_str()) == Some("txt");
    let (rms, rho, dataset_metadata) = if manifest {
        let dataset = Dataset::load(&args.reference, p.clone(), 32)?;
        let samples: usize = dataset.audio.iter().map(Vec::len).sum();
        let power = dataset
            .audio
            .iter()
            .flatten()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>();
        // Do not invent neighboring samples between separately decoded crops.
        let lag_product = dataset
            .audio
            .iter()
            .map(|audio| {
                audio
                    .windows(2)
                    .map(|w| w[0] as f64 * w[1] as f64)
                    .sum::<f64>()
            })
            .sum::<f64>();
        let rms = (power / samples as f64).sqrt() as f32;
        let rho = (lag_product / power.max(1e-20)).clamp(-0.98, 0.98) as f32;
        let concatenated: Vec<f32> = dataset.audio.iter().flatten().copied().collect();
        let metadata = serde_json::json!({
            "identity":dataset.identity,
            "samples":samples,
            "within_file_lag_pairs":samples - dataset.audio.len(),
            "power_sum_f64":power,
            "within_file_lag_product_sum_f64":lag_product,
            "concatenated_reference_descriptors":evaluate::describe(&concatenated,16000),
            "note":"RMS is sqrt(total sample power / total samples); rho uses lag products only within each decoded file and total sample power, then clamps to [-.98,.98]. Dataset identity lists each accepted source hash and rejected files. Concatenated descriptors include artificial file seams; those seams are excluded from AR1 fitting. Uniform sample weighting, not uniform file weighting.",
        });
        (rms, rho, Some(metadata))
    } else {
        // Keep the historical single-audio arithmetic and draw stream unchanged.
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
        (rms, rho, None)
    };
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
        &serde_json::json!({"reference":args.reference,"reference_sha256":data::hash_file(&args.reference)?,"preprocess":p,"reference_kind":if manifest {"manifest"} else {"single_audio"},"dataset":dataset_metadata,"sampling":sampling,"ar1_rho":rho,"ar1_rms":rms,"ar1_burnin_samples":2048,"gaussian":evaluate::describe(&gaussian,16000),"ar1":evaluate::describe(&ar1,16000),"note":"non-neural controls only; AR1 fits two training moments, Gaussian is zero learned residual; neither is presented as FERAL generation"}),
    )?;
    Ok(())
}
