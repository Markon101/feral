//! Matched held-out probes for the analytic Gaussian reference, not neural results.
use anyhow::{ensure, Result};
use clap::Parser;
use feral::{
    audio::Preprocess, data::Dataset, evaluate, parameterization::GaussianReference, rng::Rng,
    runtime,
};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 7001)]
    seed: u64,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(!a.output.exists(), "output exists");
    let p = Preprocess {
        normalize: false,
        max_seconds: 60.,
        ..Default::default()
    };
    let data = Dataset::load(&a.input, p, 8192)?;
    let model = GaussianReference {
        block: 32,
        data_scale: 0.1,
    };
    let mut rng = Rng::new(a.seed);
    let mut rows = vec![];
    for _ in 0..4 {
        let source = rng.index(data.audio.len());
        let audio = &data.audio[source];
        let start = rng.index(audio.len() - 8192 + 1);
        let clean = &audio[start..start + 8192];
        let noise: Vec<f32> = (0..8192).map(|_| rng.normal()).collect();
        for task in 0..4 {
            for t in [0.1, 0.5, 0.9] {
                let mut row = evaluate::probe(&model, clean, &noise, task, t);
                row.source_index = source;
                row.crop_start = start;
                rows.push(row);
            }
        }
    }
    let mean = rows.iter().map(|r| r.velocity_mse).sum::<f64>() / rows.len() as f64;
    runtime::write_json(
        &a.output,
        &serde_json::json!({"control":"analytic Gaussian zero learned residual","seed":a.seed,"dataset":data.identity,"mean_velocity_mse":mean,"rows":rows}),
    )?;
    println!("Gaussian control velocity MSE {mean:.6}");
    Ok(())
}
