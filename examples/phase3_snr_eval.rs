//! Common corruption levels for comparisons across differently scaled priors.
use anyhow::{ensure, Result};
use clap::Parser;
use feral::{
    audio::Preprocess,
    data::Dataset,
    evaluate,
    rng::Rng,
    runtime::{self, Checkpoint},
};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(!a.output.exists(), "output exists");
    let ck = Checkpoint::load(&a.checkpoint)?;
    let dataset = Dataset::load(
        &a.input,
        Preprocess {
            normalize: false,
            max_seconds: 60.,
            ..Default::default()
        },
        8192,
    )?;
    let mut rng = Rng::new(7001);
    let mut rows = vec![];
    for _ in 0..4 {
        let source = rng.index(dataset.audio.len());
        let audio = &dataset.audio[source];
        let at = rng.index(audio.len() - 8192 + 1);
        let clean = &audio[at..at + 8192];
        let eps: Vec<f32> = (0..8192).map(|_| rng.normal()).collect();
        let noise: Vec<f32> = eps.iter().map(|v| v * ck.flow.prior_std()).collect();
        for task in 0..4 {
            for sigma in [0.02, 0.1, 0.5] {
                let t = ck.flow.prior_std() / (ck.flow.prior_std() + sigma);
                let mut row = evaluate::probe(&ck.velocity(), clean, &noise, task, t);
                row.source_index = source;
                row.crop_start = at;
                let range = match task {
                    0 => 0..8192,
                    1 => 4096..8192,
                    2 => 2048..6144,
                    _ => 0..4096,
                };
                let common_noisy_mse = eps[range.clone()]
                    .iter()
                    .map(|v| (*v as f64 * sigma as f64).powi(2))
                    .sum::<f64>()
                    / range.len() as f64;
                rows.push(serde_json::json!({"corruption_std":sigma,"probe":row,"common_noisy_mse":common_noisy_mse}));
            }
        }
    }
    let mean = rows
        .iter()
        .map(|r| r["probe"]["endpoint"]["mse"].as_f64().unwrap())
        .sum::<f64>()
        / rows.len() as f64;
    runtime::write_json(
        &a.output,
        &serde_json::json!({"checkpoint":a.checkpoint,"flow":ck.flow,"dataset":dataset.identity,"seed":7001,"mean_endpoint_mse":mean,"rows":rows,"note":"same y+sigma*epsilon at corruption std .02,.1,.5; t depends on prior; compare endpoint metrics, not velocity MSE or path-input metrics across priors"}),
    )?;
    println!("common-corruption endpoint MSE {mean:.6}");
    Ok(())
}
