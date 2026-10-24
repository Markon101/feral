//! Numerical energy-organization comparison; not a music or audio-quality score.
use anyhow::{ensure, Result};
use clap::Parser;
use feral::{
    audio::{self, Preprocess},
    data, runtime, structure,
};
use std::path::{Path, PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 7009)]
    seed: u64,
}
fn analyze(path: &Path, seed: u64, preprocess: &Preprocess) -> Result<serde_json::Value> {
    let hash = data::hash_file(path)?;
    let waveform = audio::decode(path, preprocess)?;
    let x: Vec<_> = waveform.iter().map(|&v| v as f64).collect();
    Ok(serde_json::json!({
        "path": std::fs::canonicalize(path)?, "sha256": hash,
        "comparison": structure::compare_to_phase_null(&x, preprocess.sample_rate, seed)?,
    }))
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing existing output");
    let preprocess = Preprocess {
        sample_rate: 16000,
        max_seconds: 10.,
        normalize: false,
        channel: "mix".into(),
        silence_rms: 0.,
    };
    let input = analyze(&args.input, args.seed, &preprocess)?;
    let reference = analyze(&args.reference, args.seed, &preprocess)?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let git_commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string());
    runtime::write_json(
        &args.output,
        &serde_json::json!({
            "version": structure::VERSION, "seed": args.seed,
            "unix_seconds": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
            "git_commit": git_commit, "source_hash": runtime::source_hash()?,
            "preprocess": preprocess, "input": input, "reference": reference,
            "contract": "10ms RMS envelope; partial final frame omitted unless sole frame; overlap-energy normalized autocorrelation after whole-envelope mean subtraction; modulation uses unwindowed periodic envelope DFT. Each clip has its own same-whole-clip-spectrum phase-randomized null. No normalization, clamping, or listening surrogate output. Global DFT magnitude preservation is mathematical, with measured floating-point error. Correlation, modulation and source resemblance do not demonstrate music, pitch competence, generalization, or quality. Compare multiple seeds before interpreting a phase-null difference.",
        }),
    )?;
    println!(
        "saved numerical temporal-envelope comparisons to {}",
        args.output.display()
    );
    Ok(())
}
