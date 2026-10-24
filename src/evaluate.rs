//! Bounded, deterministic held-out diagnostics. None of these is a perceptual score.
use crate::{
    audio::Preprocess,
    data::{self, Dataset},
    model::Velocity,
    rng::Rng,
    runtime::{self, Checkpoint},
};
use anyhow::{ensure, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};
#[derive(Debug, Serialize, Deserialize)]
pub struct Acoustic {
    pub mse: f64,
    pub derivative_mse: f64,
    pub log_spectral_l1: f64,
    pub coarse_mse: Vec<f64>,
}
/// Multiresolution Hann STFT log(1+|FFT|) distance, FFT sizes 128/512/2048,
/// hop=N/4. Tail frames are zero padded. Paired signal metric, not a MOS estimate.
pub fn acoustic(reference: &[f32], estimate: &[f32]) -> Acoustic {
    assert!(!reference.is_empty() && reference.len() == estimate.len());
    let mse = reference
        .iter()
        .zip(estimate)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>()
        / reference.len() as f64;
    let derivative_mse = (1..reference.len())
        .map(|i| {
            ((reference[i] - reference[i - 1]) as f64 - (estimate[i] - estimate[i - 1]) as f64)
                .powi(2)
        })
        .sum::<f64>()
        / (reference.len() - 1).max(1) as f64;
    let coarse_mse = [4, 16, 64]
        .iter()
        .map(|&scale| {
            let mut sum = 0.;
            let mut count = 0;
            for (a, b) in reference.chunks(scale).zip(estimate.chunks(scale)) {
                let d = a
                    .iter()
                    .zip(b)
                    .map(|(&a, &b)| a as f64 - b as f64)
                    .sum::<f64>()
                    / a.len() as f64;
                sum += d * d;
                count += 1;
            }
            sum / count as f64
        })
        .collect();
    let mut planner = FftPlanner::new();
    let mut spectral = 0.;
    for size in [128, 512, 2048] {
        let fft = planner.plan_fft_forward(size);
        let mut sum = 0.;
        let mut count = 0;
        for start in (0..reference.len()).step_by(size / 4) {
            let windowed = |signal: &[f32]| {
                (0..size)
                    .map(|j| {
                        Complex::new(
                            signal.get(start + j).copied().unwrap_or(0.)
                                * (0.5
                                    - 0.5 * (std::f32::consts::TAU * j as f32 / size as f32).cos()),
                            0.,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let mut a = windowed(reference);
            let mut b = windowed(estimate);
            fft.process(&mut a);
            fft.process(&mut b);
            for (a, b) in a[..=size / 2].iter().zip(&b[..=size / 2]) {
                sum += (a.norm().ln_1p() as f64 - b.norm().ln_1p() as f64).abs();
                count += 1;
            }
        }
        spectral += sum / count as f64;
    }
    Acoustic {
        mse,
        derivative_mse,
        log_spectral_l1: spectral / 3.,
        coarse_mse,
    }
}
#[derive(Serialize)]
pub struct Probe {
    pub source_index: usize,
    pub crop_start: usize,
    pub task: &'static str,
    pub t: f32,
    pub velocity_mse: f64,
    pub zero_velocity_mse: f64,
    pub endpoint: Acoustic,
    pub zero_audio: Acoustic,
    pub noisy_input: Acoustic,
}
/// Explicit tasks and known intervals make comparisons independent of architecture/RNG use.
pub fn probe<M: Velocity>(model: &M, clean: &[f32], noise: &[f32], task: usize, t: f32) -> Probe {
    let n = clean.len();
    let mut mask = vec![0.; n];
    let name = match task {
        0 => "unconditional",
        1 => {
            mask[..n / 2].fill(1.);
            "prefix"
        }
        2 => {
            mask.fill(1.);
            mask[n / 4..3 * n / 4].fill(0.);
            "hole"
        }
        _ => {
            mask[n / 2..].fill(1.);
            "suffix"
        }
    };
    let x: Vec<f32> = clean
        .iter()
        .zip(noise)
        .zip(&mask)
        .map(|((&y, &z), &m)| if m == 1. { y } else { (1. - t) * z + t * y })
        .collect();
    let pred = model.velocity(&x, clean, &mask, t);
    assert_eq!(pred.len(), n);
    let count = mask.iter().filter(|&&m| m == 0.).count() as f64;
    let velocity_mse = pred
        .iter()
        .zip(clean)
        .zip(noise)
        .zip(&mask)
        .filter(|(_, m)| **m == 0.)
        .map(|(((&v, &y), &z), _)| (v as f64 - (y - z) as f64).powi(2))
        .sum::<f64>()
        / count;
    let zero_velocity_mse = clean
        .iter()
        .zip(noise)
        .zip(&mask)
        .filter(|(_, m)| **m == 0.)
        .map(|((&y, &z), _)| ((y - z) as f64).powi(2))
        .sum::<f64>()
        / count;
    let unknown = |v: &[f32]| {
        v.iter()
            .zip(&mask)
            .filter(|(_, m)| **m == 0.)
            .map(|(&v, _)| v)
            .collect::<Vec<_>>()
    };
    let endpoint: Vec<f32> = x.iter().zip(pred).map(|(&x, v)| x + (1. - t) * v).collect();
    let y = unknown(clean);
    let estimate = unknown(&endpoint);
    let input = unknown(&x);
    Probe {
        source_index: 0,
        crop_start: 0,
        task: name,
        t,
        velocity_mse,
        zero_velocity_mse,
        endpoint: acoustic(&y, &estimate),
        zero_audio: acoustic(&y, &vec![0.; y.len()]),
        noisy_input: acoustic(&y, &input),
    }
}
/// Nearest absolute, mean-centered normalized correlation over stride-aligned windows.
/// Decimates by 4 for bounded cost. A copying screen, not proof of non-memorization.
pub fn nearest_correlation(query: &[f32], reference: &[f32], stride: usize) -> f64 {
    if reference.len() < query.len() || query.len() < 8 {
        return 0.;
    }
    fn normalized(x: &[f32]) -> Vec<f64> {
        let mut v: Vec<f64> = x.iter().step_by(4).map(|&v| v as f64).collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        for a in &mut v {
            *a -= mean;
        }
        let norm = v.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-15);
        for a in &mut v {
            *a /= norm;
        }
        v
    }
    let q = normalized(query);
    let mut best = 0f64;
    for start in (0..=reference.len() - query.len()).step_by(stride.max(1)) {
        let r = normalized(&reference[start..start + query.len()]);
        let c = q.iter().zip(r).map(|(a, b)| a * b).sum::<f64>().abs();
        best = best.max(c);
    }
    best
}
pub fn evaluate(
    checkpoint: &Path,
    input: &Path,
    output: &Path,
    crops: usize,
    seed: u64,
) -> Result<()> {
    ensure!((1..=32).contains(&crops), "evaluation crops must be 1..32");
    ensure!(!output.exists(), "evaluation output already exists");
    let start = Instant::now();
    let ck = Checkpoint::load(checkpoint)?;
    let p = Preprocess {
        sample_rate: ck.sample_rate,
        normalize: false,
        max_seconds: 60.,
        ..Default::default()
    };
    let dataset = Dataset::load(input, p, ck.crop_samples)?;
    let mut rng = Rng::new(seed);
    let mut rows = vec![];
    for _ in 0..crops {
        let source = rng.index(dataset.audio.len());
        let a = &dataset.audio[source];
        let at = rng.index(a.len() - ck.crop_samples + 1);
        let clean = &a[at..at + ck.crop_samples];
        let noise: Vec<f32> = (0..clean.len()).map(|_| rng.normal()).collect();
        for task in 0..4 {
            for t in [0.1, 0.5, 0.9] {
                let mut row = probe(&ck.velocity(), clean, &noise, task, t);
                row.source_index = source;
                row.crop_start = at;
                ensure!(row.velocity_mse.is_finite(), "nonfinite evaluation");
                rows.push(row);
            }
        }
    }
    let mean = rows.iter().map(|r| r.velocity_mse).sum::<f64>() / rows.len() as f64;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    runtime::write_json(
        output,
        &serde_json::json!({"checkpoint":checkpoint,"checkpoint_sha256":data::hash_file(checkpoint)?,"dataset":dataset.identity,"seed":seed,"crops":crops,"crop_samples":ck.crop_samples,"training_steps":ck.completed_steps,"flow":ck.flow,"parameters":ck.model.count(),"architecture":ck.model.architecture,"context_samples":ck.model.architecture.context_samples(),"mean_velocity_mse":mean,"rows":rows,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,"note":"same-recording held-out probes are not cross-recording or perceptual quality evidence; endpoint estimates are paired denoising, not generated samples"}),
    )?;
    eprintln!(
        "evaluation {} | {} probes | mean velocity MSE {:.6}",
        output.display(),
        crops * 12,
        mean
    );
    Ok(())
}

/// Unpaired signal descriptors. These characterize audio; they do not score quality.
pub fn describe(x: &[f32], sample_rate: u32) -> serde_json::Value {
    assert!(!x.is_empty());
    let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt();
    let derivative_rms = ((1..x.len())
        .map(|i| ((x[i] - x[i - 1]) as f64).powi(2))
        .sum::<f64>()
        / (x.len() - 1).max(1) as f64)
        .sqrt();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(512);
    let mut spectrum = vec![0f64; 256];
    let mut frames = 0f64;
    for start in (0..x.len()).step_by(256) {
        let mut frame: Vec<_> = (0..512)
            .map(|j| {
                Complex::new(
                    x.get(start + j).copied().unwrap_or(0.)
                        * (0.5 - 0.5 * (std::f32::consts::TAU * j as f32 / 512.).cos()),
                    0.,
                )
            })
            .collect();
        fft.process(&mut frame);
        for (p, c) in spectrum.iter_mut().zip(&frame[1..257]) {
            *p += c.norm_sqr() as f64;
        }
        frames += 1.;
    }
    for p in &mut spectrum {
        *p /= frames.max(1.);
    }
    let power = spectrum.iter().sum::<f64>().max(1e-20);
    let mut bands = vec![0.; 4];
    for (i, &p) in spectrum.iter().enumerate() {
        let hz = (i + 1) as f64 * sample_rate as f64 / 512.;
        let b = if hz < 250. {
            0
        } else if hz < 1000. {
            1
        } else if hz < 4000. {
            2
        } else {
            3
        };
        bands[b] += p / power;
    }
    let flatness = if power <= 1e-20 {
        0.
    } else {
        (spectrum.iter().map(|p| p.max(1e-20).ln()).sum::<f64>() / 256.).exp() / (power / 256.)
    };
    let sum = x
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .max(1e-20);
    let lag1 = (1..x.len())
        .map(|i| x[i] as f64 * x[i - 1] as f64)
        .sum::<f64>()
        / sum;
    serde_json::json!({"samples":x.len(),"sample_rate":sample_rate,"rms":rms,"derivative_rms":derivative_rms,"lag1_uncentered":lag1,"spectral_band_fractions":bands,"spectral_flatness":flatness,"limiter_hits":x.iter().filter(|v|v.abs()>=0.979999).count(),"note":"FFT512 Hann power; bands <250,250-1000,1000-4000,>=4000Hz; descriptors are not perceptual quality scores"})
}
pub fn analyze(input: &Path, reference: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "analysis output already exists");
    let p = Preprocess {
        normalize: false,
        max_seconds: 60.,
        ..Default::default()
    };
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let x = audio_decode(input, &p)?;
    let r = audio_decode(reference, &p)?;
    let mut rng = Rng::new(51);
    let mut shuffled = x.clone();
    for i in (1..shuffled.len()).rev() {
        let j = rng.index(i + 1);
        shuffled.swap(i, j);
    }
    let noise: Vec<f32> = (0..x.len()).map(|_| rng.normal()).collect();
    // A train-window copy must pass this detector; random noise is the hostile null.
    ensure!(
        r.len() >= x.len(),
        "reference must be at least as long as analyzed signal"
    );
    let copy = &r[..x.len()];
    let stride = 256;
    runtime::write_json(
        output,
        &serde_json::json!({"input":input,"input_sha256":data::hash_file(input)?,"reference_path":reference,"reference_sha256":data::hash_file(reference)?,"signal":describe(&x,p.sample_rate),"reference":describe(&r,p.sample_rate),"shuffled_signal":describe(&shuffled,p.sample_rate),"noise":describe(&noise,p.sample_rate),"nearest_absolute_correlation":nearest_correlation(&x,&r,stride),"copied_reference_control":nearest_correlation(copy,&r,stride),"noise_correlation_control":nearest_correlation(&noise,&r,stride),"correlation_stride_samples":stride,"correlation_decimation":4,"limits":"correlation is a bounded screen over stride-aligned windows; low scores do not exclude transformed memorization"}),
    )?;
    Ok(())
}
fn audio_decode(path: &Path, p: &Preprocess) -> Result<Vec<f32>> {
    crate::audio::decode(path, p)
}
