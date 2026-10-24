//! Temporal-envelope diagnostics with a whole-clip spectrum-matched null.
//!
//! These describe energy organization, not music, pitch competence, or quality.
//! Phase randomization preserves each whole-clip DFT magnitude mathematically;
//! FFT/IFFT roundoff is reported explicitly. No normalization, limiter, or window
//! is applied. The periodic-clip DFT assumption can create edge artifacts.
use crate::rng::Rng;
use anyhow::{ensure, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Serialize;

pub const VERSION: &str = "envelope-phase-null-v1";
const MAX_SAMPLES: usize = 160_000;

#[derive(Debug, Serialize)]
pub struct LagCorrelation {
    pub milliseconds: usize,
    pub frame_lag: usize,
    pub overlapping_pairs: usize,
    /// None when the clip is too short; zero for a constant/silent envelope.
    pub correlation: Option<f64>,
}
#[derive(Debug, Serialize)]
pub struct Envelope {
    pub samples: usize,
    pub frame_samples: usize,
    pub frames: usize,
    pub waveform_rms: f64,
    pub mean: f64,
    pub standard_deviation: f64,
    pub coefficient_of_variation: f64,
    pub quantiles_05_25_50_75_95: [f64; 5],
    pub autocorrelation: Vec<LagCorrelation>,
    /// Fraction of demeaned envelope DFT power in [0.5,2), [2,8), [8,20) Hz.
    /// Other frequencies are retained in the denominator. Zero if constant.
    pub modulation_band_fractions: [f64; 3],
}
#[derive(Debug, Serialize)]
pub struct NullComparison {
    pub seed: u64,
    pub original: Envelope,
    pub phase_randomized: Envelope,
    pub relative_rms_error: f64,
    pub relative_dft_magnitude_l2_error: f64,
    pub relative_dft_power_l2_error: f64,
    pub maximum_dft_magnitude_absolute_error: f64,
    pub surrogate_peak_absolute: f64,
}
fn validate(x: &[f64]) -> Result<()> {
    ensure!(
        !x.is_empty() && x.len() <= MAX_SAMPLES,
        "expected 1..160000 samples"
    );
    ensure!(
        x.iter().all(|v| v.is_finite() && v.abs() <= 1e6),
        "nonfinite or extreme diagnostic input"
    );
    Ok(())
}
fn spectrum(x: &[f64]) -> Vec<Complex<f64>> {
    let mut values: Vec<_> = x.iter().map(|&re| Complex::new(re, 0.)).collect();
    FftPlanner::<f64>::new()
        .plan_fft_forward(x.len())
        .process(&mut values);
    values
}
/// Randomize positive-frequency phases, enforcing conjugate symmetry explicitly.
/// DC and the even-length Nyquist coefficient remain real and unchanged. Returns
/// f64 samples to keep numerical preservation checks independent of WAV quantizing.
pub fn phase_randomized(x: &[f64], seed: u64) -> Result<Vec<f64>> {
    validate(x)?;
    let mut bins = spectrum(x);
    let mut rng = Rng::new(seed);
    for k in 1..x.len().div_ceil(2) {
        let phase = ((rng.bits() >> 11) as f64 / (1u64 << 53) as f64) * std::f64::consts::TAU;
        bins[k] = Complex::from_polar(bins[k].norm(), phase);
        bins[x.len() - k] = bins[k].conj();
    }
    bins[0].im = 0.;
    if x.len().is_multiple_of(2) {
        bins[x.len() / 2].im = 0.;
    }
    FftPlanner::<f64>::new()
        .plan_fft_inverse(x.len())
        .process(&mut bins);
    let output: Vec<f64> = bins.iter().map(|v| v.re / x.len() as f64).collect();
    ensure!(output.iter().all(|v| v.is_finite()), "nonfinite surrogate");
    Ok(output)
}

pub fn envelope(x: &[f64], sample_rate: u32) -> Result<Envelope> {
    validate(x)?;
    ensure!((100..=192000).contains(&sample_rate), "invalid sample rate");
    let frame_samples = ((sample_rate as f64 * 0.01).round() as usize).max(1);
    // Omit a partial final frame unless it is the only frame. The last short
    // group must not receive the same modulation weight as a full 10ms frame.
    let full = x.len() / frame_samples;
    let frame: Vec<f64> = if full == 0 {
        vec![(x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()]
    } else {
        x[..full * frame_samples]
            .chunks_exact(frame_samples)
            .map(|p| (p.iter().map(|v| v * v).sum::<f64>() / frame_samples as f64).sqrt())
            .collect()
    };
    let mean = frame.iter().sum::<f64>() / frame.len() as f64;
    let centered: Vec<_> = frame.iter().map(|v| v - mean).collect();
    let variance = centered.iter().map(|v| v * v).sum::<f64>() / frame.len() as f64;
    let standard_deviation = variance.sqrt();
    let mut sorted = frame.clone();
    sorted.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        let at = p * (sorted.len() - 1) as f64;
        let lo = at.floor() as usize;
        let hi = at.ceil() as usize;
        sorted[lo] * (1. - at.fract()) + sorted[hi] * at.fract()
    };
    let mut autocorrelation = vec![];
    for milliseconds in [50, 100, 250, 500, 1000] {
        let lag = ((milliseconds as f64 * sample_rate as f64 / 1000.) / frame_samples as f64)
            .round() as usize;
        let pairs = frame.len().saturating_sub(lag);
        let correlation = if pairs == 0 {
            None
        } else {
            let a = &centered[..pairs];
            let b = &centered[lag..];
            let norm = (a.iter().map(|v| v * v).sum::<f64>()
                * b.iter().map(|v| v * v).sum::<f64>())
            .sqrt();
            Some(if norm > 1e-24 {
                (a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>() / norm).clamp(-1., 1.)
            } else {
                0.
            })
        };
        autocorrelation.push(LagCorrelation {
            milliseconds,
            frame_lag: lag,
            overlapping_pairs: pairs,
            correlation,
        });
    }
    let bins = spectrum(&centered);
    let mut bands = [0.; 3];
    let mut total = 0.;
    for (k, bin) in bins.iter().enumerate().take(frame.len() / 2 + 1).skip(1) {
        let hz = k as f64 * sample_rate as f64 / frame_samples as f64 / frame.len() as f64;
        let power = bin.norm_sqr()
            * if frame.len().is_multiple_of(2) && k == frame.len() / 2 {
                1.
            } else {
                2.
            };
        total += power;
        for (j, (lo, hi)) in [(0.5, 2.), (2., 8.), (8., 20.)].into_iter().enumerate() {
            if hz >= lo && hz < hi {
                bands[j] += power;
            }
        }
    }
    if total > 1e-24 {
        for band in &mut bands {
            *band /= total;
        }
    }
    Ok(Envelope {
        samples: x.len(),
        frame_samples,
        frames: frame.len(),
        waveform_rms: (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt(),
        mean,
        standard_deviation,
        coefficient_of_variation: if mean > 1e-12 {
            standard_deviation / mean
        } else {
            0.
        },
        quantiles_05_25_50_75_95: [
            quantile(0.05),
            quantile(0.25),
            quantile(0.5),
            quantile(0.75),
            quantile(0.95),
        ],
        autocorrelation,
        modulation_band_fractions: bands,
    })
}

pub fn compare_to_phase_null(x: &[f64], sample_rate: u32, seed: u64) -> Result<NullComparison> {
    let surrogate = phase_randomized(x, seed)?;
    let original = envelope(x, sample_rate)?;
    let phase_randomized = envelope(&surrogate, sample_rate)?;
    let a = spectrum(x);
    let b = spectrum(&surrogate);
    let mut magnitude_error = 0.;
    let mut magnitude_scale = 0.;
    let mut power_error = 0.;
    let mut power_scale = 0.;
    let mut maximum: f64 = 0.;
    for (a, b) in a.iter().zip(&b) {
        let (am, bm) = (a.norm(), b.norm());
        let (ap, bp) = (a.norm_sqr(), b.norm_sqr());
        magnitude_error += (am - bm).powi(2);
        magnitude_scale += am * am;
        power_error += (ap - bp).powi(2);
        power_scale += ap * ap;
        maximum = maximum.max((am - bm).abs());
    }
    Ok(NullComparison {
        seed,
        relative_rms_error: if original.waveform_rms > 0. {
            (original.waveform_rms - phase_randomized.waveform_rms).abs() / original.waveform_rms
        } else {
            0.
        },
        relative_dft_magnitude_l2_error: if magnitude_scale > 0. {
            (magnitude_error / magnitude_scale).sqrt()
        } else {
            0.
        },
        relative_dft_power_l2_error: if power_scale > 0. {
            (power_error / power_scale).sqrt()
        } else {
            0.
        },
        maximum_dft_magnitude_absolute_error: maximum,
        surrogate_peak_absolute: surrogate.iter().map(|v| v.abs()).fold(0., f64::max),
        original,
        phase_randomized,
    })
}
