//! Differentiable multiresolution Hann STFT log-magnitude L1 loss on the predicted clean endpoint.
//!
//! Uses `rustfft` (already a crate dependency) with smooth Charbonnier L1 and magnitude epsilon
//! so reverse-mode gradients remain finite and smooth everywhere, including at zero residual.
use crate::parameterization::FlowConfig;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub const SIZES: [usize; 3] = [128, 512, 2048];
const MAG_EPS: f32 = 1e-5;
const L1_DELTA: f32 = 1e-5;

struct ResolutionPlan {
    size: usize,
    hop: usize,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
}

/// Cached FFT plans and Hann windows for multiresolution STFT loss and gradient computation.
pub struct SpectralEngine {
    plans: Vec<ResolutionPlan>,
}

impl Default for SpectralEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralEngine {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let plans = SIZES
            .iter()
            .map(|&size| {
                let window = (0..size)
                    .map(|j| 0.5 - 0.5 * (std::f32::consts::TAU * j as f32 / size as f32).cos())
                    .collect();
                ResolutionPlan {
                    size,
                    hop: size / 4,
                    fwd: planner.plan_fft_forward(size),
                    inv: planner.plan_fft_inverse(size),
                    window,
                }
            })
            .collect();
        Self { plans }
    }

    /// Multiresolution STFT log(1 + |FFT|_eps) smooth-L1 loss and exact gradient w.r.t. `yhat`.
    pub fn loss_and_yhat_grad(&self, clean: &[f32], yhat: &[f32]) -> (f32, Vec<f32>) {
        assert_eq!(clean.len(), yhat.len());
        let n = clean.len();
        if n == 0 {
            return (0., vec![]);
        }
        let max_size = SIZES[SIZES.len() - 1];
        let mut buf_a = vec![Complex::new(0f32, 0f32); max_size];
        let mut buf_b = vec![Complex::new(0f32, 0f32); max_size];
        let mut buf_g = vec![Complex::new(0f32, 0f32); max_size];
        let mut d_yhat = vec![0f32; n];
        let mut total_loss = 0f64;
        let eps_sq = MAG_EPS * MAG_EPS;
        let delta_sq = L1_DELTA * L1_DELTA;

        for plan in &self.plans {
            let size = plan.size;
            let bins = size / 2 + 1;
            let frame_count = n.div_ceil(plan.hop);
            let scale = 1.0f32 / (self.plans.len() as f32 * (frame_count * bins) as f32);
            let mut res_loss = 0f64;

            for start in (0..n).step_by(plan.hop) {
                let len = (n - start).min(size);
                for j in 0..len {
                    let w = plan.window[j];
                    buf_a[j] = Complex::new(yhat[start + j] * w, 0.);
                    buf_b[j] = Complex::new(clean[start + j] * w, 0.);
                }
                for j in len..size {
                    buf_a[j] = Complex::new(0., 0.);
                    buf_b[j] = Complex::new(0., 0.);
                }
                plan.fwd.process(&mut buf_a[..size]);
                plan.fwd.process(&mut buf_b[..size]);

                buf_g[bins..size].fill(Complex::new(0., 0.));
                for k in 0..bins {
                    let a = buf_a[k];
                    let b = buf_b[k];
                    let mag_a = (a.norm_sqr() + eps_sq).sqrt();
                    let mag_b = (b.norm_sqr() + eps_sq).sqrt();
                    let diff = mag_a.ln_1p() - mag_b.ln_1p();
                    let rho = (diff * diff + delta_sq).sqrt();
                    res_loss += (rho - L1_DELTA) as f64;
                    let coeff = scale * (diff / rho) / ((1.0 + mag_a) * mag_a);
                    buf_g[k] = Complex::new(a.re * coeff, a.im * coeff);
                }
                // `rustfft` inverse FFT computes sum_{k=0..size-1} G_k * exp(+i 2pi k j / size).
                // With G_k populated on one-sided bins 0..=size/2 and zero elsewhere, Re(IFFT(G)_j)
                // is the exact transpose of the real-input one-sided DFT.
                plan.inv.process(&mut buf_g[..size]);
                for j in 0..len {
                    d_yhat[start + j] += plan.window[j] * buf_g[j].re;
                }
            }
            total_loss += res_loss * scale as f64;
        }
        (total_loss as f32, d_yhat)
    }

    /// Time-weighted spectral endpoint loss `spectral_weight * t * L_stft(y, y_hat)` and its
    /// gradient w.r.t. the raw network output `pred` (zero on protected positions).
    #[allow(clippy::too_many_arguments)]
    pub fn endpoint_loss_gradient(
        &self,
        config: &FlowConfig,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        pred: &[f32],
        t: f32,
        spectral_weight: f32,
    ) -> (f32, Vec<f32>) {
        let n = clean.len();
        if spectral_weight <= 0. || t <= 0. || t >= 1. || n == 0 {
            return (0., vec![0.; n]);
        }
        let (cskip, cout, _) = config.coefficients(t);
        let omt = 1. - t;
        let mut yhat = Vec::with_capacity(n);
        for i in 0..n {
            if mask[i] == 1. {
                yhat.push(clean[i]);
            } else {
                let xt = omt * noise[i] + t * clean[i];
                let v = cskip * xt + cout * pred[i];
                yhat.push(xt + omt * v);
            }
        }
        let (base_loss, g_yhat) = self.loss_and_yhat_grad(clean, &yhat);
        let weight = spectral_weight * t;
        let grad_scale = weight * omt * cout;
        let dy = g_yhat
            .into_iter()
            .zip(mask)
            .map(|(g, &m)| if m == 0. { g * grad_scale } else { 0. })
            .collect();
        (weight * base_loss, dy)
    }

    /// Batched spectral endpoint loss and gradient, evaluated in parallel across batch items.
    /// Returns the *sum* of per-item spectral losses (matching CUDA's `loss_parts` accumulation,
    /// which `feral_statistics` divides by `items`).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_endpoint_loss_gradient(
        &self,
        config: &FlowConfig,
        clean: &[f32],
        mask: &[f32],
        noise: &[f32],
        pred: &[f32],
        t: &[f32],
        spectral_weight: f32,
        dy_out: &mut [f32],
    ) -> f32 {
        let items = t.len();
        if items == 0 || spectral_weight <= 0. {
            dy_out.fill(0.);
            return 0.;
        }
        let n = clean.len() / items;
        assert_eq!(dy_out.len(), items * n);
        if items == 1 {
            let (loss, dy) = self.endpoint_loss_gradient(
                config,
                clean,
                mask,
                noise,
                pred,
                t[0],
                spectral_weight,
            );
            dy_out.copy_from_slice(&dy);
            return loss;
        }
        let mut losses = vec![0f32; items];
        std::thread::scope(|scope| {
            for (item, (loss_slot, dy_slice)) in losses
                .iter_mut()
                .zip(dy_out.chunks_exact_mut(n))
                .enumerate()
            {
                let row = item * n..(item + 1) * n;
                let ti = t[item];
                scope.spawn(move || {
                    let (loss, dy) = self.endpoint_loss_gradient(
                        config,
                        &clean[row.clone()],
                        &mask[row.clone()],
                        &noise[row.clone()],
                        &pred[row],
                        ti,
                        spectral_weight,
                    );
                    *loss_slot = loss;
                    dy_slice.copy_from_slice(&dy);
                });
            }
        });
        losses.into_iter().sum()
    }
}
