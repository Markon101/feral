//! Standalone research prototype; not linked into FERAL training.
//!
//! Let d_i = prediction_i - target_i and U = {i: known_i == 0}.
//! Fine loss: sum_U d_i^2 / |U|.
//! For each S in [2, 8, 32, 128], partition sample positions into contiguous,
//! nonoverlapping groups aligned to crop sample zero. Only groups containing
//! unknown samples are active. With K_g actual unknown samples and G active groups:
//!
//!   L_S = (1/G) sum_g (sum_{i in U_g} d_i)^2 / K_g
//!   dL_S/dd_i = 2 * sum_{j in U_g} d_j / (K_g * G), for i unknown.
//!
//! Total loss is L_fine + (weight / 4) * sum_S L_S. Protected samples have zero
//! derivative. Empty/all-known inputs define every loss and gradient as zero.
//!
//! Normalization is deliberate: each group term is the squared projection onto
//! its unit-norm constant vector, NOT the squared arithmetic mean. Independent
//! unit-variance residuals have expected L_S=1; a constant residual c has
//! L_S=c^2 * (N_unknown/G). With full unmasked groups this equals S*c^2.
//! Thus large-scale coherent errors receive much more weight than alternating
//! errors. Partial groups use actual K and masks split no positions/phase.
//!
//! For a fixed input/time/mask, the unknown-coordinate quadratic form is positive
//! definite because of the fine identity term. For nonnegative fixed weight the
//! unrestricted conditional-mean minimizer is unchanged: E[d|input]=0. This
//! observation does NOT ensure equivalent finite-network fits/optimization, audible
//! quality, or a solution to multimodal generation. Crop-aligned pooling remains
//! shift sensitive. Scale weights can dominate (constant full-group multiplier
//! 1 + weight*(2+8+32+128)/4 = 1+42.5*weight). Global gradient clipping may erase
//! the intended gain or change the effective learning rate.
//!
//! Every accumulation, division, loss and gradient is f32; no FFT, phase discard,
//! spectral magnitude target, filter synthesis, or hidden f64 training arithmetic.
//! Weight zero returns immediately after the fine pass for exact baseline parity.
//! Compile independently: rustc --test experiments/multiscale_objective.rs -O ...

pub const SCALES: [usize; 4] = [2, 8, 32, 128];

#[derive(Debug, PartialEq)]
pub struct LossAndGradient {
    pub fine: f32,
    pub coarse: [f32; 4],
    pub total: f32,
    pub gradient: Vec<f32>,
}

/// Known mask must be exactly binary: 1 protects, 0 is unknown. Nonfinite unknown
/// residuals and invalid weights are rejected. Protected residuals are unread.
pub fn loss_and_gradient(
    delta: &[f32],
    known: &[f32],
    weight: f32,
) -> Result<LossAndGradient, &'static str> {
    if delta.len() != known.len() {
        return Err("residual and mask lengths differ");
    }
    if !weight.is_finite() || weight < 0.0 {
        return Err("weight must be finite and nonnegative");
    }
    if known.iter().any(|&m| m != 0.0 && m != 1.0) {
        return Err("known mask must be binary");
    }
    let unknown = known.iter().filter(|&&m| m == 0.0).count();
    let mut out = LossAndGradient {
        fine: 0.0,
        coarse: [0.0; 4],
        total: 0.0,
        gradient: vec![0.0; delta.len()],
    };
    if unknown == 0 {
        return Ok(out);
    }
    let denominator = unknown as f32;
    for i in 0..delta.len() {
        if known[i] == 0.0 {
            if !delta[i].is_finite() {
                return Err("nonfinite unknown residual");
            }
            out.fine += delta[i] * delta[i];
            out.gradient[i] = 2.0 * delta[i] / denominator;
        }
    }
    out.fine /= denominator;
    out.total = out.fine;
    if !out.total.is_finite() || out.gradient.iter().any(|g| !g.is_finite()) {
        return Err("fine loss/gradient overflow");
    }
    if weight == 0.0 {
        return Ok(out);
    }
    let scale_weight = weight / SCALES.len() as f32;
    for (scale_index, &scale) in SCALES.iter().enumerate() {
        let active_groups = known.chunks(scale).filter(|m| m.contains(&0.0)).count();
        let groups_denominator = active_groups as f32;
        let mut coarse_sum = 0.0;
        for start in (0..delta.len()).step_by(scale) {
            let end = (start + scale).min(delta.len());
            let count = known[start..end].iter().filter(|&&m| m == 0.0).count();
            if count == 0 {
                continue;
            }
            let mut sum = 0.0;
            for i in start..end {
                if known[i] == 0.0 {
                    sum += delta[i];
                }
            }
            coarse_sum += sum * sum / count as f32;
            let derivative = scale_weight * 2.0 * (sum / count as f32) / groups_denominator;
            for (i, grad) in out.gradient.iter_mut().enumerate().take(end).skip(start) {
                if known[i] == 0.0 {
                    *grad += derivative;
                }
            }
        }
        out.coarse[scale_index] = coarse_sum / groups_denominator;
        out.total += scale_weight * out.coarse[scale_index];
    }
    if !out.total.is_finite()
        || out.coarse.iter().any(|v| !v.is_finite())
        || out.gradient.iter().any(|g| !g.is_finite())
    {
        return Err("multiscale loss/gradient overflow");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_mask_partial_groups_numerical_derivatives() {
        let mut delta: Vec<f32> = (0..139)
            .map(|i| ((i * 17 % 31) as f32 - 15.0) / 50.0)
            .collect();
        let known: Vec<f32> = (0..139)
            .map(|i| {
                if i % 7 == 0 || (32..40).contains(&i) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        for weight in [0.0, 0.015, 0.4] {
            let actual = loss_and_gradient(&delta, &known, weight).unwrap();
            let eps = 0.001;
            for i in 0..delta.len() {
                let old = delta[i];
                delta[i] = old + eps;
                let plus = loss_and_gradient(&delta, &known, weight).unwrap().total;
                delta[i] = old - eps;
                let minus = loss_and_gradient(&delta, &known, weight).unwrap().total;
                delta[i] = old;
                let numeric = (plus - minus) / (2.0 * eps);
                assert!(
                    (numeric - actual.gradient[i]).abs() <= 2e-5 + 0.003 * numeric.abs(),
                    "weight={weight} index={i} numeric={numeric} analytic={}",
                    actual.gradient[i]
                );
                if known[i] == 1.0 {
                    assert_eq!(actual.gradient[i].to_bits(), 0);
                }
            }
        }
    }

    #[test]
    fn constants_show_scale_energy_and_alternating_samples_cancel() {
        let known = vec![0.0; 256];
        let constant = loss_and_gradient(&vec![0.25; 256], &known, 0.2).unwrap();
        assert_eq!(constant.fine, 0.0625);
        for (loss, scale) in constant.coarse.iter().zip(SCALES) {
            assert_eq!(*loss, scale as f32 * constant.fine);
        }
        assert!((constant.total - constant.fine * (1.0 + 42.5 * 0.2)).abs() < 1e-6);
        let alternating: Vec<f32> = (0..256)
            .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
            .collect();
        let result = loss_and_gradient(&alternating, &known, 0.2).unwrap();
        assert_eq!(result.coarse, [0.0; 4]);
        assert_eq!(result.total, result.fine);
    }

    #[test]
    fn actual_unknown_counts_and_empty_groups() {
        let delta = [1.0, 999.0, 3.0, -777.0, 999.0, -999.0, 2.0];
        let known = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let result = loss_and_gradient(&delta, &known, 0.4).unwrap();
        assert_eq!(result.fine, 14.0 / 3.0);
        // S=2 active groups have sums/count (1/1), (3/1), (2/1).
        assert_eq!(result.coarse[0], 14.0 / 3.0);
        // Other scales encompass all unknowns: (1+3+2)^2 / 3 =12.
        assert_eq!(&result.coarse[1..], &[12.0; 3]);
        // Partial terminal group has one sample, not padded to two.
        let partial = loss_and_gradient(&[2.0; 3], &[0.0; 3], 1.0).unwrap();
        assert_eq!(partial.coarse[0], 6.0); // (2*4 +1*4)/2 active groups.
    }

    #[test]
    fn protected_values_all_known_empty_and_zero_weight_contracts() {
        let known = [0.0, 1.0, 0.0, 1.0];
        let first = loss_and_gradient(&[0.1, 2.0, -0.3, 4.0], &known, 0.1).unwrap();
        let second = loss_and_gradient(&[0.1, f32::NAN, -0.3, f32::INFINITY], &known, 0.1).unwrap();
        assert_eq!(first, second);
        let zero = loss_and_gradient(&[f32::NAN; 3], &[1.0; 3], 0.1).unwrap();
        assert_eq!(zero.total, 0.0);
        assert_eq!(zero.gradient, vec![0.0; 3]);
        assert_eq!(loss_and_gradient(&[], &[], 0.1).unwrap().gradient, vec![]);
        let delta = [0.1f32, 3.0, -0.3, 8.0];
        let result = loss_and_gradient(&delta, &known, 0.0).unwrap();
        let mut fine = 0.0f32;
        for (&d, &m) in delta.iter().zip(&known) {
            if m == 0.0 {
                fine += d * d;
            }
        }
        fine /= 2.0;
        assert_eq!(result.total.to_bits(), fine.to_bits());
        assert_eq!(result.coarse, [0.0; 4]);
        assert_eq!(
            result.gradient[0].to_bits(),
            (2.0 * delta[0] / 2.0).to_bits()
        );
        assert_eq!(
            result.gradient[2].to_bits(),
            (2.0 * delta[2] / 2.0).to_bits()
        );
        assert_eq!(result, loss_and_gradient(&delta, &known, -0.0).unwrap());
    }

    #[test]
    fn invalid_inputs_and_overflow_are_rejected() {
        assert!(loss_and_gradient(&[1.0], &[], 1.0).is_err());
        for weight in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(loss_and_gradient(&[1.0], &[0.0], weight).is_err());
        }
        assert!(loss_and_gradient(&[1.0], &[0.5], 1.0).is_err());
        assert!(loss_and_gradient(&[f32::NAN], &[0.0], 1.0).is_err());
        assert!(loss_and_gradient(&[f32::MAX], &[0.0], 1.0).is_err());
    }
}
