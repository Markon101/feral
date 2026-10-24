use feral::{rng::Rng, structure};

#[test]
fn phase_null_is_seeded_real_and_preserves_whole_clip_spectrum() {
    for n in [1usize, 2, 3, 127, 128, 1023, 1024] {
        let mut rng = Rng::new(91);
        let x: Vec<f64> = (0..n)
            .map(|i| 0.3 + rng.normal() as f64 * 0.2 + if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let a = structure::phase_randomized(&x, 12).unwrap();
        assert_eq!(a, structure::phase_randomized(&x, 12).unwrap());
        assert_eq!(a.len(), x.len());
        assert!(a.iter().all(|v| v.is_finite()));
        if n > 3 {
            assert_ne!(a, structure::phase_randomized(&x, 13).unwrap());
        }
        let report = structure::compare_to_phase_null(&x, 16000, 12).unwrap();
        assert!(report.relative_rms_error < 1e-12);
        assert!(report.relative_dft_magnitude_l2_error < 1e-12);
        assert!(report.relative_dft_power_l2_error < 1e-12);
        let mean = |y: &[f64]| y.iter().sum::<f64>() / y.len() as f64;
        assert!((mean(&a) - mean(&x)).abs() < 1e-12);
        if n.is_multiple_of(2) {
            let nyquist = |y: &[f64]| {
                y.iter()
                    .enumerate()
                    .map(|(i, v)| if i % 2 == 0 { *v } else { -v })
                    .sum::<f64>()
            };
            assert!((nyquist(&a) - nyquist(&x)).abs() < 1e-10);
        }
    }
}

#[test]
fn amplitude_modulation_is_distinguished_from_a_spectrum_matched_stationary_null() {
    let mut rng = Rng::new(901);
    let stationary: Vec<f64> = (0..64000).map(|_| rng.normal() as f64 * 0.1).collect();
    let modulated: Vec<_> = stationary
        .iter()
        .enumerate()
        .map(|(i, v)| v * (1. + 0.8 * (std::f64::consts::TAU * 2. * i as f64 / 16000.).sin()))
        .collect();
    let report = structure::compare_to_phase_null(&modulated, 16000, 15).unwrap();
    let flat = structure::envelope(&stationary, 16000).unwrap();
    println!(
        "AM CV={} phase-null CV={} stationary CV={}, AM ac250ms={:?}, null ac250ms={:?}",
        report.original.coefficient_of_variation,
        report.phase_randomized.coefficient_of_variation,
        flat.coefficient_of_variation,
        report.original.autocorrelation[2].correlation,
        report.phase_randomized.autocorrelation[2].correlation
    );
    assert!(report.original.coefficient_of_variation > 0.5);
    assert!(report.phase_randomized.coefficient_of_variation < 0.12);
    assert!(flat.coefficient_of_variation < 0.12);
    assert!(report.original.autocorrelation[2].correlation.unwrap() < -0.9);
    assert!(report.original.autocorrelation[3].correlation.unwrap() > 0.9);
    assert!(report.original.modulation_band_fractions[1] > 0.9);
    assert!(report.relative_dft_power_l2_error < 1e-12);
}

#[test]
fn silence_short_frames_and_nonfinite_inputs_are_explicit() {
    for n in [1, 17, 160, 161, 1600] {
        let report = structure::compare_to_phase_null(&vec![0.; n], 16000, 3).unwrap();
        assert_eq!(report.relative_rms_error, 0.);
        assert_eq!(report.relative_dft_power_l2_error, 0.);
        assert_eq!(report.original.coefficient_of_variation, 0.);
        assert_eq!(report.original.modulation_band_fractions, [0.; 3]);
        assert!(report
            .original
            .autocorrelation
            .iter()
            .all(|a| a.correlation.is_none_or(|v| v == 0.)));
        assert!(serde_json::to_string(&report).is_ok());
    }
    for bad in [vec![], vec![f64::NAN], vec![f64::INFINITY], vec![1e7]] {
        assert!(structure::phase_randomized(&bad, 1).is_err());
        assert!(structure::envelope(&bad, 16000).is_err());
    }
    assert!(structure::envelope(&[0.; 10], 0).is_err());
    assert!(structure::phase_randomized(&vec![0.; 160001], 1).is_err());
}
