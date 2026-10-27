use feral::{
    model::{Architecture, Model},
    parameterization::{self, FlowConfig},
    rng::Rng,
    runtime::Adam,
};
use std::process::Command;

#[test]
fn default_width_preserves_serialized_parameters_and_forward_outputs() {
    // This checks the new selection path against the unchanged legacy constructors.
    for architecture in [Architecture::default(), Architecture::pyramid()] {
        let selected = architecture.clone().with_width(64).unwrap();
        assert_eq!(
            serde_json::to_string(&architecture).unwrap(),
            serde_json::to_string(&selected).unwrap()
        );
        let original = Model::new(architecture, 42);
        let selected = Model::new(selected, 42);
        assert_eq!(original.count(), 57_952);
        assert_eq!(original.parameters(), selected.parameters());
        let x: Vec<f32> = (0..96).map(|i| (i as f32 * 0.17).sin() * 0.1).collect();
        let mask = vec![0.; x.len()];
        assert_eq!(
            original.forward(&x, &x, &mask, 0.37).0,
            selected.forward(&x, &x, &mask, 0.37).0
        );
    }
}

#[test]
fn scaled_networks_have_correct_shapes_gradients_and_updates() {
    for architecture in [Architecture::default(), Architecture::pyramid()] {
        let architecture = architecture.with_width(128).unwrap();
        architecture.validate().unwrap();
        let mut model = Model::new(architecture, 71);
        assert_eq!(model.count(), 214_176);
        let initial = model.parameters();
        let clean: Vec<f32> = (0..96).map(|i| (i as f32 * 0.13).sin() * 0.1).collect();
        let mut rng = Rng::new(73);
        let noise: Vec<f32> = (0..clean.len()).map(|_| rng.normal() * 0.1).collect();
        let mut mask = vec![0.; clean.len()];
        mask[..32].fill(1.);
        assert_eq!(
            model.forward(&noise, &clean, &mask, 0.37).0.len(),
            clean.len()
        );
        let config = FlowConfig::Preconditioned {
            data_scale: 0.1,
            prior_std: 0.1,
        };
        let (loss, gradient) =
            parameterization::loss_gradient(&model, &config, &clean, &mask, &noise, 0.37);
        assert!(loss.is_finite());
        assert_eq!(gradient.len(), model.count());
        assert!(gradient.iter().all(|g| g.is_finite()));
        let mut offset = 0;
        for layer in &model.layers {
            let weight_end = offset + layer.w.len();
            // Test the strongest weight derivative and a bias in every layer, avoiding
            // empty/padded kernel positions in this deliberately short CPU fixture.
            let weight = (offset..weight_end)
                .max_by(|&a, &b| gradient[a].abs().total_cmp(&gradient[b].abs()))
                .unwrap();
            let bias = (weight_end..weight_end + layer.b.len())
                .max_by(|&a, &b| gradient[a].abs().total_cmp(&gradient[b].abs()))
                .unwrap();
            assert!(gradient[weight].abs() > 1e-7);
            for index in [weight, bias] {
                let mut shifted = initial.clone();
                shifted[index] += 0.002;
                let mut perturbed = model.clone();
                perturbed.set_parameters(&shifted);
                let high = parameterization::loss_gradient(
                    &perturbed, &config, &clean, &mask, &noise, 0.37,
                )
                .0;
                shifted[index] -= 0.004;
                perturbed.set_parameters(&shifted);
                let low = parameterization::loss_gradient(
                    &perturbed, &config, &clean, &mask, &noise, 0.37,
                )
                .0;
                let numerical = (high - low) / 0.004;
                assert!(
                    (gradient[index] - numerical).abs() < 0.001 + 0.03 * numerical.abs(),
                    "index={index}, analytic={}, numerical={numerical}",
                    gradient[index]
                );
            }
            offset = weight_end + layer.b.len();
        }
        let mut adam = Adam::new(model.count(), 0.0003);
        assert!(adam.update(&mut model, &gradient).unwrap() > 0.);
        assert_ne!(model.parameters(), initial);
        assert!(model.parameters().iter().all(|p| p.is_finite()));
        assert_eq!(adam.step, 1);
    }
    let maximum = Architecture::pyramid().with_width(256).unwrap();
    assert_eq!(Model::new(maximum, 1).count(), 821_536);
}

#[test]
fn training_width_rejects_unsupported_sizes_before_input_io() {
    for width in ["0", "16", "33", "127", "257", "528", "NaN"] {
        let output = Command::new(env!("CARGO_BIN_EXE_feral"))
            .args([
                "train",
                "--input",
                "/nonexistent-feral-scaling-input",
                "--run",
                "/nonexistent-feral-scaling-run",
                "--width",
                width,
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("--width"));
    }
    for width in [32, 48, 64, 128, 256, 512] {
        assert!(Architecture::default().with_width(width).is_ok());
    }
    // `validate` bounds the size (1..=512) and stays permissive for the tiny architectures the
    // numerical tests use; the multiple-of-16 training rule lives in `with_width`, which the CLI
    // parsing above exercises.
    for width in [513, 1024] {
        assert!(Architecture {
            width,
            ..Architecture::default()
        }
        .validate()
        .is_err());
    }
}

#[test]
fn detail_weight_keeps_the_gradient_exact() {
    // The detail term is a stencil on the residual; its analytic gradient must match finite
    // differences, and detail = 0 must reproduce the plain objective bit for bit.
    for config in [
        FlowConfig::Velocity,
        FlowConfig::Preconditioned {
            data_scale: 0.16,
            prior_std: 1.,
        },
    ] {
        let model = Model::new(Architecture::default().with_width(64).unwrap(), 71);
        let clean: Vec<f32> = (0..256).map(|i| (i as f32 * 0.13).sin() * 0.2).collect();
        let mut rng = Rng::new(73);
        let noise: Vec<f32> = (0..clean.len()).map(|_| rng.normal()).collect();
        // Unknown samples in the middle of the window, so there are many valid difference pairs.
        let mut mask = vec![1f32; clean.len()];
        mask[64..192].fill(0.);

        let (plain, plain_grad) =
            parameterization::loss_gradient(&model, &config, &clean, &mask, &noise, 0.37);
        let (zero, zero_grad) = parameterization::loss_gradient_detailed(
            &model, &config, &clean, &mask, &noise, 0.37, 0.,
        );
        assert_eq!(
            plain.to_bits(),
            zero.to_bits(),
            "detail=0 must not change the loss"
        );
        assert_eq!(
            plain_grad, zero_grad,
            "detail=0 must not change the gradient"
        );

        let detail = 2.0f32;
        let (loss, gradient) = parameterization::loss_gradient_detailed(
            &model, &config, &clean, &mask, &noise, 0.37, detail,
        );
        assert!(loss.is_finite());
        assert!(
            loss > plain,
            "the detail term must add to the loss: {loss} vs {plain}"
        );

        let eps = 0.004f32;
        let mut checked = 0;
        let mut offset = 0;
        for layer in &model.layers {
            let weight_end = offset + layer.w.len();
            // Strongest weight plus a bias in each layer is enough to exercise the stencil.
            let index = offset
                + layer
                    .w
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
            for probe in [index, weight_end] {
                let mut high = model.clone();
                let mut params = high.parameters();
                params[probe] += eps;
                high.set_parameters(&params);
                let high_loss = parameterization::loss_gradient_detailed(
                    &high, &config, &clean, &mask, &noise, 0.37, detail,
                )
                .0;
                let mut low = model.clone();
                let mut params = low.parameters();
                params[probe] -= eps;
                low.set_parameters(&params);
                let low_loss = parameterization::loss_gradient_detailed(
                    &low, &config, &clean, &mask, &noise, 0.37, detail,
                )
                .0;
                let numerical = (high_loss - low_loss) / (2. * eps);
                assert!(
                    (gradient[probe] - numerical).abs() < 0.002 + 0.05 * numerical.abs(),
                    "detail gradient mismatch for {config:?} at {probe}: analytic {} numerical {numerical}",
                    gradient[probe]
                );
                checked += 1;
            }
            offset = weight_end + layer.b.len();
        }
        assert!(
            checked >= 4,
            "expected several gradient probes, got {checked}"
        );
    }
}

#[test]
fn frame_samples_and_dilations_cover_the_long_context_range() {
    // Frames trade time resolution for context: convolution cost scales with frames, so a larger
    // block is cheaper per sample and multiplies the receptive field by the same factor.
    assert!(Architecture::default().with_frame_samples(16).is_ok());
    assert!(Architecture::default().with_frame_samples(128).is_ok());
    for block in [8, 256] {
        assert!(Architecture::default().with_frame_samples(block).is_err());
    }
    for blocks in [1usize, 7, 8, 10] {
        assert!(Architecture::default().with_blocks(blocks).is_ok());
    }
    assert!(Architecture::default().with_blocks(11).is_err());
    assert!(Architecture {
        dilations: vec![1, 1024],
        ..Architecture::default()
    }
    .validate()
    .is_err());

    // The receptive field the context plan is built on.
    let baseline = Architecture::default().with_blocks(6).unwrap();
    assert_eq!(baseline.context_samples(), 4064); // 0.254 s
    let long = Architecture::default()
        .with_frame_samples(128)
        .unwrap()
        .with_blocks(8)
        .unwrap();
    assert_eq!(long.context_samples(), 65_408); // 4.088 s
}
