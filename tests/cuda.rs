//! CUDA bring-up checks: real device execution and CPU parity. Skipped without the feature.
#![cfg(feature = "cuda")]

use feral::{
    backend::Resident,
    model::{Architecture, Model},
    parameterization::{self, FlowConfig},
    runtime::Adam,
};

fn tiny() -> Model {
    Model::new(
        Architecture {
            block: 4,
            width: 5,
            dilations: vec![1, 2],
            scales: vec![1, 2],
        },
        7,
    )
}

#[test]
fn tiny_cuda_step_matches_cpu() {
    let n = 68usize;
    let flow = FlowConfig::Velocity;
    let model = tiny();
    let adam = Adam::new(model.count(), 0.0003);
    let mut device = feral::cuda::CudaTrainer::new(&model, &adam, n, 1, flow.clone())
        .expect("CUDA device unavailable; this test only runs on a CUDA machine");

    let clean: Vec<f32> = (0..n).map(|i| 0.2 * (i as f32 * 0.13).sin()).collect();
    let noise: Vec<f32> = (0..n).map(|i| 0.1 * ((i * 7 % 11) as f32 - 5.)).collect();
    let mut mask = vec![0f32; n];
    mask[..n / 3].fill(1.);
    let t = 0.37f32;

    let (cpu_loss, cpu_grad) =
        parameterization::loss_gradient(&model, &flow, &clean, &mask, &noise, t);
    let step = device
        .step(&clean, &mask, &noise, &[t], false)
        .expect("device forward/backward");
    let gpu_grad = device.read_gradients().expect("read gradients");

    assert!(
        (cpu_loss - step.loss).abs() <= 1e-4 + 1e-3 * cpu_loss.abs(),
        "loss parity: cpu {cpu_loss} device {}",
        step.loss
    );
    let max = cpu_grad
        .iter()
        .zip(&gpu_grad)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max <= 1e-4, "gradient parity: max abs delta {max}");
}

#[test]
fn baseline_cuda_update_matches_cpu() {
    let n = 512usize;
    let flow = FlowConfig::Velocity;
    let model = Model::new(Architecture::default(), 7);
    let adam = Adam::new(model.count(), 0.0003);
    let mut device = feral::cuda::CudaTrainer::new(&model, &adam, n, 2, flow.clone())
        .expect("CUDA device unavailable; this test only runs on a CUDA machine");

    let clean: Vec<f32> = (0..2 * n)
        .map(|i| 0.15 * (i as f32 * 0.031).sin())
        .collect();
    let noise: Vec<f32> = (0..2 * n)
        .map(|i| 0.05 * ((i * 13 % 17) as f32 - 8.))
        .collect();
    let mask: Vec<f32> = (0..2 * n)
        .map(|i| if i % 3 == 0 { 1. } else { 0. })
        .collect();
    let t = [0.2f32, 0.8f32];

    let mut cpu_model = model.clone();
    let mut cpu_adam = adam.clone();
    let mut g = vec![0f32; model.count()];
    for (item, &ti) in t.iter().enumerate() {
        let row = item * n..(item + 1) * n;
        let (_, gi) = parameterization::loss_gradient(
            &cpu_model,
            &flow,
            &clean[row.clone()],
            &mask[row.clone()],
            &noise[row.clone()],
            ti,
        );
        for (dst, src) in g.iter_mut().zip(&gi) {
            *dst += src / 2.;
        }
    }
    cpu_adam
        .update(&mut cpu_model, &g)
        .expect("cpu optimizer update");

    device
        .step(&clean, &mask, &noise, &t, true)
        .expect("device update");
    let mut dev_model = model.clone();
    let mut dev_adam = adam.clone();
    device.sync(&mut dev_model, &mut dev_adam).expect("sync");

    let max = cpu_model
        .parameters()
        .iter()
        .zip(dev_model.parameters())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max <= 5e-5, "parameter parity: max abs delta {max}");
    assert_eq!(dev_adam.step, cpu_adam.step, "optimizer step counter");
}

#[test]
fn detail_term_matches_cpu_on_device() {
    // The detail term is a stencil inside the loss kernel; at a non-zero weight the device loss and
    // gradient must still match the CPU reference exactly (up to fp32 summation order).
    use feral::backend::Resident as _;
    let n = 96usize;
    let model = tiny();
    let flow = FlowConfig::Velocity;
    let adam = Adam::new(model.count(), 0.0003);
    let clean: Vec<f32> = (0..n).map(|i| 0.2 * (i as f32 * 0.13).sin()).collect();
    let noise: Vec<f32> = (0..n).map(|i| 0.1 * ((i * 7 % 11) as f32 - 5.)).collect();
    let mut mask = vec![1f32; n];
    mask[16..80].fill(0.);
    let t = 0.37f32;
    for detail in [2.0f32, 8.0] {
        let (cpu_loss, cpu_grad) = feral::parameterization::loss_gradient_detailed(
            &model, &flow, &clean, &mask, &noise, t, detail,
        );
        // A fresh trainer per weight: the resident reports the *previous* step's statistics
        // (reported_loss_lag_steps = 1), so reusing one would compare against the wrong lambda.
        let mut device = feral::cuda::CudaTrainer::new(&model, &adam, n, 1, flow.clone())
            .expect("CUDA device unavailable; this test only runs on a CUDA machine");
        device.set_detail(detail);
        let step = device
            .step(&clean, &mask, &noise, &[t], false)
            .expect("device step with detail");
        let gpu_grad = device.read_gradients().expect("gradients");
        assert!(
            (cpu_loss - step.loss).abs() <= 1e-5 + 1e-3 * cpu_loss.abs(),
            "detail loss parity at {detail}: cpu {cpu_loss} device {}",
            step.loss
        );
        let max = cpu_grad
            .iter()
            .zip(&gpu_grad)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max <= 2e-5, "detail gradient parity at {detail}: {max}");
    }
}

#[test]
fn device_evaluation_loss_matches_cpu() {
    // `parameterization::loss` is what validation reports. It must give the same number through a
    // device-backed Velocity as through the CPU reference, for both parameterizations.
    use feral::model::Velocity as _;
    let n = 64usize;
    let model = tiny();
    let adam = Adam::new(model.count(), 0.0003);
    let known: Vec<f32> = (0..n).map(|i| 0.2 * (i as f32 * 0.11).cos()).collect();
    let mut mask = vec![0f32; n];
    mask[..n / 4].fill(1.);
    let noise: Vec<f32> = (0..n).map(|i| 0.1 * ((i * 7 % 11) as f32 - 5.)).collect();

    for flow in [
        FlowConfig::Velocity,
        FlowConfig::Preconditioned {
            data_scale: 0.1,
            prior_std: 1.,
        },
    ] {
        let trainer = feral::cuda::CudaTrainer::new(&model, &adam, n, 1, flow.clone())
            .expect("CUDA device unavailable; this test only runs on a CUDA machine");
        let device =
            feral::sampler::DeviceVelocity::new(Box::new(trainer), &flow, n, model.block())
                .expect("device sampler");
        let cpu = feral::parameterization::ConfiguredVelocity {
            model: &model,
            config: &flow,
        };
        for t in [0., 0.5, 1.] {
            let a = feral::parameterization::loss(&cpu, &flow, &known, &mask, &noise, t);
            let b = feral::parameterization::loss(&device, &flow, &known, &mask, &noise, t);
            assert!(
                (a - b).abs() <= 1e-5 + 1e-3 * a.abs(),
                "evaluation loss parity for {flow:?} at t={t}: cpu {a} device {b}"
            );
        }
    }
}

#[test]
fn large_frame_and_wide_dilation_parity() {
    use feral::model::Velocity as _;
    // The context plan raises block to 128 (388 feature channels per frame) and dilations to 128.
    // Both change kernel indexing, so check them against the CPU reference at a size that stays
    // fast: width 5, block 128, dilations 1..128.
    let n = 1024usize;
    let model = Model::new(
        Architecture {
            block: 128,
            width: 5,
            dilations: vec![1, 2, 4, 8, 16, 32, 64, 128],
            scales: vec![],
        },
        7,
    );
    // 128 frames of block 128 across dilations 1..128: a 4.1 s field at 16 kHz.
    assert_eq!(model.architecture.context_samples(), 65_408);
    let flow = FlowConfig::Velocity;
    let adam = Adam::new(model.count(), 0.0003);
    let mut device = feral::cuda::CudaTrainer::new(&model, &adam, n, 1, flow.clone())
        .expect("CUDA device unavailable; this test only runs on a CUDA machine");

    let clean: Vec<f32> = (0..n).map(|i| 0.2 * (i as f32 * 0.13).sin()).collect();
    let noise: Vec<f32> = (0..n).map(|i| 0.1 * ((i * 7 % 11) as f32 - 5.)).collect();
    let mut mask = vec![0f32; n];
    mask[..n / 3].fill(1.);
    let t = 0.37f32;

    let (cpu_loss, cpu_grad) =
        parameterization::loss_gradient(&model, &flow, &clean, &mask, &noise, t);
    let step = device
        .step(&clean, &mask, &noise, &[t], false)
        .expect("device forward/backward at block 128");
    let gpu_grad = device.read_gradients().expect("read gradients");
    let gpu_pred = device.read_prediction().expect("read prediction");

    assert!(
        (cpu_loss - step.loss).abs() <= 1e-4 + 1e-3 * cpu_loss.abs(),
        "loss parity at block 128: cpu {cpu_loss} device {}",
        step.loss
    );
    let grad_max = cpu_grad
        .iter()
        .zip(&gpu_grad)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(grad_max <= 1e-4, "gradient parity at block 128: {grad_max}");
    // The device builds x_t from clean/noise and feeds clean*mask as the known feature, which is
    // exactly `Model::velocity(x_t, clean, mask, t)`.
    let x_t: Vec<f32> = clean
        .iter()
        .zip(&noise)
        .zip(&mask)
        .map(|((&y, &z), &m)| if m == 1. { y } else { (1. - t) * z + t * y })
        .collect();
    let pred_max = model
        .velocity(&x_t, &clean, &mask, t)
        .iter()
        .zip(&gpu_pred)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        pred_max <= 1e-4,
        "prediction parity at block 128: {pred_max}"
    );
}

#[test]
fn device_sampler_field_matches_cpu() {
    // The device sampler evaluates the same field by feeding x as both clean and noise, which the
    // trainer turns into x_t = x for every position. That is only exact when the solver's x already
    // equals the known samples wherever the mask is set, which conditioning::project guarantees.
    use feral::model::Velocity as _;
    let n = 68usize;
    let model = tiny();
    let adam = Adam::new(model.count(), 0.0003);
    let known: Vec<f32> = (0..n).map(|i| 0.2 * (i as f32 * 0.11).cos()).collect();
    let free: Vec<f32> = (0..n).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
    let mut mask = vec![0f32; n];
    mask[..n / 4].fill(1.);
    let x: Vec<f32> = (0..n)
        .map(|i| if mask[i] == 1. { known[i] } else { free[i] })
        .collect();

    for flow in [
        FlowConfig::Velocity,
        FlowConfig::Preconditioned {
            data_scale: 0.1,
            prior_std: 1.,
        },
    ] {
        let trainer = feral::cuda::CudaTrainer::new(&model, &adam, n, 1, flow.clone())
            .expect("CUDA device unavailable; this test only runs on a CUDA machine");
        let device =
            feral::sampler::DeviceVelocity::new(Box::new(trainer), &flow, n, model.block())
                .expect("device sampler");
        let cpu = feral::parameterization::ConfiguredVelocity {
            model: &model,
            config: &flow,
        };
        for t in [0., 0.37, 1.] {
            let expected = cpu.velocity(&x, &known, &mask, t);
            let actual = device
                .try_velocity(&x, &known, &mask, t)
                .expect("device field");
            let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
            let max = expected
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max <= 1e-4 + 1e-3 * scale,
                "field parity for {flow:?} at t={t}: max abs delta {max}"
            );
        }
    }
}

#[test]
fn reported_loss_lags_one_step_and_is_never_zero() {
    // The resident backend defers its statistics readback so a step boundary never drains the
    // device, which makes the reported loss belong to the previous step. It must still be a real
    // loss: step 2 once reported 0.0 because the first step's statistics were never stored.
    let n = 68usize;
    let flow = FlowConfig::Velocity;
    let model = tiny();
    let adam = Adam::new(model.count(), 0.0003);
    let mut device = feral::cuda::CudaTrainer::new(&model, &adam, n, 1, flow.clone())
        .expect("CUDA device unavailable; this test only runs on a CUDA machine");
    assert_eq!(
        device.info()["reported_loss_lag_steps"].as_u64(),
        Some(1),
        "backend must declare its loss lag"
    );

    let clean_a: Vec<f32> = (0..n).map(|i| 0.2 * (i as f32 * 0.13).sin()).collect();
    let clean_b: Vec<f32> = (0..n).map(|i| 0.3 * (i as f32 * 0.07).cos()).collect();
    let noise: Vec<f32> = (0..n).map(|i| 0.1 * ((i * 7 % 11) as f32 - 5.)).collect();
    let mut mask = vec![0f32; n];
    mask[..n / 3].fill(1.);
    let t = 0.37f32;

    let loss_a = parameterization::loss_gradient(&model, &flow, &clean_a, &mask, &noise, t).0;
    let loss_b = parameterization::loss_gradient(&model, &flow, &clean_b, &mask, &noise, t).0;
    assert!((loss_a - loss_b).abs() > 1e-3, "test data must differ");

    let first = device
        .step(&clean_a, &mask, &noise, &[t], false)
        .expect("step 1");
    let second = device
        .step(&clean_b, &mask, &noise, &[t], false)
        .expect("step 2");
    let third = device
        .step(&clean_a, &mask, &noise, &[t], false)
        .expect("step 3");

    let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 + 1e-3 * a.abs();
    assert!(
        close(loss_a, first.loss),
        "step 1 reports its own loss: {} vs {loss_a}",
        first.loss
    );
    assert!(second.loss > 0., "step 2 must not report a zero loss");
    assert!(
        close(loss_a, second.loss),
        "step 2 must report step 1's loss: {} vs {loss_a}",
        second.loss
    );
    assert!(
        close(loss_b, third.loss),
        "step 3 must report step 2's loss: {} vs {loss_b}",
        third.loss
    );
}

#[test]
fn repeated_trainer_creation_stays_valid() {
    let flow = FlowConfig::Velocity;
    let model = Model::new(
        Architecture {
            block: 32,
            width: 32,
            dilations: vec![1, 2, 4, 8],
            scales: vec![],
        },
        7,
    );
    let adam = Adam::new(model.count(), 0.0003);
    let n = 8192usize;
    let clean = vec![0.01f32; n];
    let noise = vec![0.01f32; n];
    let mask = vec![0f32; n];
    for i in 0..12 {
        let mut device = feral::cuda::CudaTrainer::new(&model, &adam, n, 1, flow.clone())
            .unwrap_or_else(|e| panic!("trainer {i} creation failed: {e}"));
        device
            .step(&clean, &mask, &noise, &[0.5], true)
            .unwrap_or_else(|e| panic!("trainer {i} step failed: {e}"));
    }
}

#[test]
fn mixed_size_trainer_sequence() {
    let flow = FlowConfig::Velocity;
    let tiny = tiny();
    let adam_t = Adam::new(tiny.count(), 0.0003);
    for i in 0..9 {
        let mut d = feral::cuda::CudaTrainer::new(&tiny, &adam_t, 68, 1, flow.clone())
            .unwrap_or_else(|e| panic!("tiny {i}: {e}"));
        let clean = vec![0.01f32; 68];
        let noise = vec![0.01f32; 68];
        let mask = vec![0f32; 68];
        d.step(&clean, &mask, &noise, &[0.3], true)
            .unwrap_or_else(|e| panic!("tiny {i} step: {e}"));
    }
    let model = Model::new(
        Architecture {
            block: 32,
            width: 32,
            dilations: vec![1, 2, 4, 8],
            scales: vec![],
        },
        7,
    );
    let adam = Adam::new(model.count(), 0.0003);
    let n = 8192usize;
    let mut d = feral::cuda::CudaTrainer::new(&model, &adam, n, 1, flow.clone())
        .expect("large trainer creation");
    d.step(
        &vec![0.01f32; n],
        &vec![0f32; n],
        &vec![0.01f32; n],
        &[0.3],
        true,
    )
    .expect("large trainer step");
}

#[test]
fn batched_preconditioned_cuda_matches_cpu() {
    use feral::model::Velocity as _;
    let n = 68usize;
    let items = 3usize;
    let flow = FlowConfig::Preconditioned {
        data_scale: 0.16,
        prior_std: 0.25,
    };
    let model = tiny();
    let adam = Adam::new(model.count(), 0.0003);
    let clean: Vec<f32> = (0..items * n)
        .map(|i| 0.18 * (i as f32 * 0.071).sin())
        .collect();
    let noise: Vec<f32> = (0..items * n)
        .map(|i| 0.25 * (((i * 13 % 19) as f32 - 9.) / 9.))
        .collect();
    let mut mask = vec![0f32; items * n];
    mask[..24].fill(1.);
    mask[n + 20..n + 44].fill(1.);
    mask[2 * n + 40..].fill(1.);
    let t = [0.15f32, 0.55f32, 0.90f32];

    for detail in [0.0f32, 2.0f32] {
        let mut device = feral::cuda::CudaTrainer::new(&model, &adam, n, items, flow.clone())
            .expect("CUDA device unavailable; this test only runs on a CUDA machine");
        device.set_detail(detail);

        let mut cpu_model = model.clone();
        let mut cpu_adam = adam.clone();
        let mut cpu_loss = 0f32;
        let mut cpu_grad = vec![0f32; model.count()];
        let mut cpu_preds = Vec::with_capacity(items * n);
        for (item, &ti) in t.iter().enumerate() {
            let row = item * n..(item + 1) * n;
            let (li, gi) = feral::parameterization::loss_gradient_detailed(
                &cpu_model,
                &flow,
                &clean[row.clone()],
                &mask[row.clone()],
                &noise[row.clone()],
                ti,
                detail,
            );
            cpu_loss += li / items as f32;
            for (dst, src) in cpu_grad.iter_mut().zip(&gi) {
                *dst += src / items as f32;
            }
            let x_t: Vec<f32> = clean[row.clone()]
                .iter()
                .zip(&noise[row.clone()])
                .zip(&mask[row.clone()])
                .map(|((&y, &z), &m)| if m == 1. { y } else { (1. - ti) * z + ti * y })
                .collect();
            let (_, _, cin) = flow.coefficients(ti);
            let k: Vec<f32> = clean[row.clone()].iter().map(|v| v / 0.16).collect();
            let xx: Vec<f32> = x_t
                .iter()
                .zip(&k)
                .zip(&mask[row.clone()])
                .map(|((&x, &known), &m)| if m == 1. { known } else { x * cin })
                .collect();
            cpu_preds.extend(cpu_model.velocity(&xx, &k, &mask[row], ti));
        }

        let step = device
            .step(&clean, &mask, &noise, &t, true)
            .expect("batched preconditioned step");
        let gpu_grad = device.read_gradients().expect("read gradients");
        let gpu_pred = device.read_prediction().expect("read prediction");

        assert!(
            (cpu_loss - step.loss).abs() <= 1e-4 + 1e-3 * cpu_loss.abs(),
            "batched preconditioned loss parity (detail={detail}): cpu {cpu_loss} device {}",
            step.loss
        );
        let pred_max = cpu_preds
            .iter()
            .zip(&gpu_pred)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            pred_max <= 1e-4,
            "batched preconditioned prediction parity (detail={detail}): {pred_max}"
        );
        let grad_max = cpu_grad
            .iter()
            .zip(&gpu_grad)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            grad_max <= 2e-4,
            "batched preconditioned gradient parity (detail={detail}): {grad_max}"
        );

        cpu_adam
            .update(&mut cpu_model, &cpu_grad)
            .expect("cpu update");
        let mut dev_model = model.clone();
        let mut dev_adam = adam.clone();
        device.sync(&mut dev_model, &mut dev_adam).expect("sync");
        let param_max = cpu_model
            .parameters()
            .iter()
            .zip(dev_model.parameters())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            param_max <= 5e-5,
            "batched preconditioned parameter parity (detail={detail}): {param_max}"
        );
    }
}

#[test]
fn cached_trainer_weight_reload_and_final_step_overflow_check() {
    use feral::model::Velocity as _;
    let n = 68usize;
    let flow = FlowConfig::Preconditioned {
        data_scale: 0.16,
        prior_std: 0.25,
    };
    let model = tiny();
    let adam = Adam::new(model.count(), 0.0003);
    let mut cached = feral::cuda::CudaTrainer::new(&model, &adam, n, 1, flow.clone())
        .expect("CUDA device unavailable");
    let mut updated = model.clone();
    let mut params = updated.parameters();
    for (i, p) in params.iter_mut().enumerate() {
        *p += 0.05 * ((i % 7) as f32 - 3.);
    }
    updated.set_parameters(&params);
    cached.load_weights(&updated).expect("load_weights");
    let mut fresh =
        feral::cuda::CudaTrainer::new(&updated, &adam, n, 1, flow.clone()).expect("fresh trainer");

    let x: Vec<f32> = (0..n).map(|i| 0.15 * (i as f32 * 0.11).sin()).collect();
    let mut mask = vec![0f32; n];
    mask[..16].fill(1.);
    let p_cached = cached.predict(&x, &mask, 0.42).expect("cached predict");
    let p_fresh = fresh.predict(&x, &mask, 0.42).expect("fresh predict");
    assert_eq!(p_cached, p_fresh);

    let cpu = feral::parameterization::ConfiguredVelocity {
        model: &updated,
        config: &flow,
    };
    let wrapped = feral::sampler::DeviceVelocity::new(Box::new(cached), &flow, n, model.block())
        .expect("device velocity");
    let expected = cpu.velocity(&x, &x, &mask, 0.42);
    let actual = wrapped.try_velocity(&x, &x, &mask, 0.42).expect("field");
    let max = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max <= 1e-4, "reloaded field parity: {max}");

    // Step 1 without update sets reported_steps=1; step 2 triggers Adam overflow inside the
    // deferred readback window, which sync() / flush() must still catch and reject.
    let mut bad_adam = Adam::new(model.count(), 1.);
    bad_adam.m.fill(f32::MAX);
    let mut overflow = feral::cuda::CudaTrainer::new(&model, &bad_adam, n, 1, FlowConfig::Velocity)
        .expect("overflow trainer");
    overflow
        .step(&x, &mask, &x, &[0.5], false)
        .expect("step 1 without update succeeds");
    overflow
        .step(&x, &mask, &x, &[0.5], true)
        .expect("step 2 returns lagged step 1 stats");
    let err = overflow
        .flush()
        .expect_err("flush must catch final-step optimizer overflow");
    assert!(err.to_string().contains("nonfinite"), "{err}");
    let mut m_out = model.clone();
    let mut a_out = bad_adam.clone();
    assert!(overflow.sync(&mut m_out, &mut a_out).is_err());
}

#[test]
fn cuda_train_and_evaluate_match_cpu_with_batched_preconditioned_flow() {
    use feral::{
        audio::{self, Preprocess},
        evaluate,
        runtime::{self, Checkpoint, TrainConfig},
    };
    let root = std::env::temp_dir().join(format!("feral-cuda-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir(&root).unwrap();
    audio::synthetic(&root.join("corpus"), 16000).unwrap();

    let make = |run_name: &str, backend: &str| TrainConfig {
        input: root.join("corpus"),
        run: root.join(run_name),
        backend: backend.into(),
        steps: 4,
        seed: 19,
        crop_seconds: 0.032,
        preprocess: Preprocess::default(),
        lr: 0.0003,
        preview_every: 2,
        checkpoint_every: 2,
        resume: None,
        weights_only: false,
        grow_crop: false,
        architecture: Architecture {
            block: 4,
            width: 5,
            dilations: vec![1, 2],
            scales: vec![1, 2],
        },
        flow: FlowConfig::Preconditioned {
            data_scale: 0.16,
            prior_std: 0.25,
        },
        detail: 1.5,
        batch: 3,
        validation: Some(root.join("corpus")),
        validation_every: 2,
    };

    let cpu_ck_path = runtime::train(make("cpu", "cpu")).expect("cpu train");
    let cuda_ck_path = runtime::train(make("cuda", "cuda")).expect("cuda train");
    let cpu_ck = Checkpoint::load(&cpu_ck_path).unwrap();
    let cuda_ck = Checkpoint::load(&cuda_ck_path).unwrap();
    assert_eq!(cpu_ck.completed_steps, cuda_ck.completed_steps);
    let max_param = cpu_ck
        .model
        .parameters()
        .iter()
        .zip(cuda_ck.model.parameters())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_param <= 5e-5,
        "end-to-end preconditioned batch=3 train drift: {max_param}"
    );

    let cpu_eval = root.join("eval-cpu.json");
    let cuda_eval = root.join("eval-cuda.json");
    evaluate::evaluate_with_backend(
        &cuda_ck_path,
        &root.join("corpus/pattern-0.wav"),
        &cpu_eval,
        2,
        7001,
        "cpu",
    )
    .unwrap();
    evaluate::evaluate_with_backend(
        &cuda_ck_path,
        &root.join("corpus/pattern-0.wav"),
        &cuda_eval,
        2,
        7001,
        "cuda",
    )
    .unwrap();
    let a: serde_json::Value = serde_json::from_slice(&std::fs::read(&cpu_eval).unwrap()).unwrap();
    let b: serde_json::Value = serde_json::from_slice(&std::fs::read(&cuda_eval).unwrap()).unwrap();
    let mse_a = a["mean_velocity_mse"].as_f64().unwrap();
    let mse_b = b["mean_velocity_mse"].as_f64().unwrap();
    assert!(
        (mse_a - mse_b).abs() <= 1e-5 + 1e-4 * mse_a.abs(),
        "evaluate backend parity: cpu {mse_a} cuda {mse_b}"
    );
    assert_eq!(b["evaluation_backend"], "cuda");
    std::fs::remove_dir_all(root).unwrap();
}
