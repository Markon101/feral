//! CUDA bring-up checks: real device execution and CPU parity. Skipped without the feature.
#![cfg(feature = "cuda")]

use feral::{
    backend::Resident,
    model::{Architecture, BlockStyle, Model},
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
            sample_conv: 0,
            block_style: BlockStyle::Tanh,
            channels: 1,
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
            sample_conv: 0,
            block_style: BlockStyle::Tanh,
            channels: 1,
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
            sample_conv: 0,
            block_style: BlockStyle::Tanh,
            channels: 1,
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
            sample_conv: 0,
            block_style: BlockStyle::Tanh,
            channels: 1,
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
            sample_conv: 0,
            block_style: BlockStyle::Tanh,
            channels: 1,
        },
        flow: FlowConfig::Preconditioned {
            data_scale: 0.16,
            prior_std: 0.25,
        },
        detail: 1.5,
        spectral_weight: 0.,
        time_sampling: feral::parameterization::TimeSampling::Uniform,
        lr_schedule: feral::parameterization::LrSchedule::Constant,
        ema_decay: 0.,
        weight_decay: 0.,
        crop_horizons: vec![],
        horizon_schedule: feral::parameterization::HorizonSchedule::Fixed,
        horizon_steps: None,
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

#[test]
fn sample_conv_and_spectral_loss_cuda_match_cpu_batched() {
    use feral::model::Velocity as _;
    use feral::spectral::SpectralEngine;

    let n = 192usize;
    let items = 3usize;
    let arch = Architecture {
        block: 4,
        width: 5,
        dilations: vec![1, 2],
        scales: vec![1, 2],
        sample_conv: 7,
        block_style: BlockStyle::Tanh,
        channels: 1,
    };
    let model = Model::new(arch, 11);
    let adam = Adam::new(model.count(), 0.0003);
    let clean: Vec<f32> = (0..items * n)
        .map(|i| 0.18 * (i as f32 * 0.071).sin())
        .collect();
    let noise: Vec<f32> = (0..items * n)
        .map(|i| 0.22 * (((i * 13 % 19) as f32 - 9.) / 9.))
        .collect();
    let mut mask = vec![0f32; items * n];
    mask[..32].fill(1.);
    mask[n + 40..n + 80].fill(1.);
    mask[2 * n + 140..].fill(1.);
    let t = [0.2f32, 0.55f32, 0.85f32];
    let engine = SpectralEngine::new();

    for (flow, spectral_weight) in [
        (FlowConfig::Velocity, 0.0f32),
        (FlowConfig::Velocity, 0.4f32),
        (
            FlowConfig::Preconditioned {
                data_scale: 0.16,
                prior_std: 0.25,
            },
            0.4f32,
        ),
    ] {
        let mut device = feral::cuda::CudaTrainer::new(&model, &adam, n, items, flow.clone())
            .expect("CUDA device unavailable");
        device.set_detail(1.0);
        device.set_spectral(spectral_weight);

        let mut cpu_model = model.clone();
        let mut cpu_adam = adam.clone();
        let mut cpu_loss = 0f32;
        let mut cpu_grad = vec![0f32; model.count()];
        let mut cpu_preds = Vec::with_capacity(items * n);
        for (item, &ti) in t.iter().enumerate() {
            let row = item * n..(item + 1) * n;
            let (li, gi) = parameterization::loss_gradient_with_spectral(
                &cpu_model,
                &flow,
                &clean[row.clone()],
                &mask[row.clone()],
                &noise[row.clone()],
                ti,
                1.0,
                spectral_weight,
                &engine,
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
            let (xx, k) = match flow {
                FlowConfig::Velocity => (x_t, clean[row.clone()].to_vec()),
                FlowConfig::Preconditioned { data_scale, .. } => {
                    let (_, _, cin) = flow.coefficients(ti);
                    let k: Vec<f32> = clean[row.clone()].iter().map(|v| v / data_scale).collect();
                    let xx: Vec<f32> = x_t
                        .iter()
                        .zip(&k)
                        .zip(&mask[row.clone()])
                        .map(|((&x, &known), &m)| if m == 1. { known } else { x * cin })
                        .collect();
                    (xx, k)
                }
            };
            cpu_preds.extend(cpu_model.velocity(&xx, &k, &mask[row], ti));
        }

        let step = device
            .step(&clean, &mask, &noise, &t, true)
            .expect("sample_conv + spectral step");
        let gpu_grad = device.read_gradients().expect("read gradients");
        let gpu_pred = device.read_prediction().expect("read prediction");

        assert!(
            (cpu_loss - step.loss).abs() <= 1e-4 + 1e-3 * cpu_loss.abs(),
            "loss parity ({flow:?}, spec={spectral_weight}): cpu {cpu_loss} gpu {}",
            step.loss
        );
        let pred_max = cpu_preds
            .iter()
            .zip(&gpu_pred)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            pred_max <= 1e-4,
            "pred parity ({flow:?}, spec={spectral_weight}): {pred_max}"
        );
        let grad_max = cpu_grad
            .iter()
            .zip(&gpu_grad)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            grad_max <= 2e-4,
            "grad parity ({flow:?}, spec={spectral_weight}): {grad_max}"
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
            "param parity ({flow:?}, spec={spectral_weight}): {param_max}"
        );
    }
}

#[test]
fn cuda_e2e_train_with_all_new_features_matches_cpu() {
    use feral::{
        audio::{self, Preprocess},
        evaluate,
        parameterization::{LrSchedule, TimeSampling},
        runtime::{self, Checkpoint, TrainConfig},
    };
    let root = std::env::temp_dir().join(format!("feral-cuda-allnew-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir(&root).unwrap();
    audio::synthetic(&root.join("corpus"), 16000).unwrap();

    let make = |run_name: &str, backend: &str| TrainConfig {
        input: root.join("corpus"),
        run: root.join(run_name),
        backend: backend.into(),
        steps: 4,
        seed: 29,
        crop_seconds: 0.032,
        preprocess: Preprocess::default(),
        lr: 0.0005,
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
            sample_conv: 5,
            block_style: BlockStyle::Tanh,
            channels: 1,
        },
        flow: FlowConfig::Preconditioned {
            data_scale: 0.16,
            prior_std: 0.25,
        },
        detail: 1.0,
        spectral_weight: 0.25,
        time_sampling: TimeSampling::LogitNormal { mean: 0., std: 1. },
        lr_schedule: LrSchedule::Cosine {
            warmup_steps: 2,
            total_steps: 6,
            min_lr: 1e-4,
        },
        ema_decay: 0.9,
        weight_decay: 0.,
        crop_horizons: vec![],
        horizon_schedule: feral::parameterization::HorizonSchedule::Fixed,
        horizon_steps: None,
        batch: 2,
        validation: Some(root.join("corpus")),
        validation_every: 2,
    };

    let cpu_ck_path = runtime::train(make("cpu", "cpu")).expect("cpu train");
    let cuda_ck_path = runtime::train(make("cuda", "cuda")).expect("cuda train");
    let cpu_ck = Checkpoint::load(&cpu_ck_path).unwrap();
    let cuda_ck = Checkpoint::load(&cuda_ck_path).unwrap();
    assert_eq!(cpu_ck.completed_steps, cuda_ck.completed_steps);
    assert!(cuda_ck.has_ema());

    let max_param = cpu_ck
        .model
        .parameters()
        .iter()
        .zip(cuda_ck.model.parameters())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_param <= 5e-5, "raw parameter drift: {max_param}");

    let max_ema = cpu_ck
        .ema_weights
        .iter()
        .zip(&cuda_ck.ema_weights)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_ema <= 5e-5, "EMA parameter drift: {max_ema}");

    // Evaluate EMA vs raw on both CPU and CUDA.
    for use_ema in [true, false] {
        let tag = if use_ema { "ema" } else { "raw" };
        let c_out = root.join(format!("eval-cpu-{tag}.json"));
        let g_out = root.join(format!("eval-cuda-{tag}.json"));
        evaluate::evaluate_with_options(
            &cuda_ck_path,
            &root.join("corpus/pattern-0.wav"),
            &c_out,
            1,
            7001,
            "cpu",
            use_ema,
        )
        .unwrap();
        evaluate::evaluate_with_options(
            &cuda_ck_path,
            &root.join("corpus/pattern-0.wav"),
            &g_out,
            1,
            7001,
            "cuda",
            use_ema,
        )
        .unwrap();
        let a: serde_json::Value = serde_json::from_slice(&std::fs::read(&c_out).unwrap()).unwrap();
        let b: serde_json::Value = serde_json::from_slice(&std::fs::read(&g_out).unwrap()).unwrap();
        assert_eq!(a["using_ema"], use_ema);
        assert_eq!(b["using_ema"], use_ema);
        let mse_a = a["mean_velocity_mse"].as_f64().unwrap();
        let mse_b = b["mean_velocity_mse"].as_f64().unwrap();
        assert!(
            (mse_a - mse_b).abs() <= 1e-5 + 1e-4 * mse_a.abs(),
            "evaluate ({tag}) parity: cpu {mse_a} cuda {mse_b}"
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn gated_film_symmetric_and_gpu_stft_cuda_match_cpu_batched() {
    use feral::model::Velocity as _;
    use feral::spectral::SpectralEngine;

    // Exercise N=128, N=512, and N=2048 STFT frames with partial tails + GatedFilm + scales + sample_conv.
    let n = 2304usize;
    let items = 2usize;
    let arch = Architecture {
        block: 16,
        width: 8,
        dilations: vec![1, 2, 4, 2, 1],
        scales: vec![1, 2, 1, 2, 1],
        sample_conv: 7,
        block_style: BlockStyle::GatedFilm,
        channels: 1,
    };
    let model = Model::new(arch, 17);
    let adam = Adam::new(model.count(), 0.0003);
    let flow = FlowConfig::Preconditioned {
        data_scale: 0.16,
        prior_std: 0.16,
    };
    let clean: Vec<f32> = (0..items * n)
        .map(|i| 0.16 * (i as f32 * 0.073).sin() + 0.06 * (i as f32 * 0.19).cos())
        .collect();
    let noise: Vec<f32> = (0..items * n)
        .map(|i| 0.16 * (((i * 17 % 23) as f32 - 11.) / 11.))
        .collect();
    let mut mask = vec![0f32; items * n];
    mask[..256].fill(1.);
    mask[n + 400..n + 800].fill(1.);
    let t = [0.28f32, 0.74f32];
    let engine = SpectralEngine::new();

    let mut device = feral::cuda::CudaTrainer::new(&model, &adam, n, items, flow.clone())
        .expect("CUDA device unavailable");
    device.set_detail(0.5);
    device.set_spectral(0.35);

    let mut cpu_model = model.clone();
    let mut cpu_adam = adam.clone();
    let mut cpu_loss = 0f32;
    let mut cpu_grad = vec![0f32; model.count()];
    let mut cpu_preds = Vec::with_capacity(items * n);
    for (item, &ti) in t.iter().enumerate() {
        let row = item * n..(item + 1) * n;
        let (li, gi) = parameterization::loss_gradient_with_spectral(
            &cpu_model,
            &flow,
            &clean[row.clone()],
            &mask[row.clone()],
            &noise[row.clone()],
            ti,
            0.5,
            0.35,
            &engine,
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
        .expect("gated-film + gpu stft step");
    let gpu_grad = device.read_gradients().expect("read gradients");
    let gpu_pred = device.read_prediction().expect("read prediction");

    assert!(
        (cpu_loss - step.loss).abs() <= 2e-4 + 1e-3 * cpu_loss.abs(),
        "gated-film + GPU STFT loss parity: cpu {cpu_loss} gpu {}",
        step.loss
    );
    let pred_max = cpu_preds
        .iter()
        .zip(&gpu_pred)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(pred_max <= 1e-4, "gated-film pred parity: {pred_max}");
    let grad_max = cpu_grad
        .iter()
        .zip(&gpu_grad)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        grad_max <= 3e-4,
        "gated-film + GPU STFT grad parity: {grad_max}"
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
    assert!(param_max <= 5e-5, "gated-film param parity: {param_max}");

    // Verify GPU STFT + GatedFilm is bit-exact deterministic on a fresh trainer with same inputs.
    let mut device2 = feral::cuda::CudaTrainer::new(&model, &adam, n, items, flow)
        .expect("CUDA device unavailable");
    device2.set_detail(0.5);
    device2.set_spectral(0.35);
    device2
        .step(&clean, &mask, &noise, &t, true)
        .expect("second deterministic step");
    let mut dev_model2 = model.clone();
    let mut dev_adam2 = adam.clone();
    device2
        .sync(&mut dev_model2, &mut dev_adam2)
        .expect("sync2");
    assert_eq!(
        dev_model.parameters(),
        dev_model2.parameters(),
        "GPU STFT + GatedFilm update must be bit-exact deterministic"
    );
}

#[test]
fn cuda_e2e_train_gated_film_and_resume_matches_cpu() {
    use feral::{
        audio::{self, Preprocess},
        parameterization::{LrSchedule, TimeSampling},
        runtime::{self, Checkpoint, TrainConfig},
    };
    let root = std::env::temp_dir().join(format!("feral-cuda-gated-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir(&root).unwrap();
    audio::synthetic(&root.join("corpus"), 16000).unwrap();

    let make = |run_name: &str, backend: &str, steps: usize, resume| TrainConfig {
        input: root.join("corpus"),
        run: root.join(run_name),
        backend: backend.into(),
        steps,
        seed: 37,
        crop_seconds: 0.032,
        preprocess: Preprocess::default(),
        lr: 0.0005,
        preview_every: 2,
        checkpoint_every: 2,
        resume,
        weights_only: false,
        grow_crop: false,
        architecture: Architecture {
            block: 4,
            width: 6,
            dilations: vec![1, 2, 4, 2, 1],
            scales: vec![],
            sample_conv: 5,
            block_style: BlockStyle::GatedFilm,
            channels: 1,
        },
        flow: FlowConfig::Preconditioned {
            data_scale: 0.16,
            prior_std: 0.16,
        },
        detail: 0.5,
        spectral_weight: 0.25,
        time_sampling: TimeSampling::LogitNormal { mean: 0., std: 1. },
        lr_schedule: LrSchedule::Cosine {
            warmup_steps: 2,
            total_steps: 6,
            min_lr: 1e-4,
        },
        ema_decay: 0.9,
        weight_decay: 0.,
        crop_horizons: vec![],
        horizon_schedule: feral::parameterization::HorizonSchedule::Fixed,
        horizon_steps: None,
        batch: 2,
        validation: Some(root.join("corpus")),
        validation_every: 2,
    };

    let cpu_all = runtime::train(make("cpu-all", "cpu", 4, None)).expect("cpu train");
    let cuda_all = runtime::train(make("cuda-all", "cuda", 4, None)).expect("cuda train");
    let cuda_first = runtime::train(make("cuda-first", "cuda", 2, None)).expect("cuda first");
    let cuda_resumed =
        runtime::train(make("cuda-resumed", "cuda", 2, Some(cuda_first))).expect("cuda resume");

    let c = Checkpoint::load(&cpu_all).unwrap();
    let g = Checkpoint::load(&cuda_all).unwrap();
    let r = Checkpoint::load(&cuda_resumed).unwrap();
    let max_param = c
        .model
        .parameters()
        .iter()
        .zip(g.model.parameters())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_param <= 5e-5, "gated-film CPU/CUDA drift: {max_param}");
    let max_ema = c
        .ema_weights
        .iter()
        .zip(&g.ema_weights)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_ema <= 5e-5, "gated-film EMA CPU/CUDA drift: {max_ema}");
    assert_eq!(
        g.model.parameters(),
        r.model.parameters(),
        "CUDA gated-film + GPU STFT resume must be bit-exact"
    );
    assert_eq!(g.ema_weights, r.ema_weights);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn glu_film_adamw_and_multi_horizon_cuda_match_cpu() {
    use feral::{
        audio::{self, Preprocess},
        backend::{Batch, Trainer},
        parameterization::{HorizonSchedule, LrSchedule, TimeSampling},
        rng::Rng,
        runtime::{self, Adam, Checkpoint, TrainConfig},
    };

    // Part A: Low-level Trainer::new_with_horizons + dynamic set_shape + AdamW weight decay.
    let model = Model::new(
        Architecture {
            block: 4,
            width: 6,
            dilations: vec![1, 2, 1],
            scales: vec![],
            sample_conv: 5,
            block_style: BlockStyle::GluFilm,
            channels: 1,
        },
        53,
    );
    let flow = FlowConfig::Preconditioned {
        data_scale: 0.16,
        prior_std: 0.16,
    };
    let horizons = [(512, 4), (1024, 2), (2048, 1)];
    let adam0 = Adam::new(model.count(), 0.0005);
    let mut cpu_model = model.clone();
    let mut cpu_adam = adam0.clone();
    let mut cpu = Trainer::new_with_horizons("cpu", &cpu_model, &cpu_adam, &horizons, &flow)
        .expect("cpu trainer");
    let mut gpu_model = model.clone();
    let mut gpu_adam = adam0;
    let mut gpu = Trainer::new_with_horizons("cuda", &gpu_model, &gpu_adam, &horizons, &flow)
        .expect("cuda trainer");
    let init_p = cpu_model.parameters();
    let mut cpu_ema = init_p.clone();
    let mut gpu_ema = init_p;
    let ema_decay = 0.9f32;
    cpu.configure_ema(ema_decay, &cpu_ema).unwrap();
    gpu.configure_ema(ema_decay, &gpu_ema).unwrap();

    let mut rng = Rng::new(71);
    let mut cpu_steps = Vec::new();
    let mut gpu_steps = Vec::new();
    let schedule = [(512, 4), (1024, 2), (2048, 1), (512, 4)];
    for (step_idx, &(sn, sitems)) in schedule.iter().enumerate() {
        let total = sn * sitems;
        let clean: Vec<f32> = (0..total).map(|_| rng.normal() * 0.16).collect();
        let mask: Vec<f32> = (0..total)
            .map(|k| if (k % sn) < sn / 4 { 1. } else { 0. })
            .collect();
        let noise: Vec<f32> = (0..total).map(|_| rng.normal() * 0.16).collect();
        let t: Vec<f32> = (0..sitems).map(|i| 0.15 + 0.2 * i as f32).collect();
        let lr = 0.0005 * (1. + 0.1 * step_idx as f32);

        cpu.set_shape(sn, sitems).unwrap();
        gpu.set_shape(sn, sitems).unwrap();

        let cs = cpu
            .step(
                &mut cpu_model,
                &mut cpu_adam,
                &flow,
                Batch {
                    clean: &clean,
                    mask: &mask,
                    noise: &noise,
                    t: &t,
                    detail: 0.5,
                    spectral: 1.0,
                    lr,
                    weight_decay: 0.05,
                },
            )
            .unwrap();
        let cp = cpu_model.parameters();
        for (e, &w) in cpu_ema.iter_mut().zip(&cp) {
            *e = ema_decay * *e + (1. - ema_decay) * w;
        }
        let gs = gpu
            .step(
                &mut gpu_model,
                &mut gpu_adam,
                &flow,
                Batch {
                    clean: &clean,
                    mask: &mask,
                    noise: &noise,
                    t: &t,
                    detail: 0.5,
                    spectral: 1.0,
                    lr,
                    weight_decay: 0.05,
                },
            )
            .unwrap();
        cpu_steps.push(cs);
        gpu_steps.push(gs);
    }
    // CUDA reported loss/norm lags by 1 step after step 0 (step 0 syncs immediately).
    for (step_idx, gs) in gpu_steps.iter().enumerate() {
        let cs = &cpu_steps[step_idx.saturating_sub(1)];
        assert!(
            (cs.loss - gs.loss).abs() <= 2e-5,
            "step {step_idx} lagged loss drift cpu={} cuda={}",
            cs.loss,
            gs.loss
        );
        assert!(
            (cs.norm - gs.norm).abs() <= 5e-5,
            "step {step_idx} lagged norm drift cpu={} cuda={}",
            cs.norm,
            gs.norm
        );
    }
    gpu.sync(&mut gpu_model, &mut gpu_adam).unwrap();
    gpu.sync_ema(&mut gpu_ema).unwrap();
    let max_w = cpu_model
        .parameters()
        .iter()
        .zip(gpu_model.parameters())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_w <= 5e-5, "GluFilm AdamW weight drift: {max_w}");
    let max_m = cpu_adam
        .m
        .iter()
        .zip(&gpu_adam.m)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_m <= 5e-5, "GluFilm AdamW m drift: {max_m}");
    let max_e = cpu_ema
        .iter()
        .zip(&gpu_ema)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_e <= 5e-5, "GluFilm EMA drift: {max_e}");

    // Part B: End-to-end multi-horizon Random schedule training & resume on CPU vs CUDA.
    let root = std::env::temp_dir().join(format!("feral-cuda-mh-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir(&root).unwrap();
    audio::synthetic(&root.join("corpus"), 16000).unwrap();

    let make = |run_name: &str, backend: &str, steps: usize, resume| TrainConfig {
        input: root.join("corpus"),
        run: root.join(run_name),
        backend: backend.into(),
        steps,
        seed: 61,
        crop_seconds: 0.032,
        preprocess: Preprocess::default(),
        lr: 0.0005,
        preview_every: 2,
        checkpoint_every: 2,
        resume,
        weights_only: false,
        grow_crop: false,
        architecture: Architecture {
            block: 4,
            width: 6,
            dilations: vec![1, 2, 1],
            scales: vec![],
            sample_conv: 5,
            block_style: BlockStyle::GluFilm,
            channels: 1,
        },
        flow: flow.clone(),
        detail: 0.5,
        spectral_weight: 1.0,
        time_sampling: TimeSampling::LogitNormal { mean: 0., std: 1. },
        lr_schedule: LrSchedule::Cosine {
            warmup_steps: 2,
            total_steps: 6,
            min_lr: 1e-4,
        },
        ema_decay: 0.9,
        weight_decay: 0.02,
        crop_horizons: vec![0.032, 0.064],
        horizon_schedule: HorizonSchedule::Random,
        horizon_steps: Some(4),
        batch: 2,
        validation: Some(root.join("corpus")),
        validation_every: 2,
    };

    let cpu_all = runtime::train(make("cpu-all", "cpu", 4, None)).expect("cpu train");
    let cuda_all = runtime::train(make("cuda-all", "cuda", 4, None)).expect("cuda train");
    let cuda_first = runtime::train(make("cuda-first", "cuda", 2, None)).expect("cuda first");
    let cuda_resumed =
        runtime::train(make("cuda-resumed", "cuda", 2, Some(cuda_first))).expect("cuda resume");

    let c = Checkpoint::load(&cpu_all).unwrap();
    let g = Checkpoint::load(&cuda_all).unwrap();
    let r = Checkpoint::load(&cuda_resumed).unwrap();
    let max_param = c
        .model
        .parameters()
        .iter()
        .zip(g.model.parameters())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_param <= 5e-5,
        "multi-horizon CPU/CUDA drift: {max_param}"
    );
    let max_ema = c
        .ema_weights
        .iter()
        .zip(&g.ema_weights)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_ema <= 5e-5,
        "multi-horizon EMA CPU/CUDA drift: {max_ema}"
    );
    assert_eq!(g.model.parameters(), r.model.parameters());
    assert_eq!(g.ema_weights, r.ema_weights);
    assert_eq!(g.rng.state, r.rng.state);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn stereo_glu_film_multi_horizon_cuda_matches_cpu_and_resumes() {
    use feral::{
        audio::{self, Preprocess},
        backend::{Batch, Trainer},
        conditioning, evaluate,
        parameterization::{HorizonSchedule, LrSchedule, TimeSampling},
        rng::Rng,
        runtime::{self, Adam, Checkpoint, TrainConfig},
    };

    // Part A: Low-level Trainer::new_with_horizons + dynamic set_shape + stereo L/R/M/S STFT.
    let model = Model::new(
        Architecture {
            block: 4,
            width: 6,
            dilations: vec![1, 2, 1],
            scales: vec![],
            sample_conv: 5,
            block_style: BlockStyle::GluFilm,
            channels: 2,
        },
        97,
    );
    let flow = FlowConfig::Preconditioned {
        data_scale: 0.20,
        prior_std: 0.20,
    };
    let horizons = [(256, 4), (512, 2), (1024, 1)];
    let adam0 = Adam::new(model.count(), 0.0005);
    let mut cpu_model = model.clone();
    let mut cpu_adam = adam0.clone();
    let mut cpu = Trainer::new_with_horizons("cpu", &cpu_model, &cpu_adam, &horizons, &flow)
        .expect("cpu trainer");
    let mut gpu_model = model.clone();
    let mut gpu_adam = adam0;
    let mut gpu = Trainer::new_with_horizons("cuda", &gpu_model, &gpu_adam, &horizons, &flow)
        .expect("cuda trainer");
    let init_p = cpu_model.parameters();
    let mut cpu_ema = init_p.clone();
    let mut gpu_ema = init_p;
    let ema_decay = 0.9f32;
    cpu.configure_ema(ema_decay, &cpu_ema).unwrap();
    gpu.configure_ema(ema_decay, &gpu_ema).unwrap();

    let mut rng = Rng::new(103);
    let mut cpu_steps = Vec::new();
    let mut gpu_steps = Vec::new();
    let schedule = [(256, 4), (512, 2), (1024, 1), (256, 4)];
    for (step_idx, &(sn, sitems)) in schedule.iter().enumerate() {
        let total_scalars = sn * 2 * sitems;
        let mut clean = vec![0f32; total_scalars];
        for i in 0..sn * sitems {
            clean[2 * i] = 0.2 * (i as f32 * 0.13).sin() + 0.05 * rng.normal();
            clean[2 * i + 1] = 0.2 * (i as f32 * 0.29 + 0.7).cos() + 0.05 * rng.normal();
        }
        let mut mask = vec![0f32; total_scalars];
        for item in 0..sitems {
            let start = item * sn * 2;
            let _ = conditioning::training_mask_channels_into(
                &mut mask[start..start + sn * 2],
                2,
                &mut rng,
            );
        }
        let noise: Vec<f32> = (0..total_scalars).map(|_| rng.normal() * 0.20).collect();
        let t: Vec<f32> = (0..sitems).map(|i| 0.15 + 0.2 * i as f32).collect();
        let lr = 0.0005 * (1. + 0.1 * step_idx as f32);

        cpu.set_shape(sn, sitems).unwrap();
        gpu.set_shape(sn, sitems).unwrap();

        let cs = cpu
            .step(
                &mut cpu_model,
                &mut cpu_adam,
                &flow,
                Batch {
                    clean: &clean,
                    mask: &mask,
                    noise: &noise,
                    t: &t,
                    detail: 0.5,
                    spectral: 1.0,
                    lr,
                    weight_decay: 0.05,
                },
            )
            .unwrap();
        let cp = cpu_model.parameters();
        for (e, &w) in cpu_ema.iter_mut().zip(&cp) {
            *e = ema_decay * *e + (1. - ema_decay) * w;
        }
        let gs = gpu
            .step(
                &mut gpu_model,
                &mut gpu_adam,
                &flow,
                Batch {
                    clean: &clean,
                    mask: &mask,
                    noise: &noise,
                    t: &t,
                    detail: 0.5,
                    spectral: 1.0,
                    lr,
                    weight_decay: 0.05,
                },
            )
            .unwrap();
        cpu_steps.push(cs);
        gpu_steps.push(gs);
    }
    for (step_idx, gs) in gpu_steps.iter().enumerate() {
        let cs = &cpu_steps[step_idx.saturating_sub(1)];
        assert!(
            (cs.loss - gs.loss).abs() <= 2e-5,
            "stereo step {step_idx} lagged loss drift cpu={} cuda={}",
            cs.loss,
            gs.loss
        );
        assert!(
            (cs.norm - gs.norm).abs() <= 5e-5,
            "stereo step {step_idx} lagged norm drift cpu={} cuda={}",
            cs.norm,
            gs.norm
        );
    }
    gpu.sync(&mut gpu_model, &mut gpu_adam).unwrap();
    gpu.sync_ema(&mut gpu_ema).unwrap();
    let max_w = cpu_model
        .parameters()
        .iter()
        .zip(gpu_model.parameters())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_w <= 5e-5, "stereo GluFilm weight drift: {max_w}");
    let max_m = cpu_adam
        .m
        .iter()
        .zip(&gpu_adam.m)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_m <= 5e-5, "stereo GluFilm AdamW m drift: {max_m}");
    let max_e = cpu_ema
        .iter()
        .zip(&gpu_ema)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_e <= 5e-5, "stereo GluFilm EMA drift: {max_e}");

    // Part B: End-to-end stereo multi-horizon training, evaluation, and deterministic resume.
    let root = std::env::temp_dir().join(format!("feral-cuda-stereo-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir(&root).unwrap();
    audio::synthetic_stereo(&root.join("corpus"), 16000).unwrap();

    let make = |run_name: &str, backend: &str, steps: usize, resume| TrainConfig {
        input: root.join("corpus"),
        run: root.join(run_name),
        backend: backend.into(),
        steps,
        seed: 73,
        crop_seconds: 0.032,
        preprocess: Preprocess {
            channel: "stereo".into(),
            ..Preprocess::default()
        },
        lr: 0.0005,
        preview_every: 2,
        checkpoint_every: 2,
        resume,
        weights_only: false,
        grow_crop: false,
        architecture: Architecture {
            block: 4,
            width: 6,
            dilations: vec![1, 2, 1],
            scales: vec![],
            sample_conv: 5,
            block_style: BlockStyle::GluFilm,
            channels: 2,
        },
        flow: flow.clone(),
        detail: 0.5,
        spectral_weight: 1.0,
        time_sampling: TimeSampling::LogitNormal { mean: 0., std: 1. },
        lr_schedule: LrSchedule::Cosine {
            warmup_steps: 2,
            total_steps: 6,
            min_lr: 1e-4,
        },
        ema_decay: 0.9,
        weight_decay: 0.02,
        crop_horizons: vec![0.032, 0.064],
        horizon_schedule: HorizonSchedule::Random,
        horizon_steps: Some(4),
        batch: 2,
        validation: Some(root.join("corpus")),
        validation_every: 2,
    };

    let cpu_all = runtime::train(make("cpu-all", "cpu", 4, None)).expect("cpu train");
    let cuda_all = runtime::train(make("cuda-all", "cuda", 4, None)).expect("cuda train");
    let cuda_first = runtime::train(make("cuda-first", "cuda", 2, None)).expect("cuda first");
    let cuda_resumed =
        runtime::train(make("cuda-resumed", "cuda", 2, Some(cuda_first))).expect("cuda resume");

    let c = Checkpoint::load(&cpu_all).unwrap();
    let g = Checkpoint::load(&cuda_all).unwrap();
    let r = Checkpoint::load(&cuda_resumed).unwrap();
    assert_eq!(c.channels(), 2);
    assert_eq!(g.channels(), 2);
    let max_param = c
        .model
        .parameters()
        .iter()
        .zip(g.model.parameters())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_param <= 5e-5, "stereo CPU/CUDA drift: {max_param}");
    let max_ema = c
        .ema_weights
        .iter()
        .zip(&g.ema_weights)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_ema <= 5e-5, "stereo EMA CPU/CUDA drift: {max_ema}");
    assert_eq!(g.model.parameters(), r.model.parameters());
    assert_eq!(g.ema_weights, r.ema_weights);
    assert_eq!(g.rng.state, r.rng.state);

    let gen_spec = hound::WavReader::open(root.join("cuda-all/generated.wav"))
        .unwrap()
        .spec();
    assert_eq!(gen_spec.channels, 2);

    evaluate::evaluate_with_options(
        &cuda_all,
        &root.join("corpus"),
        &root.join("eval-cpu.json"),
        2,
        1,
        "cpu",
        true,
    )
    .unwrap();
    evaluate::evaluate_with_options(
        &cuda_all,
        &root.join("corpus"),
        &root.join("eval-cuda.json"),
        2,
        1,
        "cuda",
        true,
    )
    .unwrap();
    let ec: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("eval-cpu.json")).unwrap()).unwrap();
    let eg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("eval-cuda.json")).unwrap()).unwrap();
    let diff_eval = (ec["mean_velocity_mse"].as_f64().unwrap()
        - eg["mean_velocity_mse"].as_f64().unwrap())
    .abs();
    assert!(
        diff_eval <= 1e-5,
        "stereo evaluate CPU/CUDA drift: {diff_eval}"
    );
    std::fs::remove_dir_all(root).unwrap();
}
