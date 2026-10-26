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
