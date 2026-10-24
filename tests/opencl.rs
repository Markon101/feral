#![cfg(feature = "opencl")]
//! Explicit opt-in hardware test: run through scripts/feral-opencl.sh --exec.
use feral::{
    audio::{self, Preprocess},
    model::Architecture,
    parameterization::FlowConfig,
    runtime::{self, TrainConfig},
};
#[test]
#[ignore = "requires actual OpenCL GPU; no CPU fallback"]
fn full_gpu_training_and_checkpoint_roundtrip() {
    let root = std::env::temp_dir().join(format!("feral-opencl-test-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    audio::synthetic(&root.join("corpus"), 16000).unwrap();
    let make = |name: &str, steps, resume| TrainConfig {
        backend: "opencl".into(),
        input: root.join("corpus"),
        run: root.join(name),
        steps,
        seed: 42,
        crop_seconds: 0.032,
        preprocess: Preprocess::default(),
        lr: 0.0003,
        preview_every: 100,
        checkpoint_every: 100,
        resume,
        weights_only: false,
        architecture: Architecture {
            block: 4,
            width: 5,
            dilations: vec![1, 2],
            scales: vec![1, 2],
        },
        flow: FlowConfig::Preconditioned {
            data_scale: 0.1,
            prior_std: 0.1,
        },
    };
    let all = runtime::train(make("all", 6, None)).unwrap();
    let first = runtime::train(make("first", 3, None)).unwrap();
    let resumed = runtime::train(make("resumed", 3, Some(first))).unwrap();
    let a = runtime::Checkpoint::load(&all).unwrap();
    let b = runtime::Checkpoint::load(&resumed).unwrap();
    assert_eq!(a.model.parameters(), b.model.parameters());
    assert_eq!(a.adam.m, b.adam.m);
    assert_eq!(a.adam.v, b.adam.v);
    assert_eq!(a.rng.state, b.rng.state);
    assert_eq!(a.completed_steps, 6);
    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("all/metrics.json")).unwrap()).unwrap();
    assert_eq!(m["backend"], "opencl-f32-resident");
    assert!(root.join("all/generated.wav").exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires actual OpenCL GPU; no CPU fallback"]
fn gpu_reports_optimizer_overflow() {
    use feral::{gpu::OpenClTrainer, model::Model, runtime::Adam};
    let model = Model::new(
        Architecture {
            block: 4,
            width: 5,
            dilations: vec![1, 2],
            scales: vec![1, 2],
        },
        7,
    );
    let mut adam = Adam::new(model.count(), 1.);
    // Finite but deliberately impossible optimizer state: the first bias-corrected
    // moment overflows. Exercise actual device error reporting, not a host input gate.
    adam.m.fill(f32::MAX);
    let mut gpu = OpenClTrainer::new(&model, &adam, 68, FlowConfig::Velocity).unwrap();
    let result = gpu.train_step(&[0.1; 68], &[0.; 68], &[0.2; 68], 0.37);
    assert!(result.err().unwrap().to_string().contains("nonfinite"));
    assert_eq!(gpu.step, 0);
}
