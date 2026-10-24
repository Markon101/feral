use feral::{
    audio::{self, Preprocess},
    flow::{self, Sampling},
    model::{Architecture, Model, Velocity},
    parameterization::{self, ConfiguredVelocity, FlowConfig, GaussianReference},
    rng::Rng,
    runtime::{self, Checkpoint, TrainConfig},
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
static ID: AtomicUsize = AtomicUsize::new(0);
fn temp() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "feral-pc-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::SeqCst)
    ));
    fs::create_dir(&p).unwrap();
    p
}
fn tiny() -> Architecture {
    Architecture {
        block: 4,
        width: 5,
        dilations: vec![1, 2],
        scales: vec![1, 2],
    }
}
#[test]
fn preconditioned_coefficients_and_gradients_are_finite_at_both_endpoints() {
    let cfg = FlowConfig::Preconditioned {
        data_scale: 0.1,
        prior_std: 1.,
    };
    cfg.validate().unwrap();
    let mut m = Model::new(tiny(), 7);
    let p = m.parameters();
    let mut rng = Rng::new(8);
    let y: Vec<f32> = (0..68).map(|i| 0.1 * (i as f32 * 0.2).sin()).collect();
    let z: Vec<f32> = (0..68).map(|_| rng.normal()).collect();
    let mut mask = vec![0.; 68];
    mask[..16].fill(1.);
    for t in [0., 0.5, 0.999, 1.] {
        let (skip, out, input) = cfg.coefficients(t);
        assert!(skip.is_finite() && out.is_finite() && input.is_finite() && out > 0.);
        let (loss, g) = parameterization::loss_gradient(&m, &cfg, &y, &mask, &z, t);
        assert!(loss.is_finite() && g.iter().all(|v| v.is_finite()));
        let mut offset = 0;
        let mut indices = vec![];
        for l in &m.layers {
            indices.extend([
                offset,
                offset + l.w.len() / 2,
                offset + l.w.len(),
                offset + l.w.len() + l.b.len() - 1,
            ]);
            offset += l.w.len() + l.b.len();
        }
        for i in indices {
            let mut q = p.clone();
            q[i] += 0.002;
            m.set_parameters(&q);
            let hi = parameterization::loss_gradient(&m, &cfg, &y, &mask, &z, t).0;
            q[i] -= 0.004;
            m.set_parameters(&q);
            let lo = parameterization::loss_gradient(&m, &cfg, &y, &mask, &z, t).0;
            let fd = (hi - lo) / 0.004;
            assert!(
                (fd - g[i]).abs() < 0.001 + 0.03 * fd.abs(),
                "t={t}, i={i}, analytical={} numerical={fd}",
                g[i]
            );
        }
        m.set_parameters(&p);
    }
    let c = Sampling {
        steps: 16,
        ..Default::default()
    };
    let adapted = ConfiguredVelocity {
        model: &m,
        config: &cfg,
    };
    let x = flow::sample(&adapted, &y, &mask, &c).unwrap();
    assert_eq!(x, flow::sample(&adapted, &y, &mask, &c).unwrap());
    assert_eq!(&x[..16], &y[..16]);
    // The known/context channel still cannot leak unknown clean target samples.
    let mut altered = y.clone();
    for v in &mut altered[16..] {
        *v += 100.;
    }
    assert_eq!(
        adapted.velocity(&z, &y, &mask, 0.5),
        adapted.velocity(&z, &altered, &mask, 0.5)
    );
}
#[test]
fn gaussian_null_integrates_to_its_declared_variance() {
    let gaussian = GaussianReference {
        block: 4,
        data_scale: 0.1,
    };
    let z = vec![0.; 64];
    let c = Sampling {
        steps: 256,
        collar: 0,
        ..Default::default()
    };
    let x = flow::sample(&gaussian, &z, &z, &c).unwrap();
    let fine = Sampling {
        steps: 512,
        ..c.clone()
    };
    let fine_x = flow::sample(&gaussian, &z, &z, &fine).unwrap();
    let mut rng = Rng::new(c.seed);
    let mut coarse_error = 0f32;
    let mut fine_error = 0f32;
    for (&a, &b) in x.iter().zip(&fine_x) {
        let expected = 0.1 * rng.normal();
        coarse_error = coarse_error.max((a - expected).abs());
        fine_error = fine_error.max((b - expected).abs());
    }
    assert!(
        coarse_error < 0.0001 && fine_error < 0.4 * coarse_error,
        "Gaussian flow convergence {coarse_error} -> {fine_error}"
    );
}
#[test]
fn legacy_adapter_is_bit_exact_and_invalid_config_rejected() {
    let m = Model::new(tiny(), 9);
    let cfg = FlowConfig::Velocity;
    let adapted = ConfiguredVelocity {
        model: &m,
        config: &cfg,
    };
    let x = vec![0.; 64];
    let c = Sampling {
        steps: 7,
        ..Default::default()
    };
    assert_eq!(
        flow::sample(&m, &x, &x, &c).unwrap(),
        flow::sample(&adapted, &x, &x, &c).unwrap()
    );
    for s in [0., f32::NAN, 3.] {
        assert!(FlowConfig::Preconditioned {
            data_scale: s,
            prior_std: 1.
        }
        .validate()
        .is_err());
    }
}
fn config(root: &Path, name: &str, steps: usize, resume: Option<PathBuf>) -> TrainConfig {
    TrainConfig {
        input: root.join("corpus"),
        run: root.join(name),
        steps,
        seed: 9,
        crop_seconds: 0.032,
        preprocess: Preprocess::default(),
        lr: 0.001,
        preview_every: 100,
        checkpoint_every: 100,
        resume,
        weights_only: false,
        architecture: tiny(),
        flow: FlowConfig::Preconditioned {
            data_scale: 0.1,
            prior_std: 1.,
        },
    }
}
#[test]
fn versioned_full_resume_and_config_mismatch_gate() {
    let root = temp();
    audio::synthetic(&root.join("corpus"), 16000).unwrap();
    let all = runtime::train(config(&root, "all", 4, None)).unwrap();
    let first = runtime::train(config(&root, "first", 2, None)).unwrap();
    let next = runtime::train(config(&root, "next", 2, Some(first.clone()))).unwrap();
    let a = Checkpoint::load(&all).unwrap();
    let b = Checkpoint::load(&next).unwrap();
    assert_eq!(a.format_version, 2);
    assert_eq!(a.model.parameters(), b.model.parameters());
    assert_eq!(a.adam.m, b.adam.m);
    assert_eq!(a.rng.state, b.rng.state);
    let mut bad = config(&root, "bad", 1, Some(first));
    bad.flow = FlowConfig::Velocity;
    assert!(runtime::train(bad).is_err());
    let mut wrong = serde_json::to_value(&a).unwrap();
    wrong["format_version"] = 1.into();
    runtime::write_json(&root.join("wrong.json"), &wrong).unwrap();
    assert!(Checkpoint::load(&root.join("wrong.json")).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn calibrated_prior_sampling_matches_training_scale_and_format_gate() {
    let cfg = FlowConfig::Preconditioned {
        data_scale: 0.1,
        prior_std: 0.1,
    };
    cfg.validate().unwrap();
    assert_eq!(cfg.format_version(), 3);
    let mut model = Model::new(tiny(), 5);
    model.set_parameters(&vec![0.; model.count()]);
    let adapter = ConfiguredVelocity {
        model: &model,
        config: &cfg,
    };
    let z = vec![0.; 64];
    let c = Sampling {
        steps: 256,
        collar: 0,
        ..Default::default()
    };
    let x = flow::sample(&adapter, &z, &z, &c).unwrap();
    let mut rng = Rng::new(c.seed);
    for v in x {
        assert!((v - 0.1 * rng.normal()).abs() < 0.00001);
    }
    let root = temp();
    audio::synthetic(&root.join("corpus"), 16000).unwrap();
    let mut c = config(&root, "cal", 2, None);
    c.flow = cfg.clone();
    let path = runtime::train(c).unwrap();
    let loaded = Checkpoint::load(&path).unwrap();
    assert_eq!(loaded.format_version, 3);
    assert_eq!(loaded.flow, cfg);
    let mut bad = serde_json::to_value(&loaded).unwrap();
    bad["format_version"] = 2.into();
    runtime::write_json(&root.join("wrong-v2.json"), &bad).unwrap();
    assert!(Checkpoint::load(&root.join("wrong-v2.json")).is_err());
    let wrong = config(&root, "wrong-resume", 1, Some(path));
    assert!(runtime::train(wrong).is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn calibrated_prior_gradient_and_known_sample_projection() {
    let cfg = FlowConfig::Preconditioned {
        data_scale: 0.1,
        prior_std: 0.1,
    };
    let mut m = Model::new(tiny(), 7);
    let p = m.parameters();
    let mut rng = Rng::new(17);
    let y: Vec<f32> = (0..68).map(|i| 0.1 * (i as f32 * 0.2).sin()).collect();
    let z: Vec<f32> = (0..68).map(|_| 0.1 * rng.normal()).collect();
    let mut mask = vec![0.; 68];
    mask[..16].fill(1.);
    let (_, g) = parameterization::loss_gradient(&m, &cfg, &y, &mask, &z, 0.6);
    for i in [0, 10, 100, p.len() - 1] {
        let mut q = p.clone();
        q[i] += 0.002;
        m.set_parameters(&q);
        let hi = parameterization::loss_gradient(&m, &cfg, &y, &mask, &z, 0.6).0;
        q[i] -= 0.004;
        m.set_parameters(&q);
        let lo = parameterization::loss_gradient(&m, &cfg, &y, &mask, &z, 0.6).0;
        let fd = (hi - lo) / 0.004;
        assert!((fd - g[i]).abs() < 0.001 + 0.03 * fd.abs());
    }
    m.set_parameters(&p);
    let adapter = ConfiguredVelocity {
        model: &m,
        config: &cfg,
    };
    let x = flow::sample(
        &adapter,
        &y,
        &mask,
        &Sampling {
            steps: 8,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(&x[..16], &y[..16]);
}
