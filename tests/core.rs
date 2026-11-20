use feral::{
    audio::{self, Preprocess},
    conditioning,
    data::Dataset,
    flow::{self, Sampling},
    model::{Architecture, Model, Velocity},
    rng::Rng,
    runtime::{self, Adam, Checkpoint, Mode, RenderConfig, TrainConfig},
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
static ID: AtomicUsize = AtomicUsize::new(0);
fn temp() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "feral-test-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::SeqCst)
    ));
    fs::create_dir(&p).unwrap();
    p
}
fn tiny() -> Model {
    Model::new(
        Architecture {
            block: 4,
            width: 5,
            dilations: vec![1, 2],
            scales: vec![],
            sample_conv: 0,
        },
        7,
    )
}
fn vectors(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(5);
    let y = (0..n).map(|i| 0.4 * (i as f32 * 0.2).sin()).collect();
    let noise = (0..n).map(|_| rng.normal()).collect();
    let mut mask = vec![0.; n];
    mask[..n / 4].fill(1.);
    (y, noise, mask)
}
#[test]
fn parameter_count_and_shapes() {
    let m = Model::new(Architecture::default(), 1);
    assert_eq!(m.count(), 57952);
    let (y, z, mask) = vectors(64);
    let out = tiny().velocity(&z, &y, &mask, 0.3);
    assert_eq!(out.len(), 64);
    assert!(out.iter().all(|v| v.is_finite()));
}
#[test]
fn finite_loss_nonzero_gradient_optimizer_update() {
    let mut m = tiny();
    let (y, z, mask) = vectors(64);
    let (loss, g) = flow::loss_gradient(&m, &y, &mask, &z, 0.4);
    assert!(loss.is_finite());
    assert!(g.iter().all(|g| g.is_finite()));
    assert!(g.iter().any(|v| v.abs() > 1e-6));
    let old = m.parameters();
    Adam::new(m.count(), 0.001).update(&mut m, &g).unwrap();
    assert_ne!(old, m.parameters());
}
#[test]
fn analytic_gradients_match_finite_differences_every_layer() {
    let mut m = tiny();
    let (y, z, mask) = vectors(32);
    let (_, g) = flow::loss_gradient(&m, &y, &mask, &z, 0.37);
    let p = m.parameters();
    let mut offset = 0;
    let mut indices = vec![];
    for l in &m.layers {
        indices.extend([
            offset,
            offset + l.w.len() / 2,
            offset + l.w.len() - 1,
            offset + l.w.len(),
            offset + l.w.len() + l.b.len() - 1,
        ]);
        offset += l.w.len() + l.b.len();
    }
    for i in indices {
        let mut pp = p.clone();
        pp[i] += 0.001;
        m.set_parameters(&pp);
        let hi = flow::loss_gradient(&m, &y, &mask, &z, 0.37).0;
        pp[i] -= 0.002;
        m.set_parameters(&pp);
        let lo = flow::loss_gradient(&m, &y, &mask, &z, 0.37).0;
        let fd = (hi - lo) / 0.002;
        assert!(
            (fd - g[i]).abs() < 0.0005 + 0.02 * fd.abs(),
            "parameter {i}: analytic {} finite-difference {fd}",
            g[i]
        );
    }
    m.set_parameters(&p);
}
#[test]
fn conditional_path_and_mask_do_not_leak_unknown_clean_audio() {
    let m = tiny();
    let (y, z, mask) = vectors(64);
    let mut other = y.clone();
    for (v, &k) in other.iter_mut().zip(&mask) {
        if k == 0. {
            *v += 10.;
        }
    }
    assert_eq!(
        m.velocity(&z, &y, &mask, 0.5),
        m.velocity(&z, &other, &mask, 0.5)
    );
}
#[test]
fn seeded_sampling_finite_exact_masks_and_disabled_perturbation() {
    let m = tiny();
    let (y, _, mask) = vectors(64);
    let c = Sampling {
        steps: 5,
        collar: 0,
        ..Default::default()
    };
    let a = flow::sample(&m, &y, &mask, &c).unwrap();
    assert_eq!(a, flow::sample(&m, &y, &mask, &c).unwrap());
    assert!(a.iter().all(|v| v.is_finite()));
    for i in 0..64 {
        if mask[i] == 1. {
            assert_eq!(a[i].to_bits(), y[i].to_bits());
        }
    }
    let mut weird = c.clone();
    weird.perturb = 0.1;
    assert_ne!(a, flow::sample(&m, &y, &mask, &weird).unwrap());
    weird.perturb = 0.;
    assert_eq!(a, flow::sample(&m, &y, &mask, &weird).unwrap());
}
struct Constant;
impl Velocity for Constant {
    fn block(&self) -> usize {
        4
    }
    fn velocity(&self, x: &[f32], _: &[f32], _: &[f32], _: f32) -> Vec<f32> {
        vec![0.25; x.len()]
    }
}
#[test]
fn integration_direction_euler_and_heun() {
    for solver in ["euler", "heun"] {
        let c = Sampling {
            steps: 8,
            noise_scale: 0.,
            solver: solver.into(),
            collar: 0,
            ..Default::default()
        };
        let out = flow::sample(&Constant, &[0.; 16], &[0.; 16], &c).unwrap();
        assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }
}
#[test]
fn protected_regions_and_boundary_collar() {
    let known = vec![0.3; 128];
    let mut mask = vec![1.; 128];
    mask[32..96].fill(0.);
    let mut x = vec![-0.8; 128];
    conditioning::project(&mut x, &known, &mask);
    let before = conditioning::boundary_jumps(&x, &mask);
    conditioning::continuity(&mut x, &known, &mask, 16);
    let after = conditioning::boundary_jumps(&x, &mask);
    assert!(before.iter().all(|v| *v > 1.));
    assert_eq!(after, vec![0., 0.]);
    for i in 0..128 {
        if mask[i] == 1. {
            assert_eq!(x[i], known[i]);
        }
    }
}
#[test]
fn tasks_have_unknowns_and_cover_all_modes() {
    let mut rng = Rng::new(31);
    let mut names = std::collections::BTreeSet::new();
    for _ in 0..100 {
        let (m, name) = conditioning::training_mask(256, &mut rng);
        assert!(m.contains(&0.));
        names.insert(name);
    }
    assert_eq!(names.len(), 4);
}
#[test]
fn wav_roundtrip_exact_and_no_overwrite() {
    let dir = temp();
    let path = dir.join("signal.wav");
    let (y, _, _) = vectors(4096);
    audio::write_wav(&path, &y, 16000).unwrap();
    let r = hound::WavReader::open(&path).unwrap();
    let x: Vec<f32> = r.into_samples::<f32>().map(Result::unwrap).collect();
    assert_eq!(x, y);
    assert!(audio::write_wav(&path, &y, 16000).is_err());
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn decoder_supported_formats_and_corrupt_file() {
    let dir = temp();
    let input = dir.join("input ; literal $(name).wav");
    let (y, _, _) = vectors(4096);
    audio::write_wav(&input, &y, 16000).unwrap();
    let original = fs::read(&input).unwrap();
    let p = Preprocess {
        normalize: false,
        ..Default::default()
    };
    let x = audio::decode(&input, &p).unwrap();
    assert_eq!(x, y);
    for (ext, codec) in [
        ("flac", "flac"),
        ("mp3", "libmp3lame"),
        ("ogg", "libvorbis"),
        ("opus", "libopus"),
        ("webm", "libopus"),
        ("m4a", "aac"),
    ] {
        let path = dir.join(format!("encoded.{ext}"));
        let result = audio::decoder_command("ffmpeg")
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(&input)
            .args(["-channel_layout", "mono", "-c:a", codec])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let out = audio::decode(&path, &p).unwrap();
        assert!(!out.is_empty());
        assert!(audio::rms(&out) > 0.01);
        assert!(out.iter().all(|v| v.is_finite()));
    }
    assert_eq!(fs::read(&input).unwrap(), original);
    fs::write(dir.join("broken.mp3"), "not audio").unwrap();
    assert!(audio::decode(&dir.join("broken.mp3"), &p).is_err());
    let d = Dataset::load(&dir, p, 1024).unwrap();
    assert_eq!(d.audio.len(), 7);
    assert_eq!(d.identity.rejected.len(), 1);
    fs::remove_dir_all(dir).unwrap();
}
fn config(root: &Path, run: &str, steps: usize, resume: Option<PathBuf>) -> TrainConfig {
    TrainConfig {
        flow: feral::parameterization::FlowConfig::Velocity,
        detail: 0.,
        spectral_weight: 0.,
        time_sampling: feral::parameterization::TimeSampling::Uniform,
        lr_schedule: feral::parameterization::LrSchedule::Constant,
        ema_decay: 0.,
        backend: "cpu".into(),
        input: root.join("corpus"),
        run: root.join(run),
        steps,
        seed: 9,
        crop_seconds: 0.032,
        preprocess: Preprocess::default(),
        lr: 0.001,
        preview_every: 100,
        checkpoint_every: 100,
        resume,
        weights_only: false,
        grow_crop: false,
        batch: 1,
        validation: None,
        validation_every: 10,
        architecture: Architecture {
            block: 4,
            width: 5,
            dilations: vec![1, 2],
            scales: vec![],
            sample_conv: 0,
        },
    }
}
#[test]
fn actual_training_checkpoint_reload_full_resume_and_determinism() {
    let dir = temp();
    audio::synthetic(&dir.join("corpus"), 16000).unwrap();
    let all = runtime::train(config(&dir, "all", 4, None)).unwrap();
    let first = runtime::train(config(&dir, "first", 2, None)).unwrap();
    let next = runtime::train(config(&dir, "next", 2, Some(first.clone()))).unwrap();
    let a = Checkpoint::load(&all).unwrap();
    let b = Checkpoint::load(&next).unwrap();
    assert_eq!(a.completed_steps, 4);
    assert_eq!(a.model.parameters(), b.model.parameters());
    assert_eq!(a.adam.m, b.adam.m);
    assert_eq!(a.adam.v, b.adam.v);
    assert_eq!(a.rng.state, b.rng.state);
    let reference = dir.join("corpus/pattern-0.wav");
    let original: Vec<f32> = hound::WavReader::open(&reference)
        .unwrap()
        .into_samples::<f32>()
        .map(Result::unwrap)
        .collect();
    for (name, mode) in [("continued", Mode::Continue), ("inpainted", Mode::Inpaint)] {
        let output = dir.join(format!("{name}.wav"));
        runtime::render(RenderConfig {
            checkpoint: all.clone(),
            input: Some(reference.clone()),
            output: output.clone(),
            seconds: 0.032,
            start: 0.5,
            end: 0.55,
            sampling: Sampling {
                steps: 3,
                ..Default::default()
            },
            mode,
            backend: "cpu".into(),
            use_ema: true,
            max_input_seconds: 600.,
        })
        .unwrap();
        let rendered: Vec<f32> = hound::WavReader::open(&output)
            .unwrap()
            .into_samples::<f32>()
            .map(Result::unwrap)
            .collect();
        if name == "continued" {
            assert_eq!(&rendered[..original.len()], original);
            assert_eq!(rendered.len(), original.len() + 512);
        } else {
            assert_eq!(rendered.len(), original.len());
            assert_eq!(&rendered[..8000], &original[..8000]);
            assert_eq!(&rendered[8800..], &original[8800..]);
            assert_ne!(&rendered[8000..8800], &original[8000..8800]);
        }
    }
    assert_ne!(
        Checkpoint::load(&dir.join("all/initial.json"))
            .unwrap()
            .model
            .parameters(),
        a.model.parameters()
    );
    let c = Sampling {
        steps: 3,
        ..Default::default()
    };
    let z = vec![0.; 512];
    assert_eq!(
        flow::sample(&a.model, &z, &z, &c).unwrap(),
        flow::sample(&b.model, &z, &z, &c).unwrap()
    );
    assert!(a.clone().save(&all).is_err());
    let mut bad = serde_json::to_value(&a).unwrap();
    bad["format_version"] = 99.into();
    runtime::write_json(&dir.join("bad.json"), &bad).unwrap();
    assert!(Checkpoint::load(&dir.join("bad.json")).is_err());
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn invalid_configs_rejected() {
    let m = tiny();
    let (y, _, mask) = vectors(32);
    for scale in [f32::NAN, -1., 5.] {
        let c = Sampling {
            noise_scale: scale,
            ..Default::default()
        };
        assert!(flow::sample(&m, &y, &mask, &c).is_err());
    }
    let c = Sampling {
        steps: 0,
        ..Default::default()
    };
    assert!(flow::sample(&m, &y, &mask, &c).is_err());
}

#[test]
fn smoke_cli_creates_nested_parent_and_refuses_existing_run() {
    let dir = temp();
    let run = dir.join("new-parent/smoke");
    let invoke = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_feral"))
            .args(["smoke", "--steps", "1", "--run"])
            .arg(&run)
            .output()
            .unwrap()
    };
    let first = invoke();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(run.join("training/checkpoint.json").exists());
    assert!(run.join("training/generated.wav").exists());
    assert!(!invoke().status.success());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pyramid_gradients_shapes_and_partial_pool_groups() {
    let a = Architecture {
        block: 4,
        width: 5,
        dilations: vec![1, 2, 4],
        scales: vec![1, 2, 4],
        sample_conv: 0,
    };
    a.validate().unwrap();
    let mut m = Model::new(a, 7);
    // 17 frames deliberately leaves incomplete pooling groups.
    let (y, z, mask) = vectors(68);
    let p = m.parameters();
    let (loss, g) = flow::loss_gradient(&m, &y, &mask, &z, 0.37);
    assert!(loss.is_finite());
    let mut offset = 0;
    let mut indices = vec![];
    for l in &m.layers {
        indices.extend([
            offset,
            offset + l.w.len() / 2,
            offset + l.w.len() - 1,
            offset + l.w.len(),
            offset + l.w.len() + l.b.len() - 1,
        ]);
        offset += l.w.len() + l.b.len();
    }
    for i in indices {
        let mut pp = p.clone();
        pp[i] += 0.002;
        m.set_parameters(&pp);
        let hi = flow::loss_gradient(&m, &y, &mask, &z, 0.37).0;
        pp[i] -= 0.004;
        m.set_parameters(&pp);
        let lo = flow::loss_gradient(&m, &y, &mask, &z, 0.37).0;
        let fd = (hi - lo) / 0.004;
        assert!(
            (fd - g[i]).abs() < 0.0005 + 0.02 * fd.abs(),
            "pyramid gradient {i}: {} vs {fd}",
            g[i]
        );
    }
    m.set_parameters(&p);
    Adam::new(m.count(), 0.001).update(&mut m, &g).unwrap();
    assert_ne!(p, m.parameters());
    let c = Sampling {
        steps: 3,
        ..Default::default()
    };
    let out = flow::sample(&m, &y, &mask, &c).unwrap();
    assert_eq!(out.len(), 68);
    for i in 0..68 {
        if mask[i] == 1. {
            assert_eq!(out[i], y[i]);
        }
    }
}
#[test]
fn pyramid_adds_real_differentiable_context_without_parameters() {
    let baseline = Model::new(Architecture::default(), 7);
    let pyramid = Model::new(Architecture::pyramid(), 7);
    assert_eq!(baseline.parameters(), pyramid.parameters());
    assert_eq!(baseline.architecture.context_samples(), 992);
    assert_eq!(pyramid.architecture.context_samples(), 7488);
    let n = 8192;
    let x = vec![0.; n];
    let mut changed = x.clone();
    changed[4096 + 1024] = 1.;
    let zeros = vec![0.; n];
    let b0 = baseline.velocity(&x, &zeros, &zeros, 0.5);
    let b1 = baseline.velocity(&changed, &zeros, &zeros, 0.5);
    assert_eq!(&b0[4096..4128], &b1[4096..4128]);
    let p0 = pyramid.velocity(&x, &zeros, &zeros, 0.5);
    let p1 = pyramid.velocity(&changed, &zeros, &zeros, 0.5);
    assert!(p0[4096..4128]
        .iter()
        .zip(&p1[4096..4128])
        .any(|(a, b)| (a - b).abs() > 1e-7));
    let old = serde_json::to_value(&baseline).unwrap();
    assert!(old["architecture"].get("scales").is_none());
    let restored: Model = serde_json::from_value(old).unwrap();
    assert_eq!(b0, restored.velocity(&x, &zeros, &zeros, 0.5));
}

#[test]
fn acoustic_metrics_and_copy_detector_have_hostile_controls() {
    use feral::evaluate::{acoustic, nearest_correlation};
    let mut rng = Rng::new(12);
    let x: Vec<f32> = (0..512).map(|_| rng.normal() * 0.2).collect();
    let exact = acoustic(&x, &x);
    assert_eq!(exact.mse, 0.);
    assert_eq!(exact.derivative_mse, 0.);
    assert_eq!(exact.log_spectral_l1, 0.);
    assert!(exact.coarse_mse.iter().all(|&v| v == 0.));
    let mut changed = x.clone();
    changed[100] += 0.1;
    let m = acoustic(&x, &changed);
    assert!(m.mse > 0. && m.log_spectral_l1 > 0. && m.derivative_mse > 0.);
    let mut reference = vec![0.; 64];
    reference.extend(&x);
    reference.extend([0.; 64]);
    assert!(nearest_correlation(&x, &reference, 16) > 0.99999);
    let noise: Vec<f32> = (0..512).map(|_| rng.normal()).collect();
    assert!(nearest_correlation(&noise, &reference, 16) < 0.5);
}
#[test]
fn heldout_probe_controls_and_pyramid_full_resume() {
    let m = tiny();
    let (y, z, _) = vectors(64);
    let p = feral::evaluate::probe(&m, &y, &z, 2, 0.5);
    assert!(p.velocity_mse.is_finite() && p.endpoint.log_spectral_l1.is_finite());
    assert_eq!(p.task, "hole");
    let dir = temp();
    audio::synthetic(&dir.join("corpus"), 16000).unwrap();
    let cfg = |name, steps, resume| {
        let mut c = config(&dir, name, steps, resume);
        c.architecture.scales = vec![2, 4];
        c
    };
    let all = runtime::train(cfg("p-all", 4, None)).unwrap();
    let first = runtime::train(cfg("p-first", 2, None)).unwrap();
    let second = runtime::train(cfg("p-second", 2, Some(first))).unwrap();
    let a = Checkpoint::load(&all).unwrap();
    let b = Checkpoint::load(&second).unwrap();
    assert_eq!(a.model.parameters(), b.model.parameters());
    assert_eq!(a.adam.m, b.adam.m);
    assert_eq!(a.rng.state, b.rng.state);
    feral::evaluate::evaluate(
        &all,
        &dir.join("corpus/pattern-1.wav"),
        &dir.join("eval.json"),
        1,
        21,
    )
    .unwrap();
    assert!(feral::evaluate::evaluate(
        &all,
        &dir.join("corpus/pattern-1.wav"),
        &dir.join("eval.json"),
        1,
        21
    )
    .is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn continuity_retains_feasible_transient_slopes() {
    let mut known = vec![0.; 32];
    known[6] = 0.05;
    known[7] = 0.20;
    known[24] = 0.10;
    known[25] = -0.02;
    let mut mask = vec![1.; 32];
    mask[8..24].fill(0.);
    let mut x = vec![-0.4; 32];
    conditioning::continuity(&mut x, &known, &mask, 8);
    assert!(((x[8] - x[7]) - (known[7] - known[6])).abs() < 1e-6);
    assert!(((x[24] - x[23]) - (known[25] - known[24])).abs() < 1e-6);
    for i in 0..32 {
        if mask[i] == 1. {
            assert_eq!(x[i], known[i]);
        }
    }
}

#[test]
fn sixteen_kilohertz_validation_loss_matches_the_sample_pinned_window() {
    // The probe/validation window used to be pinned at 64,000 samples, which is exactly 4.0 s at
    // 16 kHz. Deriving it from the sample rate must leave every 16 kHz run bit-identical, so this
    // pins the loss measured with the pre-change window (crop 5 s, so the clamp is exercised).
    let dir = temp();
    let corpus = dir.join("corpus");
    fs::create_dir(&corpus).unwrap();
    let x: Vec<f32> = (0..16000 * 6)
        .map(|i| 0.2 * (i as f32 * 0.013).sin())
        .collect();
    audio::write_wav(&corpus.join("held.wav"), &x, 16000).unwrap();
    let mut c = config(&dir, "pinned", 1, None);
    c.crop_seconds = 5.;
    c.validation = Some(corpus);
    runtime::train(c).unwrap();
    let metrics: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("pinned/metrics.json")).unwrap()).unwrap();
    let loss = metrics["validation_initial_loss"].as_f64().unwrap() as f32;
    // Captured from the pre-change build (`eval_n = n.min(64_000)`) with this exact config:
    // decimal 1.4121605, bits 0x3fb4c1ad. Any difference here means the duration-derived window
    // moved a 16 kHz number and broke comparability with every existing run.
    assert_eq!(
        loss.to_bits(),
        0x3fb4c1ad,
        "16 kHz validation loss changed: {loss}"
    );
    assert_eq!(
        runtime::probe_window_samples(16_000, 80_000, 4),
        64_000,
        "the 16 kHz window must stay 64,000 samples"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn probe_window_is_four_seconds_at_every_supported_rate() {
    // 4.0 s of audio, whatever the rate: the fixed probe and held-out validation window used to be
    // 64,000 samples, which is 4 s at 16 kHz but 1.33 s at 48 kHz.
    for (rate, four_seconds) in [(16_000, 64_000), (24_000, 96_000), (48_000, 192_000)] {
        for block in [16, 32, 64, 128] {
            assert_eq!(
                runtime::probe_window_samples(rate, four_seconds * 2, block),
                four_seconds,
                "{rate} Hz window"
            );
            let short = block * 100;
            assert_eq!(
                runtime::probe_window_samples(rate, short, block),
                short,
                "{rate} Hz window must clamp to a shorter crop"
            );
        }
    }
    // A rate whose 4 s is not frame-aligned rounds down to a whole frame, never above 4 s.
    let window = runtime::probe_window_samples(44_100, 512_000, 128);
    assert!(window <= 176_400 && window.is_multiple_of(128), "{window}");
    // 16 kHz keeps the old value bit for bit, so no existing run's window moves.
    for crop in [512, 63_936, 64_000, 80_000, 512_000] {
        assert_eq!(
            runtime::probe_window_samples(16_000, crop, 64),
            crop.min(64_000)
        );
    }
}

#[test]
fn protected_samples_are_asserted_exactly_and_violations_are_refused() {
    let dir = temp();
    audio::synthetic(&dir.join("corpus"), 16000).unwrap();
    let checkpoint = runtime::train(config(&dir, "protected", 2, None)).unwrap();
    let reference = dir.join("corpus/pattern-0.wav");
    let original: Vec<f32> = hound::WavReader::open(&reference)
        .unwrap()
        .into_samples::<f32>()
        .map(Result::unwrap)
        .collect();
    let output = dir.join("continued.wav");
    runtime::render(RenderConfig {
        checkpoint: checkpoint.clone(),
        input: Some(reference),
        output: output.clone(),
        seconds: 0.032,
        start: 0.,
        end: 0.,
        sampling: Sampling {
            steps: 3,
            ..Default::default()
        },
        mode: Mode::Continue,
        backend: "cpu".into(),
        use_ema: true,
        max_input_seconds: 600.,
    })
    .unwrap();
    let rendered: Vec<f32> = hound::WavReader::open(&output)
        .unwrap()
        .into_samples::<f32>()
        .map(Result::unwrap)
        .collect();
    // Continue protects the last crop/2 samples of the input. They must survive conditioning,
    // sampling, the WAV roundtrip and the metadata write exactly, not just be reported as exact.
    let context = (Checkpoint::load(&checkpoint).unwrap().crop_samples / 2).min(original.len());
    let offset = original.len() - context;
    assert_eq!(&rendered[offset..original.len()], &original[offset..]);
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(output.with_extension("json")).unwrap()).unwrap();
    assert!(meta["protected_samples_exact"].as_bool().unwrap());
    let mask: Vec<f32> = (0..original.len())
        .map(|i| if i >= offset { 1. } else { 0. })
        .collect();
    runtime::ensure_protected(&rendered[..original.len()], &original, &mask, "continue").unwrap();
    // Negative control: one flipped mantissa bit inside the protected region must be refused,
    // and the error must name the mode and the first offending index.
    let mut violated = rendered[..original.len()].to_vec();
    violated[offset + 3] = f32::from_bits(violated[offset + 3].to_bits() ^ 1);
    let err = runtime::ensure_protected(&violated, &original, &mask, "continue").unwrap_err();
    assert!(
        err.to_string().contains(&format!(
            "continue render changed protected sample {}",
            offset + 3
        )),
        "{err}"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn collar_tangent_adapts_to_headroom_and_limiter_handles_impossible_slope() {
    let mut known = vec![0.; 64];
    known[14] = 0.85;
    known[15] = 0.90;
    let mut mask = vec![0.; 64];
    mask[..16].fill(1.);
    let mut x = vec![0.2; 64];
    conditioning::continuity(&mut x, &known, &mask, 16);
    assert!((x[16] - 0.95).abs() < 1e-6);
    assert!(x[16..].iter().all(|v| v.abs() <= 0.98));
    known[14] = 0.5;
    known[15] = 0.97;
    let c = Sampling {
        steps: 2,
        collar: 16,
        ..Default::default()
    };
    let out = flow::sample(&Constant, &known, &mask, &c).unwrap();
    assert!(out[16..].iter().all(|v| v.abs() <= 0.98));
    assert_eq!(&out[..16], &known[..16]);
}

#[test]
fn continuity_one_sample_hole_is_symmetric() {
    let mut known = vec![0.; 16];
    known[6] = 0.20;
    known[7] = 0.30;
    known[9] = 0.30;
    known[10] = 0.20;
    let mut mask = vec![1.; 16];
    mask[8] = 0.;
    let mut x = vec![-0.9; 16];
    conditioning::continuity(&mut x, &known, &mask, 16);
    assert!((x[8] - 0.40).abs() < 1e-6, "got {}", x[8]);
    for i in 0..16 {
        if mask[i] == 1. {
            assert_eq!(x[i], known[i]);
        }
    }
}

#[test]
fn variation_mode_retains_anchors_and_regenerates_gaps() {
    let dir = temp();
    audio::synthetic(&dir.join("corpus"), 16000).unwrap();
    let ck = runtime::train(config(&dir, "var-train", 2, None)).unwrap();
    let reference = dir.join("corpus/pattern-0.wav");
    let original: Vec<f32> = hound::WavReader::open(&reference)
        .unwrap()
        .into_samples::<f32>()
        .map(Result::unwrap)
        .collect();
    let output = dir.join("variation.wav");
    runtime::render(RenderConfig {
        checkpoint: ck,
        input: Some(reference),
        output: output.clone(),
        seconds: 0.032,
        start: 0.,
        end: 0.,
        sampling: Sampling {
            steps: 3,
            ..Default::default()
        },
        mode: Mode::Variation,
        backend: "cpu".into(),
        use_ema: true,
        max_input_seconds: 600.,
    })
    .unwrap();
    let rendered: Vec<f32> = hound::WavReader::open(&output)
        .unwrap()
        .into_samples::<f32>()
        .map(Result::unwrap)
        .collect();
    assert_eq!(rendered.len(), original.len());
    let stride = 16000usize / 32;
    let mut checked_anchor = false;
    let mut changed_gap = false;
    for i in 0..original.len() {
        if (i / stride).is_multiple_of(4) {
            assert_eq!(rendered[i].to_bits(), original[i].to_bits());
            checked_anchor = true;
        } else if rendered[i].to_bits() != original[i].to_bits() {
            changed_gap = true;
        }
    }
    assert!(checked_anchor && changed_gap);
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(output.with_extension("json")).unwrap()).unwrap();
    assert_eq!(meta["mode"], "variation-masked-anchors");
    assert_eq!(meta["protected_samples_exact"], true);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn spectral_endpoint_loss_matches_finite_differences_and_respects_mask() {
    use feral::parameterization::{self, FlowConfig};
    use feral::spectral::SpectralEngine;

    let engine = SpectralEngine::new();
    let n = 192usize;
    let (y, z, mut mask) = vectors(n);
    mask[..48].fill(1.);
    mask[48..].fill(0.);

    // Direct finite-difference check on SpectralEngine::loss_and_yhat_grad (broadband difference
    // so STFT bins are well away from the smooth-L1 corner at 0).
    let mut rng = Rng::new(77);
    let yhat: Vec<f32> = y.iter().map(|&v| v + 0.12 * rng.normal()).collect();
    let (base_loss, dy_yhat) = engine.loss_and_yhat_grad(&y, &yhat);
    assert!(base_loss.is_finite() && base_loss > 0.);
    for i in [0, 7, 32, 95, 127, 191] {
        let eps = 5e-4f32;
        let mut hi = yhat.clone();
        hi[i] += eps;
        let mut lo = yhat.clone();
        lo[i] -= eps;
        let fd = (engine.loss_and_yhat_grad(&y, &hi).0 - engine.loss_and_yhat_grad(&y, &lo).0)
            / (2. * eps);
        assert!(
            (fd - dy_yhat[i]).abs() < 1e-4 + 0.02 * fd.abs(),
            "spectral yhat grad[{i}]: analytic={} fd={fd}",
            dy_yhat[i]
        );
    }

    // End-to-end finite-difference check on model parameters for both Velocity & Preconditioned.
    for flow in [
        FlowConfig::Velocity,
        FlowConfig::Preconditioned {
            data_scale: 0.16,
            prior_std: 0.25,
        },
    ] {
        let mut m = tiny();
        let p = m.parameters();
        let t = 0.62f32;
        let (loss, g) = parameterization::loss_gradient_with_spectral(
            &m, &flow, &y, &mask, &z, t, 0.5, 0.5, &engine,
        );
        assert!(loss.is_finite());
        // Zero spectral weight reproduces loss_gradient_detailed bit-identically.
        assert_eq!(
            parameterization::loss_gradient_with_spectral(
                &m, &flow, &y, &mask, &z, t, 0.5, 0., &engine
            ),
            parameterization::loss_gradient_detailed(&m, &flow, &y, &mask, &z, t, 0.5)
        );
        // Check gradient at protected samples is strictly zero in endpoint_loss_gradient.
        let pred = vec![0.1; n];
        let (_, dy_spec) = engine.endpoint_loss_gradient(&flow, &y, &mask, &z, &pred, t, 0.5);
        assert!(dy_spec[..48].iter().all(|&v| v == 0.));
        assert!(dy_spec[48..].iter().any(|&v| v.abs() > 1e-6));

        for i in [0, 11, p.len() / 2, p.len() - 5, p.len() - 1] {
            let eps = 2e-3f32;
            let mut q = p.clone();
            q[i] += eps;
            m.set_parameters(&q);
            let hi = parameterization::loss_gradient_with_spectral(
                &m, &flow, &y, &mask, &z, t, 0.5, 0.5, &engine,
            )
            .0;
            q[i] -= 2. * eps;
            m.set_parameters(&q);
            let lo = parameterization::loss_gradient_with_spectral(
                &m, &flow, &y, &mask, &z, t, 0.5, 0.5, &engine,
            )
            .0;
            let fd = (hi - lo) / (2. * eps);
            assert!(
                (fd - g[i]).abs() < 1e-3 + 0.03 * fd.abs(),
                "spectral param grad ({flow:?}) [{i}]: analytic={} fd={fd}",
                g[i]
            );
        }
        m.set_parameters(&p);
    }
}

#[test]
fn sample_conv_gradients_match_finite_differences_and_no_leakage() {
    let base_arch = Architecture {
        block: 4,
        width: 5,
        dilations: vec![1, 2],
        scales: vec![1, 2],
        sample_conv: 0,
    };
    let sc_arch = base_arch.clone().with_sample_conv(7).unwrap();
    assert_eq!(
        sc_arch.context_samples(),
        base_arch.context_samples() + 4 * 6
    );
    assert!(base_arch.clone().with_sample_conv(4).is_err());
    assert!(base_arch.clone().with_sample_conv(1).is_err());
    assert!(base_arch.clone().with_sample_conv(17).is_err());

    // Legacy serialization omits sample_conv/stem/head when disabled.
    let legacy_model = Model::new(base_arch, 7);
    let legacy_json = serde_json::to_value(&legacy_model).unwrap();
    assert!(legacy_json["architecture"].get("sample_conv").is_none());
    assert!(legacy_json.get("stem").is_none());
    assert!(legacy_json.get("head").is_none());

    let mut m = Model::new(sc_arch, 7);
    m.validate().unwrap();
    let (y, z, mask) = vectors(68);
    let p = m.parameters();
    let (loss, g) = flow::loss_gradient(&m, &y, &mask, &z, 0.37);
    assert!(loss.is_finite() && g.iter().all(|v| v.is_finite()));

    let mut offset = 0;
    let mut indices = vec![];
    for l in m.layers.iter().chain(&m.stem).chain(&m.head) {
        indices.extend([
            offset,
            offset + l.w.len() / 2,
            offset + l.w.len() - 1,
            offset + l.w.len(),
            offset + l.w.len() + l.b.len() - 1,
        ]);
        offset += l.w.len() + l.b.len();
    }
    assert_eq!(offset, p.len());
    for i in indices {
        let eps = 2e-3f32;
        let mut q = p.clone();
        q[i] += eps;
        m.set_parameters(&q);
        let hi = flow::loss_gradient(&m, &y, &mask, &z, 0.37).0;
        q[i] -= 2. * eps;
        m.set_parameters(&q);
        let lo = flow::loss_gradient(&m, &y, &mask, &z, 0.37).0;
        let fd = (hi - lo) / (2. * eps);
        assert!(
            (fd - g[i]).abs() < 1e-3 + 0.03 * fd.abs(),
            "sample_conv grad[{i}]: analytic={} fd={fd}",
            g[i]
        );
    }
    m.set_parameters(&p);

    // Unknown clean audio must never leak through the sample-level stem.
    let mut altered = y.clone();
    for (v, &k) in altered.iter_mut().zip(&mask) {
        if k == 0. {
            *v += 50.;
        }
    }
    assert_eq!(
        m.velocity(&z, &y, &mask, 0.5),
        m.velocity(&z, &altered, &mask, 0.5)
    );
}

#[test]
fn logit_normal_cosine_ema_and_sample_conv_full_resume() {
    use feral::parameterization::{FlowConfig, LrSchedule, TimeSampling};

    // Unit-check TimeSampling::LogitNormal and LrSchedule::Cosine.
    let ts = TimeSampling::LogitNormal { mean: 0., std: 1. };
    ts.validate().unwrap();
    let mut rng = Rng::new(123);
    let samples: Vec<f32> = (0..256).map(|_| ts.sample(&mut rng)).collect();
    assert!(samples.iter().all(|&v| v > 0. && v < 1.));
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    assert!((mean - 0.5).abs() < 0.1, "mean={mean}");

    let sched = LrSchedule::Cosine {
        warmup_steps: 2,
        total_steps: 6,
        min_lr: 1e-4,
    };
    sched.validate(1e-3).unwrap();
    assert!((sched.lr_at(1e-3, 1) - 5e-4).abs() < 1e-7);
    assert!((sched.lr_at(1e-3, 2) - 1e-3).abs() < 1e-7);
    assert!((sched.lr_at(1e-3, 4) - 5.5e-4).abs() < 1e-6);
    assert!((sched.lr_at(1e-3, 6) - 1e-4).abs() < 1e-7);
    assert!((sched.lr_at(1e-3, 10) - 1e-4).abs() < 1e-7);

    // Full deterministic resume with all new features enabled simultaneously.
    let dir = temp();
    audio::synthetic(&dir.join("corpus"), 16000).unwrap();
    let make = |name: &str, steps: usize, resume: Option<PathBuf>| {
        let mut c = config(&dir, name, steps, resume);
        c.flow = FlowConfig::Preconditioned {
            data_scale: 0.16,
            prior_std: 0.25,
        };
        c.architecture = c.architecture.with_sample_conv(5).unwrap();
        c.spectral_weight = 0.25;
        c.time_sampling = ts;
        c.lr_schedule = sched;
        c.ema_decay = 0.9;
        c
    };
    let all = runtime::train(make("all", 4, None)).unwrap();
    let first = runtime::train(make("first", 2, None)).unwrap();
    let resumed = runtime::train(make("resumed", 2, Some(first.clone()))).unwrap();
    let a = Checkpoint::load(&all).unwrap();
    let b = Checkpoint::load(&resumed).unwrap();
    assert_eq!(a.model.parameters(), b.model.parameters());
    assert_eq!(a.ema_weights, b.ema_weights);
    assert_eq!(a.adam.m, b.adam.m);
    assert_eq!(a.adam.v, b.adam.v);
    assert_eq!(a.rng.state, b.rng.state);
    assert!(a.has_ema());
    assert_ne!(a.ema_weights, a.model.parameters());
    assert_ne!(
        a.sampling_model(true).parameters(),
        a.sampling_model(false).parameters()
    );

    // Mismatched resume gates reject altered spectral_weight / ema_decay.
    let mut bad_spec = make("bad-spec", 1, Some(first.clone()));
    bad_spec.spectral_weight = 0.;
    assert!(runtime::train(bad_spec).is_err());
    let mut bad_ema = make("bad-ema", 1, Some(first));
    bad_ema.ema_decay = 0.;
    assert!(runtime::train(bad_ema).is_err());

    // Render with EMA vs raw weights both succeed and record `using_ema`.
    for (name, use_ema) in [("ema", true), ("raw", false)] {
        let out = dir.join(format!("gen-{name}.wav"));
        runtime::render(RenderConfig {
            checkpoint: all.clone(),
            input: None,
            output: out.clone(),
            seconds: 0.032,
            start: 0.,
            end: 0.,
            sampling: Sampling {
                steps: 3,
                ..Default::default()
            },
            mode: Mode::Generate,
            backend: "cpu".into(),
            use_ema,
            max_input_seconds: 600.,
        })
        .unwrap();
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(out.with_extension("json")).unwrap()).unwrap();
        assert_eq!(meta["using_ema"], use_ema);
    }
    fs::remove_dir_all(dir).unwrap();
}
