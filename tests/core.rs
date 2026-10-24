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
        architecture: Architecture {
            block: 4,
            width: 5,
            dilations: vec![1, 2],
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
    bad["format_version"] = 2.into();
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
