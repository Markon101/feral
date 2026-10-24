use feral::{
    audio::{self, Preprocess},
    data::{Dataset, DatasetIdentity, Source},
    rng::Rng,
};
use std::path::PathBuf;

fn fixture(audio: Vec<Vec<f32>>, silence_rms: f32) -> Dataset {
    let sources = audio
        .iter()
        .enumerate()
        .map(|(i, samples)| Source {
            path: PathBuf::from(format!("fixture-{i}.wav")),
            sha256: format!("fixture-{i}"),
            samples: samples.len(),
            rms: audio::rms(samples),
        })
        .collect();
    Dataset {
        audio,
        identity: DatasetIdentity {
            sources,
            rejected: vec![],
            preprocess: Preprocess {
                silence_rms,
                ..Preprocess::default()
            },
            manifest_hash: "in-memory-fixture".into(),
        },
    }
}

// Frozen pre-trace algorithm: this intentionally stays independent of crop_traced.
fn original_crop(data: &Dataset, n: usize, rng: &mut Rng) -> Vec<f32> {
    for _ in 0..16 {
        let a = &data.audio[rng.index(data.audio.len())];
        let start = rng.index(a.len() - n + 1);
        let x = &a[start..start + n];
        if audio::rms(x) >= data.identity.preprocess.silence_rms {
            return x.to_vec();
        }
    }
    vec![]
}

#[test]
fn trace_preserves_original_stream_and_identifies_source_samples() {
    let data = fixture(
        vec![
            (0..64).map(|i| 0.1 + i as f32 * 0.001).collect(),
            (0..85).map(|i| -0.2 - i as f32 * 0.001).collect(),
            (0..101).map(|i| 0.4 + i as f32 * 0.001).collect(),
        ],
        0.01,
    );
    let mut traced_rng = Rng::new(42);
    let mut original_rng = Rng::new(42);
    let mut compatibility_rng = Rng::new(42);
    let mut selected = [false; 3];
    for _ in 0..64 {
        let traced = data.crop_traced(16, &mut traced_rng).unwrap();
        let expected = original_crop(&data, 16, &mut original_rng);
        assert_eq!(traced.samples, expected);
        assert_eq!(data.crop(16, &mut compatibility_rng), expected);
        assert_eq!(traced_rng.state, original_rng.state);
        assert_eq!(compatibility_rng.state, original_rng.state);
        assert_eq!(traced.attempts, 1);
        assert_eq!(traced.silence_rejections, 0);
        assert_eq!(
            traced.samples,
            data.audio[traced.source_index][traced.start_sample..traced.start_sample + 16]
        );
        assert_eq!(
            data.identity.sources[traced.source_index].samples,
            data.audio[traced.source_index].len()
        );
        selected[traced.source_index] = true;
    }
    assert!(selected.into_iter().all(|was_selected| was_selected));
}

#[test]
fn trace_records_silence_retries_without_consuming_extra_randomness() {
    let mut samples = vec![0.; 64];
    samples[24..].fill(0.25);
    let data = fixture(vec![samples], 0.05);
    let mut observed_retry = false;
    for seed in 0..32 {
        let mut traced_rng = Rng::new(seed);
        let mut original_rng = Rng::new(seed);
        let traced = data.crop_traced(8, &mut traced_rng).unwrap();
        assert_eq!(traced.samples, original_crop(&data, 8, &mut original_rng));
        assert_eq!(traced_rng.state, original_rng.state);
        assert_eq!(traced.silence_rejections, traced.attempts - 1);
        assert!((1..=16).contains(&traced.attempts));
        assert_eq!(
            traced_rng.state,
            seed.wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(2 * traced.attempts as u64))
        );
        observed_retry |= traced.silence_rejections > 0;
    }
    assert!(
        observed_retry,
        "fixture must actually exercise silence rejection"
    );
}

#[test]
fn all_silent_failure_preserves_sixteen_attempt_limit_and_rng() {
    let data = fixture(vec![vec![0.; 64]], 0.05);
    let mut traced_rng = Rng::new(123);
    let mut original_rng = Rng::new(123);
    let mut compatibility_rng = Rng::new(123);
    assert!(data.crop_traced(8, &mut traced_rng).is_none());
    assert!(original_crop(&data, 8, &mut original_rng).is_empty());
    assert!(data.crop(8, &mut compatibility_rng).is_empty());
    assert_eq!(traced_rng.state, original_rng.state);
    assert_eq!(compatibility_rng.state, original_rng.state);
    assert_eq!(
        traced_rng.state,
        123u64.wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(32))
    );
}
