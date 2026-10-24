use crate::{
    audio::{self, Preprocess},
    rng::Rng,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    pub path: PathBuf,
    pub sha256: String,
    pub samples: usize,
    pub rms: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetIdentity {
    pub sources: Vec<Source>,
    pub rejected: Vec<String>,
    pub preprocess: Preprocess,
    pub manifest_hash: String,
}
/// A successful canonical crop and the sampling decisions that selected it.
/// Source indices refer directly to `DatasetIdentity::sources` and `Dataset::audio`.
#[derive(Clone, Debug, PartialEq)]
pub struct Crop {
    pub samples: Vec<f32>,
    pub source_index: usize,
    pub start_sample: usize,
    /// Includes the successful attempt; at most sixteen candidates are drawn.
    pub attempts: usize,
    pub silence_rejections: usize,
}
pub struct Dataset {
    pub audio: Vec<Vec<f32>>,
    pub identity: DatasetIdentity,
}
pub fn hash_file(p: &Path) -> Result<String> {
    let mut f = fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut b = [0u8; 65536];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn select(input: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = if input.is_dir() {
        fs::read_dir(input)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|s| s.to_str())
                        .map(|s| {
                            [
                                "wav", "flac", "mp3", "ogg", "opus", "webm", "m4a", "aac", "aif",
                                "aiff", "wma", "mp4",
                            ]
                            .contains(&s.to_ascii_lowercase().as_str())
                        })
                        .unwrap_or(false)
            })
            .collect::<Vec<_>>()
    } else if input.extension().and_then(|s| s.to_str()) == Some("txt") {
        let root = input.parent().unwrap_or(Path::new("."));
        fs::read_to_string(input)?
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty() && !s.starts_with('#'))
            .map(|s| root.join(s))
            .collect()
    } else {
        vec![input.to_path_buf()]
    };
    paths.sort();
    paths.dedup();
    ensure!(
        !paths.is_empty() && paths.len() <= 128,
        "select 1..128 files per run; directory selection is nonrecursive; use a .txt manifest"
    );
    Ok(paths)
}
impl Dataset {
    pub fn load(input: &Path, p: Preprocess, crop: usize) -> Result<Self> {
        let mut audio = vec![];
        let mut sources = vec![];
        let mut rejected = vec![];
        let mut total = 0;
        for path in select(input)? {
            let result = (|| -> Result<_> {
                let x = audio::decode(&path, &p)?;
                ensure!(x.len() >= crop, "shorter than one crop");
                ensure!(audio::rms(&x) >= p.silence_rms, "silent file");
                let source = Source {
                    path: fs::canonicalize(&path)?,
                    sha256: hash_file(&path)?,
                    samples: x.len(),
                    rms: audio::rms(&x),
                };
                Ok((x, source))
            })();
            match result {
                Ok((x, s)) => {
                    total += x.len();
                    ensure!(total<=32_000_000,"decoded dataset exceeds 128 MB; reduce --max-file-seconds or file selection");
                    audio.push(x);
                    sources.push(s);
                }
                Err(e) => {
                    eprintln!("skip {}: {e}", path.display());
                    rejected.push(format!("{}: {e}", path.display()));
                }
            }
        }
        ensure!(
            !audio.is_empty(),
            "no usable audio: inspect decoder errors, crop length and silence threshold"
        );
        let bytes = serde_json::to_vec(&(&sources, &p)).context("dataset identity")?;
        let identity = DatasetIdentity {
            sources,
            rejected,
            preprocess: p,
            manifest_hash: format!("{:x}", Sha256::digest(bytes)),
        };
        Ok(Self { audio, identity })
    }
    /// Preserve the original uniform-file sampling and random stream exactly.
    pub fn crop(&self, n: usize, rng: &mut Rng) -> Vec<f32> {
        self.crop_traced(n, rng)
            .map(|crop| crop.samples)
            .unwrap_or_default()
    }

    /// Sample with provenance, without changing candidate weighting or RNG draws.
    /// As with `crop`, callers must use the crop length validated by `load` (or a
    /// smaller length). Sixteen rejected silent candidates return `None`.
    pub fn crop_traced(&self, n: usize, rng: &mut Rng) -> Option<Crop> {
        for attempts in 1..=16 {
            let source_index = rng.index(self.audio.len());
            let a = &self.audio[source_index];
            let start_sample = rng.index(a.len() - n + 1);
            let x = &a[start_sample..start_sample + n];
            if audio::rms(x) >= self.identity.preprocess.silence_rms {
                return Some(Crop {
                    samples: x.to_vec(),
                    source_index,
                    start_sample,
                    attempts,
                    silence_rejections: attempts - 1,
                });
            }
        }
        None
    }
}
