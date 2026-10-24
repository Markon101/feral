use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    path::Path,
    process::Command,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Preprocess {
    pub sample_rate: u32,
    pub max_seconds: f32,
    pub normalize: bool,
    pub channel: String,
    pub silence_rms: f32,
}
impl Default for Preprocess {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            max_seconds: 30.,
            normalize: true,
            channel: "mix".into(),
            silence_rms: 0.0001,
        }
    }
}
impl Preprocess {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (8000..=48000).contains(&self.sample_rate),
            "sample rate must be 8000..48000"
        );
        ensure!(
            self.max_seconds.is_finite() && self.max_seconds > 0. && self.max_seconds <= 600.,
            "decode limit must be >0 and <=600 seconds"
        );
        ensure!(
            ["mix", "left", "right"].contains(&self.channel.as_str()),
            "channel must be mix, left or right"
        );
        ensure!(
            self.silence_rms.is_finite() && self.silence_rms >= 0.,
            "invalid silence threshold"
        );
        Ok(())
    }
}
/// Work around the verified Termux libplacebo/C++ runtime mismatch, without a shell.
pub fn decoder_command(tool: &str) -> Command {
    let mut c = Command::new(tool);
    let cpp = Path::new("/data/data/com.termux/files/usr/lib/libc++_shared.so");
    if cpp.exists() {
        let current = std::env::var("LD_PRELOAD").unwrap_or_default();
        c.env(
            "LD_PRELOAD",
            format!(
                "{}{}{}",
                cpp.display(),
                if current.is_empty() { "" } else { ":" },
                current
            ),
        );
    }
    c
}
pub fn decode(path: &Path, p: &Preprocess) -> Result<Vec<f32>> {
    p.validate()?;
    let path = fs::canonicalize(path).with_context(|| format!("input {}", path.display()))?;
    let mut cmd = decoder_command("ffmpeg");
    cmd.args(["-nostdin", "-v", "error", "-threads", "1", "-i"])
        .arg(&path)
        .args([
            "-t",
            &p.max_seconds.to_string(),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
        ]);
    if p.channel == "left" {
        cmd.args(["-af", "pan=mono|c0=c0"]);
    }
    if p.channel == "right" {
        cmd.args(["-af", "pan=mono|c0=c1"]);
    }
    let out = cmd
        .args([
            "-ac",
            "1",
            "-ar",
            &p.sample_rate.to_string(),
            "-f",
            "f32le",
            "pipe:1",
        ])
        .output()
        .context("run ffmpeg; install it with: pkg install ffmpeg")?;
    if !out.status.success() {
        bail!(
            "decoder rejected {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    ensure!(out.stdout.len() % 4 == 0, "truncated PCM");
    let mut x: Vec<f32> = out
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(
        !x.is_empty() && x.iter().all(|x| x.is_finite()),
        "empty or nonfinite decoded audio"
    );
    ensure!(
        x.len() <= (p.sample_rate as f32 * p.max_seconds).ceil() as usize + 8192,
        "decoder exceeded duration bound"
    );
    if p.normalize {
        let peak = x.iter().fold(0f32, |a, v| a.max(v.abs()));
        if peak > 1e-8 {
            for v in &mut x {
                *v *= 0.8 / peak;
            }
        }
    }
    Ok(x)
}
pub fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}
/// IEEE-float WAV retains canonical protected samples bit-for-bit. Never replaces files.
pub fn write_wav(path: &Path, x: &[f32], sr: u32) -> Result<()> {
    ensure!(
        !x.is_empty() && x.iter().all(|v| v.is_finite()),
        "refusing empty/nonfinite WAV"
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create new output {} (will not overwrite)", path.display()))?;
    let mut w = hound::WavWriter::new(
        std::io::BufWriter::new(file),
        hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for &v in x {
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}
pub fn synthetic(dir: &Path, sr: u32) -> Result<()> {
    fs::create_dir_all(dir)?;
    for kind in 0..4 {
        let mut rng = crate::rng::Rng::new(100 + kind);
        let x: Vec<f32> = (0..sr * 2)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let env = (std::f32::consts::PI * (t % 0.5) / 0.5).sin().powi(2);
                match kind {
                    0 => 0.5 * (std::f32::consts::TAU * 220. * t).sin(),
                    1 => {
                        0.3 * ((std::f32::consts::TAU * 330. * t).sin()
                            + 0.5 * (std::f32::consts::TAU * 660. * t).sin())
                    }
                    2 => 0.6 * env * (std::f32::consts::TAU * (110. * t + 180. * t * t)).sin(),
                    _ => 0.15 * rng.normal() * env,
                }
            })
            .collect();
        write_wav(&dir.join(format!("pattern-{kind}.wav")), &x, sr)?;
    }
    Ok(())
}

/// Probe before conditioning so a decoder duration ceiling never silently drops a tail.
pub fn duration(path: &Path) -> Result<f64> {
    let path = fs::canonicalize(path)?;
    let out = decoder_command("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(path)
        .output()
        .context("run ffprobe")?;
    ensure!(
        out.status.success(),
        "duration probe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let seconds: f64 = String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .context("unknown duration; use feral decode to make an explicit bounded WAV crop first")?;
    ensure!(
        seconds.is_finite() && seconds > 0.,
        "invalid source duration"
    );
    Ok(seconds)
}
