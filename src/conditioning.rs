use crate::rng::Rng;
/// 0 = generated, 1 = protected. Tasks: unconditional, prefix, hole, suffix.
pub fn training_mask(n: usize, rng: &mut Rng) -> (Vec<f32>, &'static str) {
    training_mask_channels(n, 1, rng)
}
/// Channel-aware training mask with aligned `[L, R]` pairs per time frame.
pub fn training_mask_channels(
    n_frames: usize,
    channels: usize,
    rng: &mut Rng,
) -> (Vec<f32>, &'static str) {
    let mut m = vec![0.; n_frames * channels];
    let task = training_mask_channels_into(&mut m, channels, rng);
    (m, task)
}
/// Writes a training mask into an existing buffer without allocating.
pub fn training_mask_into(m: &mut [f32], rng: &mut Rng) -> &'static str {
    training_mask_channels_into(m, 1, rng)
}
/// Writes a frame-aligned multi-channel training mask into an existing interleaved buffer.
pub fn training_mask_channels_into(m: &mut [f32], channels: usize, rng: &mut Rng) -> &'static str {
    assert!(channels >= 1 && m.len().is_multiple_of(channels));
    let n = m.len() / channels;
    m.fill(0.);
    match rng.index(4) {
        0 => "unconditional",
        1 => {
            let end = n / 4 + rng.index(n / 2);
            m[..end * channels].fill(1.);
            "prefix"
        }
        2 => {
            m.fill(1.);
            let a = n / 8 + rng.index(n / 4);
            let b = (a + n / 4 + rng.index(n / 4)).min(n - 1);
            m[a * channels..b * channels].fill(0.);
            "hole"
        }
        _ => {
            let start = n / 4 + rng.index(n / 2);
            m[start * channels..].fill(1.);
            "suffix"
        }
    }
}
pub fn project(x: &mut [f32], known: &[f32], mask: &[f32]) {
    for ((v, &k), &m) in x.iter_mut().zip(known).zip(mask) {
        if m == 1. {
            *v = k;
        }
    }
}
/// Unknown-side raised-cosine collars; exact protected samples never change.
/// Feasible boundary slopes are retained; phase, timbre and music need separate checks.
fn tangent(edge: f32, delta: f32, j: usize) -> f32 {
    // For tau >= 1, max((j+1)*exp(-j/tau)) <= tau. Adapt to available
    // headroom so a feasible first step does not force later tangent clipping.
    let room = if delta >= 0. {
        0.98 - edge
    } else {
        edge + 0.98
    };
    let tau = (room / delta.abs().max(1e-9)).clamp(1., 16.);
    edge + delta * (j + 1) as f32 * (-(j as f32) / tau).exp()
}
pub fn continuity(x: &mut [f32], known: &[f32], mask: &[f32], collar: usize) {
    continuity_channels(x, known, mask, collar, 1);
}
/// Apply unknown-side tangent collar blending independently along each audio channel.
pub fn continuity_channels(
    x: &mut [f32],
    known: &[f32],
    mask: &[f32],
    collar: usize,
    channels: usize,
) {
    assert!(channels >= 1 && x.len().is_multiple_of(channels));
    if collar == 0 {
        project(x, known, mask);
        return;
    }
    let frames = x.len() / channels;
    for ch in 0..channels {
        let mut a = 0;
        while a < frames {
            if mask[a * channels + ch] == 1. {
                a += 1;
                continue;
            }
            let mut b = a + 1;
            while b < frames && mask[b * channels + ch] == 0. {
                b += 1;
            }
            let width = collar.min((b - a) / 2).max(1);
            if b - a == 1 && a > 0 && b < frames {
                let left_slope = if a > 1 {
                    known[(a - 1) * channels + ch] - known[(a - 2) * channels + ch]
                } else {
                    0.
                };
                let right_slope = if b + 1 < frames {
                    known[(b + 1) * channels + ch] - known[b * channels + ch]
                } else {
                    0.
                };
                let left = tangent(known[(a - 1) * channels + ch], left_slope, 0);
                let right = tangent(known[b * channels + ch], -right_slope, 0);
                x[a * channels + ch] = 0.5 * (left + right);
                a = b;
                continue;
            }
            if a > 0 {
                let slope = if a > 1 {
                    known[(a - 1) * channels + ch] - known[(a - 2) * channels + ch]
                } else {
                    0.
                };
                for j in 0..width {
                    let w = 0.5 - 0.5 * (std::f32::consts::PI * j as f32 / width as f32).cos();
                    let anchor = tangent(known[(a - 1) * channels + ch], slope, j);
                    let at = (a + j) * channels + ch;
                    x[at] = anchor * (1. - w) + x[at] * w;
                }
            }
            if b < frames {
                let slope = if b + 1 < frames {
                    known[(b + 1) * channels + ch] - known[b * channels + ch]
                } else {
                    0.
                };
                for j in 0..width {
                    let w = 0.5 - 0.5 * (std::f32::consts::PI * j as f32 / width as f32).cos();
                    let anchor = tangent(known[b * channels + ch], -slope, j);
                    let at = (b - 1 - j) * channels + ch;
                    x[at] = anchor * (1. - w) + x[at] * w;
                }
            }
            a = b;
        }
    }
    project(x, known, mask);
}
pub fn boundary_jumps(x: &[f32], mask: &[f32]) -> Vec<f32> {
    boundary_jumps_channels(x, mask, 1)
}
pub fn boundary_jumps_channels(x: &[f32], mask: &[f32], channels: usize) -> Vec<f32> {
    assert!(channels >= 1 && x.len().is_multiple_of(channels));
    let frames = x.len() / channels;
    let mut jumps = Vec::new();
    for t in 1..frames {
        for ch in 0..channels {
            let cur = t * channels + ch;
            let prev = (t - 1) * channels + ch;
            if mask[cur] != mask[prev] {
                jumps.push((x[cur] - x[prev]).abs());
            }
        }
    }
    jumps
}
