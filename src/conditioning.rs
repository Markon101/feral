use crate::rng::Rng;
/// 0 = generated, 1 = protected. Tasks: unconditional, prefix, hole, suffix.
pub fn training_mask(n: usize, rng: &mut Rng) -> (Vec<f32>, &'static str) {
    let mut m = vec![0.; n];
    match rng.index(4) {
        0 => (m, "unconditional"),
        1 => {
            let end = n / 4 + rng.index(n / 2);
            m[..end].fill(1.);
            (m, "prefix")
        }
        2 => {
            m.fill(1.);
            let a = n / 8 + rng.index(n / 4);
            let b = (a + n / 4 + rng.index(n / 4)).min(n - 1);
            m[a..b].fill(0.);
            (m, "hole")
        }
        _ => {
            let start = n / 4 + rng.index(n / 2);
            m[start..].fill(1.);
            (m, "suffix")
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
/// This reduces signal jumps; it cannot guarantee phase, timbral or musical coherence.
pub fn continuity(x: &mut [f32], known: &[f32], mask: &[f32], collar: usize) {
    if collar == 0 {
        project(x, known, mask);
        return;
    }
    let mut a = 0;
    while a < x.len() {
        if mask[a] == 1. {
            a += 1;
            continue;
        }
        let mut b = a + 1;
        while b < x.len() && mask[b] == 0. {
            b += 1;
        }
        let width = collar.min((b - a) / 2).max(1);
        if a > 0 {
            let slope = if a > 1 {
                (known[a - 1] - known[a - 2]).clamp(-0.05, 0.05)
            } else {
                0.
            };
            for j in 0..width {
                let w = 0.5 - 0.5 * (std::f32::consts::PI * j as f32 / width as f32).cos();
                let anchor = known[a - 1] + slope * (j + 1) as f32 * (-(j as f32) / 16.).exp();
                x[a + j] = anchor * (1. - w) + x[a + j] * w;
            }
        }
        if b < x.len() {
            let slope = if b + 1 < x.len() {
                (known[b + 1] - known[b]).clamp(-0.05, 0.05)
            } else {
                0.
            };
            for j in 0..width {
                let w = 0.5 - 0.5 * (std::f32::consts::PI * j as f32 / width as f32).cos();
                let anchor = known[b] - slope * (j + 1) as f32 * (-(j as f32) / 16.).exp();
                x[b - 1 - j] = anchor * (1. - w) + x[b - 1 - j] * w;
            }
        }
        a = b;
    }
    project(x, known, mask);
}
pub fn boundary_jumps(x: &[f32], mask: &[f32]) -> Vec<f32> {
    (1..x.len())
        .filter(|&i| mask[i] != mask[i - 1])
        .map(|i| (x[i] - x[i - 1]).abs())
        .collect()
}
