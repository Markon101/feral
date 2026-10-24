use serde::{Deserialize, Serialize};
/// Versioned deterministic SplitMix64 stream; normal draws use Box-Muller.
#[derive(Clone, Serialize, Deserialize)]
pub struct Rng {
    pub state: u64,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn bits(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f32 {
        (((self.bits() >> 40) as f32) + 0.5) / 16777216.0
    }
    pub fn index(&mut self, n: usize) -> usize {
        (self.bits() % n as u64) as usize
    }
    pub fn normal(&mut self) -> f32 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f32::consts::TAU * self.uniform()).cos()
    }
}
