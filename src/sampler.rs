//! Sampling on an accelerator: the same flow solver, with the velocity field evaluated on device.
//!
//! `flow::sample` is generic over [`Velocity`], so this type adapts a resident trainer to that
//! trait. The solver, the conditioning projection, the collar and the limiter stay in `flow`, which
//! means a device take and a CPU take differ only by the summation order inside the kernels.
use crate::{backend::Resident, model::Velocity, parameterization::FlowConfig};
use anyhow::{ensure, Result};
use std::cell::RefCell;

/// A [`Velocity`] backed by a resident device trainer.
///
/// `Velocity::velocity` takes `&self`, while driving a device needs `&mut`, hence the `RefCell`.
/// Sampling is single-threaded by construction (one window, one trajectory), so this never aliases.
pub struct DeviceVelocity<'a> {
    device: RefCell<Box<dyn Resident + 'a>>,
    config: FlowConfig,
    n: usize,
    block: usize,
}

impl<'a> DeviceVelocity<'a> {
    pub fn new(
        device: Box<dyn Resident + 'a>,
        config: &FlowConfig,
        n: usize,
        block: usize,
    ) -> Result<Self> {
        ensure!(
            n > 0 && n.is_multiple_of(block),
            "device sampling window must be a positive multiple of the block"
        );
        Ok(Self {
            device: RefCell::new(device),
            config: config.clone(),
            n,
            block,
        })
    }
}

impl Velocity for DeviceVelocity<'_> {
    fn prior_std(&self) -> f32 {
        self.config.prior_std()
    }
    fn block(&self) -> usize {
        self.block
    }
    fn velocity(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> Vec<f32> {
        match self.try_velocity(x, known, mask, t) {
            Ok(v) => v,
            Err(e) => panic!("device velocity failed: {e:#}"),
        }
    }
    fn try_velocity(&self, x: &[f32], known: &[f32], mask: &[f32], t: f32) -> Result<Vec<f32>> {
        ensure!(
            x.len() == self.n && known.len() == self.n && mask.len() == self.n,
            "device sampler window mismatch: expected {} samples",
            self.n
        );
        // `feral_features` builds x_t = mask ? clean : (1-t)*noise + t*clean, applies the
        // parameterization scaling itself, and feeds clean*mask as the known-side feature. Passing
        // the same buffer as clean and noise makes every x_t equal to x without touching a kernel;
        // x*mask equals known*mask because the solver keeps x equal to the known samples wherever
        // the mask is set. That equality is an invariant of the solver, so check it rather than
        // silently evaluating a different field if a caller ever breaks it.
        ensure!(
            known
                .iter()
                .zip(x)
                .zip(mask)
                .all(|((&k, &x), &m)| m == 0. || k == x),
            "device sampling needs x to hold the known samples wherever the mask is set"
        );
        let mut device = self.device.borrow_mut();
        device.step(x, mask, x, &[t], false)?;
        let f = device.read_prediction()?;
        ensure!(
            f.len() == x.len(),
            "device prediction length {} != {}",
            f.len(),
            x.len()
        );
        let (skip, out, _) = self.config.coefficients(t);
        Ok(f.iter().zip(x).map(|(&f, &x)| skip * x + out * f).collect())
    }
}
