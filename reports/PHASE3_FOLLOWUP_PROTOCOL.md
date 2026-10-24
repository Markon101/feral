# Fixed Phase 3 follow-up

The three-seed 3,000-step comparison is complete and preserved. Before a new final
checkpoint is evaluated, extend the predetermined seed 42 preconditioned model by
3,000 updates to 6,000 total. Keep all settings, source identity, optimizer and RNG.
The old raw-velocity 6,000-step model is the historical listening/control reference.
New validation source seconds [160,168), test [184,192). No tuning or checkpoint
selection on those sections. Retain the initial 3,000-step checkpoint.

This is a single-model practical follow-up, separate from the replicated primary
experiment. Verify identical initial neural parameters and paired RNG traces with
the historical raw-velocity 6,000-step run; changed parameterization/loss remains the
experimental difference. Add exact matched Gaussian reference evaluation so the
analytic prior's contribution is visible. Retain the training-fitted AR(1) control.
Prepare ten-second neural audio, source/old-model/control comparisons, and common
RMS listening copies without claiming that level/spectral gains establish music.
