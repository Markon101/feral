# Calibrated-prior follow-up, frozen before outcomes

Unit-prior preconditioning improved the three-seed 3,000-step comparison, but the
predetermined seed-42 6,000-step continuation barely changed its generated spectrum.
Its next-section velocity error was .30884 versus .31255 for the old raw 6,000-step
model, and validation was slightly worse. Keep that result. Input normalization
alone did not change the path's physical SNR.

Add a checkpointed noise prior std .1, used consistently for training draws,
solver initialization, coefficient derivation and evaluation. For noise variance B:

```
D = t²*A + (1-t)²*B
c_skip = (t*A-(1-t)*B)/D
c_out = sqrt(A*B/D)
c_in = 1/sqrt(D)
```

Checkpoint v3 is required for non-unit priors; v1/v2 remain loadable, and a config
cannot masquerade as an older version. CLI `--noise-scale` remains an OOD multiplier
of the checkpoint prior; it does not alter the training distribution. Raw velocity
remains fixed to unit prior and the default remains unchanged.

Fresh 3,000-update calibrated models, seeds 42/43/44, same source [0,24), weights,
optimizer, masks, crop and standardized RNG draws as the unit-prior counterparts.
New validation [216,224), new test [232,240). No checkpoint/seed selection.
Compare calibrated, unit-preconditioned and raw models at equal update budget.

Velocity MSE at equal t is not comparable across different priors. Use common
corruption probes y+sigma*epsilon with sigma .02/.1/.5. For prior std p choose
`t=p/(p+sigma)`; the endpoint estimate is physical waveform audio. Preserve all rows
and compare endpoint MSE/derivative/spectral metrics and silence/common-noise
controls, with noise-level-specific results. Keep waveform generation diagnostics
and level-matched listening separate. One-song, three-seed limits still apply.

`scripts/phase3_calibrated.py --run runs/phase3-calibrated-v1` executes sequentially.
This is a follow-up motivated by earlier outcomes, not a replacement for the
original preconditioning experiment or a claim that either paper was reproduced.
