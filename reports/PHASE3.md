# Phase 3 — scale-aware waveform flow

The owner's listening feedback was that the 6,000-step model still sounded mostly
like broadband noise. This phase changes the flow parameterization and then the
training noise scale, keeping the same waveform network. New ten-second examples
and explicit noise controls were copied to `/sdcard/FERAL_Phase3_20261024/`.
This is a partial spectral improvement with reconstruction tradeoffs, not a claim
that FERAL now generates coherent music.

## Implementation and checks

The optional preconditioned network learns a residual on a Gaussian reference
velocity. Inputs are variance-normalized across time; the residual output is
converted back to physical waveform velocity. The baseline/default remains raw
velocity, with unchanged 57,952 parameters. The full equations, research references
and preregistered comparisons are in PHASE3_PROTOCOL.md and
PHASE3_CALIBRATED_PROTOCOL.md. No pretrained model, external dataset or new package
was downloaded. Original recordings and older checkpoints remain untouched.

A checkpointed prior std now controls training noise, preview noise, generation,
and evaluation consistently. `--noise-scale` is a multiplier of that prior, not a
way to redefine the training distribution. A `.1` prior thus starts with noise
RMS about `.1` at multiplier 1. Masked context is normalized internally and always
projected back to the exact physical reference samples. Backend interfaces remain
flat waveform velocity and masks; no device type or general tensor engine added.

Checkpoint v1 is legacy velocity, v2 is unit-prior preconditioning, v3 is a
non-unit prior. Older checkpoints still load; older software rejects newer versions.
Resume rejects parameterization/prior mismatches. Unit-prior outputs were reproduced
byte-for-byte after the prior extension, and the old raw checkpoint also passed an
actual render parity check. Use `checkpoint.velocity()` when writing inference
helpers; `checkpoint.model` alone intentionally denotes raw neural network output.

All **25 tests** pass: prior/sampling consistency, finite-difference gradients,
finite endpoints, actual optimizer changes, exact protected samples, deterministic
sampling, full Adam/RNG resume for unit and calibrated priors, version/mismatch
gates, and analytic Gaussian convergence, plus prior audio/flow tests. Formatting,
strict Clippy, release build and example builds pass. Tests and training were
sequential after the owner's unrelated background-job OOM restart. Peak trainer
RSS was about 10.2 MiB; no new OOM occurred. Scheduling/thermals were uncontrolled.

## Experiment 1: preconditioning, same unit prior

Three fresh paired model seeds, 3,000 updates each, training source [0,24) seconds,
validation [112,120), held-out [136,144). Same initial weights, parameter count,
optimizer, crop/mask/time/noise RNG streams and source gain. No seed/checkpoint
selection. Physical velocity MSE remains comparable because the paths use the
same prior. Normalized residual training losses are not numerically comparable
with raw-velocity losses.

| Seed | Raw test velocity MSE | Preconditioned test MSE | Raw 4–8 kHz fraction | Preconditioned fraction |
|---|---:|---:|---:|---:|
| 42 | .330849 | .226200 | .4140 | .2286 |
| 43 | .328285 | .214658 | .3977 | .1342 |
| 44 | .330672 | .226106 | .4253 | .2263 |
| Mean | .329935 | .222321 | .4124 | .1964 |

Held-out velocity error improves 32.6%, in all three pairs. Mean generated spectral
flatness decreases .9091→.5341. These are fixed seed-42, .512 s, Heun-32 generations
from each final model; raw audio, initial controls and predictions are retained.
The zero-neural Gaussian reference scores .405029 on the same held-out probes:
the learned correction does more than merely apply that analytic reference.

The predetermined seed-42 model was then continued to 6,000 updates. It essentially
plateaued: generated 4–8 kHz fraction .2286→.2278, flatness .6168→.6169. On new source
[184,192), velocity MSE was .308841 versus .312551 for the old raw 6,000-step model;
validation [160,168) was slightly worse (.336173 versus .334102). Both beat the
Gaussian reference, but no general advantage is established by this follow-up.
Initial neural weights and all 6,000 standardized RNG traces match the historical
raw run exactly; the parameterization/loss is the experimental change.

A solver diagnostic before changing training also showed little difference between
Heun 32 and 256: old-checkpoint flatness .76363→.76255. More integration steps alone
did not fix the measured noise-like spectrum.

## Experiment 2: actual prior calibration

Three fresh calibrated models, 3,000 updates each, same training data and settings,
but prior std .1 and data scale .1. The recording's training RMS is .1068. New
validation [216,224), held-out [232,240). Initial neural weights and standardized
crop/mask/time/RNG streams match the unit-prior counterparts. This changes the path
SNR coverage, not the output gain or a post-generation filter.

| Seed | Native generated RMS | 4–8 kHz fraction | Flatness | Aligned-copy correlation |
|---|---:|---:|---:|---:|
| 42 | .08560 | .12979 | .44580 | .10492 |
| 43 | .09205 | .10682 | .38282 | .10444 |
| 44 | .09112 | .12139 | .42138 | .10932 |

The training source's 4–8 kHz fraction is .02556, so excess high-frequency energy
remains. The AR(1) control fits only training RMS and lag-1 correlation yet gets a
fraction about .0296. A darker spectrum is therefore insufficient evidence of
useful learned structure. Low aligned-copy correlation does not rule out transformed
memorization. No subjective listening judgment is claimed here.

Across different priors, velocity losses at fixed time are not comparable. The
common-corruption evaluator uses the exact same noisy audio `y+sigma*epsilon`,
choosing `t=prior_std/(prior_std+sigma)`. It reports endpoint waveform error at three
physical corruption levels, with silence/noisy controls and all raw rows.

| Corruption std | Raw velocity | Unit preconditioned | Calibrated preconditioned |
|---|---:|---:|---:|
| .02 | .0004623 | .0008320 | .0004784 |
| .1 | .0074354 | .0059762 | .0071108 |
| .5 | .0332687 | .0396193 | .0471603 |
| Mean | .0137221 | .0154758 | .0182498 |

These are held-out endpoint MSE means across the three models, four masks and
four fixed crops. The calibrated model does not win this reconstruction comparison,
especially at high corruption. Generation's spectrum and paired denoising are
separate outcomes; neither proves musical competence. One song and three seeds
also do not establish cross-recording generalization. All positive and negative
results remain, and neither new formulation replaces the default automatically.

## Listening pack and preservation

`/sdcard/FERAL_Phase3_20261024/` contains 17 PCM16 WAVs, all exactly ten seconds.
All 32 exported files were SHA-256 compared with the repository copies.

Start with `level_matched/`: new calibrated 3,000-step model, old raw 6,000-step
model, unit-prior preconditioned 6,000-step model, source, Gaussian white noise and
training-fitted AR(1) noise. All six are RMS .08 using constant gain only, with
no EQ, filtering or mastering of neural outputs. `comparisons_native/` preserves
original levels. Gaussian/AR(1) are explicitly non-neural controls, not generators
substituted for the trained network. The source/old/current examples and seeds are
predetermined, not selected after listening.

Top-level 00–06 WAVs provide source, new normal generation, OOD control, perturbed
"weird" generation, continuation, inpainting and experimental anchored variation.
Generated ten-second tracks are five seconds from noise plus five seconds of
learned continuation, never loops or time stretches. The continuation retains the
first five source seconds; inpainting regenerates seconds 3–7 and retains outside
samples exactly. Exact preservation was checked on float originals AND PCM16
exports. Variation's disconnected mask is still experimental. Ten-second output
remains outside the .512-second training duration; local context is only 468 ms.

Float originals, argv, sidecars, level-match gains and verification are under
`runs/phase3-listening-v1/`. New checkpoint:
`runs/phase3-calibrated-v1/calibrated-42/checkpoint.json` (v3, 3,000 updates).
The old 6,000-step checkpoint and all earlier packs remain available.

## Reproduce and inspect

```sh
cargo test --locked --offline --release -j 8 -- --test-threads=1
cargo build --locked --offline --release -j 8 --examples
python scripts/phase3.py --run runs/new-unit-comparison
python scripts/phase3_calibrated.py --run runs/new-calibrated-comparison
```

The calibrated protocol intentionally compares against the existing frozen
`runs/phase3-v1/` checkpoints; its script records exact references/hashes. It is a
follow-up, not a replacement primary experiment. The recorded runs used clean
commits a5a921d (unit comparison), 4a95c72 (unit continuation), and 4c384ce (calibrated
comparison). Raw per-step losses and every fixed held-out row remain in the run
folders. `phase3-measurements.json` retains tracked seed-level results, plans and
audio verification. No data, models or papers were downloaded into the project.

Next, test a phase-preserving multiresolution/frequency representation or stronger
acoustic objective against the same noise controls and matched listening. The
current tiny waveform network has learned spectral structure, but useful harmonic,
transient and longer musical organization remain unestablished. Increasing steps
alone has already shown diminishing returns on these runs.
