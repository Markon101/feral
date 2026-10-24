# Reproducible experiments

Keep runs new, preserve original audio, and retain raw losses and negative results.
Git ignores audio/checkpoints but tracks code and compact validation reports.
Run commands from the repository root. Use `doctor` first.

## Diagnostic pipeline

```
cargo test --locked --offline --release -j 8
./target/release/feral smoke --run runs/smoke --steps 20
```

The four deterministic patterns are a sine, harmonic pair, chirp pulse and noisy
pulse (seeds 100–103). This tests neural learning and sound production only.
Compare `untrained-or-resumed-initial.wav` with `generated.wav`: the same noise,
solver and seed isolate changed model weights. `initial.json` retains the control.
Fixed probe seed 901 checks four fixed crops/noises/masks at t=.5 before and after
training. Probe loss is a training-distribution check, not held-out evaluation.
No best-of seed selection occurs. Every training step is retained in losses.jsonl.

## First real song

The owner selected `AROUND (The Absence of Ladder)v2.wav` in Downloads.
Decode only the first 12 seconds for the initial bounded run, train on random
0.512 s crops, compare identical-noise outputs before/after. Source identity and
preprocessing are recorded by the program; original remains untouched.

```
./target/release/feral train \
  --input '/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav' \
  --run runs/around-v1 --steps 120 --max-file-seconds 12 \
  --preview-every 40 --checkpoint-every 60
./target/release/feral generate --checkpoint runs/around-v1/checkpoint.json \
  --seed 42 --seconds 0.512 --output runs/around-v1/seed42.wav
```

Validation records actual commands and results in reports/VALIDATION.md. Real-song
training is self-supervised on acoustic samples; no labels or transcription.

## Controls worth running next

1. Evaluate fixed unknown masks/noises on held-out song sections and other songs.
   Compare initialized/trained checkpoints and a noise-only control; do not select
   seeds by attractiveness. Listen at matched levels and preserve all samples.
2. Compare baseline versus perturbed generation at matched seeds/solver/steps.
   Record perturbation and noise-scale separately; clipping is not quality.
3. Compare no-collar versus collar boundary jumps, derivative jumps, short-window
   energy and spectrogram discontinuities. Exact sample retention is a hard gate;
   independently assess timbral and musical continuity by listening.
4. Before scaling CPU/GPU, compare forward outputs, objective, gradients, Adam
   updates and full trajectories. Warm up and measure transfers and peak RSS.

Recurrence, attractors, soft invariants, multiscale time and latent interpolation
are research directions, not implemented baseline claims.

## Phase 2: controlled pooled-context experiment

See [the frozen protocol](reports/PHASE2_PROTOCOL.md) and
[the complete outcome](reports/PHASE2.md). `scripts/phase2.py` trains three paired
baseline/pyramid seeds on a fixed source interval and evaluates disjoint sections.
`scripts/phase2_budget.py` is a separately frozen follow-up with twice the pooled
updates and new held-out sections. `scripts/summarize_phase2.py` aggregates model
seeds without treating crop probes as independent models.

The equal-step result is negative; the approximate compute-budget result has a
modest denoising gain and no established generation-quality gain. All results and
controls remain. The next experiment should challenge noise/path scaling and
actual generated spectra rather than enlarging the architecture reflexively.

## 6,000-step continuation

The owner-requested fivefold continuation, fixed held-out comparison and exported
10-second A/B pack are recorded in [TRAINING_6000.md](reports/TRAINING_6000.md).
The 1,200-step control, optimizer/RNG state and prior listening pack are preserved.

## Phase 3: preconditioning and calibrated prior

See [PHASE3.md](reports/PHASE3.md), the frozen protocols, `scripts/phase3.py` and
`scripts/phase3_calibrated.py`. Nine paired-model runs plus the unit-prior follow-up
preserve regression gains, reconstruction losses and generated audio separately.
Audio/control examples at common RMS are exported to `/sdcard/FERAL_Phase3_20261024/`.
