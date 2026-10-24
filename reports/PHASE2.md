# FERAL Phase 2 — context, learning controls, and transient continuity

2026-10-24. The instrument now has an optional multirate velocity network,
reproducible held-out evaluation, copying/null controls, and a measured transient
continuity fix. The default network is preserved. There is a modest compute-budget
denoising gain, not an established improvement in generated audio quality.

## What changed

* `--network pyramid`: differentiable average-pool/conv/repeat residual branches
  at scales 1/4/16/64, with full-resolution skips. Same 57,952 parameters as baseline;
  structural context span 468 ms versus 62 ms. A perturbation outside the baseline
  receptive field changes the pyramid's output. This tests a computation path,
  not usefulness of learned memory or long-range music.
* `evaluate`: fixed crop identities, mask tasks and noise/time draws; velocity MSE,
  paired endpoint waveform/derivative/coarse/STFT metrics and silence/noisy controls.
* `analyze`: RMS, limiter hits, spectral bands/flatness, local correlation, shuffled
  output control and a bounded aligned-copy screen with literal-copy/noise controls.
* Transient continuity: remove the arbitrary .05 source-slope cap and adapt tangent
  decay to amplitude headroom. Protected waveform samples remain exact. The
  generated-sample safety limiter remains authoritative for infeasible slopes.
* Reproducible three-seed protocols, per-step losses/RNG states, raw held-out rows,
  initial/final checkpoints, source hashes and playable baseline/creative examples.

The waveform representation, rectified-flow objective, Adam settings and training
mask distribution did not change. No model download, package installation, paid
service, remote, Titan import or existing-repository mutation occurred. RustFFT and
its seven transitive packages were already cached; dependency count is now 48.
Research motivations and primary sources are in PHASE2_PROTOCOL.md. This is a small
engineering experiment, not a reproduction of Stable Audio or Parallel WaveGAN.

## Equal updates: a negative result worth keeping

Train on [0,24) seconds of the owner's `AROUND (The Absence of Ladder)v2.wav`.
Validation [32,40), test [48,56), gaps beyond either receptive field. Mono 16 kHz,
original gain throughout, .512 s crops, batch one, 600 fresh updates, lr .0003.
The two architectures start from identical weight values for each model seed.
Training task/time/RNG traces matched exactly within all three pairs. No seed,
checkpoint or hyperparameter was selected from validation/test performance.

| Model seed | Baseline test velocity MSE | Pyramid test velocity MSE |
|---|---:|---:|
| 42 | .330350 | .371201 |
| 43 | .332802 | .373711 |
| 44 | .341404 | .378284 |
| Mean across models | .334852 | .374399 |

The pyramid is **11.8% worse at equal updates**, in all three pairs. More context
alone did not solve learning. Mean step latency across individual runs: baseline
31.10–33.75 ms; pyramid 12.61–16.42 ms. Parameter count is identical; cheaper
coarse computation trades off fine-resolution processing. Existing defaults stay.

## Compute-budget follow-up, with new held-out sections

After the negative result, a second plan was frozen before new evaluation. Resume
each pyramid for 600 more steps (1,200 total); retain the 600-step baseline. New
validation [64,72), new test [80,88). No model tuning followed these new results.
This is a practical compute comparison, not an equal-update architecture ablation.

| Seed | Baseline 600 test MSE | Pyramid 600 test MSE | Pyramid 1,200 test MSE | Baseline update time | Pyramid total update time |
|---|---:|---:|---:|---:|---:|
| 42 | .423203 | .453779 | .411864 | 20.253 s | 20.010 s |
| 43 | .435462 | .459365 | .421665 | 18.658 s | 15.359 s |
| 44 | .433510 | .462434 | .410625 | 19.910 s | 19.758 s |
| Mean | .430725 | .458526 | .414718 | 19.607 s | 18.376 s |

The 1,200-step pyramid improves fresh test velocity error **3.7%**, with the same
paired direction in all three seeds. It received twice the updates and training
crops. This cannot establish that long context caused the advantage. CPU scheduling
and thermals were uncontrolled; observed update times exclude previews, checkpoint
writes and decoding. They are not a controlled maximum-performance speedup claim.

Fresh test velocity MSE by task, mean across models:

| Task | Baseline 600 | Pyramid 1,200 |
|---|---:|---:|
| Unconditional | .431803 | .416441 |
| Prefix | .450708 | .436770 |
| Hole | .427682 | .409856 |
| Suffix | .412707 | .395805 |

The metric averages 4 fixed crops × 4 tasks × 3 times per model. Crops/probe rows
are not model replications; there are only three trained models per architecture.
One recording's unseen sections do not demonstrate cross-recording generalization.

## Acoustic evidence and controls

For the fresh test, paired endpoint waveform MSE (mean across model seeds/tasks):

| Path time | Baseline 600 | Pyramid 1,200 | Zero audio | Noisy input |
|---|---:|---:|---:|---:|
| .1 | .172895 | .125804 | .051749 | .850186 |
| .5 | .052436 | .044823 | .051749 | .262403 |
| .9 | .008690 | .009095 | .051749 | .010496 |

At t=.5 the pyramid retains some target information beyond silence and reduces
spectral log distance (.568→.456) and derivative MSE (.03882→.02280). At t=.9 it is
slightly worse than baseline; at t=.1 silence beats both. The last result is not a
contradiction of flow matching: a conditional mean estimate at very high noise is
ambiguous. It prevents calling every lower regression loss useful reconstruction.
No matched Wiener/linear-filter control was run, so structure beyond elementary
spectral shrinkage is not established. These estimates use noisy held-out audio;
they are denoising probes, not unconditional generated samples.

Unconditional sampling does use fresh noise. The 1,200-step pyramid outputs have
RMS .208–.213 and 1–2 limiter hits per 8,192 samples. Their spectral flatness is
.952–.955 versus .131 for the training section. Power in the 4–8 kHz band is about 45.5%,
versus 2.56% in training audio. The 600-step baseline's corresponding fraction is
36–40%, also much too high. The pooled candidate is therefore **not a generated
spectral-quality win**, despite better compute-budget denoising. It stays opt-in.

Shuffling generated samples preserves their amplitudes but moves their power
fractions toward the white-noise proportions. Generated signals have weak temporal
correlation, but nothing close to the source's structure. Nearest training-window
absolute correlations are .087–.089 for the final pyramid; literal-copy controls
score approximately 1.0. This rules out obvious aligned copies at stride 256 and
decimation 4, not shifted/transformed/recombined memorization. There is no human
listening score, learned perceptual embedding, FAD/MOS, or musical-quality claim.

## Continuity: first differences matter too

A held-out prefix ended on a strong transient. The old .05 slope cap produced a
boundary derivative change of .106954. The revised tangent gives **2.24e-8** with
the same model, seed, context and solver. Absolute boundary step increases from
.05 to .156954 because it now matches the actual source slope; minimizing a sample
jump alone would hide that distinction. Adaptive decay respects available headroom
when the edge and first step are feasible. Tests cover feasible transient slopes,
near-ceiling headroom and impossible slopes where the limiter must win.

Independent RIFF payload comparison confirms exact retention of the complete 8 s
canonical prefix plus 8,192 new samples. Inpainting [1,1.25) seconds changes that
region and retains every outside sample exactly. New inpaint boundary steps are
.074459 and .014410; their derivative errors are 0 and 7.45e-9. Old and revised renders are both retained. This is local
waveform/slope evidence, not verified inaudibility, phase-coherent generated
content, timbral consistency or musical continuation.

## Verification and resources

`cargo fmt --check`, strict Clippy and release build pass. **19 integration tests
pass** (1.46 s execution on the final Rust run), including:

* original decoder format, gradients, optimizer, sampler and checkpoint checks;
* every-layer finite-difference checks for pooled gradients, including partial
  pooling groups; finite sampling and exact masks;
* actual distant input influence at the same parameter count;
* full pooled Adam/RNG resume matching uninterrupted training exactly;
* spectral metric identity/change controls and copied/noise correlation controls;
* new feasible-transient/headroom/impossible-slope tests.

An original phase-1 checkpoint rendered a byte-identical copy of its stored seed-42
WAV using Phase 2 baseline code. CPU remains the reference. There is no OpenCL
implementation or acceleration claim; this launch still reports no platforms via
clinfo, and the prior namespace blocker has not been changed.

Peak trainer RSS in the experiment was at most 10,072 KiB (~9.84 MiB), excluding
FFmpeg and compilation. Four logical CPUs were available to the process; neural
work uses one. Background/thermal conditions were uncontrolled, and no foreground
or eight-thread maximum-speed test was needed. The 3 GB working ceiling was not
approached by the model. A final one-second 32-step Heun render took 1.469 s;
.512 s 16-step previews took .225–.337 s in the resumed runs. Revised continuation
sampling took 1.119 s and local inpainting .712 s. No large training run was used:
3,600 initial updates plus 1,800 follow-up updates total.

## Reproduce and inspect

```sh
cargo build --locked --offline --release -j 8
cargo test --locked --offline --release -j 8
python scripts/phase2.py \
  --input '/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav' \
  --run runs/new-phase2
python scripts/phase2_budget.py --base runs/new-phase2 --run runs/new-budget
python scripts/summarize_phase2.py --base runs/new-phase2 \
  --budget runs/new-budget --output runs/new-budget/summary.json
```

The initial protocol ran from clean commit `21083f47ab18cd571f5b3fc11f8320d3f0c473ed`;
the follow-up ran from clean `e0fdcdab73f9f24028b3c73cfadb11601b6db475`. Both have neural
source snapshot `d049e0841417f4cb50dc5a8160d276043da9b31cc9b00153e9e3c2541e62b03f`.
The subsequent continuity change is commit `d8c5b35`; it does not change neural
training, weights, losses or unconditional sampling. Script/decoder argv, binary
hashes, source identity, data hashes, checkpoint state, raw rows, timings and seeds
are retained under the new run directories. Source recording SHA-256 remained
`cad6b5725121914b6f0f1e932e14d6c394b68e2df9d2f472b6f0675a3e92e026`.

All paths below are relative to `/data/data/com.termux/files/home/projects/feral/`:

* `runs/phase2-v1/plan.json`, `commands.jsonl`, command logs and `results.json`:
  fixed equal-update experiment; all six model runs, controls and source crops.
* `runs/phase2-budget-v1/plan.json`, command logs and `results.json`: follow-up.
* `reports/phase2-measurements.json`: compact tracked seed-level evidence;
  full per-crop/task/time rows remain in each `eval-*.json` or follow-up case JSON.
* `runs/phase2-v1/baseline-42/generated.wav`: fixed-seed baseline, not selected best.
* `runs/phase2-budget-v1/pyramid-42/checkpoint.json`: 1,200-step candidate.
* `runs/phase2-budget-v1/pyramid-42/generated.wav`: .512 s baseline-noise sample.
* `runs/phase2-budget-v1/pyramid-42/one-second.wav`: 1 s, seed 42, Heun 32.
* `runs/phase2-budget-v1/pyramid-42/ood-control.wav`: seed 73, Euler 48, noise 1.4.
* `runs/phase2-budget-v1/pyramid-42/weird.wav`: same OOD settings, perturb .08;
  209/16,000 limiter hits. This rough result remains beside its metadata/control.
* `runs/phase2-budget-v1/pyramid-42/continued-transient.wav` and
  `inpainted-transient.wav`: revised continuity on the fresh held-out source crop.
* `runs/phase2-budget-v1/pyramid-42/transient-verification.json`: independent exact
  sample and boundary checks; earlier `continued.wav`/`inpainted.wav` are preserved.

## What deserves the next experiment

Keep the baseline and evaluation controls. Test data-calibrated training noise
and late-time coverage as a small, separate ablation: training RMS here is .107
while unit Gaussian noise dominates much of the path. Noise changes must be
consistent in training/sampling/checkpoints, not merely an inference knob. Pair
this with held-out recordings, matched runtime, amplitude-matched listening and
linear-filter controls. If generation still remains spectrally white, test a
phase-preserving multiresolution auxiliary on short generated trajectories while
measuring diversity and collapse. Do not conclude that larger context or lower
endpoint MSE alone is the answer. Persistent state/recurrence can wait for useful
information-flow evidence.
