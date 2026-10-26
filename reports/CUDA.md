# CUDA backend and the 1M-parameter audio run

Scope: an optional CUDA training backend for Pascal-class hardware (GTX 1080 Ti, sm_61) and one
reproducible 1,018,400-parameter experiment on real recorded audio. The CPU implementation remains
the numerical reference; every claim below is backed by a JSON artifact in this repository.

## Why a separate backend was needed

`src/gpu` targets OpenCL and the phone. Desktop Pascal exposes CUDA, and CUDA 13 dropped sm_61, so
the backend must be compiled with a 11.x/12.x `nvcc`. `build.rs` therefore:

* honours `FERAL_NVCC` and `FERAL_CUDA_ARCH`,
* otherwise searches CUDA 11.x/12.x installs and refuses CUDA 13+ for sm_61,
* fails the build with an explicit message when no suitable `nvcc` exists.

Nothing is downloaded and no new crate is added. The trainer talks to the driver API through a
local `dlopen("libcuda.so.1")` binding declared in `src/cuda/mod.rs`; `--features cuda` has no
dependencies.

## Design constraints kept from the CPU reference

* fp32 only, `fmad` disabled, same operation order as `src/gpu/kernels.cl` and `src/model.rs`.
* Kernels (`src/cuda/kernels.cu`) are compiled ahead of time to `feral.cubin`, with `feral.ptx`
  retained as a JIT fallback. The artifact actually used is reported as `backend_info.kernel_build`.
* Explicit backend selection (`Trainer::{Cpu,OpenCl,Cuda}`); there is no silent CPU fallback. A
  missing device, missing feature or driver error is a hard failure.
* One current CUDA context per thread: every device entry point re-binds its own context with
  `cuCtxSetCurrent`. Without this, a second live trainer invalidated the first one's handles
  (`cuLaunchKernel ... invalid resource handle`), which the audit exposed.
* Batch support (`--batch`) and a held-out validation corpus (`--validation`,
  `--validation-every`) live in the backend-neutral runtime, not in the CUDA path.

## Build and gates

```
cargo fmt --check
cargo clippy --locked --offline --all-targets --features cuda -- -D warnings
cargo test  --locked --offline --release -j 8                 # 34 tests
cargo test  --locked --offline --release --features cuda -j 8 # 39 tests, incl. 5 in tests/cuda.rs
```

`tests/cuda.rs` covers tiny-architecture forward/backward parity, batched Adam parity against the
CPU accumulation, and repeated/mixed trainer creation (the context-rebinding regression).

`feral gpu-audit --backend cuda` runs the full parity audit against a checkpoint: analytic loss,
gradient and forward parity at t ∈ {0, 0.37, 1} for three flow parameterizations, Adam parameter
and moment parity, multi-step drift, bit-exact device resume, and CPU continuation from a device
checkpoint.

## The experiment

Corpus: 7 of the owner's 48 kHz/16-bit stereo recordings, split deterministically and copied into
the run tree (originals untouched):

| split | files |
| --- | --- |
| train | 70_57_17, Cartoon Chaos, Cosmic Embrace, Crimson Shadows, Digital Echoes ext v2.1.2 |
| validation | Clubhouse Antics, FISREAL |

`experiments/audio-1m-cuda-01/corpus/` holds the copies; `run.json` records the SHA-256 of each
file plus the manifest hash, so the split is auditable without re-reading the originals.

Command:

```
feral train \
  --input     experiments/audio-1m-cuda-01/corpus/train \
  --validation experiments/audio-1m-cuda-01/corpus/validation \
  --run       experiments/audio-1m-cuda-01/run-1m-6000 \
  --backend cuda --batch 4 --width 256 --blocks 5 \
  --steps 6000 --validation-every 250 --checkpoint-every 1000 --preview-every 1000
```

Architecture: baseline residual stack, block 32, width 256, dilations 1/2/4/8/16.

## Result (artifacts, not impressions)

| quantity | value | source |
| --- | --- | --- |
| trainable parameters | 1,018,400 | `metrics.json: parameters` |
| backend | `cuda-f32-resident` on GTX 1080 Ti, cc 6.1, sm_61, cubin | `metrics.json: backend_info` |
| steps | 6,000 × 4 crops of 8,192 samples (32,768 samples/step) | `metrics.json` |
| step time | mean 6.49 ms, median 6.42 ms, max 8.82 ms | `metrics.json` |
| throughput | 51.1 steps/s, 1.67 M samples/s | `metrics.json` |
| validation loss (2 held-out files) | 1.1958 → 0.0439 | `metrics.json: validation_curve` |
| fixed-probe loss | 1.2242 → 0.0665 | `metrics.json` |
| held-out evaluation | 384 probes, mean velocity MSE 0.1657 | `evaluate-validation.json` |
| device memory resident | 42.8 MB (budget 1 GiB) | `metrics.json: backend_info.allocated_bytes` |
| host peak RSS | 180 MB, inside the 3 GB ceiling | `metrics.json: peak_rss_kib` |
| audit speedup vs CPU, same model | 320.0 ms → 2.44 ms per step (131×) | `audit.json` |

Preprocessing resamples to 16 kHz mono and truncates each file at the 30 s platform default, so
the corpus is ~150 s of training audio; the model sees 0.512 s crops.

Artifacts under `experiments/audio-1m-cuda-01/run-1m-6000/`: `checkpoint.json`,
`checkpoint-001000..006000.json`, `losses.jsonl`, `validation.jsonl`, `metrics.json`, `run.json`,
`preview-*.wav`, `generated.wav`, `untrained-or-resumed-initial.wav`, `audit.json`, `audit.wav`,
`audit.ckpt`, `evaluate-validation.json`.

## Scaling runs (desktop, 8 GiB device budget)

`FERAL_CUDA_BUDGET_MB` raises the resident ceiling for the desktop path (256..=9216, default
1024); the value in force is reported as `backend_info.resident_budget_bytes`. Two long runs were
made at 1,018,400 parameters, `--batch` chosen so each step sees ~130 k samples:

| run | crop | batch | steps | ms/step | steps/s | samples/s | validation | held-out MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `run-60k-b16-c512` | 0.512 s | 16 | 60,000 | 29.2 | 34.2 | 4.48 M | 1.1958 → 0.0352 | 0.11967 |
| `run-50k-b8-c1000` | 1.0 s | 8 | 50,000 | 30.8 | 32.5 | 4.16 M | 1.1997 → 0.0398 | 0.11973 |

Doubling the crop costs ~5 % throughput at equal samples/step, so the longer window is close to
free on this hardware. Device-resident memory is 114–117 MB: at 1 M parameters the 8 GiB ceiling is
headroom for larger batches, longer crops or wider nets, not something these runs use.

### Throughput sweep (1,018,400 parameters, 120-step runs)

| crop | batch 1 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.512 s | 3.01 | 4.37 | 5.06 | 5.05 | 4.84 | 5.09 |
| 1.0 s | 3.68 | 5.04 | 4.98 | — | — | — |
| 2.0 s | — | 5.13 | — | — | — | — |
| 4.0 s | — | starved (2.80 at b2) | 5.24 | — | — | — |

Values are M samples/s. Throughput saturates near 5.1 M samples/s once a step carries ≥ ~250 k
samples; batch beyond 8 buys nothing and costs VRAM linearly (419.8 MB at b64). The CLI bound on
`--crop-seconds` is therefore raised from 1 s to 4 s: crops are context, not speed, and 4 s ×
batch 8 is the highest-throughput configuration measured (5.24 M samples/s, 97.7 ms/step). Long
crops with a small batch starve the tile grid — 4 s × batch 2 falls to 2.8 M samples/s.

### Where the remaining time goes

Two controlled measurements bound the opportunity:

* width 128 (263,456 parameters, a quarter of the MACs) runs at 9.92 ms/step against 25.39 ms at
  width 256 for the same 128 k samples. Solving the two-term model gives ≈ 20.6 ms of GEMM work and
  ≈ 4.8 ms of fixed per-step cost (features, loss, norm, Adam, transfers, synchronisation).
* 5.1 M samples/s at 1.018 M parameters is ≈ 1.1 TFLOP/s, roughly 10 % of the GTX 1080 Ti's
  11.3 TFLOP/s fp32 peak. The 16×16-tile GEMM kernels are the limit, not the GPU, the batch size
  or the crop.

### Register-blocked GEMM kernels (shipped)

The three GEMM kernels (`feral_conv_forward`, `feral_weight_grad`, `feral_input_grad`) now live in
`src/cuda/conv32.cuh`: 32×32 output tiles per 256-thread block, each thread holding a 2×2 register
block, 33-column padded shared tiles. That halves the global loads per FLOP and the
`__syncthreads()` count per useful FLOP relative to the 16×16 version.

| configuration | before | after | speedup |
| --- | --- | --- | --- |
| 1.0 s × batch 8 | 25.69 ms (4.98 M/s) | 14.43 ms (8.87 M/s) | 1.78× |
| 1.0 s × batch 16 | 49.53 ms (5.17 M/s) | 27.63 ms (9.26 M/s) | 1.79× |
| 4.0 s × batch 8 | 97.70 ms (5.24 M/s) | 54.60 ms (9.38 M/s) | 1.79× |

Validation, in the order it was done:

* a standalone harness compiles the candidate kernels to a cubin and drives them through libcuda,
  comparing against a numpy reference of the same math — forward 6.3e-7, dW 1.2e-6, dX 2.9e-6 max
  abs difference on a 3×64×12→20, k=3, d=2 case;
* `tests/cuda.rs` (CPU parity for a tiny pooled architecture, batched Adam, repeated trainers);
* `gpu-audit --backend cuda` against the 1.018 M checkpoint: every parity gate, bit-exact device
  resume, CPU continuation from the device checkpoint, 2.22 ms per device step.

A build-integrity bug was found on the way and fixed: `build.rs` emitted `rerun-if-changed` only
for `kernels.cu`, so editing an included `.cuh` silently reused the previous cubin. It now tracks
every file under `src/cuda`.

### 64×64 tiles with 4×4 blocking: parity exact, no speed gain

A second variant (`feral_*64`, 64×64 output tile, 4×4 register block, 65-column padded shared
tiles) was built and measured against the same numpy reference:

```
8 items x 256 frames, ci = co = 256, k = 3, d = 8      fwd     dW     dX    total
tile 32 (2x2 blocking, shipped)                       0.40   0.42   0.36   1.18 ms
tile 64 (4x4 blocking)                                0.42   0.29   0.45   1.17 ms
```

Both are bit-comparable to the reference (max 5.5e-6). The dW kernel gains 30 %, forward and dX lose
the same amount, net zero. That places the GEMMs past the shared-memory-bound regime: the first
rewrite (16×16/1 accumulator → 32×32/2×2) bought 1.79× by cutting shared-memory traffic, and a
further 2× of tile reuse buys nothing, so the limit is now global-memory traffic and occupancy, not
tile arithmetic intensity. The 32×32 version stays: same speed, fewer registers, already audited.

### Host/device pipelining: the GPU was idling 20 % of every step

The owner noticed GPU utilisation dipping between ~80 % and idle. Instrumenting the trainer with
CUDA events (device time between the first and last kernel of a step) and reporting the split in
`backend_info` (`last_step_gpu_ms`, `last_step_host_ms`) showed why: each step ended with a
blocking statistics readback (`cuMemcpyDtoH`), which drains the device, and only then could the
next step's 6 MB upload be enqueued. Measured at 4 s × batch 8, 1.215 M parameters: 58 ms/step,
46.6 ms device, ~11 ms stall, ≈ 80 % duty.

Two changes removed the stall:

* the step payload is uploaded straight from the caller's crop/mask/noise slices at their device
  offsets (`cuMemcpyHtoDAsync_v2`) instead of being concatenated into a fresh 6 MB host buffer and
  copied twice;
* the statistics readback is deferred by one step into pinned host memory, so the device never
  drains at a step boundary. The reported loss/norm therefore lag one step, which
  `backend_info.reported_loss_lag_steps = 1` records.

| 4 s × batch 8, 1.215 M params | before | after |
| --- | --- | --- |
| wall per step | 58.0 ms | 52.2 ms |
| device time | 46.6 ms | 50.1 ms |
| host/stall | ~11 ms | ~0 ms |
| GPU duty | ≈ 80 %, visible dips | 96–98 %, steady |

`Resident::flush()` exists so timing harnesses (`gpu-audit`) can wait for pipelined work; without
it a speedup claim would measure enqueue time instead of device time.

Remaining ceiling: the GEMMs run at ≈ 2 TFLOP/s of the card's 11.3 TFLOP/s fp32, and the step is
now ~96 % device-bound. Getting past that needs a different data-flow (fused im2col GEMM with
double-buffered shared memory, or holding the weight tile resident across many more frames per
block), not another tile-size tweak.

## 1,215,264 parameters on 61 recordings (10,000 steps, 4 s crops)

One million parameters is a design point, not a ceiling. This run doubles the
audio context (0.512 s → 4 s) and adds a sixth residual block, giving
1,215,264 parameters, on the largest leak-free split available: 61 of the
owner's recordings for training, 2 held out (`experiments/audio-corpus-64`; one
further file is rejected as empty and recorded in `run.json`).

```
feral train --input experiments/audio-corpus-64/train \
  --validation experiments/audio-corpus-64/validation --validation-every 1000 \
  --run experiments/big-1m4-c4000/run-10k-b8-c4000 \
  --backend cuda --batch 8 --crop-seconds 4.0 --width 256 --blocks 6 \
  --steps 10000 --checkpoint-every 1000 --preview-every 2000
```

| quantity | value | source |
| --- | --- | --- |
| trainable parameters | 1,215,264 | `metrics.json: parameters` |
| steps × crops × samples | 10,000 × 8 × 64,000 = 5.12e9 samples | `metrics.json` |
| step time | mean 52.57 ms, median 52.77 ms, p90 54.56 ms, max 76.35 ms | `metrics.json` |
| throughput while stepping | 19.02 steps/s, 9.74 M samples/s | `metrics.json` |
| end-to-end (previews included) | 9.84 steps/s, 1,016.5 s wall | `metrics.json` |
| held-out validation loss | 1.2340 → 0.0331 | `validation.jsonl` |
| fixed-probe loss | 1.2409 → 0.0674 | `metrics.json` |
| held-out evaluation | 384 probes, mean velocity MSE 0.1277 | `evaluate-validation.json` |
| CPU/device audit | all gates pass, 6.0e-8 parameter drift, 6.9e-6 loss diff | `audit.json` |
| observed device speedup | 387.1× (CPU 2,944.6 ms vs device 7.61 ms per step) | `audit.json` |
| device memory resident | 498.9 MB of an 8 GiB budget | `run.json: backend_info` |
| host peak RSS | 314 MB, inside the 3 GB ceiling | `metrics.json: peak_rss_kib` |

Validation was flat after the first thousand steps (0.0397 at 1 k, 0.0348 at
5 k, 0.0331 at 10 k) — the same shape as the 0.512 s runs, at a quarter of the
number of optimizer steps, which is what a longer crop buys: more signal per
step. The per-`t` breakdown of the held-out probes is 0.040 / 0.060 / 0.285 at
t = 0.1 / 0.5 / 0.9, so nearly all of the remaining error sits at the noisy end,
which is the expected rectified-flow profile rather than a plateau of the model.

Sampling, not training, now dominates wall clock: the 16-step previews cost
72 s each and the final sample another 72 s, so 508 s of the 1,016 s run was the
CPU sampler while the GPU idled. Training itself used 525.7 s.

### Bugs this run exposed

* **The second step of every CUDA run reported a zero loss.** The deferred
  statistics readback seeds `prev_stats` on the step after the first, but the
  first step's own statistics were only read into a local, so step 2 reported
  `loss 0.0, grad 0.0` in `losses.jsonl` (visible at step 2 of this run; the
  older pre-pipelining run has a real value there). The device-side optimizer
  was never affected — only the reported row. Fixed in `src/cuda/mod.rs`, and
  `tests/cuda.rs::reported_loss_lags_one_step_and_is_never_zero` fails without
  the fix.
* **The audit compared a lagged device loss against the current CPU loss.**
  With `reported_loss_lag_steps = 1`, `multi_step_loss_max_abs` read 0.889 on
  this checkpoint while parameters agreed to 6e-8. `src/audit.rs` now aligns the
  two series by the lag the backend declares and records `loss_alignment`; the
  same checkpoint reports 6.9e-6 over 21 aligned pairs.
* **The 64 MB checkpoint cap rejected the larger model.** Checkpoints are
  decimal JSON, roughly 22 bytes per number, and store weights plus both Adam
  moments: a 1.215 M-parameter model is 80 MB, and the largest architecture the
  CLI permits (width 256, 7 blocks) is about 110 MB. `Checkpoint::load` now
  allows 192 MB — a guard against untrusted files rather than a model-size
  limit — and `README.md` says so.

### Takes

Four unconditional 4 s takes (seeds 101–104, Heun, 32 flow steps) and one
`continue` take built on a 2 s protected prefix of the held-out *Clubhouse
Antics* recording. Every take records `protected_samples_exact: true`; the
continue take's largest boundary jump is 2.3e-8, i.e. the join is continuous to
float precision. Sampling cost was 144.6–146.4 s per 4 s take and 221.1 s for
the continue take.

The copying screen against a training recording gives nearest absolute
correlations of 0.081–0.107 (mean 0.0865) against a self-copy control of 1.0000
and a noise control of 0.032; the continue take scores 0.0683 against 0.0088.
That is the same bounded result as the earlier runs — not excerpts of the corpus
— and, as before, it is a screen, not proof of non-memorisation.

Artifacts: `experiments/big-1m4-c4000/run-10k-b8-c4000/` plus
`experiments/big-1m4-c4000/takes/`. A copy for listening, with the metrics that
belong to it, is in `~/Downloads/feral-big-1m-20261025-1841/`.

## Sampling on the device (250x faster, same trajectory)

The run above spent 508 s of its 1,016 s wall clock in the CPU sampler, so the next
piece of work was to evaluate the velocity field where the weights already live.
`flow::sample` is generic over `Velocity`, so the change is one `Velocity`
implementation backed by a resident trainer (`src/sampler.rs`) plus a
`--backend cpu|cuda` switch on `generate`, `continue`, `inpaint` and `variation`.

Two things made it small instead of invasive:

* `feral_features` already builds `x_t` and applies the parameterization scaling on
  the device (`pre ? (m == 1 ? y/scale : xt*cin) : xt`), and `read_prediction`
  already returns the network output — the audit has used both for forward parity
  since it existed.
* The solver, the conditioning projection, the collar and the safety limiter stay in
  `flow`. Only the field evaluation moves.

| mode | window | CPU | CUDA | speedup | max abs delta | correlation |
| --- | --- | --- | --- | --- | --- | --- |
| generate 4 s, 32 Heun steps | 64,000 | 144.6 s | 0.583 s | 248x | 2.4e-7 | 1.00000000 |
| continue, 2 s protected + 4 s | 96,000 | 221.1 s | 0.778 s | 284x | 3.0e-7 | 1.00000000 |
| inpaint, 1 s hole in a 3 s window | 48,000 | 107.3 s | 0.462 s | 232x | 1.8e-7 | 1.00000000 |

The deltas are fp32 summation order, not a different computation: 15,551 of the
64,000 samples in the generate take are bit-identical, and the whole CLI command
(checkpoint load included) takes 1.3 s. `protected_samples_exact` stays true in
every mode, and the continue take's boundary jump matches the CPU value exactly
(2.3018195e-8). `tests/cuda.rs::device_sampler_field_matches_cpu` pins the field
itself — both parameterizations, t in {0, 0.37, 1} — against `ConfiguredVelocity`.

How the field is fed without a kernel change: passing the same buffer as `clean`
and `noise` makes `x_t = x` at every position, and the known-side feature comes out
as `x*mask`, which equals `known*mask` because the solver keeps `x` equal to the
known samples wherever the mask is set. That invariant is now checked on every
evaluation rather than assumed.

A trap worth recording: the kernels apply the parameterization themselves, so
pre-scaling the inputs on the host — the obvious reading of `ConfiguredVelocity` —
double-scales the preconditioned variant. It fails at t = 0 with 0.18 error while
the legacy parameterization still matches, which is how the mistake was found.
The host feeds raw `x`; the device does the rest.

## Status at pause (2026-10-25)

`experiments/audio-1m-cuda-02/run-80k-c4000-b8` is the first run on the full corpus: 62 recordings
symlinked into `experiments/audio-corpus-64/train` (61 usable; one rejected as empty and recorded in
`run.json`), 2 held out, 4 s crops, batch 8, 1,018,400 parameters, register-blocked kernels. It ran
to step ~12,250 and was stopped on request; the last checkpoint is `checkpoint-010000.json`.

Measured before the stop: 52 ms/step ≈ 9.4 M samples/s (1.79× the pre-kernel-rewrite rate),
validation 1.1752 → 0.0376 (2k) → 0.0362 (4k) → 0.0318 (12k), GPU ~2.3 GiB total, 69 °C.

Resume with `--resume experiments/audio-1m-cuda-02/run-80k-c4000-b8/checkpoint-010000.json` into a
fresh run directory, keeping every other flag identical (crop, batch, width, blocks, steps).

An earlier attempt at the 1,215,264-parameter configuration was launched twice into one run
directory; both trainers wrote checkpoints into it and the directory was discarded rather than
analysed. The run documented above is the single-process rerun. Two trainers must never share a run
directory: a fresh run directory is created exclusively, so the failure mode is relaunching after
deleting a directory while the previous process is still alive.

Open items, in the order the measurements point at them:

* **In-training previews still sample on the CPU.** `generate`/`continue`/`inpaint`/`variation` now
  run the field on the device (below), but `train` still calls `flow::sample` with the CPU field, so
  508 s of the 1,016 s run above was previews. Note that this must be a *second* resident trainer
  with its weights uploaded per preview, not the training trainer: reusing it would leave the
  sampler's statistics in the deferred readback, and the next step would report the sampler's loss —
  the same failure class as the step-2 bug fixed above.
* A device field evaluation currently pays for a backward pass it does not need
  (`step(update = false)`): 7.6 ms per evaluation against roughly 2.5 ms for a forward-only entry.
  At 64 evaluations per take that is 0.58 s versus about 0.2 s — worth doing only once previews
  move to the device, where it would matter 5 times per run rather than once per command.
* The residual error sits at t ≈ 0.9 (0.285 MSE versus 0.040 at t = 0.1); if that matters, it is a
  loss-weighting question rather than a capacity one, since validation stopped improving after
  1,000 steps at both 1.018 M and 1.215 M parameters.
* The ≈ 4.8 ms fixed per-step cost measured earlier is still there, but at 4 s crops it is under
  10 % of a step; it only matters for short crops.
* 64×64 tiles with 4×4 register blocking were measured (above) and rejected: same speed, more
  registers. The GEMMs are global-memory bound, so the next step there is a fused im2col GEMM with
  double-buffered shared memory, not another tile size.

Copying screen on 5 s takes (`analyze`, reference = one training file): nearest absolute
correlation 0.098 (0.512 s run) and 0.095 (1.0 s run) against a self-copy control of 1.0 and a
noise control of 0.026. The takes are not excerpts of the corpus, but this is a bounded screen,
not proof of non-memorisation.

## What these numbers do not say

Loss on 0.512 s crops, held-out waveform MSE and the audit speedup establish that the pipeline
learns and that the device path reproduces the CPU reference. They say nothing about perceptual
quality, musical continuity or generalization. The 131× figure is a short foreground measurement
of the audit workload (one crop per step), not a thermally controlled peak or a sustained
throughput guarantee; the training figure above (51.1 steps/s at batch 4) is the honest
steady-state number for this configuration.

## Reproducing

Deterministic given the corpus copies and `--seed 42` (default). Two independent checks:

* in-process: the audit compares a reloaded device trainer against the live one bit-for-bit, and
  the CPU reference can continue from the device checkpoint within 5e-6 relative tolerance;
* end-to-end: `run-1m-resume-check` resumes `checkpoint-003000.json` for the remaining 3,000 steps
  and lands on the identical model — `model_hash bb7954e1a49ccc642c5c329a`, identical layer
  weights, identical Adam moments, identical RNG state and the same validation loss 0.043909.

`experiments/audio-1m-cuda-01/run-probe-200` is the 200-step capacity probe at width 192 (579,488
parameters) that motivated the width-256 configuration; it is kept as a control, not as a result.
