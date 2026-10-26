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

### Continuation to 50,000 steps

Resuming `checkpoint.json` for 40,000 further steps with identical flags (`run-50k-b8-c4000`) is
the first experiment the new sampler made cheap: 20 previews cost 7.2 s where the CPU field would
have cost about 500 s. Held-out validation keeps falling, so the flatness after 1,000 steps in the
shorter runs was not this configuration's limit:

| step | 10 k | 20 k | 30 k | 40 k | 50 k |
| --- | --- | --- | --- | --- | --- |
| held-out validation | 0.0331 | 0.0298 | 0.0277 | 0.0241 | 0.0237 |
| fixed probe | 0.0674 | — | — | — | 0.0627 |

The 40,000 steps took 2,549.7 s wall: mean 51.2 ms/step, median 50.2 ms, 15.7 steps/s end to end
including previews, 500 MB device-resident, host peak RSS 433 MB. The larger probe change is small
(0.0073 output delta MSE) because the model was already fitting this corpus; the held-out loss is
the number that moved. `audit.json` in that run directory passes every gate against the CPU
reference (386.6x, 6.3e-6 aligned loss difference, 1.8e-7 parameter drift, exact device resume and
CPU continuation), which also covers the forward/backward refactor below.

Held-out evaluation on the same 384 probes improves from 0.1277 at 10 k to 0.1050 at 50 k (-18 %).

Four 4 s takes and one continue take from this checkpoint (seeds 101–104 and 201, 32 Heun steps)
are the `take50k-*`/`continue50k-*` files in `experiments/big-1m4-c4000/takes/`, produced in
0.29–0.39 s each. The copying screen is unchanged from the 10 k takes — nearest absolute
correlation 0.059–0.123 against a noise control of 0.032 — so 40,000 more steps did not make the
model more likely to reproduce the corpus by this measure.

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

### Previews inside `train`

Training previews and the final take now sample on the training backend too, so the
508 s of previews in the 10k run above become a rounding error. The sampler gets a
*second* resident trainer built from the current checkpoint, not a borrow of the
training one: a shared trainer would push the sampler's statistics into its deferred
readback and the next step would report the sampler's loss — the same failure class
as the step-2 bug fixed earlier. Weights upload once per sample (5 MB here).

Two measurements, both against the CPU path on identical training:

| configuration | previews | CPU path | device | speedup |
| --- | --- | --- | --- | --- |
| width 32, 2 blocks, 4 s crops, 40 steps, 3 previews | 3 | 1,785 ms | 282 ms | 6.3x |
| width 256, 6 blocks (1.215 M), 4 s crops, 3 samples | 3 | 218 s | 1.40 s | 155x |

The tiny model is the pessimistic case — its CPU step is only 2.8 ms, so the sampler
was most of that run's wall clock (1,446 ms -> 441 ms end to end, 3.3x). At the real
configuration the same three samples cost 72.6 s each on the CPU, which is exactly
the 508 s the 10k run spent: that run would now take about 590 s instead of 1,016 s.

The in-loop preview is the same audio, not merely similar: `preview-000020.wav` from
the device-preview run against `feral generate --backend cpu` on the checkpoint that
run wrote at step 20 gives max |Δ| 3.6e-7, correlation 1.00000000.

### Forward-only device evaluation

`step(update = false)` computed gradients nobody read, which cost more than the forward
itself. The upload-and-forward half of `execute` is now `forward_pass`, shared by both
paths; `CudaTrainer::predict` stops after it, and `Resident::predict` keeps the old
behaviour as the default for backends that have no cheaper entry.

| path | 4 s take, 32 Heun steps | per field evaluation |
| --- | --- | --- |
| CPU reference | 144.6 s | ~2.3 s |
| device, `step(update = false)` | 0.583 s | 7.6 ms |
| device, forward-only | 0.293–0.317 s | 2.9 ms |

That is 1.96x over the step-based sampler and about 490x over the CPU reference. The
output is not merely close: the same checkpoint and seed produce the identical WAV
sha256 (`24dcc8c7…`) through both device paths, which is the regression test that
matters for this refactor. `gpu-audit` on the 50k checkpoint re-verifies gradients,
Adam updates and resume through the shared `forward_pass`.

### Checkpoints are written compact

Pretty-printing a checkpoint costs about 39 % more bytes for no reader: a 1.215 M-parameter
checkpoint was 79.9 MB and is now 49.6 MB, and `Checkpoint::save` takes 146 ms instead of 220 ms for
the same state. Artifacts (`metrics.json`, `run.json`, take metadata) keep the readable writer, and
both forms parse identically, so every checkpoint written before this change still loads — the
40,000-step continuation above wrote its checkpoints in the old format.

Measured on this machine with a throwaway test that saved both forms of the 50k checkpoint and
reloaded them: identical `model_hash`, identical `completed_steps`.

## Generation test at 50,000 steps (29 takes)

The weights are worth listening to now, and a take costs 0.3 s on the device, so the test is a
matrix rather than one sample: 29 takes, every one a real CLI invocation, with the per-take
descriptors in `reports/generation-test-01.json` and the audio in
`experiments/big-1m4-c4000/generation-test-01/takes/`. Groups: three training states on the same
four seeds, eight seeds at 50 k, solver and step counts, prior amplitude, duration, and conditioning
on both held-out files plus one training file.

| group | RMS | flatness | lag-1 | limiter hits | bands <250 / 250-1k / 1k-4k / >4k |
| --- | --- | --- | --- | --- | --- |
| untrained (step 0), 4 seeds | 0.760 | 0.988 | 0.017 | 108,774 | 0.03 / 0.09 / 0.38 / 0.50 |
| 10 k, 4 seeds | 0.162 | 0.217 | 0.879 | 4 | 0.55 / 0.24 / 0.17 / 0.04 |
| 50 k, 8 seeds | 0.152 | 0.305 | 0.828 | 3 | 0.53 / 0.19 / 0.23 / 0.06 |
| corpus reference | 0.301 | 0.105 | 0.930 | — | 0.68 / 0.14 / 0.16 / 0.02 |

What the numbers support:

* **Training turned noise into audio-shaped output.** Untrained takes are near-white (flatness
  0.988), sit on the 0.98 limiter (27,000 hits per take) and have no temporal correlation (lag-1
  0.017). Both trained checkpoints have lag-1 0.83–0.88, flatness 0.22–0.31 and essentially no
  limiter activity.
* **It has not learned the corpus's spectral tilt.** Flatness is 3× the reference and energy sits
  higher: 23 % in 1–4 kHz against 16 %, 6 % above 4 kHz against 2 %. Broadband texture and level
  are learned; the tilt is not. That is the concrete gap, and it is measurable.
* **Seeds are independent.** Mean pairwise correlation across eight seeds is −0.008 (−0.034 to
  +0.034), so there is no mode collapse. The same seed at 10 k and 50 k correlates +0.85: the 50 k
  model is a continuation, slightly brighter (flatness 0.217 → 0.305) at the same level.
* **Solver and prior amplitude are load-bearing.** Heun converges by 8 steps (RMS 0.137 at 8, 0.132
  at 64); Euler is under-integrated at 8 (RMS 0.073) and still short at 32 (0.111). `--noise-scale`
  moves output level almost linearly (0.5 → RMS 0.013, 1.5 → 0.500 with 4,134 limiter hits), so it
  is an amplitude control, not a creativity knob. Duration 1/2/4/5 s gives RMS 0.118/0.137/0.15/0.136.
* **Conditioning holds.** `protected_samples_exact` is true in all four conditioned takes, with
  boundary jumps 0.0 (Clubhouse Antics), 0.0083 (FISREAL), 0.0 (training file) and 0.247/0.025 at
  the two inpaint joins.
* **The copying screen behaves.** Generated takes score 0.068–0.124 nearest absolute correlation
  against a 0.032 noise control, and the continue take built on a training recording scores 1.0000 —
  a positive control the screen passes.
* **Peaks above full scale (1.30, 1.40)** appear only in continue takes and come from the protected
  prefix, which is preserved bit-exactly by contract; generated samples are clamped to 0.98. A float
  WAV carries it, a playback chain with a hard limiter would clip the prefix.

What this does not say: descriptors are not perceptual quality, and the screen is bounded. The
evidence supports "the model learned audio structure, is seed-diverse, conditions correctly and is
too bright", not "it sounds good". Listening is still the only test for the latter.

## The receptive field was the ceiling (0.254 s)

The 29-take test above showed audio-shaped output that is too bright and has no long-range
structure. The reason is structural, not an optimisation shortfall: `Architecture::context_samples`
for the trained stack is 4,064 samples, i.e. **0.254 s at 16 kHz**. Whatever the crop length, an
output sample cannot depend on anything further away than that.

| architecture | receptive field |
| --- | --- |
| baseline 6 blocks, block 32 (everything trained so far) | 4,064 samples / 0.254 s |
| baseline 7 blocks, block 32 (previous CLI maximum) | 8,160 / 0.510 s |
| pyramid as shipped (scales 1/4/16/64) | 7,488 / 0.468 s |
| block 128, 8 dilations (probe below) | 65,408 / 4.088 s |
| block 128, 10 dilations (new maximum) | 262,016 / 16.376 s |

So the caps had to move before any of "more capacity" or "longer crops" could mean anything:
`--block` now sets 16..128 samples per frame, widths run to 512, dilations to 512 across up to 10
layers, and crops to 32 s through one shared constant (`MAX_WINDOW_SAMPLES = 512_000`) used by
checkpoint validation, crop validation, previews and render windows alike.

Larger frames are also how this is paid for. Convolution cost scales with *frames*, so block 128
gives 4x fewer frames per sample; that is what makes width 512 affordable instead of four times
slower.

Probe at the intended shape — block 128, 8 dilations, width 512, 16 s crops x batch 2 (512 k
samples/step, the same sample budget as today):

| quantity | value |
| --- | --- |
| parameters | 6,560,384 (5.4x the trained model) |
| step time | mean 125.2 ms, median 130.7 ms |
| device memory | 420 MB resident of an 8 GiB budget |
| host peak RSS | 538 MB |
| loss, 20 steps from scratch | 1.312 → 0.640 |

Correctness at the new shapes was checked twice, because block 128 changes the feature layout to
388 channels per frame and exercises a 128-frame dilation in the kernels:

* `tests/cuda.rs::large_frame_and_wide_dilation_parity` — loss, gradient and prediction parity
  against the CPU reference at block 128 with dilations 1..128;
* a full `gpu-audit` on a small checkpoint built at the new shapes: every gate passes, gradients and
  Adam match, device resume is exact, 16.74x on that reference workload.

## What actually fixed the sound: corpus, not capacity

Five arms at equal sample budget (512 k samples/step), each generating four takes at seeds 11/22/33/44.
Per-take numbers are in `reports/arm-comparison-01.json`; the audio is in
`~/Downloads/feral-arm-compare-20261026-1656/`.

| recipe | params | corpus | steps | held-out | RMS | flatness | lag-1 | bands <250/250-1k/1k-4k/>4k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline, 4 s | 1.2 M | 31 min | 50 k | 0.0237 | 0.137 | 0.370 | 0.792 | 0.45 / 0.20 / 0.27 / 0.08 |
| D, 4 s | 1.2 M | 189 min | 20 k | 0.0322 | 0.160 | 0.142 | 0.922 | 0.71 / 0.16 / 0.10 / 0.03 |
| E, 16 s, block 128 | 6.5 M | 189 min | 20 k | 0.0388 | 0.150 | 0.146 | 0.921 | 0.63 / 0.24 / 0.10 / 0.03 |
| curriculum 4→8→16→24 s | 6.5 M | 189 min | 20 k | 0.0431 | 0.167 | 0.145 | 0.922 | 0.66 / 0.20 / 0.11 / 0.03 |
| C, preconditioned | 1.2 M | 31 min | 20 k | — | 0.252 | 0.327 | 0.812 | 0.49 / 0.25 / 0.18 / 0.08 |
| F, preconditioned | 1.2 M | 189 min | 20 k | — | 0.247 | 0.093 | 0.952 | 0.75 / 0.17 / 0.06 / 0.02 |
| real corpus | — | — | — | — | 0.301 | 0.105 | 0.930 | 0.68 / 0.14 / 0.16 / 0.02 |

* **The corpus was the defect.** At fixed architecture and steps, 31 → 189 minutes of audio took
  flatness from 0.370 to 0.142 against real audio's 0.105, lag-1 from 0.792 to 0.922 against 0.930,
  and limiter hits to zero. The 50 k-step baseline on the short corpus is the noisiest model in the
  set — more steps on 31 minutes does not compete with less training on 189.
* **Capacity and context did not pay at this budget.** 6.5 M parameters with a 4.09 s field and
  16 s crops land on the same descriptors as the 1.2 M/4 s model, are worse on held-out loss at
  equal steps (0.0388 vs 0.0322), and cost 3x more per take (0.9 s vs 0.3 s). With 189 minutes of
  audio, 6.5 M parameters at 20 k steps is undertrained rather than capable.
* **The curriculum added nothing measurable.** Validation ran 0.0431 → 0.0415 → 0.0432 → 0.0431
  across 4/8/16/24 s stages. That is what a 4.09 s receptive field predicts: a crop longer than the
  field only adds padding behind each prediction. Note also that the 32 s stage had to be run at
  24 s: at 32 s the 30 s file becomes "shorter than one crop", which changes the dataset identity and
  makes a full resume refuse — the guard is right, the curriculum was simply out of useful range.
* **Preconditioning fixes level, then over-smooths.** F reached RMS 0.247 against real 0.301 and
  flatness 0.093 against 0.105 — but its 1–4 kHz share fell to 0.06 against real 0.16, with too much
  below 250 Hz. That is MSE regression to the conditional mean: right level, missing detail.
* **What is left is the objective.** Every full-corpus arm is short of real audio in the 1–4 kHz
  band (0.10, 0.10, 0.11, 0.06 against 0.16) while sitting at or above it in level. More data and
  more parameters will not close that; a detail-aware objective (spectral weighting, or a
  high-pass-weighted residual) is the next real piece of work.

Two limits were hit and fixed on the way: a 6.5 M checkpoint is 267 MB of decimal JSON, so the load
cap went 192 → 768 MB (base64 would cut that ~3x), and `--grow-crop` exists because a curriculum
step changes the crop but not the weights.

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

* Previews construct a resident trainer per sample (~0.3 s of sampling plus ~0.2 s of context
  setup and weight upload at this size). Caching one preview trainer and uploading weights before
  each sample needs an upload path that does not exist yet; it is worth building only if previews
  become much more frequent than every 2,000 steps.
* The residual error sits at t ≈ 0.9 (0.285 MSE versus 0.040 at t = 0.1). Forty thousand extra
  steps cut held-out validation by 28 % (0.0331 → 0.0237) but moved the fixed probe only 7 %
  (0.0674 → 0.0627), so the endpoint is not simply under-trained — a loss weighting that spends
  more capacity near t = 1 is the next thing to try, not more steps.
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
