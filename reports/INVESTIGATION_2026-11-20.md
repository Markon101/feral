# FERAL Investigation Report — 2026-11-20

**Scope:** Diagnosis of the stalled 400k-step mono `GatedFilm` run; implementation and parity verification of native 2-channel interleaved stereo (`--channel stereo`), non-saturating depth-scaled `GluFilm` residual blocks, decoupled AdamW weight decay, single-file `FERALCK1` binary checkpoints, and equal-compute multi-scale crop horizons; clean deduplicated stereo AI-music corpus curation; 4-way multi-horizon ablation; capacity scaling probes; 50,000-step 24.6M stereo validation run; and scaled `FERALCK1` resumability proof.

> [!IMPORTANT]
> Per `AGENTS.md`, lower training/validation losses, lower multiresolution STFT L1 errors, smoother spectral flatness, and stereo L/R correlations **do not establish perceptual quality, timbral fidelity, or musical continuity**. Human listening to the generated stereo takes is required before deciding whether to resume the 50k checkpoint for a longer training run.
>
> Desktop working-memory ceiling: **8 GB** (mobile/Termux 3 GB target is separate). Measured peak host RSS (`peak_rss_kib`) is reported against the desktop ceiling throughout.

---

## Executive Summary

1. **Why the 400k mono run stalled (`arm-o-gated-film-sym-gpu-stft-4m`, preserved untouched):** Held-out validation plateaued early (`1.1727` -> `0.1201` at 100k -> `0.1152` at 200k -> `0.1139` at 300k -> `0.1132` at 405k). Inspection revealed severe activation saturation under `GatedFilm` (`tanh(a) * sigmoid(b)` without weight decay), weight growth up to `max |w| = 7.058`, weak spectral loss weighting (`0.25`), heterogeneous/duplicated training files (`19.7%` `rust_*`/`titan*` files plus train/validation song-family leakage), and JSON checkpoints (`187.1 MB` at 3.56M parameters) that made scaling much further impractical.
2. **Architecture & pipeline upgrades (CPU oracle + CUDA `sm_61`, `fmad=false`):**
   - **Native interleaved stereo (`--channel stereo`):** End-to-end 2-channel `[L0, R0, ...]` PCM layout, stereo `stem`/`head`, frame-aligned conditioning masks, per-channel collars, and 4-stream **Left / Right / Mid / Side** multiresolution STFT loss (`1/4` equal weight per stream, zero gradient on protected samples). Legacy mono models remain bit-identical.
   - **Non-saturating `GluFilm` blocks (`--block-style glu-film`):** Replaces bounded `tanh(a)` content with `a * sigmoid(b)` scaled by `1 / sqrt(num_residual_blocks)`.
   - **Decoupled AdamW (`--weight-decay`):** Applies decay selectively to weight tensors while keeping all biases unregularized (`Model::weight_decay_mask()`).
   - **Single-file `FERALCK1` binary checkpoints:** Automatically used above 4,000,000 parameters (`tensors_sha256` covering contiguous little-endian f32 `weights + adam.m + adam.v [+ ema_weights]`), cutting 24.6M-parameter checkpoint size to `394.33 MB` and load latency to `~69 ms`.
   - **Equal-compute multi-scale horizons (`--crop-horizons`, `--horizon-schedule`):** Maintains roughly constant `crop_frames * batch` per step across durations (`0.512s@16`, `1.024s@8`, `2.048s@4`, `4.096s@2`).
3. **Curated clean stereo corpus (`157` train / `19` val):** Deduplicated by SHA-256 and mirror cleanup from `~/Music` and `~/Downloads` (read-only originals), removing non-music/noise/short outliers and splitting strictly by title family with zero cross-split family overlap. All 176 tracks are native 2-channel stereo (`169` @ 48 kHz, `7` @ 44.1 kHz); median post-preprocess stereo RMS is `0.1602`, motivating `--data-scale 0.16 --prior-std 0.16`.
4. **Equal-compute multi-horizon ablation (4 × 4,000 steps, 5.60M stereo params):** `short-to-long` outperformed `fixed`, `random`, and `long-to-short` on **every** held-out `feral evaluate` metric (`vel_mse = 0.023335`, `ep_stft_l1 = 0.247571`, `ep_mse = 0.008787`, `ep_deriv = 0.002964`), while `long-to-short` was worst on every metric.
5. **Scaled 24.6M stereo validation run (50,000 steps on GTX 1080 Ti):**
   - **24,643,684 parameters** (**6.91×** the 3.56M `arm-o` baseline; **17.4×** earlier 1.4M runs; **425×** the 57,952-param v1 baseline), fitting in `833.2 MiB` resident VRAM (`10,260.1 MiB` free) and `peak_rss_kib = 4,796,888` (`4.575 GiB` vs 8 GB desktop ceiling), running at `82.20 ms/step` mean (`12.06 steps/s`, `69.12 min` wall).
   - Held-out validation decreased monotonically throughout training (`1.609800` -> `0.273870`, `-83.0%`). Fixed held-out `feral evaluate` probes improved continuously across `10k -> 50k` (`vel_mse 0.020840 -> 0.015429`, `ep_stft_l1 0.224020 -> 0.157600`, `ep_deriv 0.002606 -> 0.001681`), with EMA consistently beating raw weights.
   - `GluFilm` + `weight_decay=0.01` kept `max |w| = 0.857`, `content_sat = 0.0%` across all 15 blocks, and `mean_sig_sat = 14.37%`.
   - **Full resumability verified at 24.6M scale:** Splitting steps `40001–40002` across two `FERALCK1` checkpoints matched uninterrupted 2-step execution across all `394,298,944` binary tensor bytes (`weights + adam.m + adam.v + ema_weights`), `tensors_sha256`, `model_hash`, `adam.step`, and `rng.state`.

---

## 1. Audit of the Preserved 400k Mono Baseline (`arm-o-gated-film-sym-gpu-stft-4m`)

Preserved directory: `experiments/big-1m4-c4000/arm-o-gated-film-sym-gpu-stft-4m/` (stopped before 409k and retained untouched; last logged line step `408,820` in `losses.jsonl`, last periodic save `checkpoint-400000.json`).

### Configuration & Plateau Trajectory
- **Architecture:** `3,564,482` parameters, mono `mix`, `block = 64`, `width = 192`, `dilations = [1, 2, 4, 8, 16, 32, 64, 128, 64, 32, 16, 8, 4, 2, 1]` (15 symmetric `gated-film` residual blocks), `sample_conv = 7`.
- **Training objective & schedule:** `batch = 8`, `crop_samples = 64000` (4.0 s at 16 kHz), `preconditioned` (`data_scale = 0.16`, `prior_std = 0.16`), `spectral_weight = 0.25`, `weight_decay = 0.0`, `ema_decay = 0.9995`, `logit-normal` `t`, cosine LR (`min_lr = 1e-5`, 4,000 warmup, 4,000,000 total steps).
- **Held-out validation (`validation.jsonl`, 82 points):**

| Step | 0 | 100,000 | 200,000 | 300,000 | 405,000 |
|---|---:|---:|---:|---:|---:|
| `validation_loss` | 1.172701 | 0.120129 | 0.115247 | 0.113853 | 0.113245 |

Between steps `100k` and `405k`, validation loss fell by only `0.006884` (`-5.7%` relative over 305k steps).

- **Preserved baseline held-out evaluation (`experiments/investigations/2026-11-20-baseline-400k/eval-val.json`, 96 probes):**
  - `mean_velocity_mse`: `0.012820`
  - `endpoint log_spectral_l1`: `0.153397`
  - `endpoint mse`: `0.004916`
  - `endpoint derivative_mse`: `0.001788`
- **4 s 64-step Heun mono generation (`gen-heun64.wav`, `analyze-gen-heun64.json`):** `780.9 ms` CUDA sampling, `peak_rss_kib = 228,764`, `rms = 0.1636`, `spectral_flatness = 0.0939`, `lag1_uncentered = 0.9455`, spectral bands `[<250, 250-1000, 1000-4000, >=4000 Hz]` = `[0.739, 0.140, 0.102, 0.018]`, `limiter_hits = 0`.

### Diagnosed Root Causes
Using a uniform Gaussian forward probe (`t = 0.5`, `prior_std = 0.16`, saturation defined as `sigmoid < 0.05 or > 0.95` and `|content_pre| > 2.0`):
1. **Severe activation saturation & weight growth:** Without weight decay, unscaled `0.5 * (tanh(a) * sigmoid(b))` across 15 `gated-film` blocks drove `max |w| = 7.058` (`mean_layer_w_rms = 0.1340`), `mean_tanh_sat_pct = 28.89%` (peaking at `49.28%` in block 2 and `40.41%` in block 3), and `mean_sig_sat_pct = 22.26%` (peaking at `36.75%` in block 4 and `23.49%` in block 7). Residual RMS ranged from `0.466` (block 0) to `0.643` (block 14).
2. **Weak spectral objective weighting:** `--spectral-weight 0.25` was small relative to normalized residual MSE on complex music waveforms.
3. **Heterogeneous corpus & train/validation leakage:** Of the `61` loaded mono training sources in `run.json` (`1` rejected: `titan_prime_0s.wav`), `12/61` (`19.7%`) were `rust_*`/`titan*` files (often synthetic/experimental renders with RMS down to `0.0352`), alongside non-song clips (`70_57_17.wav`), multiple duplicate masters/versions (`Hyperlearn Odyssey ... (1).wav`, three `Path to Connection` files), and direct train/validation leakage (`FISREAL (1).wav` and `FISREAL (2).wav` in training while `FISREAL.wav` sat in validation).
4. **Capacity & serialization ceiling:** `checkpoint-400000.json` already weighed `187.1 MB` in compact JSON; scaling to 20M+ parameters in JSON would exceed `1 GB` per checkpoint and add multi-second serialization stalls.

---

## 2. Implemented Architectural & Pipeline Changes

All additions default to off / mono / `tanh` so legacy mono model hashes (`channels = 1` omitted by serde) and trajectories remain bit-identical unless opted in. Both CPU (numerical oracle) and CUDA (`sm_61`, `fmad=false`) match to fp32 summation order and pass finite-difference derivative checks:

- **Native 2-channel interleaved stereo (`--channel stereo`):**
  - `src/audio.rs`, `src/data.rs`: Decodes via `ffmpeg -ac 2`, jointly peak-normalizes to preserve stereo L/R balance, writes stereo float32 WAV (`WAVE_FORMAT_EXTENSIBLE` via `hound`), adds synthetic stereo fixtures (`synthetic_stereo`), raises file selection cap from `128` to `512`, and makes the decoded RAM cap configurable (`FERAL_MAX_DATASET_MB` `256..=4096` MB, default `1024` MB, 4 bytes per scalar float).
  - `src/model.rs`, `src/conditioning.rs`, `src/flow.rs`: Interleaves `[L0, R0, ...]` into `frame_elems = block * channels` (`3 * frame_elems + 4` input channels per frame) so the dilated frame convolution stack executes at the exact same time-frame count `T / block` for mono and stereo. Sample-level `stem` (`3*C -> 16 -> C`) and `head` (`C -> 16 -> C`) operate over `time_steps = scalars / C`. Masks and tangent collars (`training_mask_channels_into`, `continuity_channels`) remain frame-aligned so Left and Right share protection and exact canonical sample retention.
  - `src/spectral.rs`, `src/cuda/kernels.cu`, `src/cuda/mod.rs`: Evaluates multiresolution Hann STFT log-magnitude Charbonnier L1 (`N = 128/512/2048`, `hop = N/4`) across **four equal-weight streams (`1/4` each)** — **Left**, **Right**, **Mid** `(L+R)/sqrt(2)`, and **Side** `(L-R)/sqrt(2)` — backpropagating exact chain-rule gradients only to unmasked positions.
  - `src/evaluate.rs`, `src/audit.rs`, `src/sampler.rs`: Stereo-aware forward/backward parity audits, per-channel detail finite-difference stencil (`i + ch`), stereo held-out evaluation probes, and CUDA/CPU `DeviceVelocity::channels()`.
- **Non-saturating depth-scaled GLU (`--block-style glu-film`):** `h + (a * sigmoid(b)) / sqrt(num_blocks)` eliminates the outer `tanh(a)` saturation wall and keeps residual scale balanced across deep symmetric stacks (`src/model.rs`, `src/cuda/kernels.cu`).
- **Decoupled AdamW (`--weight-decay`):** Implemented via `Model::weight_decay_mask()` (`1.0` on weights, `0.0` on biases) in `Adam::update_with_lr_wd` (`src/runtime.rs`) and `feral_adam` (`src/cuda/kernels.cu`).
- **Single-file `FERALCK1` binary checkpoints (`src/runtime.rs`):** Auto-selected above `4,000,000` parameters and transparently detected by `Checkpoint::load` (capped at `2 GiB` on load). Stores `b"FERALCK1"`, little-endian `u64` header length, UTF-8 JSON metadata (including `tensors_sha256`, architecture, Adam config, EMA metadata, and RNG state), followed by contiguous little-endian `f32` payloads (`weights`, `adam.m`, `adam.v`, and optional `ema_weights`).
- **Equal-compute multi-scale crop horizons (`--crop-horizons`, `--horizon-schedule`, `--horizon-steps`):** `HorizonSchedule::{Fixed, Random, ShortToLong, LongToShort}` in `src/parameterization.rs` pairs each duration with an inversely scaled batch item count to maintain near-constant `crop_frames * batch` per step. `CudaTrainer::new_with_horizons` and `set_shape(n, items)` preallocate maximum workspace capacities once at startup with zero per-step reallocation.
- **Generation window expansion (`src/model.rs`, `src/runtime.rs`):** Raises `generate` `--seconds` cap to `16.0` s (`MAX_WINDOW_SAMPLES = 512_000` time frames = `32 s` at 16 kHz including conditioning context).
- **OpenCL boundary (`src/gpu/mod.rs`, `src/backend.rs`):** Explicitly rejects stereo (`channels != 1`), non-zero `weight_decay`, and multi-horizon schedules (`OpenClTrainer` stays mono-only).

---

## 3. Curated Clean Stereo AI Music Corpus

All candidate audio under `~/Music` and `~/Downloads` (excluding FERAL output directories; originals strictly read-only) was scanned and filtered:
- **Initial scan:** `262` audio files -> `238` unique SHA-256 hashes (`1` undecodable: `titan_prime_0s.wav`).
- **Cleaning:** Removed `28` non-music/noise/speech/short outliers and `33` `OLD_WAVS/*.wav` mirror duplicates -> **`176` unique clean stereo AI-music tracks**.
- **Manifest splits (zero shared title families across splits after normalized title-family grouping):**
  - **Train (`experiments/manifests/ai-music-clean-train.txt`):** `157` files (100 title families), 100% 2-channel stereo (`151` @ `48 kHz`, `6` @ `44.1 kHz`), `455.11 min` at `--max-file-seconds 300`.
  - **Validation (`experiments/manifests/ai-music-clean-val.txt`):** `19` files (16 title families), 100% 2-channel stereo (`18` @ `48 kHz`, `1` @ `44.1 kHz`), `52.47 min` at `--max-file-seconds 300`.
- **Stereo corpus load probe (`experiments/investigations/2026-11-20-stereo-corpus-probe/metrics.json`):**
  - Loaded `157 / 157` train files, `0` rejected, `436,908,237` stereo frames (`~3.495 GB` f32 with `FERAL_MAX_DATASET_MB=4096`), `peak_rss_kib = 2,745,036` (`2.618 GiB` vs 8 GB desktop ceiling).
  - Post-preprocess peak-normalized (`0.8`) stereo RMS: **median `0.1602`**, mean `0.1630`, min `0.1104`, max `0.2496` — directly calibrating `--data-scale 0.16 --prior-std 0.16`.

---

## 4. Equal-Compute Stereo Multi-Horizon Ablation (4 × 4,000 Steps)

**Setup (`experiments/investigations/2026-11-20-stereo-horizon-ablation/{fixed,random,short-to-long,long-to-short}`):**
`5,603,172` parameters (`width = 256`, `block = 128`, `blocks = 7` symmetric [13 blocks], `glu-film`, `sample_conv = 7`, `stereo`, `batch = 4` @ `2.048 s` -> `131,072` frames/step across `0.512s×16`, `1.024s×8`, `2.048s×4`, `4.096s×2`), `spectral_weight = 2.0`, `weight_decay = 0.01`, `ema_decay = 0.995`, cosine LR `3e-4 -> 3e-5` (200 warmup, 4,000 steps), `FERALCK1` `89.68 MB`, CUDA resident alloc `272.5–272.8 MiB`, `27.64–27.86 ms/step`.

> [!NOTE]
> Periodic `validation_loss` inside `train` probes `probe_window_samples(sr, max_crop_frames, block)` — `2.048 s` for `fixed` vs `4.0 s` for multi-horizon runs — so cross-arm quality must be compared on **identical `2.048 s` probes** via `feral evaluate` (`192` held-out crops per arm).

| Arm | Peak RSS | Step (ms) | Eval Vel MSE | Eval EP STFT L1 | Eval EP MSE | Eval EP Deriv | 4s Take L/R Corr | 4s Side/Mid | 4s Flatness | Max \|w\| | Sig Sat | Content Sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed` (`2.048s×4`) | 4,227,240 KiB (4.031 GiB) | 27.64 | 0.023500 | 0.251663 | 0.008805 | 0.003066 | 0.8110 | 0.3230 | 0.1888 | 0.535 | 4.00% | 0.0% |
| `random` | 4,388,004 KiB (4.185 GiB) | 27.84 | 0.023531 | 0.252549 | 0.008839 | 0.003102 | 0.8056 | 0.3282 | 0.1910 | 0.503 | 4.39% | 0.0% |
| **`short-to-long`** | **4,388,020 KiB (4.185 GiB)** | **27.85** | **0.023335** | **0.247571** | **0.008787** | **0.002964** | **0.8101** | **0.3239** | **0.1864** | **0.579** | **6.82%** | **0.0%** |
| `long-to-short` | 4,383,108 KiB (4.180 GiB) | 27.86 | 0.023674 | 0.255812 | 0.008862 | 0.003190 | 0.8002 | 0.3332 | 0.1999 | 0.543 | 4.29% | 0.0% |

**Result:** At identical compute, `short-to-long` (`0.512s -> 1.024s -> 2.048s -> 4.096s`) won every single held-out `feral evaluate` metric and spectral flatness, while `long-to-short` finished worst on all held-out metrics.

---

## 5. Stereo Capacity Scaling Probes (`experiments/investigations/2026-11-20-stereo-scale-probes/`)

Measured over 20 steps with `block = 128`, `blocks = 8` symmetric (`15` residual blocks, dilations `1..128..1`), `glu-film`, `sample_conv = 7`, `stereo`, `short-to-long` (`0.512s,1.024s,2.048s,4.096s`), `spectral_weight = 2.0`, `weight_decay = 0.01`:

| Probe | Width | Batch (@2.048s) | Params | Resident VRAM | Device Free VRAM | Peak Host RSS (vs 8 GB) | Mean / Median Step | Last GPU Step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `14m-b4` | 384 | 4 | 14,059,492 | 535.7 MiB | 10,472.1 MiB | 4,520,236 KiB (4.311 GiB) | 48.80 / 51.12 ms | 49.11 ms |
| `24m-b2` | 512 | 2 | 24,643,684 | 698.6 MiB | 10,396.1 MiB | 4,792,940 KiB (4.571 GiB) | 43.48 / 45.39 ms | 42.33 ms |
| **`24m-b4`** | **512** | **4** | **24,643,684** | **833.2 MiB** | **10,260.1 MiB** | **4,792,568 KiB (4.571 GiB)** | **77.33 / 80.71 ms** | **76.32 ms** |

`24m-b4` (**6.91×** the 3.56M `arm-o` baseline, **17.4×** 1.4M models) used under `7.5%` of GTX 1080 Ti VRAM and remained comfortably under the 8 GB host RSS ceiling while doubling samples/step over `b2`, so it was selected for the 50,000-step validation run.

---

## 6. Scaled 50,000-Step 24.6M Stereo Validation Run (`experiments/investigations/2026-11-20-stereo-scaled-50k/`)

### Configuration & Runtime Telemetry
```sh
FERAL_MAX_DATASET_MB=4096 ./target/release/feral train \
  --backend cuda --channel stereo \
  --input experiments/manifests/ai-music-clean-train.txt \
  --validation experiments/manifests/ai-music-clean-val.txt \
  --block 128 --blocks 8 --dilation-pattern symmetric --width 512 \
  --block-style glu-film --sample-conv 7 --max-file-seconds 300 \
  --parameterization preconditioned --data-scale 0.16 --prior-std 0.16 \
  --time-sampling logit-normal --lr-schedule cosine --warmup-steps 1000 \
  --learning-rate 2e-4 --min-lr 2e-5 --weight-decay 0.01 \
  --spectral-weight 2.0 --ema-decay 0.999 --seed 42 \
  --crop-seconds 2.048 --crop-horizons 0.512,1.024,2.048,4.096 \
  --horizon-schedule short-to-long --horizon-steps 50000 --schedule-steps 50000 \
  --batch 4 --steps 50000 --validation-every 2000 --checkpoint-every 10000 --preview-every 10000
```
- **Model:** `24,643,684` parameters, `model_hash = 6d04f5991e9190e2a012fc1c15ca73100e9a44b74298ac063fe8e88a7ce59760`, `checkpoint.json` size `394,331,891` bytes (`FERALCK1`, `tensors_sha256 = 6f2d0fae36e0ec832f28ac0825457a929e9eb0d4cdcc4e5b8985d552c8639e5b`).
- **Speed & memory (`metrics.json`):** `preprocessing_ms = 60,273.4 ms`, `step_mean_ms = 82.20 ms` (`median 82.22 ms`, `max 87.98 ms`), `12.06 steps/s`, `4,147,169.9 ms` (`69.12 min`) training wall with previews, `833.2 MiB` resident VRAM (`10,260.1 MiB` free), **`peak_rss_kib = 4,796,888` (`4.575 GiB` vs 8 GB desktop ceiling)**.

### Validation Curve & Held-Out Trajectory (`validation.jsonl` + `feral evaluate`)
Fixed probe normalized residual MSE fell from `1.485726` to `0.363506` (`-75.5%`). Held-out `validation_loss` (`4.0 s` window) improved monotonically at every single checkpoint across all four horizon quarters (`12,500` step transitions cleanly visible between `12k->14k` and `24k->26k`):

| Step | 0 | 2,000 | 4,000 | 10,000 | 12,000 | 14,000 | 20,000 | 24,000 | 26,000 | 30,000 | 40,000 | 50,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `validation_loss` | 1.609800 | 0.519825 | 0.446478 | 0.394886 | 0.384468 | 0.327045 | 0.308654 | 0.304022 | 0.289104 | 0.280685 | 0.275681 | **0.273870** |

Independent fixed-`2.048 s` held-out evaluation (`feral evaluate --crops 16 --seed 1`, `192` probes across all four mask tasks & times):

| Checkpoint | Step | EMA | `mean_velocity_mse` | Endpoint `log_spectral_l1` | Endpoint `mse` | Endpoint `derivative_mse` |
|---|---:|:---:|---:|---:|---:|---:|
| `checkpoint-010000` | 10,000 | Yes | 0.020840 | 0.224020 | 0.007341 | 0.002606 |
| `checkpoint-020000` | 20,000 | Yes | 0.017030 | 0.178291 | 0.006441 | 0.001904 |
| `checkpoint-030000` | 30,000 | Yes | 0.015882 | 0.162200 | 0.006186 | 0.001724 |
| `checkpoint-040000` | 40,000 | Yes | 0.015573 | 0.159047 | 0.006091 | 0.001699 |
| **`checkpoint` (final)** | **50,000** | **Yes** | **0.015429** | **0.157600** | **0.006050** | **0.001681** |
| `checkpoint --no-ema` | 50,000 | No | 0.015582 | 0.158908 | 0.006058 | 0.001688 |

Note that `10k -> 50k` held-out `ep_stft_l1` dropped `29.6%` (`0.224020 -> 0.157600`) on held-out stereo recordings in just 50k updates, without the plateau seen in `arm-o`. Direct raw numbers vs `arm-o` evaluate across different channel layouts (`2ch` stereo vs `1ch` mono) and different validation corpora (`ai-music-clean-val` vs `audio-corpus-64`).

### Weight & Saturation Health vs. 400k Mono Baseline (`analyze_runs.py` Gaussian probe)

| Run | Style | Weight Decay | Steps | `max \|w\|` | `mean w_rms` | `h_rms` (first -> last) | Mean / Max Gate Sig Sat | Mean Content Sat |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `arm-o` (3.56M mono) | `gated-film` | 0.0 | 400,000 | 7.058 | 0.1340 | `0.466 -> 0.643` | 22.26% / 36.75% | **28.89%** (max 49.28%) |
| **Stereo 50k (24.6M)** | **`glu-film`** | **0.01** | **50,000** | **0.857** | **0.0309** | **`0.543 -> 1.256`** | **14.37% / 46.51%** | **0.00%** (all 15 blocks) |

### Stereo Takes & Conditioning Diagnostics
All outputs are native `2-channel` float32 `16 kHz` WAVs generated on CUDA from `experiments/investigations/2026-11-20-stereo-scaled-50k/checkpoint.json` with `protected_samples_exact = true` and `lim_total = 0` (zero limiter hits across all generated and conditioned takes):

| Artifact | Scope Analyzed | Gen/Render Time | Peak RSS | RMS | L/R Corr | Side/Mid Ratio | Spectral Flatness | Lag-1 Corr | Band Energy (`<250 / 250-1k / 1k-4k / >=4k`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| *Held-out reference prefix* (`VextorDreamsDigital.wav`) | Full prefix (278.88 s, 4,462,080 fr) | — | — | 0.2045 | 0.8490 | 0.2859 | 0.1274 | 0.9197 | `[0.556, 0.296, 0.121, 0.026]` |
| `take-4s-32s-ema.wav` | Full 4.0 s (32 Heun, EMA) | 1,629.9 ms | 889,704 KiB | 0.1732 | 0.8017 | 0.3318 | 0.2541 | 0.8475 | `[0.510, 0.251, 0.185, 0.055]` |
| `take-4s-32s-raw.wav` | Full 4.0 s (32 Heun, `--no-ema`) | 1,572.9 ms | 889,704 KiB | 0.1825 | 0.8101 | 0.3240 | 0.2350 | 0.8579 | `[0.581, 0.204, 0.163, 0.052]` |
| `take-4s-64s-ema.wav` | Full 4.0 s (64 Heun, EMA) | 2,380.8 ms | 889,864 KiB | 0.1790 | 0.8073 | 0.3266 | 0.2490 | 0.8506 | `[0.527, 0.241, 0.178, 0.054]` |
| `take-8s-64s-ema.wav` | Full 8.0 s (64 Heun, EMA) | 3,928.4 ms | 889,748 KiB | 0.1775 | 0.8035 | 0.3301 | 0.2533 | 0.8452 | `[0.581, 0.189, 0.172, 0.057]` |
| `continue-val0-4s.wav` | **Generated suffix only** (last 4.0 s) | 2,734.2 ms | 925,616 KiB | 0.1589 | 0.8102 | 0.3238 | 0.2405 | 0.8553 | `[0.565, 0.214, 0.170, 0.051]` |
| `inpaint-val0-4to6s.wav` | **Generated hole only** (`4.0–6.0 s`) | 2,043.3 ms | 924,964 KiB | 0.1580 | 0.8400 | 0.2950 | 0.1826 | 0.8941 | `[0.587, 0.240, 0.137, 0.036]` |

Conditioned join & exact-retention checks:
- `continue-val0-4s.wav`: `protected_samples_exact = true`, `boundary_jumps = [0.0000979, 0.0002070]`, `channels = 2`.
- `inpaint-val0-4to6s.wav`: `protected_samples_exact = true`, `boundary_jumps = [0.001555, 0.021594, 0.028396, 0.008084]`, `channels = 2`.

### Scaled 24.6M `FERALCK1` Bit-Exact Resume Proof
Location: `experiments/investigations/2026-11-20-stereo-resume-proof/` (`two-steps` uninterrupted 2 steps from `checkpoint-040000.json` vs `step1` -> `step2` resumed across an intermediate `FERALCK1` save):
- **Payload verified:** `394,298,944` raw contiguous bytes = `4 × 24,643,684 × 4` (`weights + adam.m + adam.v + ema_weights`).
- **Exact match across `two-steps/checkpoint.json` and `step2/checkpoint.json`:**
  - Raw binary payload bytes (`394,298,944` B): **bit-identical**
  - `tensors_sha256`: `b038c6239b23473251c9c85a5c8cc5cf4a24622600b1da2f8453fbb1807a13d6` (**exact match**)
  - `model_hash`: `6121c9e4d45cc6d3280d02cbb2118c920a840d7a215223207b05914f569f7bde` (**exact match**)
  - `completed_steps`: `40002` (**exact match**)
  - `adam.step`: `40002` (**exact match**)
  - `rng.state`: `15545984694852522593` (**exact match**)
  - Fixed probe trajectory: `0.368401 -> 0.368398` (both paths), `output delta MSE = 0.000000`.

---

## 7. Verification Gates & Open Limitations

All checks ran on idle GPU after the final source, test, and documentation edits (`export PATH="/home/anon/micromamba/envs/llama-turbo-61/bin:$PATH"`, `rustc 1.100.0`, `cargo 1.100.0`, `nvcc sm_61`):

- `cargo fmt --check`: passed.
- `cargo clippy --locked --offline --all-targets -- -D warnings`: passed (0 warnings).
- `cargo clippy --locked --offline --all-targets --features cuda -- -D warnings`: passed (0 warnings).
- `cargo test --locked --offline --release -j 8`: passed (**50 passed, 0 failed**: `core` 33, `data_trace` 3, `parameterization` 6, `scaling` 5, `structure` 3).
- `cargo test --locked --offline --release --features cuda -j 8`: passed (**68 passed, 0 failed**: `core` 33, `cuda` 18, `data_trace` 3, `parameterization` 6, `scaling` 5, `structure` 3).
- Note: `--features opencl` stays uncompiled offline in this environment because `libc v0.2.191` is absent from the offline Cargo cache; `OpenClTrainer` explicitly rejects stereo, weight decay, and multi-horizon runs.

### Remaining Limitations
- All conclusions above rest on numerical flow losses, STFT distances, stereo correlations, and exact protected-sample assertions — **no subjective human listening test has been scored** on the `50k` 24.6M stereo takes.
- At `50,000` steps (`~69 min` on GTX 1080 Ti), validation loss (`0.2739`) and held-out `ep_stft_l1` (`0.1576`) were still monotonically decreasing; resuming `experiments/investigations/2026-11-20-stereo-scaled-50k/checkpoint.json` for a multi-hundred-thousand-step run is recommended only after listening inspection of the current takes.
