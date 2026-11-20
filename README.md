# FERAL 👻🤠

**FERAL — Freeform Evolutionary Resonant Audio Laboratory**

A local experimental instrument that learns from unlabeled audio, generates new
waveforms with a neural rectified flow, and regenerates masked regions while
retaining protected canonical PCM exactly. Rust, CPU first, no pretrained models,
no labels, no services. The first model is small and the first results are pipeline
evidence, not evidence of realistic music quality.

## Build and check

Rust and FFmpeg/FFprobe must already be installed. There are 60 locked Cargo
packages including optional OpenCL bindings; CPU builds need no ML framework,
Python runtime, GPU SDK, or model download.

```sh
cargo build --locked --offline --release -j 8
cargo test --locked --offline --release -j 8
./target/release/feral doctor
./target/release/feral --help
```

If the dependencies are not cached, `cargo build --locked --release -j 8` fetches
small Rust dependencies. If tools are missing on Termux, the owner can install
them with `pkg install rust ffmpeg`; FERAL does not install them. The FFmpeg
adapter prepends Termux's existing `libc++_shared.so` to `LD_PRELOAD` to address a
verified libplacebo symbol mismatch. Subprocesses use argument arrays, never a
shell. Other platforms invoke the installed FFmpeg normally.

## First sound

All commands below run from the repository root. Run/output names must be new.
Audio and checkpoints remain local and are ignored by Git.

```sh
./target/release/feral smoke --run runs/smoke --steps 20
# runs/smoke/training/generated.wav and checkpoint.json

# One real file; approximately 0.512 s crops from its first 30 seconds.
./target/release/feral train --input /path/to/recording.flac \
  --profile phone --steps 100 --run runs/first
./target/release/feral generate --checkpoint runs/first/checkpoint.json \
  --seconds 0.512 --seed 42 --output runs/first/new.wav
```

A directory selects supported extensions nonrecursively (1–128 files). A `.txt`
manifest selects files with one path per line, relative to the manifest directory;
blank lines and `#` comments are ignored. A single input file can use any format
FFmpeg recognizes. WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus, WebM Opus and M4A AAC are
covered by actual decode tests with this installation. Decoder failures are
reported and skipped for training; no usable files is an error.

```sh
./target/release/feral decode --input /path/to/music.opus \
  --output runs/decoded.wav --max-file-seconds 10
./target/release/feral train --input /path/to/collection.txt \
  --run runs/collection --steps 100 --crop-seconds 0.512 \
  --max-file-seconds 30 --sample-rate 16000 --channel mix
```

Canonical PCM is finite mono f32, normally 16 kHz. `--channel mix|left|right`
controls stereo ingestion; stereo generation is not implemented. Training
normalizes each selected file to peak 0.8 unless `--no-normalize` is given.
`--silence-rms` rejects silent files and retries silent crops. Each file is decoded
once into RAM; the total decoded dataset is capped at 128 MB. Source files are
never altered or copied into the run. Hashes, absolute source paths, preprocessing
and decode rejections are recorded.

## Transform and extend

```sh
./target/release/feral continue --checkpoint runs/first/checkpoint.json \
  --input /path/to/reference.opus --seconds 0.512 \
  --output runs/first/continued.wav
./target/release/feral inpaint --checkpoint runs/first/checkpoint.json \
  --input /path/to/reference.mp3 --start 1.0 --end 1.25 \
  --output runs/first/inpainted.wav
./target/release/feral variation --checkpoint runs/first/checkpoint.json \
  --input runs/first/new.wav --output runs/first/variation.wav
```

Training mixes unconditional, prefix, interior-hole and suffix tasks equally in
expectation. These commands perform learned masked flow integration. Known
samples are projected at every solver step. Final WAVs preserve protected samples
exactly **after FFmpeg decoding/resampling/downmixing**, not the original compressed
file's encoding. Conditioning disables gain normalization. The float WAV writer
retains these samples bit-for-bit. Continuation includes the existing decoded
prefix plus the newly generated suffix. Repeated calls provide short context for
subsequent segments. Inpainting leaves all canonical samples outside the interval
untouched. Conditioning probes source duration and rejects inputs beyond
`--max-input-seconds` (default and maximum 600 seconds), instead of truncating
known source tails. Use an explicitly decoded crop for longer recordings. Context is local; these commands do not promise coherent music.

A 128-sample raised-cosine collar changes only generated samples near boundaries.
It reduces sample jumps and records measured boundary jumps in the output JSON;
it does not establish phase, timbral or long-range musical continuity. Disable it
with `--collar 0`. Variation uses repeated short protected anchors and is
experimental: that disconnected mask pattern is not separately trained.

## Explore, inspect, resume

```sh
./target/release/feral generate --checkpoint runs/first/checkpoint.json \
  --seed 73 --solver euler --flow-steps 48 --noise-scale 1.4 \
  --perturb 0.08 --output runs/first/strange.wav
./target/release/feral train --input /path/to/recording.flac \
  --resume runs/first/checkpoint.json --steps 50 --run runs/resumed
```

`--backend` selects where the velocity field is evaluated: `cpu` (the default
reference) or `cuda`, which runs the same network and the same solver on the
device. The two agree to fp32 summation order — measured max |Δ| 1.8e-7–3.0e-7,
correlation 1.00000000, identical protected samples and identical boundary
jumps — while a 4 s take at 32 Heun steps drops from 144.6 s to 0.58 s.

`--noise-scale` multiplies the checkpoint prior amplitude (0–4); it is not a
calibrated temperature. Legacy checkpoints use prior std 1.
`--perturb` adds a seeded bounded velocity term with a sine time envelope (0–0.5).
Zero perturbation restores the baseline exactly. Euler and Heun both integrate
from noise at t=0 to audio at t=1. Trajectories reject nonfinite or extreme values;
new audio has an explicit hard safety limiter at 0.98. No file is overwritten.
Output JSON records checkpoint/input/output hashes, seed, solver, timings, masks'
boundary metrics and preservation checks. Unusual outputs remain beside their
metadata instead of being silently discarded. Generation is bounded to 5 seconds
and 160000 samples, solver steps to 512. Longer windows are outside training's
scale and may be poor even if finite.

Full resume restores weights, Adam moments/step and the training RNG, requires the
same dataset/preprocessing/crop/learning rate, and writes a fresh run. `--weights-only`
explicitly resets optimizer/RNG and permits another dataset. `--steps` always means
additional steps. Full resume restores the original RNG stream, so `--seed`
only selects a new stream for fresh or weights-only training. Tests compare
uninterrupted and resumed parameters exactly on CPU. Checkpoints are versioned, portable JSON, hash checked, written compact (a
1.215M-parameter checkpoint is 49.6 MB; artifacts such as metrics.json stay readable), and capped at 192 MB (enough for
the largest architecture the CLI allows; the cap is a guard against untrusted files, not a model limit).
Run metadata includes architecture, parameter count, source/Git identity, data
hash, objective, preprocessing, optimizer and seed. `losses.jsonl` records every
step, task, loss, gradient norm, time and RNG state. Previews use a fixed seed.

Read [ARCHITECTURE.md](ARCHITECTURE.md), [LIMITATIONS.md](LIMITATIONS.md),
[EXPERIMENTS.md](EXPERIMENTS.md), [ROADMAP.md](ROADMAP.md) and
[reports/VALIDATION.md](reports/VALIDATION.md) for implementation contracts,
measured evidence and the OpenCL decision.

## Phase 2: longer context and held-out diagnostics

The original network remains the default. `--network pyramid` selects an optional
57,952-parameter pooled residual CNN with a 468 ms structural context span instead
of 62 ms. The waveform representation, flow loss and mask tasks stay the same.
Pooling is differentiable; the full-resolution skip path retains sample detail.
Old checkpoints still load, and an old unconditional render was reproduced exactly.

```sh
./target/release/feral train --input recording.wav --run runs/pyramid \
  --network pyramid --steps 600 --no-normalize
./target/release/feral evaluate --checkpoint runs/pyramid/checkpoint.json \
  --input heldout.wav --output runs/pyramid/heldout.json
./target/release/feral analyze --input runs/pyramid/generated.wav \
  --reference recording.wav --output runs/pyramid/signal.json
```

`evaluate` records fixed crops, all four mask tasks at three times, waveform and
multiresolution spectral endpoint diagnostics, and zero-audio/noisy-input controls.
`analyze` records signal descriptors, shuffled/noise controls and a bounded aligned
copying screen. Both use unnormalized 16 kHz mono audio (evaluation uses checkpoint
sample rate), capped at the first 60 seconds; explicitly split longer inputs.
These are numerical checks, not listening scores. Do not train on held-out files.

The three-seed test found the pyramid **worse at equal updates**. At twice the
updates and similar observed CPU update time it modestly improved fresh held-out
denoising, while generated spectra remained overly noise-like. It stays opt-in.
The continuity collar now retains feasible source slopes on transients, with
headroom-aware decay; protected canonical samples remain exact. Amplitude limits
can make exact slope matching impossible. See [Phase 2 results](reports/PHASE2.md)
and the frozen [protocol](reports/PHASE2_PROTOCOL.md).

## Phase 3: scale-aware flow

Optional preconditioning and a checkpointed prior are implemented and tested:

```sh
./target/release/feral train --input runs/phase2-v1/train.wav \
  --run runs/preconditioned --network pyramid --parameterization preconditioned \
  --data-scale 0.1 --prior-std 0.1 --no-normalize --steps 3000
./target/release/feral generate --checkpoint runs/preconditioned/checkpoint.json \
  --seconds 1 --output runs/preconditioned/generated-new.wav
```

These scales were chosen from the current training section (RMS .1068); calibrate
only from training audio for other datasets. Generation automatically uses the
saved prior. `--noise-scale 1.4` means 1.4 times that prior (.14 for prior std .1).
Resume requires the same parameterization/data scale/prior flags. Switching them
requires fresh training; weights are never silently reinterpreted.

V1 legacy and v2 unit-prior checkpoints still load. Non-unit priors use v3, rejected
by older binaries. `evaluate` follows the checkpoint path; across priors compare
common-corruption endpoint metrics via `examples/phase3_snr_eval.rs`, not fixed-time
velocity losses. Lower regression loss and darker spectra are separate outcomes.
See [PHASE3.md](reports/PHASE3.md) for the three-seed results, failed/plateaued
follow-up, and level-matched listening pack at `/sdcard/FERAL_Phase3_20261024/`.
The raw-velocity default remains intact.

## Optional resident OpenCL training

The CPU remains the default numerical reference. GPU training is explicit:

```sh
cargo build --locked --offline --release --features opencl -j 8
scripts/feral-opencl.sh gpu-audit --checkpoint runs/preconditioned/checkpoint.json \
  --output runs/new-gpu-audit/report.json --steps 30
scripts/feral-opencl.sh train --backend opencl --input recording.wav \
  --network pyramid --parameterization preconditioned --data-scale 0.1 \
  --prior-std 0.1 --no-normalize --steps 2000 --run runs/new-gpu
```

The Android launcher selects an existing vendor runtime without installation or
system edits. Other platforms can run the feature-enabled binary with their
normal OpenCL runtime. Forward/backward, clipping and Adam execute on GPU with
resident buffers; decoding, previews and generation remain CPU work. Discovery
failure is an error, never silent CPU fallback. `gpu-audit` requires a checkpoint
whose dataset is still available. Hardware tests are explicit and ignored in
ordinary test runs. Read the [protocol](reports/OPENCL_TRAINING_PROTOCOL.md) for
parity gates, measured warm-phone timings and the checkpointed cooling supervisor.

The verified longer run reaches **100,000 total updates** from the calibrated3k
checkpoint. Same-song held-out endpoint MSE improved43.3%; this is not a musical
quality score. Ten-second listening packs are at
`/sdcard/FERAL_OpenCL_100k_20261024/` (seed42) and
`/sdcard/FERAL_OpenCL_100k_Take2_20261024/` (fresh seed99). Compare
`level_matched/current.wav` with `baseline.wav` and the labeled noise controls.
The checkpoint is `runs/opencl-long100k-v1/final-checkpoint.json`.
See [full results and reproduction](reports/OPENCL_TRAINING.md).

## Optional resident CUDA training (Pascal desktops)

`--features cuda` adds a driver-API backend for hardware that has no vendor OpenCL runtime, built
ahead of time by `build.rs`. CUDA 13 dropped sm_61, so the build requires a CUDA 11.x/12.x `nvcc`
(override with `FERAL_NVCC`; arch with `FERAL_CUDA_ARCH`). No crate is added and nothing is
downloaded.

```sh
cargo build --locked --offline --release --features cuda -j 8
feral gpu-audit --backend cuda --checkpoint runs/<run>/checkpoint.json \
  --output runs/<new-audit>/report.json --steps 30
feral train --backend cuda --batch 4 --width 256 --blocks 5 \
  --input corpus/train --validation corpus/validation \
  --steps 6000 --run runs/<new-run>
```

Forward/backward, clipping and Adam stay resident on the device in fp32 with `fmad` disabled;
`--batch` and `--validation` are backend-neutral runtime features. Discovery, kernel-load or
launch failure is an error, never silent CPU fallback. `--blocks` selects 1..10 power-of-two
dilations for the baseline stack (5 at width 256 is 1,018,400 parameters).

`--block` sets samples per frame (16/32/64/128) and `--crop-seconds` accepts up to 32; together they
decide the receptive field as `block * (1 + 2 * sum(dilations))`. The default 32 with four blocks is
0.254 s at 16 kHz; `--block 128 --blocks 8` is 4.09 s and `--block 128 --blocks 10` is 16.4 s. Larger
frames are cheaper per sample (convolution cost scales with frames), which is what makes a wider,
longer-context model affordable.

`--detail-weight <w>` adds an optional detail term to the objective:
`mean(r^2) + w * mean((r[i+1] - r[i])^2)` over unknown samples, where `r` is the residual. It is a
3-tap stencil rather than a transform, so it weights detail without an FFT; `0` (the default)
reproduces the plain objective bit for bit. It is checkpointed, must be repeated on a full resume,
and is implemented on the CPU and CUDA backends.

`--spectral-weight <w>` adds a non-quadratic multiresolution Hann STFT log-magnitude smooth-L1
loss (`FFT sizes 128/512/2048`, `hop = N/4`) on the predicted clean endpoint
`y_hat = x_t + (1 - t) * v_pred`, weighted by `w * t` with strictly zero gradient on protected
samples. CPU uses `rustfft`; CUDA executes a resident radix-2 shared-memory STFT kernel with zero
PCIe sync and deterministic 4-slot overlap accumulation. `0` (default) disables it with zero
overhead.

Training schedule, time sampling and EMA controls (all opt-in, checkpointed, and deterministic on
resume; defaults preserve v1–v3 behavior bit for bit):
- `--time-sampling uniform|logit-normal` with `--time-mean` (default `0.0`) and `--time-std`
  (default `1.0`) concentrates flow-time updates around mid-trajectory `t` where structure forms.
  With `--parameterization preconditioned`, pair `--prior-std` with `--data-scale` so `0 dB` SNR
  aligns with `t = 0.5`.
- `--lr-schedule constant|cosine` with `--warmup-steps`, `--min-lr` (default `1e-5`) and optional
  `--schedule-steps` (defaults to `--steps`) applies linear warmup and cosine decay to Adam.
- `--ema-decay <d>` (e.g. `0.999`; `0` disables) maintains checkpointed exponential-moving-average
  weights on CPU and CUDA. Training previews, validation, `generate`/`continue`/`inpaint`/`variation`
  and `evaluate` automatically use EMA weights when present; pass `--no-ema` to inspect raw weights.

Architecture options (opt-in; CPU and CUDA):
- `--dilation-pattern ascending|symmetric|cycles` controls the residual dilation schedule for
  `--network baseline`: `ascending` (`1..2^(b-1)`, legacy default), `symmetric`
  (`1..2^(b-1)..1`, U-Net coarse-to-fine refinement after global context), or `cycles` (two full
  `1..2^(b-1)` passes).
- `--block-style tanh|gated-film` selects either the legacy `h + 0.5 * tanh(Conv(h))` block or
  pre-RMSNorm + per-block 32-dim sinusoidal FiLM time conditioning + WaveNet gated activation
  `h + 0.5 * (tanh(a) * sigmoid(b))`.
- `--sample-conv <K>` (`0` disables; odd `3..=15` enables) wraps the `block`-sample frame stack with
  two-layer residual sample-domain 1D convolutions (16 hidden channels) before packing and after
  unpacking, bridging frame boundaries with shift-equivariant sample context while preserving exact
  protected-sample masking.

Long-context configurations are checked the same way as any other: `--block 128` changes the feature
layout and exercises 128-frame dilations, so `tests/cuda.rs::large_frame_and_wide_dilation_parity`
pins loss, gradient and prediction parity there, and `gpu-audit` passes on a checkpoint built at the
new shapes.

Sampling and evaluation run on the device as well: `generate`, `continue`, `inpaint`,
`variation` and `evaluate` accept `--backend cuda`, and training previews and validation probes
follow the training backend. On the 1,215,264-parameter run a
4 s take at 32 Heun steps drops from 144.6 s to 0.30 s with max |delta| 2.4e-7 against the CPU
reference, identical protected samples and the same WAV bytes as the step-based device path.
Sampling uses a forward-only device entry, so a field evaluation costs 2.9 ms rather than 7.6 ms.
See [reports/CUDA.md](reports/CUDA.md) for the parity gates and the measured 1,018,400-parameter
run (GTX 1080 Ti: 51.1 steps/s at batch 4, held-out validation 1.1958 -> 0.0439).

## Controlled scaling experiments

`train --width 128` selects 214,176 parameters (the default 64 stays 57,952). CLI
widths 32..512 in multiples of 16 are supported; resume must repeat the checkpoint width.
This is a capacity control, not a claim that a bigger model makes music.

`scripts/prepare_scaling_data.py` creates sample-exact audio-window manifests with
source hashes and holdout guards; `scripts/level_match_scaling_data.py` optionally
creates separately recorded constant-gain copies. Original recordings/raw crops
remain unchanged. Training logs now identify each crop's source, offset and silence
retry count, with the original RNG stream retained. The[scaling protocol](reports/SCALING_PROTOCOL.md)
keeps data and width as separate axes; [data contracts](reports/SCALING_DATA_CONTRACT.md)
describe provenance and limits.

The `structure_eval` example compares10ms envelopes with seeded phase-randomized
surrogates preserving the whole-clip spectrum. These diagnostics test organization
beyond a spectral envelope; they are not music/quality scores. The owner still
hears broadband noise from the100k model. Stereo remains deferred.
