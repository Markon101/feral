# FERAL validation — 2026-10-24

A real CPU flow generator was built, trained, sampled, conditioned and committed
locally. No pretrained model, system package, large resource, paid service or
remote repository was used. Other projects were read only. The owner's selected
recording was hashed before/after; it is unchanged.

## Machine and build

Android Linux 6.6.127, ARM64, Rust/Cargo 1.99.1; FFmpeg/FFprobe 9.0.2. Approximately
10.85 GiB physical RAM, 2–2.5 GiB available during inspection; owner authorizes a
3 GB working ceiling. Eight CPU entries are present; this process reports four
available logical CPUs. Training/sampling use one CPU thread. Termux may have been
in the background. No foreground, thermal, affinity or maximum-performance claim.
KGSL sysfs reports `Adreno830v2`; OpenCL queries are blocked in this process's linker
namespace (see OPENCL.md). About 22 GB storage was available before the build.
No home Cargo config or explicit Rust/Cargo target/flags/jobs override was found.

`cargo build --locked --offline --release -j 8` passes; the first release build took
26.86 seconds. `cargo fmt --check` and strict
`cargo clippy --locked --offline --all-targets -- -D warnings` pass.
`cargo test --locked --offline --release -j 8` passes **13 integration tests**
(1.20 seconds test execution on the final run). These include:

* exact float WAV round trip and overwrite refusal;
* FFmpeg WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus, WebM Opus and M4A AAC decoding,
  source integrity, literal shell-metacharacter filenames, corrupt-file skips;
* network shape and exact 57,952-parameter count; finite loss/nonzero gradients;
* finite-difference checks of weights/biases in every layer, tolerance
  `abs(error) < 5e-4 + .02*abs(numerical_gradient)`;
* actual Adam parameter changes; Euler/Heun integration direction;
* finite deterministic seeded sampling, disabled perturbation baseline parity;
* no leakage of unknown clean samples through context; all four training masks;
* exact protected samples, reduced collar jumps, actual learned continuation and
  inpainting through checkpoint reload;
* full Adam/RNG resume identical to uninterrupted training, checkpoint validation
  and overwrite refusal; actual nested-directory smoke CLI training/sampling.

Resolved issues encountered during verification: the Opus encoder fixture needed
explicit mono channel layout; the first standalone smoke invocation exposed a
missing parent-directory creation, now covered by a regression test; the Python
artifact reader needed WAVE_FORMAT_EXTENSIBLE float GUID support. Early incomplete
or unusual local artifacts remain under runs; none is represented as a success.
The Rust neural source used by all reported runs matches the current source hash.

## Actual training and phone timings

The model has 57,952 f32 parameters: 100→64 input projection; four 64-channel,
kernel-3 residual convolutions at dilations 1/2/4/8; 64→32 output projection. Lossless
32-sample waveform blocks; unknown-sample rectified-flow velocity MSE. Batch one,
Adam lr .0003, gradient norm clip 1. Default sample rate 16 kHz mono.

| Measurement | Synthetic 20 steps | Song 120 steps | Song +380 steps (500 total) |
|---|---:|---:|---:|
| Crop duration | .128 s | .512 s | .512 s |
| Mean step | 9.22 ms | 34.33 ms | 25.07 ms |
| Median step | 9.16 ms | 34.68 ms | 24.24 ms |
| Max step | 11.63 ms | 42.32 ms | 42.26 ms |
| Neural forward/backward total | 167.16 ms | 4000.03 ms | 9221.47 ms |
| Adam total | 15.71 ms | 82.73 ms | 221.06 ms |
| Decode/preprocess + source hash | 246.97 ms | 491.51 ms | 392.67 ms |
| Training wall, including previews/checkpoints | .528 s | 6.184 s | 11.000 s |
| Final sampling, 16-step Heun | 146.00 ms | 669.83 ms | 756.22 ms |
| Trainer peak resident memory | 7,564 KiB | 8,612 KiB | 8,696 KiB |
| Fixed training-distribution probe before→after | 1.442→1.331 | 1.211→.724 | .724→.158 |

Peak RSS comes from the trainer's `/proc/self/status` VmHWM; it excludes FFmpeg's
separate decoder process and compiler memory. No whole-device memory claim. Values
vary with background scheduling; the differing step times are not an optimization
speedup. The 500-step continuation took 380 additional updates, not 500 more.

Neural forward/backward is 96.8% of measured update wall time in the +380 run.
No per-layer profiler was added. Structurally, the residual k=3 convolutions account
for 85.3% of forward MACs (49,152 of 57,600 weights used per frame); this identifies
a likely acceleration target, not a measured per-operator attribution. Forward
arithmetic is approximately 14.75 million MACs per default crop, with input and
parameter backward passes adding roughly twice that. OpenCL is deferred.

Separate 32-step Heun sampling of .512 seconds took 965.0 ms on the first recorded
500-step render; a repeated exact-seed render took 1310.8 ms. One second of output
took 2774.6 ms. Continuation took 1186.7 ms; local inpainting 728.2 ms. All are CPU,
without controlled foreground or thermal state. No inference-only GPU claim.

## Source, controls and outputs

Training source: `/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav`,
310.4 seconds, stereo PCM16, 48 kHz. Only the first 12 seconds were decoded for
training, downmixed/resampled to 16 kHz and peak-normalized to .8. Random crops
are 8,192 samples. The song's SHA-256 before/after was:
`cad6b5725121914b6f0f1e932e14d6c394b68e2df9d2f472b6f0675a3e92e026`.

All training used clean commit `ef41af30cab2fd1a563bdcceba2644e7d155046b` and source
hash `6627d90947962ef22c2b7c2ded7bbbcea8490163bd5af74212d5d111aa6dfb07`.
Final checkpoint SHA-256:
`abf14d60bd3677b36b6d27d6845bbad8dd6a45c1a9c83ffb6b03ffae58fc1e08`.
Final model payload hash:
`2001a0a8baffac332872f7cdce66c8700b03322aace77003ec6b2f72c3c9bd2b`.

Paths below are relative to `/data/data/com.termux/files/home/projects/feral/`:

* `runs/smoke-v1/training/checkpoint.json` and `generated.wav`: actual synthetic
  diagnostic neural learning (four patterns), never a realistic-music claim.
* `runs/around-v1/initial.json`: untrained real-song run control.
* `runs/around-v1/untrained-or-resumed-initial.wav`: fixed-seed untrained control.
* `runs/around-v1/checkpoint.json`: 120 steps; `generated.wav`: same-noise sample.
* `runs/around-v2/checkpoint.json`: **500 total real-song training steps**.
* `runs/around-v2/generated.wav`: .512 s, fixed seed 42, 16-step Heun.
* `runs/around-v2/seed42.wav`: .512 s, seed 42, 32-step Heun.
* `runs/around-v2/seed42-repeat.wav`: byte-identical repeat, checked with `cmp`.
* `runs/around-v2/one-second.wav`: 1 s, seed 42, 32-step Heun; longer than crops.
* `runs/around-v1/reference-12s.wav`: unnormalized canonical conditioning crop.
* `runs/around-v2/continued.wav`: 12.512 s, original canonical prefix plus .512 s
  of learned suffix. The prefix is byte-for-byte exact in float sample payload.
* `runs/around-v2/inpainted.wav`: 12 s, interval [1,1.25) regenerated; all samples
  outside it remain byte-for-byte exact and the interval actually changed.
* `runs/around-v2/variation.wav`: experimental disconnected-anchor mask variation.
* `runs/around-v2/strange.wav`: seed 73, Euler 48 steps, noise scale 1.4, perturb .08.

Each render has an adjacent JSON with input/output/checkpoint hashes, configuration,
seed, solver, timing and preservation metrics. Runs retain initial/final/interim
checkpoints, fixed-seed previews, full raw `losses.jsonl`, dataset manifest/hash and
optimizer/RNG metadata. No initialized control or recording was overwritten.

`ffprobe` verifies generated WAVs as 16 kHz mono `pcm_f32le`, with correct durations.
`scripts/verify_audio.py` independently checks RIFF payloads and preservation.
The same-noise MSE change after 120 updates was .09333 and from 120→500 was .17143.
These establish an observable neural training effect. They do not establish useful
music or generalization. The final .512 s baseline has RMS .30428 and 10 of 8,192
samples at the .98 limiter, versus 1,782 after 120 steps. Internal adjacent sample
differences still reach 1.6493: rough/noisy audio and artifacts remain possible.
The OOD sample has 431 limiter hits; this unusual result is retained, not filtered.
No subjective listening judgment is claimed.

## Continuity evidence

Same checkpoint, seed and solver, with versus without the 128-sample unknown-side
collar (known samples exact in both):

| Boundary | No collar absolute jump | Collar absolute jump |
|---|---:|---:|
| Real-song prefix → new suffix | .229791 | .016825 |
| Inpaint left | .993264 | .021747 |
| Inpaint right | .118937 | .005871 |

At the continuation join, the change of the first difference is exactly zero after
the collar in the stored float output. This is evidence of local sample/slope
continuity. It is not evidence of inaudibility, generated phase coherence, stable
timbre, or long-range musical continuity. Collar-disabled results are also retained.

## Reproduce the recorded real runs

```sh
./target/release/feral smoke --run runs/new-smoke --steps 20
./target/release/feral train \
  --input '/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav' \
  --run runs/new-around-v1 --steps 120 --max-file-seconds 12 \
  --preview-every 40 --checkpoint-every 60
./target/release/feral train \
  --input '/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav' \
  --run runs/new-around-v2 --steps 380 --max-file-seconds 12 \
  --preview-every 100 --checkpoint-every 100 \
  --resume runs/new-around-v1/checkpoint.json
./target/release/feral generate --checkpoint runs/new-around-v2/checkpoint.json \
  --seconds 1 --seed 42 --output runs/new-around-v2/one-second.wav
./target/release/feral decode \
  --input '/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav' \
  --output runs/new-around-v2/reference.wav --max-file-seconds 12 --no-normalize
./target/release/feral continue --checkpoint runs/new-around-v2/checkpoint.json \
  --input runs/new-around-v2/reference.wav --seconds .512 \
  --output runs/new-around-v2/continued.wav
./target/release/feral inpaint --checkpoint runs/new-around-v2/checkpoint.json \
  --input runs/new-around-v2/reference.wav --start 1 --end 1.25 \
  --output runs/new-around-v2/inpainted.wav
python scripts/verify_audio.py runs/new-around-v2/one-second.wav \
  --reference runs/new-around-v2/reference.wav \
  --continued runs/new-around-v2/continued.wav \
  --inpainted runs/new-around-v2/inpainted.wav --start 1 --end 1.25
```

Verified CLI operations: doctor, corpus (through actual deterministic corpus
creation), decode, train, full resume, generate, continue, inpaint, variation,
smoke, command help; Euler, Heun, noise scaling and bounded perturbation. Full resume and an explicit one-step weights-only warm start on a new synthetic
manifest were both exercised. The warm start retained the initial model weights,
reset Adam/RNG, changed dataset, and completed step 1. See LIMITATIONS.md and ROADMAP.md for what remains.

Phase 2 follow-up: the historical results above remain unchanged. The current
19-test suite, pooled network, held-out experiments, negative controls and transient
continuity fix are documented in [PHASE2.md](PHASE2.md).

Phase 3: the current 25-test suite, versioned scale-aware flow, replicated comparisons
and level-matched audio/control export are documented in [PHASE3.md](PHASE3.md).
