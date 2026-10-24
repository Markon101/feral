# FERAL 👻🤠

**FERAL — Freeform Evolutionary Resonant Audio Laboratory**

A local experimental instrument that learns from unlabeled audio, generates new
waveforms with a neural rectified flow, and regenerates masked regions while
retaining protected canonical PCM exactly. Rust, CPU first, no pretrained models,
no labels, no services. The first model is small and the first results are pipeline
evidence, not evidence of realistic music quality.

## Build and check

Rust and FFmpeg/FFprobe must already be installed. There are 40 locked Cargo
packages; no ML framework, Python runtime, GPU SDK, or model download is required.

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

`--noise-scale` is initial noise amplitude (0–4), not a calibrated temperature.
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
uninterrupted and resumed parameters exactly on CPU. Checkpoints are versioned, portable JSON, hash checked, and capped at 64 MB.
Run metadata includes architecture, parameter count, source/Git identity, data
hash, objective, preprocessing, optimizer and seed. `losses.jsonl` records every
step, task, loss, gradient norm, time and RNG state. Previews use a fixed seed.

Read [ARCHITECTURE.md](ARCHITECTURE.md), [LIMITATIONS.md](LIMITATIONS.md),
[EXPERIMENTS.md](EXPERIMENTS.md), [ROADMAP.md](ROADMAP.md) and
[reports/VALIDATION.md](reports/VALIDATION.md) for implementation contracts,
measured evidence and the OpenCL decision.
