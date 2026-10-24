# Complete OpenCL training and 100,000-update continuation

2026-10-24. Full neural FP32 GPU training is implemented and verified on this
phone's Adreno 830. The calibrated 3,000-step checkpoint was continued to 100,000
updates without changing its audio representation, model, data, masks, prior,
optimizer settings or RNG stream. Held-out endpoint error improved; coherent music
and generalization to other recordings are not established. Listen to the exports.

## Backend and prior-project inspection

The optional `opencl` Cargo feature adds 12 cached packages (60 locked in total).
No models, datasets, drivers or system packages were downloaded or installed.
The existing CPU path remains complete and default. `train --backend opencl`
explicitly selects actual GPU discovery; failure is an error, never CPU fallback.
The network still has 57,952 parameters: block32, width64, input100->64 tanh,
four residual k3 64->64 branches with pooling1/4/16/64 and dilation1, output64->32.
The representation is phase-preserving f32 waveform packing; the objective is
masked preconditioned rectified-flow residual MSE, uniform t, data scale .1 and
training/sampling prior std .1. Adam lr .0003, clip1, 8,192 samples, 16kHz mono.

Read-only inspection covered Titan Image's `src/opencl.rs` and OpenCL notes, and
RoachAudioMasterer's OpenCL host code, kernels and docs. Useful patterns were
persistent packed weights/tapes, explicit queue/event ownership, reductions and
CPU parity gates. Their renderer/inference code does not implement FERAL's
convolution training; there are no cross-repository dependencies or copied model
architectures. Existing source recordings and repositories were not changed.

The standard ICD's stale vendor path/linker namespace failed in the earlier audit.
The existing `/vendor/lib64/libOpenCL.so` works with process-local vendor/system
library paths and `/system/lib64/libc++.so` preloading. The launcher preserves the
original FFmpeg child environment: otherwise vendor `libbase` breaks decoding.
Device: Qualcomm Adreno 830, 12 CUs, OpenCL3.0, driver0800.64.8, compiler
E031.47.18.56, local32KiB, maxworkgroup1024. Kernels use CL1.2 FP32 without relaxed
math. A source/options/device/driver keyed binary cache is local and ignored.

All convolutions, pooling, tanh/residuals, reverse-mode input/weight/bias
derivatives, masked loss, norm clipping and Adam execute on GPU. Parameters,
moments, gradients and tapes remain resident. Per-step transfers are crop, noise,
mask and time features plus four status scalars. CPU still constructs random
inputs and handles audio, previews, diagnostics and inference. Checkpoints sync
full weights/moments and use the existing portable JSON schema. Checkpoint origin
fields retain their original lineage; each chunk's run.json and the supervisor
plan record the actual training commit/source hash. No inference acceleration or
optimal hardware utilization is claimed.

## Correctness and reproducibility

Formatting and strict Clippy pass for CPU and OpenCL builds. All 25 reference tests
pass, including analytic/numerical derivatives, updates, formats, seeded sampling,
protected regions and CPU full resume. Two explicitly run hardware tests pass:
actual GPU six-step training versus 3+3 full-state resume, and a deliberately
overflowing Adam state returning an error before writing a checkpoint.

Each 11-case GPU audit checks both architectures, three flow parameterizations,
endpoint/interior times, dilation, partial pooling and active clipping. Maximum
absolute forward/gradient/Adam-parameter errors were 5.07e-7 / 2.98e-7 / 3.62e-7.
Declared tolerances: forward and gradients 1e-5+1e-3 relative; parameters
5e-6+1e-4 relative. Multi-step updates, exact GPU save/reload/next update, and CPU
continuation of a GPU checkpoint all pass. The final 100k checkpoint also passes.
A post-run error-path hardening removes a nonatomic read of the atomic overflow
flag; normal finite arithmetic is unchanged. Cache size is checked before reading.

Every new step3001..100000 was verified present, ordered and finite. Source and
initial checkpoint hashes are unchanged. All chunks use one model-code source
hash and the same dataset/configuration. Frozen controls and earlier checkpoints,
new raw losses/RNG traces, commands, snapshots and failures remain local under
runs/. Compact measured evidence is committed in OPENCL_RESULTS.json.

## Phone timings and memory

Owner confirmed foreground, phone initially warm; affinity0-7 was visible in the
audits. Thirty alternating paired CPU/GPU updates after two warmups averaged
4.744ms CPU and 1.807ms GPU: observed2.63x, including GPU enqueues/transfers/status
but excluding preprocessing/checkpoint I/O. Not thermally controlled peak speed.
The post-run audit observed4.68x under hotter/different CPU conditions; retain it
for correctness evidence, not a stronger acceleration claim.

The 97,000-update continuation took269.41s across49 fresh chunks, including
startup/decoding, previews and checkpoints, no cooling waits. Mean runtime step
including crop/noise construction was2.183ms; backend step1.694ms. GPU event totals:
76.63s neural/reduction work and2.23s Adam; audio preprocessing2.95s. The remaining
wall time includes host dispatch, waits, random inputs, logging, sampling and I/O.
Peak reported host RSS31,216KiB; explicit GPU buffers2,241,200bytes. These are
separate measurements, not complete driver memory accounting. Device buffer
allocation is capped256MiB; no OOM occurred and the3GB ceiling was not approached.

The supervisor samples GPU/battery sensors before/after2,000-step chunks. Highest
observed readings during training were51.1C GPU and31.8C battery. Pause/resume
settings60/55C GPU and41/39C battery are operator thresholds, not manufacturer
ratings; they do not bound unobserved in-chunk peaks. No clocks or thermal settings
changed. Training is finished; no training job is left running.

## Learning evidence and limitations

Fixed four crops x four masks x three matched physical corruption levels. The
training recording section is[0,24)s, validation[216,224), test[232,240). These
are one song, one model lineage, correlated probes, not independent model seeds
or evidence of cross-recording generalization. The test section was already used
in Phase3; it is a regression comparison, not a newly untouched confirmatory set.

| Split | Initialized endpoint MSE | 3,000 updates | 100,000 updates | Relative reduction vs3k |
|---|---:|---:|---:|---:|
| Train | .007476 | .004676 | .003558 | 23.9% |
| Validation | .021885 | .014533 | .008504 | 41.5% |
| Test | .028557 | .019214 | .010893 | 43.3% |

All three corruption levels improve on validation/test. These are one-step flow
endpoint reconstruction metrics, not an ODE-generated music score. Raw rows keep
masks, crops, corruption levels, spectral errors and silence/noisy controls.

Matched seed42/43 ten-second free generation: RMS.08517->.08642, 4–8kHz energy
fraction.13021->.03344, flatness.45356->.18017. No limiter hits. The training-fitted
non-neural AR1 control has high-band fraction.02805 and flatness.15934, so darker
spectra alone do not demonstrate superior acoustic structure. The new neural
output overconcentrates250–1000Hz energy(.7443 versus source.5516), with too little
<250Hz energy(.0717 versus.3123). Narrowband collapse/colored-noise explanations
remain plausible. No perceptual claim is made. The bounded10s copying screen gives
.0229 correlation versus.0247 for3k and1.0 for a real copy; it misses transformed,
short or unaligned memorization.

Continuation and inpainting preserve known canonical float samples and exported
PCM16 regions exactly. Collar/no-collar jumps: continuation.02284/.12061;
inpaint left.02604/.02038, right.13320/.08902. The collar matches the neighboring
source slope and can increase jump magnitude on transients. Right slope mismatch
is about4e-9; this does not establish click-free or phase/timbral/musical continuity.

## Artifacts and reproduction

* Final checkpoint: `runs/opencl-long100k-v1/final-checkpoint.json`
  SHA256 `4c1244e926301cfd6b3612e6f56deb5b4713865ee0f89d0b3b66727ffff40a6f`.
* Training plan, all49 chunks, verification and summary: `runs/opencl-long100k-v1/`.
* Both GPU audits: `runs/opencl-audit-final/`, `runs/opencl-audit-100k/`.
* Raw held-out/descriptor rows: `runs/opencl-100k-eval-v1/`.
* Seed42 float renders/commands: `runs/opencl-100k-listening-v1/`;
  verified PCM16 pack `/sdcard/FERAL_OpenCL_100k_20261024/`.
* Fresh seed99 take requested afterward: `runs/opencl-100k-listening-take2-v1/`;
  hash-verified PCM16 pack `/sdcard/FERAL_OpenCL_100k_Take2_20261024/`
  (33 files, 17 ten-second WAVs; all canonical preservation checks pass).

Each main pack contains10s reference, free generation, OOD control, weird,
continuation, inpainting and variation, with baseline/initialized/Gaussian/AR1
comparisons at native and common RMS. All generated clips use learned weights;
non-neural controls are explicitly labeled. Free generation is5s from noise plus
5s learned continuation, not a10s learned context. Audio/RNG checkpoint lineage
is retained. Float originals stay in the repository; PCM16 is for easy playback.

```sh
cargo build --locked --offline --release --features opencl --examples -j 8
scripts/feral-opencl.sh gpu-audit --checkpoint runs/opencl-long100k-v1/final-checkpoint.json \
  --output runs/new-audit/report.json --steps 30
scripts/feral-opencl.sh --exec cargo test --locked --offline --release \
  --features opencl -j 8 --test opencl -- --ignored --test-threads=1
python scripts/opencl_long.py --checkpoint runs/phase3-calibrated-v1/calibrated-42/checkpoint.json \
  --run runs/new-long --total-steps 100000 --chunk-steps 2000
python scripts/listening_pack.py --checkpoint runs/opencl-long100k-v1/final-checkpoint.json \
  --baseline runs/phase3-calibrated-v1/calibrated-42/checkpoint.json \
  --initialized runs/phase3-calibrated-v1/calibrated-42/initial.json \
  --reference '/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav' \
  --training-reference runs/phase2-v1/train.wav --seed 99 --run runs/new-listening
```

Next: (1) paired phase-preserving multiresolution endpoint-objective experiment,
checking diversity, noise/linear-filter controls and genuinely new held-out audio;
(2) expand diverse training windows through indexed decoding without consuming
held-out sections; (3) profile host/dispatch overhead and test optional recorded
queues with CPU/gradient/resume parity before claiming another speedup. The device
advertises `cl_qcom_recordable_queues`; this optimization is deferred. Qualcomm's
[OpenCL optimization guide](https://developer.qualcomm.com/qfile/33472/80-nb295-11_c.pdf)
describes recorded dispatch. FP16, codecs, recurrence, stereo and long musical
structure remain future work.
