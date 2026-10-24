# Explicit audio-window preparation

`scripts/prepare_scaling_data.py` prepares one explicitly selected source into
new, provenance-recorded split files. It uses installed FFmpeg/FFprobe with argv
arrays, no shell, package installation, labels, normalization or source writes.

```sh
python scripts/prepare_scaling_data.py \
  --source '/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav' \
  --output runs/scaling-data-v1 \
  --train-window 0:104 --train-window 244:304 \
  --validation-window 216:224 --test-window 232:240
```

This plans 164 seconds of training audio, 8 seconds validation and 8 seconds test.
All six historical evaluation windows, 112–120, 136–144, 160–168, 184–192,
216–224 and 232–240 seconds, are always excluded from training with a two-second
guard by default. Additional `--exclude-window START:END` values accumulate.
Training/evaluation windows may not overlap, and windows outside source duration
are errors. Half-open boundaries and guard checks use integer 16 kHz sample indices.
This protects evaluation sections rather than treating other sections from the
same recording as proof of cross-recording generalization.

Each selected window is subdivided into files of at most 20 seconds by default;
`--chunk-seconds` is capped at 30 seconds so the existing loader's default
30-second decode limit does not silently discard expanded audio. `manifest.txt`
contains training crops; `validation.txt` and `test.txt` remain separate. Their
paths are relative to their manifest directory. Pass `--no-normalize` when training
on this manifest to retain the existing calibrated scale. Expanding the dataset
requires an explicit weights-only warm start or fresh run, never full optimizer
resume under a changed dataset identity. The original frozen 24-second training
file is untouched and remains the paired small-data control.

FFmpeg resamples and trims the original stream at canonical sample indices; it
does not seek approximately to codec keyframes. Each output is verified as finite
mono 16 kHz IEEE f32 WAV with exactly the requested sample count. `provenance.json`
records absolute original path, original SHA-256 before/after, probe/decode argv,
source-relative sample intervals and seconds, preprocessing, split manifest
hashes, crop hashes, RMS and peak. No original recording is copied into the run.
The destination directory must not exist, FFmpeg uses `-n`, and an interrupted
attempt leaves its plan and partial artifacts for inspection. Only a completed
attempt publishes split manifests and final provenance.

Bounded limits: 128 total prepared files, 128 MB total canonical PCM, 30 seconds
per output file, 120 seconds wall timeout per decoder invocation. Crops are decoded
separately, trading redundant source decode work for bounded RAM and clear
provenance. This helper currently accepts one source per invocation. Multiple
recordings need separate explicit preparations and a provenance-preserving combined
manifest; automatic source discovery and combined-corpus orchestration are deferred.
Historical exclusions conservatively apply even if another source is selected.

Verification:

```sh
python -m unittest discover -s tests -p test_prepare_scaling_data.py -v
```

Four tests passed on the current device, including real installed FFmpeg decode,
resampling, exact sample counts, source hashes, literal shell-metacharacter paths,
overwrite refusal, guarded splits, invalid windows and memory bounds. The tests
use temporary synthetic audio; no original song was decoded by the test run.

## Optional constant RMS gain control

The raw preparation remains unchanged. To isolate expanded content from the
amplitude/SNR confound, derive a new directory with a separately recorded scalar
gain per crop:

```sh
python scripts/level_match_scaling_data.py \
  --input runs/scaling-data-v1 --output runs/scaling-data-rms-v1 \
  --target-rms 0.1067798255
```

The target above is predetermined from the frozen 24-second training baseline,
not validation/test audio. Each gain is target RMS divided by that raw crop's RMS.
Every output sample is exactly the f32 rounding of `raw_sample * gain`; sample
count, source intervals and waveform shape are retained. There is no clipping,
compression, EQ or amplitude safety limiter. IEEE-float peaks may consequently
exceed one and are reported rather than silently limited. Silence, nonfinite gain
results and invalid targets are rejected. This is an explicit experimental data
transformation, not the training loader's peak normalization: use `--no-normalize`.

Raw crops remain in the original directory, unchanged. The derived provenance
records their absolute paths, hashes, RMS and peaks, scalar gains, requested target,
resulting RMS/peak/hash, original source hash/offsets and parent provenance SHA-256.
All splits are emitted with the same declared operation. Primary paired evaluation
should continue using the original frozen held-out WAVs; using the newly matched
validation/test files would change the evaluation condition. Per-crop gain changes
global amplitude relationships across segments and is not a claim of perceptual
normalization or improved music quality.

The same four-test Python suite now also checks every sample of real-decoded,
level-matched fixtures, expected f32 scalar-gain results, unchanged parents,
provenance ancestry, exact sample counts, target RMS and overwrite refusal.
