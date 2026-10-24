# Continuation from 1,200 to 6,000 steps

Owner requested approximately five times the training, followed by the same
10-second listening tests copied to /sdcard. Completed 4,800 additional updates,
6,000 total, with the same 57,952-parameter pooled waveform network, training audio,
0.512 s crops, mask mixture, flow objective and Adam learning rate (.0003).

```sh
./target/release/feral train --input runs/phase2-v1/train.wav \
  --run runs/phase2-long-6000 --network pyramid --steps 4800 --seed 42 \
  --no-normalize --resume runs/phase2-budget-v1/pyramid-42/checkpoint.json \
  --preview-every 1200 --checkpoint-every 1200
```

Training used clean code commit `89c4fad`. It completed in 51.484 s including
previews/checkpoints, mean update 10.421 ms, peak trainer RSS 9,784 KiB (excluding
FFmpeg). CPU scheduling/thermal conditions were uncontrolled. No code, architecture
or preprocessing change was made for this run, and no maximum-speed claim applies.

Verified initial model/Adam/RNG/dataset state equals the old checkpoint exactly;
all 4,800 losses and gradient norms are finite; log steps are exactly 1201–6000;
final Adam/model step is 6000 and parameters changed. Evaluation and rendering both
successfully reload the final checkpoint. Raw logs, intermediate checkpoints at
2400/3600/4800, previews, final checkpoint and run provenance remain under
`runs/phase2-long-6000/`. This is a full resume, not a weights-only restart.

| Metric | 1,200 steps | 6,000 steps |
|---|---:|---:|
| Fixed training-distribution probe | .104003 | .089297 |
| Held-out validation velocity MSE, source seconds 64–72 | .383291 | .299873 |
| Held-out test velocity MSE, source seconds 80–88 | .411864 | .334691 |
| Generated sample RMS, seed 42 / Heun 16 / .512 s | .212645 | .144296 |
| Generated 4–8 kHz power fraction | .455655 | .305669 |
| Generated spectral flatness | .952431 | .767204 |
| Generated lag-1 correlation | .096759 | .383462 |
| Aligned-copy screen absolute correlation | .088815 | .106413 |

The fixed held-out test error fell 18.7%, validation 21.8%. The new probe rows use
the same seed 7001, four crops, four masks and three path times as the old rows.
Held-out source sections never enter training. One continued model on one song,
with reused evaluation sections, does not establish broad generalization or a
seed-replicated improvement. Generated high-frequency power remains excessive
(training reference fraction .025558), and perceptual quality is for listening.
The lower RMS alone is not quality evidence. Waveform/spectral temporal changes
are measured separately; low aligned correlation does not rule out transformed
memorization. Complete metrics are in training-6000-metrics.json and raw rows in
the run's eval-validation.json/eval-test.json.

## Listening pack and actual export

All fourteen PCM16 WAVs were copied and SHA-256 verified through the normal
sandboxed executor to:

`/sdcard/FERAL_6000steps_20261024/`

Canonical path: `/storage/emulated/0/FERAL_6000steps_20261024/`.
No permission escalation was used. The earlier failed export attempt concerned
the escalation route; the normal bounded copy succeeded on this request.
All 36 files, including guides/manifests/render metadata, were checked after copying.

Seven top-level clips use 6,000 steps; `comparison_1200_steps/` contains the prior
seven clips with identical seeds/settings. Each WAV is exactly ten seconds, mono
16 kHz PCM16. No mastering, loudness normalization, looping or time stretching.
The generated/normal, OOD and weird tests use five seconds of generation plus five
seconds of learned continuation, as in the prior pack. Longer spans are exploratory
relative to training crops. The experimental variation mask is still not a separately
trained task. Protected source samples match the reference in both float originals
and PCM16 exports; inpainting actually changes its unknown interval.

* `00_reference_AROUND_80-90s.wav`: source reference.
* `01_generated_10s.wav`: seeds 42/43, Heun 32, noise scale 1.
* `02_ood_control_10s.wav`: seeds 73/74, Euler 48, noise scale 1.4.
* `03_weird_10s.wav`: same OOD setup with perturbation .08.
* `04_continuation_source5s_then_generated5s.wav`: source first half, generated second half.
* `05_inpainting_generated_3-7s.wav`: generated middle, protected outside.
* `06_variation_experimental_anchors.wav`: short protected anchors plus regenerated audio.

Local render argv/logs/script/raw float audio and exports are retained under
`runs/listening-10s-6000-v1/`; the previous pack is untouched. Original AROUND audio
remains unchanged. The final checkpoint is `runs/phase2-long-6000/checkpoint.json`.
