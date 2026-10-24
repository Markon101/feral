# Phase 2 protocol: pooled temporal context

Frozen before outcome inspection. Baseline starting commit: 6407f2e. No old run or
checkpoint is changed. The phase-1 waveform representation, flow objective,
optimizer and four training mask distributions remain unchanged.

## Question and candidate

Does longer differentiable context improve held-out audio prediction at the same
parameter count, without losing exact conditioning or making phone training costly?
The optional `--network pyramid` uses 57,952 parameters and the same seeded weight
initialization as baseline. Each residual branch is:

```
h <- h + .5 * repeat(tanh(Conv3(average_pool(h, scale))), scale)
```

Scales are 1/4/16/64, each convolution has dilation 1. The skip remains at full
resolution. Transposes of repeat and average-pool implement true backward flow
through coarse context. Partial final groups divide by their actual sizes.
Structural interior support spans 7,488 samples (468 ms at 16 kHz), versus baseline
992 (62 ms). Crop padding can shorten effective support. This is a pooled residual
CNN, not a U-Net, codec, recurrent memory, or a reproduction of another model.
Aligned pooling may create block artifacts and shift sensitivity; evaluate them.

The first implementation sketch combined pooling and increasing dilation. A causal
support test exposed holes at intermediate distances. Before training, it was
replaced by adjacent (dilation-1) coarse convolutions and wider pooling groups.
Long context here is an explicit neural computation path, not optimizer accumulation.

## Fixed experiment

`python scripts/phase2.py --input '/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav' --run runs/phase2-v1`

* Same owner-selected recording, read only. Train [0,24), validation [32,40),
  held-out test [48,56) seconds. Gaps exceed either receptive field. No waveform
  overlap across splits. No gain normalization: all splits use original gain.
* 600 fresh updates per architecture, model seeds 42/43/44; paired input crop,
  noise, masks, times, learning rate, crop length and parameter count. The script
  checks identical task/time/RNG traces after training. Order alternates by seed.
* Default 16 kHz mono, .512 s, batch one, Adam .0003, no new objective or mask task.
  Baseline remains the default network regardless of results.
* Same seed 7001 and 4 fixed crops per split. All four masks, each at t=.1/.5/.9.
  Preserve all 48 raw rows per checkpoint/split, not only mean loss. All checkpoints
  are final fixed-step checkpoints: no best-of seed or validation selection.
* Paired diagnostics: velocity MSE; endpoint unknown-sample waveform/derivative MSE;
  coarse waveform MSE at 4/16/64 samples; multiresolution Hann STFT log(1+magnitude)
  distance at FFT sizes 128/512/2048. Zero-audio and noisy-input controls per row.
  These endpoint predictions are denoising probes, not stochastic generation.
* Unconditional generation starts from fresh seeded noise. Retain initial and
  trained WAVs. Inspect RMS, limiter hits, differences, spectral bands/flatness and
  lag-1 correlation. Shuffle generated samples to retain marginal amplitudes while
  destroying ordering. Noise and literal train-copy controls calibrate diagnostics.
* Nearest absolute mean-centered correlation scans training windows at stride 256,
  decimation 4. Low correlation only excludes obvious aligned copies at this scan
  resolution. It cannot rule out shifted, transformed or recombined memorization.
* Report per-model results and paired directions across three seeds. Crop rows are
  not independent model replications. One song cannot establish cross-recording
  generalization; objective gains cannot establish musical or perceptual quality.

## Research basis and scope

* Lipman, Chen, Ben-Hamu, Nickel and Le, [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
  (2022/2023): vector-field regression on continuous probability paths is the
  baseline method; this experiment keeps that objective fixed.
* Evans, Parker, Carr, Zukowski, Taylor and Pons, [Stable Audio Open](https://arxiv.org/abs/2407.14358)
  (2024), sections 2 and 5: learned multirate audio representations, latent
  diffusion and separate exact-copy analysis are useful design/evaluation context.
  Its autoencoder/text/DiT scale is inappropriate for this small CPU experiment;
  no weights, labels, text encoder or codec are imported.
* Yamamoto, Song and Kim, [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)
  (2019/2020): multiresolution spectral comparisons motivate evaluating multiple
  time/frequency scales. FERAL's diagnostic is a simple log(1+FFT magnitude) L1,
  not that paper's full spectral convergence/log loss, and is not a training loss.

The specific pooled residual network is an engineering hypothesis. These papers
are context, not empirical evidence for its performance. No new spectral auxiliary
objective, adversarial training, learned codec or diffusion path is mixed into the
architecture comparison. Such changes require their own ablations.
