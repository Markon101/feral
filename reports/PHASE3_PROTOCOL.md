# Phase 3 — preconditioned waveform flow

The owner heard mostly broadband noise after 6,000 raw-velocity updates. Preserve
that checkpoint and listening pack. Increasing Heun from 32 to 256 on the same
seed changed flatness .76363→.76255 and 4–8 kHz fraction .30354→.30291, so additional
integration steps alone did not resolve the measured broadband output.

## Fixed change

Keep the 57,952-parameter pooled waveform network, masks, source gain, crops, unit
Gaussian prior, uniform time draws and Adam learning rate. Introduce optional
`--parameterization preconditioned --data-scale 0.1`:

```
A = sigma_data^2
D = (1-t)^2 + t^2*A
c_in = 1/sqrt(D)
c_skip = (t*A-(1-t))/D
c_out = sigma_data/sqrt(D)
v_theta = c_skip*x_t + c_out*f_theta(c_in*x_t, known/sigma_data, mask, t)
target_f = (clean-noise-c_skip*x_t)/c_out
L = mean_unknown((f_theta-target_f)^2)
```

Known positions in the network's x channel use known/sigma_data as well. Sampling
and evaluation consume physical waveform velocity, retain exact masks, and share
the checkpointed configuration. No source waveform is gain-normalized. The fixed
sigma .1 is close to training RMS .1068 and uses no held-out calibration. All three
coefficients remain finite at t=0 and t=1. This is a derived straight-path Gaussian
reference plus learned residual. The weighted velocity loss is exactly residual
MSE, or physical velocity MSE/c_out²; its numeric training loss cannot be compared
directly with raw-velocity training loss.

The Gaussian reference with f=0 generates scaled white noise and is explicitly a
non-neural control. It must not be claimed as learned audio improvement. An AR(1)
control fits only RMS and lag-1 correlation from training audio, to challenge a
mere spectral-tilt improvement. Neither control is the FERAL generator.

New parameterizations use checkpoint format 2. Legacy format 1 remains loadable;
old software rejects version 2. Full and weights-only resume reject a changed
parameterization rather than silently reinterpreting weights. The CPU velocity
network, optimizer and waveform preservation contracts remain the reference.

## Fixed experiment, before outcomes

`scripts/phase3.py --run runs/phase3-v1`

* Fresh raw-velocity vs preconditioned models, three seeds 42/43/44, 3,000 updates
  each. Same initial neural parameter values, .512 s crops, batch one, lr .0003.
  All jobs sequential. Verify matching training task/time/RNG traces within pairs.
* Train only existing source [0,24) seconds. New validation [112,120), new held-out
  test [136,144). No best-checkpoint or seed selection. Sample seed 42 for listening
  examples is predetermined. Keep old 6,000-step checkpoint as historical control.
* Fixed evaluation seed 7001, four crops, four masks and t=.1/.5/.9. Report physical
  velocity error and endpoint controls separately from normalized training loss.
* Generate from noise with Heun 32. Inspect RMS and spectral shape separately;
  retain shuffled/copy/noise diagnostics. Level-match listening derivatives and
  include Gaussian and AR(1) controls, so quieter/low-pass noise cannot be called
  musical progress. Preserve original unprocessed neural WAVs.
* One song cannot establish broad generalization. Three model seeds provide only
  small replication; no listening-quality or musical-coherence claim follows from
  a regression, spectral or copying screen.

## Research context (not a reproduction)

Karras, Aittala, Aila and Laine, [Elucidating the Design Space of Diffusion-Based
Generative Models](https://arxiv.org/abs/2206.00364), 2022, motivates separating
input/output preconditioning from architecture. FERAL's coefficients above are
derived for its straight flow path, not copied diffusion sampler coefficients.

Zhou et al., [WavFlow: Audio Generation in Waveform Space](https://arxiv.org/abs/2605.18749),
2026, discusses low waveform energy, amplitude lifting and clean-signal prediction
under flow matching. This supports investigating signal scale; FERAL does not
implement that paper's x-prediction, transformer, multimodal conditioning or data
pipeline. No pretrained model, dataset or paper code was downloaded.

The owner's unrelated background jobs caused an OOM/restart during implementation.
No experiment had started; recovered files were inspected, processes checked, and
all tests rerun sequentially before training. The 3 GB working ceiling remains.
