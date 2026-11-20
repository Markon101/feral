# Architecture

FERAL models acoustic samples directly. An f32 mono or interleaved stereo
(`--channel stereo`, `[L0, R0, ...]`) waveform is losslessly packed into consecutive
frames of `block * channels` scalar values (`frame_elems`). This is a layout, not a
learned codec or magnitude spectrogram: phase and sample identities remain present.
A complex STFT would add consistency/windowing/reconstruction requirements; direct
waveform is the smaller reliable first implementation on this CPU. Candle is used by
existing projects but was absent from the local registry cache. FERAL needs only
dense and short convolution derivatives, so a specialized network with explicit
reverse-mode backpropagation avoids framework downloads and a general-purpose tensor
engine. Numerical finite-difference tests check weights and biases in every layer.

## Modules and contracts

* `audio`: FFmpeg adapter, bounded decoding/resampling/channel selection (`mix`,
  `left`, `right`, or interleaved `stereo`), optional joint peak normalization,
  finite f32 PCM, exclusive float WAV output.
* `data`: file/directory/manifest selection (up to 512 files), streaming SHA-256,
  decoded RAM cache (1024 MB default, adjustable via `FERAL_MAX_DATASET_MB` in
  `256..=4096`), deterministic frame-aligned crops and silence rejection. No
  genre/text/score annotations.
* `model`: time-major `[frame, channel]` neural operations and parameter state.
  `Velocity` (including `channels()`) is the backend boundary used by the solver. No
  device types escape.
* `conditioning`: frame-aligned multi-channel sample masks, exact projection and
  unknown-side per-channel boundary collar.
* `flow`: conditional path, unknown-sample MSE, Euler/Heun and bounded perturbation.
* `runtime`: Adam/AdamW, gradient clipping, versioned JSON (`<= 4M` params) and
  single-file `FERALCK1` binary (`> 4M` params) full-state checkpoints, run
  metadata, evaluation probes, multi-horizon training loop and rendering.
  `backend::Trainer` explicitly selects CPU, OpenCL or CUDA; audio stays independent.
* `rng`: versioned SplitMix64 + Box-Muller normal stream; checkpointed training
  state and separate seeded preview/sampling streams.
* `main`: clap CLI and configuration. Research definitions live in EXPERIMENTS.md.

## Exact current network

Default baseline phone profile: **57,952 trainable f32 parameters**, batch size one.
For each frame, concatenate 32 current samples, 32 protected context samples
(context times mask), 32 per-sample mask values, and four time features:
`t`, `1-t`, `sin(2*pi*t)`, `cos(2*pi*t)`. This gives 100 input channels.

1. Pointwise convolution 100→64, bias, tanh: 6,464 parameters.
2. Four residual blocks: `h + 0.5*tanh(Conv1d(h))`, each 64→64,
   kernel 3, zero same padding, dilations 1, 2, 4, 8. Each: 12,352 parameters.
3. Pointwise linear output 64→32 with bias: 2,080 parameters.

Total: 6,464 + 4×12,352 + 2,080 = 57,952. Receptive field: 31 blocks,
992 samples, 62 ms at 16 kHz. The default 8,192-sample crop lasts 0.512 s,
but local receptive field limits modeling of global structure. The network is
noncausal: it may use known content on both sides for inpainting. Initialization
is seeded normal with fan-in scaling; biases start at zero. No oscillator bank or
nonlearned synthesis participates in generation. Synthetic signals only test it.

## Conditional rectified flow

For binary known mask m (1=protected), clean waveform y, z~N(0,I), t~U(0,1):

```
x_t = m*y + (1-m)*((1-t)*z + t*y)
u_t = (1-m)*(y-z)
L = sum((1-m)*(v_theta(x_t, m*y, m, t) - (y-z))^2) / sum(1-m)
```

The context input never exposes unknown clean samples. The network's known-region
velocity outputs are unused. Each update randomly selects unconditional, prefix,
interior-hole or suffix masks. Adam uses beta1=.9, beta2=.999, eps=1e-8,
lr=3e-4 by default, global gradient norm clipped to 1. Full reverse-mode derivatives
propagate through every convolution, tanh and residual branch.

Sampling draws z, replaces known positions with y, and integrates v from t=0 to
1 using positive dt. Every Euler stage, Heun predictor and corrector projects known
samples. The optional fixed seeded texture perturbation is
`p*sin(pi*t)*tanh(texture)` and has absolute magnitude <=p. It has no optimizer
role; p=0 yields the baseline. Noise scaling deliberately changes the initial
sampling distribution; it is an OOD control, not a temperature theorem.

## Continuity contract

Exact waveform retention means canonical known f32 samples are copied exactly.
It does not mean matching compressed file bytes, stereo samples, or another sample
rate. FFmpeg preprocessing defines the reference. Unknown-side collars blend
network samples toward short, decaying local tangent estimates at boundaries.
Known audio is never crossfaded. Boundary absolute differences are measured.
Limiter/NaN checks are separate numerical safeguards. Float WAV avoids quantizing
protected samples. Smooth joins do not prove phase/signal fidelity, timbral
continuity, or long-range musical continuity. Those require separate evidence.

Continuation uses the last half-training-crop as protected context and generates
a suffix; the untouched earlier prefix is reattached. Inpainting sends a bounded
local window with a quarter-crop context on each side and reattaches outside
samples unchanged. Model context and generated window are explicit, not hidden
recurrent state. Suffix masks also train the primitive needed for future public
outpainting commands; there is no public bidirectional outpaint command yet.

## Backend and evolution boundary

The CPU implementation is single-threaded and uses no custom general-purpose
tensor engine. Its explicit convolution forward and input/weight/bias derivatives
are the numerical oracle. Flow only needs `Velocity`, flat waveform/mask buffers
and a block size; a device backend can batch/retain buffers internally. The model
serialization uses portable `Model` and Adam state. `backend::Trainer` selects
CPU or optional `gpu::OpenClTrainer`; the latter retains packed parameters,
moments, activations and derivatives, synchronizing before previews/checkpoints.
Loss, reverse-mode gradients, clipping and Adam are GPU operations. Seeded input
construction, decoding, diagnostics and inference remain CPU operations. Device
initialization errors do not silently fall back. The CPU finite-difference tests,
GPU parity audit and exact GPU resume tests guard this boundary.

Alternative velocity networks can implement `Velocity`; objectives and samplers
can change within `flow`. A complex STFT or learned codec should introduce an
explicit encode/decode contract and conservation tests, while waveform-domain
projection remains the last authority on protected samples. Hybrid waveform/latent
models can retain a sample correction head. Multiscale encoders and recurrent
latent context could add longer receptive fields without replacing decoder,
manifest, provenance or mask contracts. Learned codecs, recurrence and hybrid latent models remain future work. The pooled
waveform variant below is the first implemented multirate experiment.

## Optional pooled residual network (Phase 2)

`Architecture::pyramid()` keeps the same projections, widths, kernel sizes and
57,952 parameters, but uses residual branch pooling scales 1/4/16/64 and dilation
1 throughout. The branch is `.5*repeat(tanh(Conv3(avg_pool(h,s))),s)`, added to the
full-resolution `h`. Backpropagation sums through repeat and divides through average
pooling using actual group lengths; parameter derivatives merge into the same Adam
contract. All four branches participate in one differentiable forward/backward
pass. This is not hidden state or gradient accumulation.

For an interior frame, tracing aligned pooling groups and kernel neighbors gives a
maximum support span of 234 frames, 7,488 samples (468 ms at 16 kHz). Zero padding
and finite crops can shorten it. Pooling loses fine ordering in the residual branch;
the skip preserves fine-resolution activations but does not guarantee fine audio.
Alignment introduces shift sensitivity. Network and sampler remain separate.

An empty `scales` vector is omitted from model JSON and means all scales are one.
This preserves v1 baseline model hashes/checkpoints. New checkpoints with explicit
scales require Phase 2 code; old code rejects their hash instead of silently running
a different network. Full resume requires the same architecture (`--network`).

The `evaluate` module adds forward-only held-out probes, paired endpoint waveform,
derivative/coarse/STFT metrics, unpaired signal descriptors and a bounded correlation
screen. RustFFT (eight cached additional Cargo packages) is used only for diagnostics,
not to replace the waveform generator or add a new training objective. Research
basis and metric differences are explicit in reports/PHASE2_PROTOCOL.md.

The final waveform continuity operator uses uncapped source edge differences and
headroom-adapted tangent decay. It retains the first unknown step's feasible slope;
the .98 safety limiter still wins when a slope cannot fit. Raised-cosine blending
changes only unknown samples. Signal continuity, perceptual continuity and musical
structure continue to require different evidence.

## Checkpointed flow parameterization and prior (Phase 3)

`parameterization::FlowConfig` separates the neural output from physical velocity.
With data variance A and prior variance B, D=t²A+(1-t)²B:
`c_in=1/sqrt(D)`, `c_skip=(t*A-(1-t)*B)/D`, `c_out=sqrt(A*B/D)`.
The CNN learns a normalized residual and physical velocity is `c_skip*x+c_out*f`.
Known context uses data-scale normalization. The target is
`(clean-noise-c_skip*x)/c_out` on unknown positions; its MSE equals physical
velocity MSE/c_out². All coefficients remain finite at both endpoints. This is an
engineering adaptation of preconditioning ideas; research scope is in the Phase 3
protocols, not a reproduction claim.

`Velocity::prior_std` has a legacy default of 1. `Checkpoint::velocity()` supplies
the saved configuration for solver initialization and field evaluation. Runtime
training, previews, evaluation and rendering all use that configuration; passing
`checkpoint.model` directly is only correct for raw neural output experiments.
V1 means raw velocity; v2 means unit-prior preconditioning; v3 permits non-unit
priors. Loading and resume validate the version/configuration contract. Missing
legacy fields retain their original semantics and outputs. GaussianReference and
AR(1) examples are non-neural falsification controls, not the primary generator.

## Explicit capacity and data diagnostics (Phase4)

Training width is configurable through Architecture::with_width and --width;
64 remains the default,128 has214,176 parameters,256 has821,536. The existing
network topology, scalar objective, flow prior and checkpoint format do not change.
CPU finite differences and actual GPU forward/backward/Adam parity gate widening.
Same-width old seeded renders remain byte-identical. Wider full resume requires
an explicit matching CLI width.

Dataset::crop_traced exposes accepted source/start/attempts without extra RNG
operations. Its legacy crop wrapper remains exact. Runtime records those traces
and source/retry counts; sampling remains uniform-file then uniform-start. Prepared
equal-duration files make the primary diversity experiment's weighting explicit.
Original source intervals and hashes are recorded by the preparation sidecar;
loader dataset hashes identify the exact resulting PCM files.

structure is forward-only diagnostic code using existing RustFFT. Its f64
phase null preserves DC, Nyquist and conjugate symmetry; magnitude/power/RMS errors
are measured. Ten-millisecond envelopes, modulation and normalized overlap
correlations describe time variation. The surrogate never enters the neural
training path or replaces learned generation. Phase-null differences are not
proof of music, pitch competence, useful memory or generalization.

## Optional spectral objective, schedules, EMA, gated-FiLM blocks and sample-domain stem/head

All additions are opt-in and omitted from checkpoint JSON when at their legacy defaults
(`spectral_weight = 0`, `time_sampling = uniform`, `lr_schedule = constant`, `ema_decay = 0`,
`sample_conv = 0`, `block_style = tanh`, `dilation_pattern = ascending`), preserving v1–v3 model
hashes, serialization and numerical behavior bit for bit.

* `spectral`: differentiable multiresolution Hann STFT (`N = 128, 512, 2048`, `hop = N/4`)
  log-magnitude Charbonnier smooth-L1 loss on `y_hat = m*y + (1-m)*(x_t + (1-t)*v_pred)`, scaled
  by `spectral_weight * t`. CPU computes the exact one-sided real-DFT transpose via `rustfft`
  inverse FFT. CUDA runs a resident shared-memory radix-2 STFT kernel (`feral_stft_frame`) using
  the exact 1-FFT real-pair packing of `yhat_w + i * clean_w`, writing gradients into 4
  non-overlapping `hop = N/4` slots (`feral_stft_accum`) with zero PCIe sync and zero atomics so
  CUDA resume stays bit-exact.
* `parameterization::TimeSampling` and `LrSchedule`: optional logit-normal `t` sampling
  (`sigmoid(mean + std * N(0, 1))`, clamped to `[1e-5, 1 - 1e-5]`) and linear-warmup cosine LR
  decay (`lr_at(base_lr, step)`). Both are checkpointed and validated on full resume.
* EMA shadow weights (`Checkpoint::ema_weights`): updated after each optimizer step
  (`ema = decay * ema + (1 - decay) * w`) on CPU and via `feral_ema` on CUDA, synced on
  checkpoints/previews/validation, and selected by default for inference and evaluation (`--no-ema`
  selects raw weights).
* Multi-cycle / symmetric dilations (`Architecture::with_dilation_pattern`): supports up to 24
  residual blocks (`ascending`, U-Net `symmetric`, or 2-pass `cycles`) so fine/medium layers refine
  local pitch and transients after global receptive-field layers.
* Gated FiLM residual blocks (`BlockStyle::GatedFilm`): each residual block applies parameter-free
  pre-RMSNorm modulated by a per-block FiLM projection (`time_proj`, `32 -> 2*width`) of a
  deterministic 16-frequency log-spaced sinusoidal time embedding, followed by a `width -> 2*width`
  dilated 1D conv and WaveNet gated activation `h + 0.5 * (tanh(a) * sigmoid(b))`.
* Sample-level residual 1D conv stem and head (`Architecture::sample_conv = K`, odd `3..=15`):
  adds `stem` (`3*C -> 16 -> C`, tanh, `x_eff = x + 0.5 * s1`) before frame packing and `head`
  (`C -> 16 -> C`, tanh then linear, `out = h_raw + 0.5 * h1`) after frame unpacking, adding
  `4 * (K - 1)` time steps of shift-equivariant context across frame boundaries while keeping
  unknown clean audio masked to zero (`known * mask`). Parameter order is
  `layers`, `time_proj`, `stem`, `head`. CPU finite differences and batched CUDA parity tests
  verify every frame, time-projection, stem and head parameter.

## Native stereo, GluFilm, AdamW, multi-scale horizons and FERALCK1 checkpoints

All additions preserve legacy mono model hashes and checkpoint behavior bit for bit when at
their defaults (`channels = 1`, `block_style = tanh`, `weight_decay = 0`, single fixed horizon).

* Native interleaved stereo (`Architecture::channels = 2`): frames hold `frame_elems = block * C`
  scalars (`3 * frame_elems + 4` input features per frame), so the dilated frame convolution stack
  runs at the same time-frame count `T / block` for mono and stereo while `stem`/`head` run over
  interleaved time steps `time_steps = total_scalars / C`. Serde omits `channels` when `1`. Masks
  and tangent collars (`conditioning::{training_mask_channels_into, continuity_channels}`) align to
  channel pairs, and `spectral::endpoint_loss_gradient_channels` evaluates Left, Right, Mid
  `(L+R)/sqrt(2)`, and Side `(L-R)/sqrt(2)` uniformly (`1/4` each) on both CPU (`rustfft`) and
  CUDA (`feral_stft_frame` + `feral_stft_accum_ms`) with zero gradient on protected samples.
* Non-saturating depth-scaled GLU (`BlockStyle::GluFilm`): replaces the double-bounded `tanh(a)`
  content branch of `GatedFilm` with `a * sigmoid(b)` scaled by `1 / sqrt(num_blocks)`.
* Decoupled AdamW (`Adam::update_with_lr_wd`, CUDA `feral_adam`): `Model::weight_decay_mask()`
  marks layer weights (`1.0`) while exempting all biases (`0.0`).
* Multi-scale horizons (`parameterization::HorizonSchedule::{Fixed, Random, ShortToLong,
  LongToShort}`): varies crop duration across `--crop-horizons` while adjusting per-horizon batch
  size to hold `crop_frames * batch` roughly fixed (`set_shape` switches preallocated CUDA buffers
  without reallocation). `eval_n = probe_window_samples(sr, max_crop_frames, block)` probes up to
  the maximum training horizon; `feral evaluate` pins its window from `checkpoint.crop_samples`.
* Single-file `FERALCK1` binary checkpoints: automatically used when `model.count() > 4_000_000`
  (transparently detected by `Checkpoint::load`, capped at 2 GiB). Format: 8-byte magic
  `FERALCK1`, little-endian u64 header byte length, UTF-8 JSON metadata (including architecture,
  optimizer metadata, RNG state and `tensors_sha256`), followed by contiguous little-endian f32
  arrays (`weights`, `adam.m`, `adam.v`, and optional `ema_weights`).
* `OpenClTrainer` remains mono-only and explicitly errors if given `channels != 1`, non-zero
  `weight_decay`, or multi-horizon shapes.
