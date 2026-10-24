# Architecture

FERAL models acoustic samples directly. An f32 mono waveform is losslessly packed
into consecutive blocks of 32 samples. This is a layout, not a learned codec or
magnitude spectrogram: phase and sample identities remain present. A complex STFT
would add consistency/windowing/reconstruction requirements; direct waveform is
the smaller reliable first implementation on this CPU. Candle is used by existing
projects but was absent from the local registry cache. FERAL needs only dense and
short convolution derivatives, so a specialized network with explicit reverse-mode
backpropagation avoids framework downloads and a general-purpose tensor engine.
Numerical finite-difference tests check weights and biases in every layer.

## Modules and contracts

* `audio`: FFmpeg adapter, bounded decoding/resampling/channel selection, optional
  peak normalization, finite mono f32 PCM, exclusive float WAV output.
* `data`: file/directory/manifest selection, streaming SHA-256, decoded RAM cache,
  deterministic crops and silence rejection. No genre/text/score annotations.
* `model`: time-major `[frame, channel]` neural operations and parameter state.
  `Velocity` is the backend boundary used by the solver. No device types escape.
* `conditioning`: sample masks, exact projection and unknown-side boundary collar.
* `flow`: conditional path, unknown-sample MSE, Euler/Heun and bounded perturbation.
* `runtime`: Adam, gradient clipping, full-state checkpoints, run metadata,
  evaluation probes, training loop and rendering. `backend::Trainer` explicitly
  selects CPU or optional resident FP32 OpenCL training; audio stays independent.
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
