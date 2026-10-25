# Phase 4: data and capacity, frozen before new training outcomes

Owner listening assessment: the100k model remains mostly broadband noise and is
not music. Reconstruction improvements cannot override that observation.

Prerequisite measurements: the historical same-song validation endpoint MSE at
3k/11k/21k/41k/61k/81k/100k is .014533/.012229/.012127/.010205/.008365/.009155/.008504.
Late updates have no clear monotonic benefit; do not simply extend this line.

Primary experiment is a bounded2x2 diagnostic: hidden width64 versus128
(57,952 versus214,176 parameters), old24s versus expanded200s, fresh model seeds
42 and43,20k updates each. All8 arms keep network pyramid,16k mono,8192sample
crops, batch1, prior/data scale.1, uniform-time residual FM loss, Adam lr.0003 and
clip1. Initial weights match exactly between dataset arms at a given width/seed;
width changes necessarily change parameter counts and initialization draws.
Training remains CPU-referenced optional FP32 OpenCL, with new-width parity first.
Widths are opt-in; existing defaults/checkpoints remain intact. Stereo and longer
context are deferred to avoid mixing variables before useful sound is learned.

Expanded training: AROUND[0,100),[244,304), Cosmic Serenity[0,20), In the Circuit
[0,20). Ten equal20s canonical crops, total200s, three recordings. Each crop receives
one explicitly recorded constant gain to match the frozen baseline RMS.1067798255;
this removes gross amplitude/SNR differences as a confound, while changing global
amplitude relationships between segments. Raw crops are retained. Primary evaluation
uses raw original held-out audio, never the newly gain-matched evaluation files.
Uniform-file selection thus equals duration weighting across these files, apart
from excluded crossing-boundary crop starts. Historical AROUND eval intervals
112:120,136:144,160:168,184:192,216:224,232:240 and two-second guards are excluded.
Each output has original source hash, exact sample interval, decode argv and hash.
The original24s baseline file is unchanged. No source recordings are modified.

Regression validation/test: AROUND216:224/232:240. Additional within-recording
clips Cosmic/Circuit24:27 and29:32 stay out of training. A fourth recording,
Glass Antenna Dreaming, is entirely absent from the training manifest:40:48 is
validation,60:68 is a final cross-recording test for this experiment. Its prepared
0:20 crop is deliberately unused. No previously uninspected-data claim is made
beyond its absence from all FERAL training manifests. Two training seeds and a
single unseen recording are limited evidence; source effects are not seed effects.

Record physical endpoint MSE/derivative/log-spectral errors separately by recording,
corruption level and mask, with all raw rows and noise/silence controls. The common
corruption evaluator uses four fixed crops, four masks, sigma .02/.1/.5; these are
correlated probes, not48 independent replications. No hyperparameter tuning using
the final test. Compare all arms at20k steps and at predeclared10k snapshots; use
actual wall times to interpolate an approximate equal-compute comparison and
label it approximate rather than claim equal updates imply equal compute.

Generate fixed seeds42 and99 from each final model (10s as5s noise+5s learned
continuation); preserve initialized controls. Measure waveform/spectral descriptors,
copying screens and10ms envelope statistics against same-whole-clip-spectrum
phase-randomized surrogates. Include source, Gaussian and fitted AR1 controls.
A darker spectrum, phase-null difference, smooth join or improved denoising is
not evidence of musical structure. Report collapse or worse results without
promoting a model. If data/capacity still yield stationary noise, prefer a bounded
objective/representation investigation over scaling stereo or more hours.

Runtime observes actual cpuset and CPU affinity repeatedly. /top-app is evidence
of foreground scheduling class, not proof of screen visibility. Reduced/unknown
scheduling chooses500step chunks instead of2000; builds use at most2 jobs when
reduced,4 when foreground-class. Record temperatures at least once per second
while training; stop/defer on60C GPU/41C battery/512MiB available-memory operator
limits. These are not manufacturer safety ratings. Trainer RSS limit2GiB leaves
room below the owner's3GB ceiling; device buffers remain capped256MiB. Preserve
partial checkpoints and all failed attempts. Do not alter system performance or
thermal controls; run only one trainer at a time.

Agents: native workers edit disjoint files in this shared checkout; coordinator
integrates/tests/commits. Hermes MCP review is advisory, not evidence. First review
hit an iteration cap and contained incorrect metric/initialization claims; the
bounded follow-up corrected those claims but also confused clip counts and some
source/control descriptors. All conclusions must be checked against actual code
and measurements. Agents cannot validate each other by agreement.

Research context: RFWave uses complex-spectrogram multiband rectified flow for
conditioned waveform reconstruction (https://arxiv.org/abs/2403.05010); WaveFM
uses a mel-conditioned prior and auxiliary multiresolution losses for speech
vocoder training (https://aclanthology.org/2025.naacl-long.110/). Neither establishes
that a tiny unconditional FERAL waveform model will generate music; no models or
code are downloaded, and no reproduction claim is made.

## Resource amendment after initial warm-device trials

The initial policy was60C GPU, and two larger-model attempts
stopped at observed60.3C with partial files retained. Subsequent read-only sysfs
inspection found gpuss passive trip temperatures105C/135C and hot125C. Those are
configured kernel trip values, not user-comfort ratings or permission to approach
them. For subsequent invocations the job-local GPU stop threshold is65C; battery
41C, memory limits and Android thermal management remain unchanged. No system
trip, driver, governor or clock is modified. The purpose is to avoid repeatedly
aborting on short internal-die60C excursions while retaining a substantial
job-level limit; this is not a certified safety claim. Snapshot metadata now
includes the operator limits. Background/moderate scheduling still uses500step
chunks and two build jobs at most. The raw trip reads are preserved in
runs/scaling-diagnostics-v1/read-only-thermal-trips.json. The earlier failed
attempts remain part of the resource report and do not count as successful arms.
