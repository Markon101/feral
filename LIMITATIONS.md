# Limits and failure modes

Implemented: real differentiable neural rectified-flow training; CPU Adam updates;
WAV/FFmpeg ingestion; mono f32 output; unconditional and masked sampling; exact
canonical protected samples; learned prefix continuation and hole inpainting;
experimental anchored variation; Euler/Heun; seeded noise scaling and bounded
velocity perturbations; versioned full Adam/RNG checkpoints and weights-only
warm starts; tests and small actual generated WAVs.

Not established: pleasant sound, music competence, held-out generalization,
source-identity retention, phase-coherent generated transitions, timbral continuity,
beat alignment, long-range musical structure, or quality after repeated transforms.
Do not mistake decoder success, finite loss, lower training probes, or changed
output for any of those results. No subjective listening evaluation is claimed.

* The CNN sees 62 ms at 16 kHz. It cannot plan musical form. Outputs after very
  short training can be noisy, clipped, strange or unrecognizable. Unknown-region
  hard clipping can introduce distortion; counts are a future diagnostic.
* Training is currently single-threaded batch one. The 3 GB working ceiling is
  respected by a much smaller profile. Thermal state, background scheduling and
  device load are uncontrolled; measured latency is not a maximum-speed claim.
* Only mono is modeled. Stereo inputs are downmixed or one channel selected.
* Every selected file is cropped at its beginning to `--max-file-seconds`
  (default 30); random training crops are taken inside that decoded section.
  There is no disk cache, streaming corpus loader, recursive traversal or
  offset-indexed whole-collection sampling yet. At most 128 files and 128 MB PCM.
* Conditioning accepts complete inputs up to `--max-input-seconds` (default and
  maximum 600). A duration probe rejects longer inputs before decoding. Unknown
  durations require an explicitly decoded WAV crop; there is no streaming input
  or seekable long-file conditioning yet.
* Generation bounds: 5 s and 160000 samples; conditional windows include context
  and padding. Training crops <=1 s. Long generation is OOD and can be poor.
* Continuation and inpainting are trained mask operations, but a tiny dataset/run
  cannot establish useful repair or coherent continuation. Exact retention is
  confirmed independently of perceptual quality. Smoothing is a local operation,
  not a continuity guarantee. Inpainting collars can dominate very short holes.
* Variation's repeated mask anchors are experimental and outside the contiguous
  mask patterns used in training. No scalar audio-to-audio strength control.
* No public backward outpaint command, latent interpolation, recurrent latent
  dynamics, multiscale model, learned codec, diffusion, acoustic invariant engine,
  evolutionary search or automated OOD selection. No pretrained models or services.
* Checkpoint format v1 is readable JSON and includes weights, Adam and RNG. CPU
  deterministic resume is tested on this compiler/architecture; cross-platform
  floating-point bit parity is not promised. Hashes detect model payload changes,
  not malicious edits with recomputed hashes. Only load trusted local artifacts.
* Run creation and output writes are exclusive. An interrupted write may leave an
  incomplete artifact; it is retained and rejected on reload. Use a new run/name.
  FERAL never automatically deletes partial files or overwrites outputs.
* FFmpeg/FFprobe must be available; codec support depends on their build. The
  tests exercise six transcoded formats on this installation. Unsupported input
  is an error/recorded skip, not fake training success.
* GPU/OpenCL training is not implemented. Existing OpenCL work was inspected
  read-only. See reports/OPENCL.md for runtime discovery, reusable patterns and a
  concrete acceleration gate. There is no hidden CPU/GPU switch.
