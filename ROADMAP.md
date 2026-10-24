# Priorities after Phase 2

1. **Improve generated audio, not just denoising probes.** Phase 3 implemented
   preconditioning and a consistent calibrated prior with measured tradeoffs. Next
   test phase-preserving multiresolution representations/objectives and richer
   local acoustic structure, alongside late-time coverage. Keep the same network, three model seeds, matched compute,
   amplitude-matched listening, linear-filter controls and new held-out recordings.
   Generated spectra remain overly white despite a modest pooled denoising gain.
2. **Test useful context and stronger acoustic objectives.** Compare coarse-context
   lesions/shuffles with matched disruption controls before attributing gains to
   memory/context. Test a phase-preserving multiresolution auxiliary on short flow
   trajectories, checking diversity and collapse. Pooled context is real but its
   causal contribution is not established. Keep baseline as default.
3. **Scale the instrument carefully.** Seekable full-song crop indexing, stereo,
   streaming preserved tails, chunked extension and trained disconnected masks for
   variation. Separate signal, phase, timbral and musical continuity measurements.
   Preserve the new transient slope/headroom tests and exact known samples.

Complete optional FP32 OpenCL training now passes discovery, forward/backward/Adam
parity and resume tests on Adreno830. The100k continuation improves same-song
held-out reconstruction. Profile host/dispatch overhead next; recorded queues,
FP16 and other devices need separate correctness and performance gates. See
reports/OPENCL_TRAINING.md.

Later: learned codec + sample correction, recurrent latent context, coarse musical
structure/fine waveform heads, alternative continuous objectives, interpolation,
controlled extrapolation, attractor logging and explicit acoustic constraints. None
of those research directions is substituted for the current audible waveform path.
