# Priorities after Phase 2

1. **Improve generated audio, not just denoising probes.** Test data-calibrated
   noise/path schedules and late-time coverage with consistent train/sample
   conventions. Keep the same network, three model seeds, matched compute,
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

Optional OpenCL remains behind actual discovery and forward/backward/update parity.
The smaller pooled computation reduces CPU work; quantify end-to-end benefit before
adding kernels, transfer synchronization or a new optimizer backend.

Later: learned codec + sample correction, recurrent latent context, coarse musical
structure/fine waveform heads, alternative continuous objectives, interpolation,
controlled extrapolation, attractor logging and explicit acoustic constraints. None
of those research directions is substituted for the current audible waveform path.
