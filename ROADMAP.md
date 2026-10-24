# Priorities after the first instrument

1. **More useful learning and honest quality measurements.** Train on diverse
   held-out sections, add a downsample/upsample multiscale waveform network with a
   larger receptive field, review same-seed matched-level WAVs, quantify clipping
   and spectral structure. Retain the current CPU baseline and raw predictions.
2. **Full-song data and continuity.** Add seekable streaming/crop indexing, seek offsets, stereo representation, streaming full-tail preservation,
   chunked extension and richer masks. Measure signal derivatives, phase/timbre
   and musical joins separately. Add public backward outpainting when tested.
3. **Measured acceleration.** Implement optional OpenCL packed convolution forward
   plus input/weight/bias gradients, then Adam, only after runtime access is fixed
   and parity gates pass. Use persistent buffers and benchmark end-to-end speed,
   not inference-only kernels described as training acceleration.

Later: learned codec + sample correction; recurrent latent context; coarse musical
structure and fine waveform heads; alternative continuous objectives; controlled
interpolation/extrapolation; repeated transformations and attractor logging;
explicit acoustic constraints; portable checkpoint migrations. Keep these as small
experiments behind stable representation/mask/provenance contracts.
