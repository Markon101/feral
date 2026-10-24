# OpenCL readiness audit

Read-only bounded inspection on 2026-10-24:

* `../titan_image_ecosystem/src/opencl.rs`: device queries, `opencl3` context and
  in-order queue, f32 time/pixel-major MLP buffers, persistent training tapes,
  residual derivatives, input VJPs, parameter-gradient reductions, shared packed
  weights/biases and explicit readback. The architecture-specific renderer is not
  FERAL's temporal convolution. No code copied or cross-repository dependency.
* `../RoachAudioMasterer/src/opencl.rs` and `src/kernels.cl`: bounded batch capacity,
  optional device selection, persistent weights/scratch, profiling events for
  transfers and kernels, explicit completion and backend isolation. Again, MLP
  operations do not implement this model's dilated convolution backward pass.
* Existing Cargo manifests use optional `opencl3` dynamic loading and CL1.2. That
  crate and its dependencies are cached. Runtime kernel construction is present;
  no persistent binary-kernel disk cache was identified in the inspected paths.

Current machine: ARM64 Android, vendor `/vendor/lib64/libOpenCL.so` exists. Plain
`clinfo --list` returns status 0 and no devices/platforms. Explicit
`OCL_ICD_VENDORS=/vendor/lib64/libOpenCL.so OCL_ICD_ASSUME_ICD_EXTENSION=1 clinfo --list`
reports that the vendor library is inaccessible to this process's Android linker
namespace. This establishes a runtime-access blocker **for this launch context**,
not an absence of OpenCL hardware or proof that the user's normal Termux shell
cannot access it. No runtime, drivers, system configuration or other repository
was modified. The readable KGSL sysfs gpu_model reports `Adreno830v2`; OpenCL device, driver
and precision queries remain unavailable from this process. Older runtime notes
are not treated as current discovery.

Decision: **defer the optional OpenCL backend**. FERAL runs complete CPU training
and sampling. Existing discovery/queue/buffer/profiling/correctness approaches are
reusable, but temporal convolution forward plus all backward reductions and an
optimizer need new model-specific kernels. Inference-only acceleration would not
solve measured training costs. Fixing namespace access or installing a runtime is
outside the authorized package/configuration scope of this build.

Next implementation gate:

1. Obtain successful device discovery from an ordinary owner-launched Termux
   process without changing system drivers. Record device, driver, CL version,
   memory/workgroup limits and precision. No silent CPU fallback.
2. Keep f32, time-major `[frame, channel]`, packed weights and a shared in-order
   queue. Compile fixed-shape CL1.2 kernels once; keep activations/derivatives and
   weight-gradient buffers resident. Cache programs in process; any future binary
   cache key must include source, options, driver/device and shape.
3. Implement pointwise projection and dilated k=3 convolution with their input,
   weight and bias derivatives. The CPU layer math is the oracle. Start tolerances
   at max absolute 1e-4 plus relative 1e-3, report actual errors, then compare
   objective, clipped gradients, Adam updates and seeded trajectories. Mixed
   precision remains disabled until separate correctness evidence exists.
4. Measure warm kernel, transfer, queue and end-to-end step costs at the actual
   crop width. Persistent buffers avoid repeated allocations. Add device memory
   bounds, checked errors and explicit backend selection. Initialization failure
   must fail unless the user explicitly permits CPU fallback.
5. Implement training acceleration only if forward+backward savings exceed
   synchronization/transfers. Preserve CPU tests/checkpoints and report speedup
   and peak memory under matched foreground/thermal conditions. No such speedup
   is claimed by this MVP.

CPU measurements and their scheduling limitations are in VALIDATION.md.
