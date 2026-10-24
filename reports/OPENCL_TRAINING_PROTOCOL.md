# Resident FP32 OpenCL training protocol

2026-10-24. CPU math remains the reference. Preserve the calibrated 3,000-step
checkpoint, initialized controls and source recordings. Continue the same model,
dataset, prior, Adam and RNG to 100,000 total updates. More updates alone do not
establish improved audio or generalization.

The optional `opencl` feature dynamically loads the existing Qualcomm Adreno 830
runtime (driver 0800.64.8 / E031.47.18.56). All neural forward, pooling, reverse
derivatives, masked loss, clipping and Adam updates execute in FP32 on the GPU.
Parameters, moments, gradients and tapes remain resident. Each step transfers
crop/noise/mask/time input and scalar status. Checkpoints synchronize full state.
Decoding, crop selection, previews, evaluation and generation remain CPU work.
There is no silent fallback or claim of optimal hardware utilization.

The launcher changes only its process library environment; FFmpeg children
restore the decoder environment. No packages, drivers or system settings change.
Kernel binaries are cached by source, options, device and driver. Buffers are
capped at 256 MiB; the current model/crop allocates 2,241,200 device bytes.
Total working memory must remain below the owner-approved 3 GB ceiling.

Read-only prior-project inspection informed packed persistent buffers, in-order
queues, tiled reductions and event profiling. Titan Image's renderer derivatives
and RoachAudioMasterer's inference kernels are not imported into FERAL.

Pre-training gates: 11 CPU/GPU cases cover both networks, three parameterizations,
endpoint times, partial pooling, dilation and clipping. Forward/gradient tolerance
is 1e-5 absolute + 1e-3 relative; Adam parameters use 5e-6 + 1e-4 relative. Check
multi-update drift, exact GPU resume and CPU continuation of GPU checkpoints.
The hardware integration test compares six continuous steps with 3+3 resumed
steps exactly. CPU finite differences and protected-sample tests remain required.

Final audit: `runs/opencl-audit-final/report.json`, 30 paired steps after two
warmups, CPU 4.744 ms versus GPU 1.807 ms: observed 2.63x. GPU times include
enqueues, transfers and status synchronization, excluding preprocessing and I/O.
Owner confirmed foreground; phone was warm. This short comparison is not a cold
maximum-performance measurement or a sustained speed guarantee.

`scripts/opencl_long.py` records commands, hashes, chunk metrics, temperatures
and UTC timestamps. Between chunks it pauses at GPU >=60 C, battery >=41 C or
available memory <512 MiB; resumes at GPU <=55 C and battery <=39 C. After five
minutes without recovery it stops with checkpoints preserved. These are operator
settings, not manufacturer safety ratings. Before/after readings do not measure
unobserved in-chunk peaks. No clocks or thermal controls change. Compare final
and frozen 3,000-step models using matched seeds/solvers and untouched held-out
audio; preserve failures and noise-like results.
