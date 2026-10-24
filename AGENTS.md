# Working on FERAL

Read README.md, ARCHITECTURE.md, LIMITATIONS.md and reports/VALIDATION.md first.
The CPU implementation is the numerical reference. Keep real neural learning,
canonical audio conditioning and exact protected samples as hard contracts.

Keep changes inside this repository. Source recordings and existing research
repositories are read only. Preserve initial controls, existing runs, checkpoints,
seeds and failed experiments; use new output names. Do not publish or add remotes
without an explicit owner request. Do not download models, datasets or system
packages automatically. The current owner-approved working memory ceiling is 3 GB.

Use the locked dependencies and small CPU checks before costly training:

```
cargo fmt --check
cargo clippy --locked --offline --all-targets -- -D warnings
cargo test --locked --offline --release -j 8
```

For changes to model math, check analytic gradients against numerical derivatives,
real parameter updates, initialized/trained controls and deterministic resume. For
conditioning changes, test exact canonical sample retention and measure joins.
Never infer perceptual quality or musical continuity from those numerical gates.
OpenCL requires actual discovery and forward/backward/update parity before any
acceleration claim. Background timings are not maximum-performance measurements.
See reports/OPENCL.md before changing backend behavior.
