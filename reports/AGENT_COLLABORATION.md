# Shared-directory agent trial,2026-10-24

Owner explicitly authorized multiple agents, including write access to the same
FERAL directory. The coordinator retained integration, expensive training and Git
commit ownership. All writes stayed within FERAL; prior projects/audio were read-only.

Two native workers operated in this checkout with disjoint file ownership:

* Model worker: optional CLI width and architecture validation, scaled numerical
  gradient/default-parity tests; then a separate structure module/example/tests.
* Data worker: sample-exact guarded split preparation, optional constant RMS gain,
  provenance tests; then crop tracing API and old-RNG equivalence tests.
* Coordinator: runtime log integration, scheduling/memory/thermal observation,
  sweep orchestration, integration review, independent validation and documentation.

Hermes ran through the installed agent-orchestra MCP bridge using its existing
adapter. The bridge supports one sequential external worker. Native workers and
coordinator progressed while Hermes reviewed; GPU training remained serial.
First Hermes task3dce2ef7d6d94a8bbaf45d30087666a2 failed at an iteration cap but
returned partial critique. Second task8ebd099275a14c1697bfa8df2381de51 completed.
Raw terminal results are preserved in runs/scaling-diagnostics-v1/hermes-review-*.json.
Partial output is labeled partial rather than a completed review. The reviews
contained useful sampling/context cautions alongside mistakes. Coordinator checked
actual metric provenance:43.3% was physical endpoint-waveform MSE versus3k, not
normalized training residual. Fan-in initialization does not automatically shrink
variance when width increases. Follow-up still confused200seconds with200clips
and some source/control descriptors. No conclusion rests on agent agreement.

There were no overlapping-file edits or lost changes. One interim compile was
blocked while runtime referenced crop_traced before the data worker landed it.
The interface was agreed explicitly, then targeted checks and the complete suite
passed after both edits landed. This is a shared-tree integration hazard: use
ownership and a build barrier rather than assuming every intermediate checkout
must compile. No worker commits, reverts of another worker or hidden worktree
merges occurred. Tests exercise actual new-width math and old behavior, not merely
that two agents report success. The old calibrated render is byte-identical.

This trial supports controlled parallel read/write engineering on the phone. It
does not establish unrestricted agent concurrency, optimal throughput or better
model science. The3GB working ceiling, measured resource gates and one trainer
remain in force. No agent installed packages, downloaded models or added remotes.
