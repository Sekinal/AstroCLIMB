# Final analysis handoff — September 21

## Purpose and fixed boundaries

Improve the paper's evidence, not another competition submission. The selected
Kaggle model and its 0.75149 public score remain unchanged. See
[the frozen plan](final-analysis-plan.md) for the matched controls and reporting
rules. The operational submission deadline is September 22 03:59 UTC; experimental
collection stops by September 22 00:00 UTC. Do not rely on later advertised AoE
wording or the portal grace period.

## Completed analyses

- `analysis/paired-development.json` and `paired-findings.md`: paired class-level
  corrections/regressions and conditional inherited-group bootstrap intervals.
- `analysis/public-supervision-diagnostics.json` and
  `public-supervision-findings.md`: TRAIN-only exact-input checks. No duplicate
  inputs or conflicting labels at the checked identity; 19/80,000 public and
  1/2,230 gold captions exceed the 2,400-character limit. These checks do not
  establish semantic correctness or diagnose DEV truncation.
- `analysis/summarize_matched_runs.py`: checks completion, scheduled validations,
  reload guards, arm/seed identities, and both selected/final-epoch comparisons.
- Wrapper relocation supports an unchanged parent's historical model identity
  while loading the pinned snapshot at its new location. It rejects ambiguous
  flag abbreviations and avoids printing the archival model path in its receipt.
- Full CPU suite: 62 tests and six subtests passed. Historical training code is
  unchanged. Pi/Muse implementation/review metadata is preserved in
  `analysis/implementation-review-receipt.json`; private raw sessions are excluded.

## Running experiment

Two RTX 3090 GPUs run gold-only and public4k-initialized CXI with the same
three-epoch, 837-update, seed-7 schedule. Both passed gradient/update and saved
checkpoint replay guards. At the first 279-update validation, logged CXI macro-F1 is 0.6184316863
for gold-only and 0.7276152394 for public4k initialization. These are interim
training-log values; full three-epoch results and fresh exports remain pending.
They do not constitute a new Kaggle result or a completed matched comparison.

The first setup attempt exposed a model-path identity mismatch in the transfer
arm, before transfer training began. Its parallel gold-only attempt was stopped
at about 30 updates. Both are preserved as setup attempts and excluded from the
comparison. The working pair explicitly passes the parent's archival identity to
both arms and physically loads the same pinned base snapshot. No checkpoint
metadata or archived training source was rewritten.

All 2,954 required gold images were recovered as original verified bytes using
only the competition TRAIN CSV. Recovery streams the CSV to locate the selected
TRAIN/DEV rows; it does not use nonselected labels. The full 2.27 GB image archive
has a locally verified SHA-256 backup. Download credentials were removed from the
GPU machine after staging.

## Remaining work

1. Require both arms to finish all 837 updates and their scheduled reload checks.
2. Fresh-process batch-one export selected and final checkpoints, recompute aligned
   DEV macro-F1, and compare against logged values before reporting any result.
3. If measured runtime fits the collection cutoff, repeat paired gold-stage seeds
   19 and 37, conditional on the same seed-7 public parent. The decision is based
   on runtime, never favorable scores. Report every completed pair.
4. Preserve logs, export receipts, and checkpoint artifacts. Available local disk
   is limited: do not blindly download all optimizer states or duplicate caches.
5. Revise the paper only after verified results; distinguish the matched new-runtime
   controls from the historical submitted model. Update limitations, figures,
   reproduction status, metadata receipts, and the source ZIP consistently.
6. Compile both paper variants, check citations and page limits, and visually inspect
   rendered pages. The author still handles OpenReview submission and consents.

Operational paths, launcher scripts, private logs and prediction tables live in
ignored local output/harness directories and on the GPU machine. Do not publish
raw session logs, authentication files, row-level gold data, or training probes.

## Continuation staged and running

A guarded coordinator waits for seed 7, requires completion and matching fresh
exports, then launches seeds 19 and 37 sequentially. Each replication pair has a
two-hour outer training limit and must have at least four hours remaining before
the collection cutoff. Export subprocesses have 30-minute timeouts. A failed
completion, replay, or score check stops continuation; scores never gate whether
a replication is attempted. The coordinator holds an exclusive process lock.
Remote preflight verified 794 unique label-free inference rows and the actual
gold label field before launch. Final experiment results are still pending.
