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

## Completed matched experiment

All six runs completed 837 gold updates; the guarded campaign finished at
2026-09-21 09:18:20 UTC. Fresh batch-one exports reproduce every selected and
final checkpoint score, with zero maximum logit difference on all replay probes.
Retrieved training logs independently regenerate all three aggregate receipts;
prediction/log SHA-256 checks pass. Within each pair, ready configurations differ
only in output path and parent initialization, with identical training-data hashes.

| Gold seed | Gold only | Public4k | Paired gain |
|---|---:|---:|---:|
| 7 | 0.689062 | 0.756193 | +0.067131 |
| 19 | 0.676980 | 0.748111 | +0.071132 |
| 37 | 0.681808 | 0.745694 | +0.063885 |

The mean selected-checkpoint gain is 0.067383; the mean fixed-final-epoch gain is
0.067716. Seed 19 selects step 558 in both arms; seeds 7 and 37 select 837.
These are new-runtime CXI development controls, not new Kaggle scores. The public
parent is fixed at its original seed 7; only gold-stage variability is measured.
Public adaptation adds compute, and repeatedly consulted DEV remains a limitation.
Receipts: `analysis/matched-gold-controls.json` and
`analysis/matched-gold-verification.json`. The standalone PDF/PNG plot includes
all three pairs and has been visually checked.

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

## Preservation and GPU shutdown completed

Original gold images, all selected/final inference checkpoints, processors,
private replay probes, predictions, logs, and export/environment receipts are
preserved locally. Archive and member-level integrity checks passed. Optimizer
states were intentionally excluded from the final inference-only backup.
Private training probes and row-level data are not public release artifacts.

The author authorized termination after verified preservation. Lium accepted
termination of the exact project two-GPU instance at **2026-09-21 17:43:18 UTC**
(HTTP 200); the subsequent account query confirmed it absent from active pods.
No unrelated instance was touched. The private provider receipt is
`output/final_analysis_20260921/lium-termination.json`.

## Final manuscript integration

The matched-control table and interpretation are integrated into the paper.
Submission fields are synchronized; both PDFs compile cleanly with four content
pages and six total. All 28 citation keys resolve and remain credited. All twelve
rendered pages and the standalone three-seed plot passed visual inspection.
The source ZIP is refreshed from the final committed snapshot.
The author still handles OpenReview submission and its profile/consent fields.
No submission has been created. Fresh CXI controls do not constitute independent
end-to-end reproduction of the full three-modality submitted system.

The final prose pass develops each comparison around its purpose and result;
see [editorial references and checks](prose-revision.md). A separate factual
review found no material drift; the identified averaging ambiguity was repaired.
Both PDF variants and the submission abstract include this revision.
