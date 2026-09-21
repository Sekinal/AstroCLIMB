# Matched gold adaptation: verified results so far

Status: seed 7 complete; seeds 19 and 37 are planned replications and are not
included in these results. This is a CXI development comparison, not a Kaggle
submission or an independent test result.

| Gold updates | Gold only | Public4k then gold | Difference |
| ---: | ---: | ---: | ---: |
| 279 | 0.618432 | 0.727615 | +0.109184 |
| 558 | 0.667065 | 0.733450 | +0.066385 |
| 837 | 0.689062 | 0.756193 | +0.067131 |

Both arms select their final checkpoint, so the selected and fixed-final-epoch
comparisons coincide. Fresh-process batch-one exports reproduce both logged
final macro-F1 scores exactly on the same 794 development pairs. Checkpoint replay
also reproduces the stored probe logits exactly. Downloaded prediction tables and
export logs match their recorded SHA-256 hashes; recomputing the summary from
retrieved training logs reproduces the remote training receipt.

The ready-event configurations differ only in output location and parent
initialization; training-data hashes match. Both use the frozen three-epoch,
837-update gold schedule. The public arm starts from the verified public4k
adapter and classifier head. That parent represents additional training compute:
this is not an equal-total-compute comparison.

The public-initialized arm leads at every recorded epoch in this pair. Its
advantage remains after three gold epochs, so the observed gain is not confined
to the first epoch. This does not establish what would happen with a longer
matched gold schedule. All four class F1 values improve, with the largest changes
in same-paper and related-paper classification. The individual class values and
confusion matrices are in `matched-gold-seed7-verification.json`.

One gold-stage seed does not establish robustness. Repeated development use and
inherited split provenance remain limitations. Replications retain the same
seed-7 public parent; they will measure gold-stage variability conditional on
that parent, not public-pretraining variability. The new runtime and retrained
models also differ from the historical submitted artifacts. Do not substitute
this score for the historical CXI score in the submitted-system results.

Sources: `matched-gold-controls.json`, `matched-gold-seed7-verification.json`,
and the frozen protocol in `../docs/final-analysis-plan.md`.
