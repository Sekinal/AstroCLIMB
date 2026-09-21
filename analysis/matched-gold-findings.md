# Matched CXI gold-only controls

All three pairs completed and passed fresh-export verification on September 21, 2026.

| Gold-stage seed | Gold only | Public4k | Difference |
|---|---:|---:|---:|
| 7 | 0.689062 | 0.756193 | +0.067131 |
| 19 | 0.676980 | 0.748111 | +0.071132 |
| 37 | 0.681808 | 0.745694 | +0.063885 |

Mean selected-checkpoint difference: **+0.067383**. Mean fixed-final-epoch
difference: **+0.067716**. Seed 19 selects update 558 in both arms; seeds 7 and
37 select 837. All six runs complete 837 updates, with validations at 279,
558, and 837. All eight unique selected/final checkpoint exports reproduce
logged scores exactly; fresh replay maximum logit differences are zero.

Within each pair, configurations differ only in output path and initialization;
training-data hashes match. Both arms use the same pinned base and fresh RTX3090
runtime, rank16 adapters, and the same gold schedule. Public adaptation adds
compute. The public parent is fixed at original seed7: the comparison checks
gold-stage variability, not public-pretraining variability. Scores use the same
repeatedly consulted 794-pair CXI DEV set. Three positive differences do not
establish statistical significance or independent test performance.

These new controls are distinct from the historical selected submission. They
neither replace its artifacts nor produce a new Kaggle score. The earlier
unmatched .71367 gold-only reference used a different gold schedule and is not
part of this comparison. Setup attempts stopped before the matched campaign
are documented in the handoff and excluded from the completed pairs.

Sources: `matched-gold-controls.json`, `matched-gold-verification.json`, and
`docs/final-analysis-plan.md`. Private logs and predictions are preserved locally.
