# What changed, and what did not

Post-hoc analysis completed 2026-09-21 using saved canonical batch-one predictions
on the same 794 CXI development examples. No model was trained, tuned or selected
in this analysis. Source hashes and complete counts are in `paired-development.json`.

## Public adaptation improved relation classes beyond figure identity

| Class F1 | Public 2k | Public 4k | Public 8k |
| --- | ---: | ---: | ---: |
| Same figure | 0.93233 | 0.92732 | 0.92611 |
| Same paper | 0.60172 | 0.65363 | 0.60436 |
| Related papers | 0.53810 | 0.60976 | 0.62754 |
| Unrelated papers | 0.76190 | 0.80285 | 0.81340 |
| Four-class macro-F1 | 0.70851 | 0.74839 | 0.74285 |

The 2k-to-4k gain does not come from better same-figure recognition. All three
paper-relation class F1s improve. Among 136 changed predictions, 81 become correct,
49 become incorrect, and six change between two incorrect classes: net 32 more
correct examples. This is a descriptive localization of the gain, not proof of
which representation or data mechanism caused it.

The 8k model has a lower macro-F1 but higher related- and unrelated-paper F1 than
4k. Same-paper F1 loses 0.04927. Of 103 changed predictions, 45 are corrected,
47 are broken, and 11 remain incorrect. This is a class tradeoff; it does not
establish exhausted model capacity, uniform degradation, or an overfitting mechanism.
The gold checkpoints were selected at different epochs within the same schedule.

On the 400 true related/unrelated examples, original four-way correct predictions
increase from 273 to 294 to 309 across 2k/4k/8k. Predictions escaping to same-figure
or same-paper fall from 44 to 41 to 25. These are post-hoc slices; no binary
renormalization, retraining or retuning was used. Other classes can regress while
this slice improves, as the table demonstrates.

## RL changes the error distribution

| Replacement | Corrected | Broken | Changed, both wrong | Net correct |
| --- | ---: | ---: | ---: | ---: |
| Answer SFT to GSPO512 | 26 | 19 | 8 | +7 |
| Matched short SFT to GSPO32 | 11 | 15 | 1 | -4 |
| Selected classifier to GSPO512 | 14 | 26 | 8 | -12 |

For answer SFT to GSPO512, 25 of 26 corrections are true related-paper cases;
all 19 regressions occur outside that class. This supports describing a class
tradeoff, not a broad reasoning improvement. A matched-duration long SFT control
is absent. Every saved generative evaluation reports 794 valid canonical answers,
794 EOS completions, and zero fallback; the script verifies those receipts and
does not silently discard malformed output. Thus these comparisons are not merely
counts of formatting failures.

Whole-task F1 is recomputed with the identical selected IXI/CXC confusion counts.
The verified scores reproduce answer SFT 0.74630, matched short SFT 0.74670,
GSPO32 0.74494, GSPO512 0.75115, and selected classifier 0.75620. Do not confuse
these with CXI-only F1 or historical whole scores using older IXI/CXC branches.

## Conditional uncertainty, not independent confirmation

We sample the 278 inherited CXI DEV groups with replacement, 10,000 times, seed
20260921. Every member receives its group's sampled multiplicity; each draw is
shared across all models. Macro-F1 is recomputed from each resulting confusion
matrix. We do not average group-level F1. Percentiles below concern CXI only.

| Comparison | Observed delta macro-F1 | Paired group-resampling 95% interval |
| --- | ---: | ---: |
| 2k to 4k | +0.03988 | [0.00707, 0.07418] |
| 4k to 8k | -0.00554 | [-0.03608, 0.02278] |
| Answer SFT to GSPO512 | +0.01487 | [-0.00095, 0.03227] |
| Matched short SFT to GSPO32 | -0.00669 | [-0.02021, 0.00678] |
| Classifier to GSPO512 | -0.01333 | [-0.02873, 0.00155] |

These describe conditional sampling variation of already selected predictions.
They are **not** selection-corrected confidence statements, training-seed
variability, multiple-comparison-adjusted tests, or proof of causal superiority.
The inherited groups may also miss dependencies. In particular, neither the 8k
comparison nor the long-RL comparison establishes a robust gain/loss merely from
its point estimate. The larger 2k-to-4k change is clearer within this conditional
analysis, while still requiring independent confirmation.

## Reproduction and next evidence

Run `uv run python analysis/paired_development.py --manifest inputs.json --out
analysis/paired-development.json` as one command. The manifest contains `labels`
(selected CXI DEV parquet), `groups` (released folds parquet),
`fixed_other_modalities` (`development-errors.json`), `models` (name to path and,
for generated answers, `metadata` receipt), and `comparisons` (ordered name pairs).
Optional `expected_macro_f1`/`expected_whole` values enforce historical replay.
No input paths or row-level labels are written to the output. Private saved input
artifacts are required and are not distributed in the source release.

Run `uv run --extra paper python analysis/plot_paired.py` for the standalone
`paired-development.pdf` and PNG. It plots descriptive class F1 and paired counts,
not bootstrap intervals. Invalid-generation inputs are deliberately rejected;
future datasets with nonzero fallback need an explicit matching historical policy.

The next experiment is a same-runtime gold-only versus public4k initialization
control, with the final epoch also reported to reduce dependence on endpoint
selection. See `docs/final-analysis-plan.md`. It is a proposed/running experiment
until validated outputs exist, not an already established causal result.
