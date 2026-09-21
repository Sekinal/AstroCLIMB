# Final analysis plan — 2026-09-21

Purpose: strengthen the system paper's explanations and controls, not optimize a
new Kaggle submission. Preserve the selected model and canonical inference.
Earlier project-level exposure and repeated DEV selection remain disclosed.

## Cheap phase

1. Replay saved native batch-one CXI 2k/4k/8k decisions on identical 794 DEV rows.
2. Report class F1, paired corrected/broken counts, and original four-way behavior
   on the true related/unrelated subset (including escapes to other classes).
3. Replay saved answer-SFT, matched-SFT, GSPO32 and GSPO512 decisions. Verify zero
   fallback from historical generation receipts and recompute whole scores with
   the same selected IXI/CXC confusion counts.
4. Resample inherited DEV paper groups, paired across models, 10,000 draws with
   seed 20260921. Intervals describe conditional variation of selected predictions;
   they do not remove selection bias or measure training-seed variability.

Analyses are post hoc. Group provenance is inherited and may miss dependence.
Keep all outcomes; do not select another model using these diagnostics.

## GPU phase, frozen before launch

Highest-value control: gold-only vs public4000-initialized CXI, both on the new
machine/runtime, one model per RTX 3090. Same base revision, 2,230 TRAIN rows,
794 DEV rows, seed 7, BF16, rank 16/alpha 32, classifier head, three epochs,
837-update horizon, batch 1/accumulation 8, LR 3e-5/head 1e-4, original pixels
and caption limits. Evaluate and save at 279, 558, 837. Select highest DEV macro-F1,
break ties toward earliest epoch; also report the fixed-final-epoch comparison.
The only intentional treatment difference is verified public-parent initialization
(adapter and head). Parent pretraining is additional compute, so this is not an
equal-total-compute comparison. Matching runtime avoids attributing machine/backend
changes to adaptation. One seed still does not establish seed robustness.

Use unchanged archived training code through the location wrapper. Verify corpus
and image bytes, exact parent artifacts including its replay probe, and frozen
base revision. Preserve export batch one and BF16 inference context. No 4-bit
substitution. Run short memory/throughput gates before committing to full runs;
if a memory-related implementation change is required, apply it to both arms and
record it before the full comparison. Do not silently compress scheduler horizons.

Environment/data staging is allowed during the cheap phase. Before training,
resolve private image-cache and parent-probe availability. If unavailable, report
that blocker rather than bypassing replay guards or inventing a matched result.

## Conditional seed replication, specified before matched-run results

If the first pair completes successfully and measured runtime permits all work
before the collection cutoff, repeat the same paired comparison with gold-stage
seeds 19 and 37, in that order, one pair per wave. Decide whether to run these
waves from elapsed runtime and remaining time, not the sign or size of observed
F1 differences. Preserve the fixed seed-7 public4000 parent in every transfer arm:
this tests gold-stage seed sensitivity conditional on that parent, not variation
in public pretraining. Report all completed seeds, selected and final-epoch
scores, and individual paired differences. Three seeds are a limited robustness
check, not a broad population claim or an excuse to select the best seed.

## Time discipline

At the Sep21 check, live OpenReview due time is Sep22 03:59 UTC (Sep21 21:59
Mexico City), earlier than the advertised AoE deadline. Finish final experimental
collection by Sep22 00:00 UTC, leaving nearly four hours for export checks,
analysis, paper revision and submission. Stop starting long jobs if measured
throughput puts completion beyond that point. Preserve completed partial runs as
such; never present them as completed three-epoch controls. External submission
and consent are still the author's actions.
