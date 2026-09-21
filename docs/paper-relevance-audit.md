# Relevance audit of `paper/sections.tex`

Scope: remove implementation mishaps, internal bookkeeping, and unreported dead ends that do not help readers understand the reported result. No numerical results, citations, evaluation restrictions, failed experiments, uncertainty statements, inference-setting sensitivity, or the historical-vs-fresh discrepancy were removed. Abstract, introduction, tables, figure, and acknowledgments are unchanged. Line count is unchanged (125); word count drops from 2,071 to 2,054.

## Edits made (six)

1. **Earlier-exploration disclosure (Data and Evaluation, "Public supervision").** Dropped "and relied on misaligned OCR" (a discarded alignment bug with no bearing on reported results). Replaced the vague "everything it touched is excluded" with "the models, features, and results affected by that exploration are excluded from this paper." Retained: scoring-information and structural-shortcut exposure, synthetic-gold overlap, label-dependent sampling, no competition test labels, and "the exposure cannot be undone."
2. **OCR results (Results, "OCR and auxiliary transfer").** Removed "correctly aligned" before PP-OCRv6. The negative OCR results and all scores (0.44142, 0.68987, 0.69513) remain.
3. **Reproducibility paragraph (System).** Removed the standalone sentence "Historical replay probes remain private because they contain training examples." This is an archive/repository detail. Retained: public source, prompts, configs, and weights; packaged inference manifest-checked but not GPU-validated; full retraining not independently verified.
4. **Conclusion.** Removed "Billing does not isolate experiment costs." No cost comparison is made anywhere in the paper.
5. **GSPO results.** Removed ", though both verifiably update parameters" (a debugging sanity check). Retained the actual GSPO32/GSPO512 results, the SFT-control mismatch, and the missing-control caveat.
6. **Additional (not in the candidate list): scoring description (Data and Evaluation, "Selection and reporting").** Replaced "the four-class logger scoring the absent class as zero" with "with the four-class score counting the absent class as zero." "Logger" names an internal training-script component; the scoring convention itself is unchanged and still stated.

## Verification

Python comparison of original vs revised: identical sets of numeric tokens (none lost, none added) and identical sets of `\cite*` keys (none lost, none added).

## Material disclosures deliberately retained

- Earlier exposure of scoring information and a structural shortcut; synthetic data overlapping gold; exposed fold 0 excluded but not a clean test set; no independent benchmark claim.
- Batch-size sensitivity (11 of 794 predictions changed, max probability difference 0.380) and backend sensitivity (vLLM rollouts vs. native evaluation).
- The unexplained L40S vs. RTX 3090 discrepancy (0.74839 vs. 0.75619) and the restriction of fresh controls to one runtime.
- Compute not matched between arms; single public parent; single-seed status of all other neural comparisons.
- Bootstrap intervals and their stated limitations; failed public-to-gold transfer for the frozen-feature IXI classifier; failed dynamic-sampling trial; initial 13/64 valid-answer rate motivating answer-tag SFT.
- Hashing and fresh-replay policy before export (supports the "fresh exports exactly reproduce" claim in Results).

## Optional candidates considered and kept

- "superseded whole-system baseline" (OCR paragraph): mildly bookkeeping-flavored, but it correctly warns the reader that the 0.68987 → 0.69513 comparison is not on the final system. Kept.
- Model label "public2k-to-gold837" in the batch-size sensitivity sentence: internal naming, but it precisely identifies which model the sensitivity measurement was made on. Kept.
- "without shortening the schedule" (public adaptation): clarifies that update 4,000 is a mid-schedule checkpoint of a 10,000-update cosine schedule, which matters for interpreting the duration comparison. Kept.
- The 1,024-pair / 128-update answer-tag SFT warmup and the 256-context TRAIN monitor numbers: these define the RL setup and screen, not bookkeeping. Kept.

Coordinator validation: both PDFs retain four content pages including acknowledgments, with references on pages five and six. Changed page renders inspected; unchanged page images match the previously inspected renders. Numerical token sets, cited keys, opening, table and figure environments are unchanged.
