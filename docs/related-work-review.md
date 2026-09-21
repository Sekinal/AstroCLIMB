# Related-work review and revision record

**September 21 update:** the previously missing matched gold-only comparison
is now complete for three gold-stage seeds, conditional on one fixed public
parent. The mean selected-checkpoint CXI DEV gain is 0.06738; all fresh exports
and replay checks pass. See [the final handoff](final-analysis-handoff.md).
The historical review below records the earlier evidence state; its independent
evaluation and long-SFT-control concerns remain unresolved.

Targeted review completed 2026-09-14 against primary publisher records and papers.
This is related-work positioning added during writing, not a claim that all these
papers inspired the historical experiments. The broader historical inspiration
inventory remains in `paper/inspiration-audit.md`.

## Published precedents

| Work | Relevant precedent | Boundary and manuscript action |
| --- | --- | --- |
| [Look, Read and Enrich, K-CAP 2019](https://doi.org/10.1145/3360901.3364420) | Learning figure–caption correspondence from scientific publications, including enriched representations. | Added: figure–caption learning is established prior work, not our invention. Published metadata verified against Crossref and the official K-CAP accepted-paper list. |
| [MedICaT, Findings EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.191/) | Medical figures, captions, textual references and alignment. | Added: scientific image–text alignment precedent; different domain and target from citation-relation classification. |
| [SPECTER, ACL 2020](https://aclanthology.org/2020.acl-main.207/) | Citation-informed scientific document representation learning. | Added: graph-derived supervision precedent; document text embeddings differ from our figure/caption classification. |
| [SciCap, Findings EMNLP 2021](https://aclanthology.org/2021.findings-emnlp.277/) | Scientific figure captioning data and baselines. | Added: public figure/text supervision context; caption generation scores are not AstroCLIMB baselines. |
| [SciOL and MuLMS-Img, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Tarsi_SciOL_and_MuLMS-Img_Introducing_a_Large-Scale_Multimodal_Scientific_Dataset_and_WACV_2024_paper.html) | Large-scale scientific multimodal pretraining and image–text evaluation. | Added: direct precedent for scientific-domain adaptation. Our contribution is the AstroCLIMB recipe and empirical evaluation, not general multimodal pretraining. |
| [TRACS overview, WASP 2025](https://aclanthology.org/2025.wasp-main.2/) | Earlier astronomy shared-task evaluation. | Added: venue/task history. Telescope references and categorization are different prediction targets; no numerical cross-task ranking is justified. |
| [SciCap+, SDU at AAAI 2023](https://ceur-ws.org/Vol-3656/paper13.pdf) | Figure captioning augmented with OCR and textual context. | Reviewed; optional future-work context. Does not establish that OCR improves our relation task or authorize additional paper text at inference. |
| [SciCap Challenge analysis, TACL 2026](https://aclanthology.org/2026.tacl-1.12/) | Scientific figure captioning with multimodal models and human evaluation. | Reviewed; optional broader context. Caption quality and four-way relation accuracy are different outcomes. |

Exact-name searches found the current task announcements, dataset and competition,
but no earlier published AstroCLIMB system paper. This is a search result, not
proof of absence. Unrelated games with the same name were excluded.

The TRACS overview's Table 6 describes micro averaging, whereas the historical
shared-task website describes macro-F1. We do not resolve that source discrepancy
by guessing, and do not import its scores into AstroCLIMB comparisons.

## Response to the internal critical review

See `critical-review.md` for the independent agent's original findings.

1. **Exposure disclosure addressed:** the manuscript now explicitly states prior
   released scoring-information and structural-shortcut exposure, exclusion of
   affected branches, and the distinction between selected-model inputs and
   project-level blindness.
2. **RL interpretation addressed:** GSPO512 exceeds the displayed answer-SFT
   reference but does not beat the classifier. No matched 512-update SFT control
   establishes a causal gain. The short matched control and long comparison are
   distinguished.
3. **Research positioning addressed:** six published precedents added, with
   publisher BibTeX for ACL papers and verified venue metadata for K-CAP/WACV.
4. **Split/exclusion scope addressed:** inherited group construction was not
   independently reconstructed; label-free test content participates in exclusion.
   Unknown foundation-model pretraining exposure is now explicit.
5. **Selection limitations retained:** single seeds, repeated development
   selection, unmatched gold-only reference, and differing selected gold epochs
   remain disclosed. No new independent evaluation is claimed.
6. **Availability addressed:** source/configuration/split preparation is separated
   from incomplete public usable-checkpoint release and clean GPU reproduction.
7. **Class-level analysis deferred:** no new per-class statistic or error example
   was invented. Aggregate and modality results remain the verified analysis.
8. **Terminology addressed:** citation relatedness is explicitly an edge in either
   direction. The existing workshop-resource citation remains correctly scoped;
   a separate competition author record was not inserted without a retrievable
   primary citation record during this pass.

Remaining scientific gaps are matched gold-only and long-SFT controls, multiple
seeds, independent evaluation, and richer verified class-level error analysis.
These limit claims; the wrap-up does not restart experiments to fill them.

## Validation

Both anonymous and author PDFs were rebuilt. Content and acknowledgments fit
within four pages; references occupy pages five and six. Bibliography keys were
checked for duplicates and unresolved citations, and changed layouts inspected.
Training code and recorded performance artifacts were not modified.

Layout inspection also caught the installed lineno v5.7 two-column regression.
The build now pins upstream v5.9; review numbers were visually verified in the
outer margins. See `paper/vendor/LINENO_SOURCE.md`.
