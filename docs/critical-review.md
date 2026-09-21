# Internal critical review of the AstroCLIMB manuscript

**September 21 update:** the previously missing matched gold-only comparison
is now complete for three gold-stage seeds, conditional on one fixed public
parent. The mean selected-checkpoint CXI DEV gain is 0.06738; all fresh exports
and replay checks pass. See [the final handoff](final-analysis-handoff.md).
The historical review below records the earlier evidence state; its independent
evaluation and long-SFT-control concerns remain unresolved.

Reviewed 2026-09-14, at the author's request. This is an author-side review of a
public-facing draft, not a confidential external review. References to manuscript
lines identify the version reviewed before the current revision. No additional
model runs, private reasoning, audit labels, or test labels were accessed.

## Overall assessment

The draft is a credible **shared-task system report**, with a useful selected
recipe and unusually explicit engineering evidence. It is not yet persuasive as
a new-method paper or as a general result against reinforcement learning. Its
main weaknesses are incomplete disclosure in the manuscript relative to the
repository, insufficient positioning in scientific-document research, and the
limited independence of the evaluation. Those weaknesses should be addressed
through clear scope and reporting; another rushed training run is not required.

## Prioritized findings

### 1. High: disclose the prior scoring-information exposure plainly in the paper

**Evidence:** `paper/sections.tex:37` says that earlier structurally derived
submissions and exposed synthetic representations are excluded. In contrast,
`docs/history-and-limitations.md:5` explicitly records historical inspection of
released scoring information and a structural shortcut, and states that the
benchmark was not wholly unseen throughout the project. `docs/data.md:24–28`
also warns that historical bundle extraction could include a scoring key.

The manuscript wording is insufficient for a reader who only receives the PDF.
Excluding affected model branches is valuable, but does not erase project-level
human/agent exposure. This does **not** establish that the selected content
classifier trained on test labels; it establishes a disclosure distinction that
the draft needs to preserve.

**Suggested fix:** include one direct sentence, such as: “Earlier project
exploration exposed released scoring information and a structural shortcut;
affected submissions and model branches are excluded, but the benchmark cannot
be regarded as wholly unseen at the project level.” Retain the narrower,
evidence-backed statement about the submitted model's training inputs. Avoid
unqualified “leakage-free” language in the abstract, cover text, or repository.

### 2. High: distinguish beating the classifier from improvement within RL

**Evidence:** `paper/sections.tex:78–89` and `paper/figures/results.json:10–16`.
The figure reports answer SFT at 0.74630, GSPO32 at 0.74494, and GSPO512 at
0.75115, compared with the classifier's 0.75620. Thus the displayed long RL
result is higher than the answer-SFT comparator, although it does not displace
the classifier. The 32/512 branches differ in context pool as well as length.
The text and caption already acknowledge several of these boundaries.

**Suggested fix:** explicitly state that the long generative RL branch improves
over the displayed answer-SFT reference but remains below the selected
classifier; do not infer a clean causal RL gain without verifying parent and
evaluation equivalence. Keep “did not improve the selected system,” rather than
“RL does not work.” The matched SFT control supports the **short** comparison;
there is no displayed matched-exposure 512-update SFT control. Identify the
comparison scopes in one sentence or a small experiment manifest.

This is a matter of fair interpretation, not a request to suppress negative
results. Switching from a linear classification head to generated answers
changes both optimization and the output interface. These experiments answer a
practical replacement question more directly than they isolate RL's causal
effect on an otherwise identical classifier.

### 3. Medium-high: the manuscript is missing its closest research positioning

**Evidence:** `paper/references.bib` and `paper/sections.tex:7–9,88–89` contain
many relevant RL/software citations, but no substantive scientific-figure or
citation-supervision related-work paragraph. New algorithmic novelty is not
claimed, which is appropriate, but the reader still needs to understand what
the system adds relative to the established task families.

Primary published work that directly helps position this paper:

- **SPECTER (ACL 2020):** learns scientific document representations from
  citation-informed training. It is relevant precedent for graph-derived
  supervision, while its title/abstract document embeddings differ from this
  work's figure/caption relation classifiers. [Publisher record](https://aclanthology.org/2020.acl-main.207/)
- **MedICaT (Findings EMNLP 2020):** studies medical figure/text alignment and
  image-text matching, including compound-figure structure. It establishes
  prior scientific figure–text matching, but not AstroCLIMB's cross-paper
  citation-relation task. [Publisher record](https://aclanthology.org/2020.findings-emnlp.191/)
- **SciCap (Findings EMNLP 2021):** provides scientific figure/caption data and
  caption-generation baselines. It is relevant to public figure supervision,
  although caption generation is a different target from relation
  classification. [Publisher record](https://aclanthology.org/2021.findings-emnlp.277/)
- A recent **TACL 2026 SciCap Challenge analysis** evaluates scientific figure
  captioning with multimodal models. This is optional context rather than a
  required direct baseline. [Publisher record](https://aclanthology.org/2026.tacl-1.12/)

**Suggested fix:** add a concise paragraph linking these task families and
stating that the contribution is an AstroCLIMB system study of public
relation-supervision transfer, modality specialization, and failed replacements.
Do not describe these works as direct score competitors, and do not imply they
were historical inspirations unless the historical record actually says so.
Prior shared-task TRACS systems are organizational precedent, but are less
direct scientific grounding than figure/text alignment and citation learning.

### 4. Medium: make the split and label-free transductive boundary explicit

**Evidence:** `paper/sections.tex:14,35–40` describes a fixed paper-component
partition and protected boundaries. `docs/data.md:29–34` explains that the
2,616-group split was inherited and supplied, not independently reconstructed.
`docs/data.md:101` explicitly uses all gold **and test content**, without labels,
for exclusion.

**Suggested fix:** say that the inherited component assignment is released and
that its original construction was not independently reconstructed. Also name
label-free test content as an exclusion input. This use does not automatically
constitute label leakage, but readers should not mistake the workflow for a
strictly inductive pipeline that never accesses test objects. Retain the
incomplete identity-recovery limitation, and mention unknown foundation-model
pretraining exposure if space permits.

### 5. Medium: the protocol supports selection, not statistical superiority

**Evidence:** `paper/sections.tex:40,63,68,93` already states repeated
development selection, one seed, and lack of intervals. These are strengths of
the current disclosure. There is no matched gold-only control for the central
public-adaptation claim, and no untouched local test result. The 8k endpoint's
best gold epoch differs from 2k/4k, despite a common three-epoch search schedule.

**Suggested fix:** preserve the existing modest wording. Call 4k the best of
the **tested selected endpoints**, not an intrinsic adaptation optimum. The
observed 2k-to-4k difference is a useful result; it does not isolate synthetic
data quality from exposure, prove public adaptation beats a matched gold-only
run, or establish a scaling law. If adding uncertainty later, resample paper
components rather than pretend pairs are independent, and disclose that this
still does not correct repeated-selection bias.

No additional experiment is necessary for an honest system paper; a stronger
causal claim would require matched controls and new independent evidence.

### 6. Medium: reproducibility claims in the PDF need the repository's caveats

**Evidence:** `paper/sections.tex:58` describes strong preserved hashes and
replay checks. `docs/reproduction.md:7,169,181–183` states that the source
release has not undergone clean GPU retraining, pretrained archives have no
public URLs, and private replay probes are omitted from staged weight archives.
`docs/release-validation.md` verifies CPU/package/document checks, not a fresh
GPU run or corpus equality.

**Suggested fix:** add one availability sentence in the manuscript stating
that source/configuration and split artifacts are prepared, while an
independent end-to-end reproduction and public usable checkpoint release have
not been completed. Do not imply that byte hashes make absent artifacts
available. Source release readiness and pretrained-model usability are distinct.

Minor methods details are recoverable from the code but missing from the paper:
AdamW weight decay 0.01 (`scripts/vlm_cxi_peft.py:276–281`), gradient clipping
1.0, capped 3% warmup, and a cosine multiplier with a 0.05 floor
(`scripts/vlm_cxi_peft.py:435–440`). Put exact settings in a linked configuration
table if manuscript space is tight. “Warmup/cosine” alone is not the exact
scheduler used.

### 7. Medium: include task-relevant error analysis if already available

**Evidence:** `paper/sections.tex:72` emphasizes modality scores; the scientific
distinction between same-paper, citation-related, and unrelated cases remains
largely qualitative. A high SF score can obscure a weak REL class in an overall
summary. The whole-task macro-F1 is correctly defined, so this is a scientific
interpretability opportunity rather than a metric error.

**Suggested fix:** if verified per-class scores or confusion counts already
exist, add a small class-level table or one sentence identifying the main
remaining confusion. Prefer that to more generic RL motivation. Do not invent
visual examples or inspect closed evaluation labels to produce the analysis.

### 8. Minor: bibliographic and terminology precision

The current AstroCLIMB bibliography entry names “WASP Organizers,” whereas the
official Kaggle page provides a citation to **Felix Grezes, 2026**. Retain the
workshop resource if desired, but add the competition/dataset citation rather
than substituting invented authorship. The official task also defines a
**symmetric citation relation**, not generic semantic relatedness; spell that
out near the label definitions. [Official task and citation](https://www.kaggle.com/competitions/astroclimb/overview)

Many bibliography records use legitimate arXiv editions. A compilation without
undefined citations establishes syntactic completeness, not publication-status
accuracy or that all inspirations are credited. Prefer publisher versions for
established papers when verified, without adding citations merely to increase
the count. The AI-assistance disclosure is neutral and compatible with a
sole-author manuscript; these tools are not listed as authors.

## Strengths worth preserving

- Concrete data counts, separate modality recipes, fixed class order, original
  scheduler horizons, and selection endpoints make the system understandable.
- The public 2k/4k/8k comparison is unusually careful about inference context and
  avoiding an incorrectly compressed schedule.
- IXI symmetry is implemented as probability averaging, not label voting, and
  the unsuccessful CXC analogue is disclosed.
- Negative RL and mining outcomes are retained, with TRAIN-monitor exposure
  and single-seed limitations explicitly acknowledged.
- The repository is candid about unavailable artifacts, original split
  reconstruction limits, and the difference between CPU validation and GPU
  reproduction. Bringing those same boundaries into the PDF will improve it.

## Recommended revision order

First fix the project-level exposure disclosure and the interpretation of the
long RL comparison. Then add concise scientific-figure/citation related work,
the label-free test-exclusion boundary, and the availability limitation. Add
class-level error information only from existing verified results. Do not
restart the experimental campaign merely to satisfy speculative reviewer
requests; the appropriate submission claim is a transparent system report.
