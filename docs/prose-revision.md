# Prose revision — September 21, 2026

This pass uses published empirical papers as references for exposition. It does
not copy their wording, reproduce an individual author's voice, or claim that
these readings motivated the earlier experiments.

## Editorial references

- Gururangan et al. (2020), [Don't Stop Pretraining: Adapt Language Models to
  Domains and Tasks](https://aclanthology.org/2020.acl-main.740/).
  The introduction establishes a specific uncertainty, and the experiments
  progressively narrow it. Comparison design precedes interpretation, and the
  discussion distinguishes results from proposed explanations. Applied here:
  connect the public-initialization control to the duration comparison rather
  than present them as unrelated entries in an experiment inventory.
- Pruksachatkun et al. (2020), [Intermediate-Task Transfer Learning with Pretrained
  Language Models: When and Why Does It Work?](https://aclanthology.org/2020.acl-main.467/).
  The paper separates evidence about performance from evidence about mechanisms.
  Applied here: report the positive matched comparison directly while retaining
  its fixed-public-parent and added-compute boundaries. Treat relation-class
  changes and rollout signal as observations, without claiming a demonstrated
  transfer mechanism or a general limit of reinforcement learning.

## Revision criteria

Each paragraph should develop one point: the comparison being made, the result,
and what that result permits us to conclude. Methods should explain the model
and training sequence before enumerating settings. The conclusion must state
the positive matched-control finding as well as the unsuccessful alternatives.
Use concrete subjects and verbs, and vary sentence length where the argument
calls for it. Preserve the author's first-person-plural convention.

All existing citation keys, numerical outcomes, method settings, and material
limitations must survive the edit. In particular, retain historical exposure,
repeated DEV selection, fixed public pretraining seed, additional compute,
TRAIN-monitor ancestor exposure, missing long-SFT control, and incomplete full
reproduction/checkpoint release. Do not imply a new Kaggle submission.

No detector scores guide this revision. Evaluation consists of factual comparison
against the prior manuscript, citation and build checks, and visual PDF review.

## Verification

A separate Pi/Muse review found no material factual drift or citation omissions.
It identified one ambiguity introduced by shortening the IXI prediction rule;
the final text restores the arithmetic mean of class probabilities explicitly.
All scientific numeric tokens and all 28 manuscript citation keys are retained.
Both PDF variants compile cleanly, with four content pages and six total, and
rendered pages were visually inspected. The submission abstract is synchronized.
