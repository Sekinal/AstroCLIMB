# Submission handoff

The PDF and fields are prepared locally; no submission has been created.

## Deadline conflict: use the earlier portal deadline until resolved

Verified September 21, 2026 via the public OpenReview API. The current
[submission invitation](https://openreview.net/invitation?id=aclweb.org/AACL-IJCNLP/2026/Workshop/WASP/-/Submission)
has a due date of **September 22, 2026 at 03:59 UTC**, which is
**September 21 at 21:59 in Mexico City**. Its expiration is 30 minutes later;
do not treat that grace period as the submission deadline.

The official workshop CFP/index instead extends papers to September 21 AoE.
These sources conflict. The earlier portal date is the conservative operational
cutoff, not proof that organizers withdrew the extension. Recheck the portal
before submitting. The public form exposes no system-track selector, separate
system-paper page limit, or explicit system-paper anonymity exception.

## Ready-to-copy fields

**Title:** Public Scientific-Figure Adaptation for AstroCLIMB with a 4B Vision–Language Model

**Author:** Irving Ernesto Quezada Ramírez — Independent Researcher.
The form requires an existing OpenReview profile ID; name/email alone do not
satisfy its author field. Use your own profile associated with the intended
contact address, research@irvingernesto.com. No profile ID is guessed here.

**Keywords:** scientific figures, multimodal classification, astronomy,
parameter-efficient fine-tuning, reinforcement learning

**TL;DR:** Public scientific-figure adaptation improves CXI development macro-F1 by 0.0674 across three matched gold-stage seeds; the submitted modality-specific system scores 0.75149 on Kaggle.

**Abstract:**

We ask whether public scientific figures and captions improve a 4B vision–language model on AstroCLIMB. Supervised adaptation on graph-derived public pairs improves caption–image development macro-F1 in all three matched gold-training runs, with a mean gain of 0.0674. These runs share one public checkpoint. The submitted system combines separate modality classifiers and obtains a Kaggle public score of 0.75149. Increasing public adaptation from 2,000 to 4,000 updates improves paper-relation classification, but 8,000 updates bring no further gain. Neither answer generation nor the tested reinforcement-learning variants improve the selected classifier. Repeated development selection limits these findings to the evaluated setting.

**PDF:** `paper/build/main.pdf` is the conservative anonymous review version;
`paper/build/author.pdf` includes author/funding metadata. Both have at most
four content pages, with references continuing through page six. Use the
anonymous PDF unless the system track explicitly instructs otherwise.

## Form declarations for the author

The live form requires acknowledgment that author emails are shared with
Program Chairs and that accepted submissions and author names become public.
Its paper-license field currently offers CC BY 4.0. These declarations require
the author's review; recording their wording here does not submit or accept them.
The PDF field permits up to 50 MB. It is technically optional in the current
form schema, which does not override the workshop's paper requirement.

The AI-assistance disclosure and Lium acknowledgment are already in the paper;
the identifying funding statement is suppressed in review mode. The source
repository contains author metadata and is not an anonymous review supplement.

`submission-form-snapshot.json` preserves the relevant public form schema and
timestamps. Fresh matched CXI GPU runs are verified for three gold-stage seeds. Full data
reconstruction and independent end-to-end reproduction of all three submitted
modalities from the source release remain incomplete.

AI-policy check: see `docs/ai-policy-check.md`. The September 21 live form has
no separate AI-use field; the PDF acknowledgments disclose the scope of assistance.
