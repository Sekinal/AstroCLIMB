# Citation audit — September 21, 2026

Three independent Muse Spark 1.3 Contributor workers checked research-paper
metadata, software/model resources, and claim-to-source support. The coordinator
reviewed their evidence and resolved findings against primary sources.

All 26 cited references exist. The two uncited planning references (Dr. GRPO and
STaR) also exist; they remain in the source bibliography but are absent from the
rendered references. No missing BibTeX key or invented work was found.

## Changes and resolved findings

- Cite the dedicated PP-OCRv6 paper with its 16 named authors instead of only the
  model collection: https://arxiv.org/abs/2606.13108.
- Add SciOL's DOI and use its IEEE/Crossref pagination (4548–4559). CVF's open-access
  copy uses 4560–4571; the old entry matched that copy, so it was not a nonexistent
  or unrelated reference. DOI: https://doi.org/10.1109/WACV57701.2024.00450.
- Match the publisher's title separator for Look, Read and Enrich; its author
  preprint independently supports the figure–caption correspondence claim:
  https://arxiv.org/abs/1909.09070.
- Protect LoRA, DAPO and LLM capitalization in BibTeX.
- Retain both Yang Yue authors: the paper explicitly identifies two people with
  the same English name, not an accidental duplicate.
- Retain Matplotlib: it is already cited in `paper/main.tex`, not `sections.tex`.
  The resource reviewer searched too narrow a source file for that finding.
- Retain official software citations for PEFT and TRL, and the exact Qwen model
  revision. BGE's official model citation is accompanied by C-Pack as recommended.

## Claim support and scope

Related-work statements match the original abstracts and descriptions. The
coordinator additionally checked DAPO section 3.2: unanimous-reward groups produce
no policy-gradient signal and dynamic sampling filters all-correct/all-wrong
prompts. GSPO is cited as the optimizer, while DAPO is credited only for sampling
inspiration. No claim is made that our trial reproduces all of DAPO.

Inspirational citations do not establish that those algorithms were implemented.
The support review's broad inventory accidentally included Dr. GRPO and STaR
among used methods; that wording was rejected. Neither is claimed as implemented.
Our numerical results remain our own measurements, not results attributed to the
cited papers. This audit does not independently replicate those experiments.

Some sites gate browser access. Official API records, bibliographic exports,
author preprints and publisher-deposited DOI metadata supplied alternate primary
evidence. In particular, the official AstroCLIMB page is intermittently gated;
the task is also corroborated by its official dataset and competition pages.

See `analysis/citation-audit.json` for the complete per-reference source catalog.
This is documented verification, not a claim that all possible bibliographic or
interpretive errors are impossible. Raw web downloads and worker sessions are not
redistributed in the source package.
