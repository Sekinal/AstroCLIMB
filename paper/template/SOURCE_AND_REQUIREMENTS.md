# Official WASP 2026 / AstroCLIMB template and requirements

Retrieved 2026-09-14. This directory contains the unmodified official ACL template checkout in `acl-style-files/`. Start with `acl_latex.tex` (pdfLaTeX) or `acl_lualatex.tex` (LuaLaTeX/XeLaTeX). Review mode is `\usepackage[review]{acl}`; it anonymizes and adds line numbers. Do not modify acl.sty or conference layout. The upstream example is documentation, not a completed system paper. No compilation was attempted.

## Provenance

- Official venue website source: https://github.com/adsabs/WIESP/tree/20be856329f2f6c5f53d7d1b6928f3cb62336b53/2026 . `_config.yml` binds repository adsabs/WIESP to https://ui.adsabs.harvard.edu/ with baseurl /WIESP. The rendered 2026 CFP/shared-task pages return an anti-bot interstitial, so the official static source was inspected instead.
- CFP: https://github.com/adsabs/WIESP/blob/20be856329f2f6c5f53d7d1b6928f3cb62336b53/2026/call_for_papers.md
- Shared task: https://github.com/adsabs/WIESP/blob/20be856329f2f6c5f53d7d1b6928f3cb62336b53/2026/shared_task.md
- Both explicitly link the ACL template: https://github.com/acl-org/acl-style-files . Local checkout commit d5adc823ff0f80f98c80405ca0ab66c68e684409.
- Submission portal: https://openreview.net/group?id=aclweb.org/AACL-IJCNLP/2026/Workshop/WASP

## Verified instructions and scope

The template is ACL, not CEUR. CFP says shared-task system descriptions receive light peer review and accepted papers enter WASP proceedings in ACL Anthology. General long/short submissions are double-blind: long papers at most8 content pages, short at most4, references unlimited; final versions allow one additional content page. Anonymity, preprints and simultaneous submissions follow AACL2026 policy.

Unresolved: neither the shared-task page nor CFP provides a separate system-description page cap or explicit statement that system-description review is double-blind. Therefore the general8/4 limits and anonymous review mode are a conservative working format, not a verified system-track-specific rule. OpenReview's public rendered page exposes no form fields in this inspection; mandatory declarations, exact track choice, supplements, and any track-specific anonymity exceptions remain unverified. Do not claim appendices or limitations are exempt from limits based solely on this CFP. Shared-task instructions also say to join Kaggle and register for AACL-IJCNLP; exact attendance/registration conditions were not independently resolved.

## Current official dates and conflict

Updated CFP AND2026 index explicitly strike old dates and replace them:

- AstroCLIMB system outputs: September20,2026.
- System papers and workshop papers: September21,2026.
- Acceptance notification: October5,2026.
- Camera-ready: October12,2026.
- Workshop: November9–10,2026.

Submission deadlines are23:59 UTC−12(AoE). Paper deadline corresponds to September22,11:59UTC /05:59Mexico City. System-output deadline corresponds to September21,11:59UTC /05:59Mexico City. The shared_task.md timeline is stale and still lists September13/14; updated CFP/index supersede it on the official site source. Kaggle enforcement and the submission portal deadline were not verified here; do not silently extend compute spending based solely on this finding.

## License documentation

The upstream ACL repository at this commit contains NO top-level LICENSE/COPYING file, and no blanket repository license was identified in README. Do not invent MIT/Apache/CC terms. The repository expressly distributes these files as the official author template. `acl_natbib.bst` retains its own copyright notices and permission terms (LaTeX Project Public License version1 or later; see its header). All upstream files and notices were preserved unchanged. Source website snapshots are provenance/reference copies; no additional license is asserted over them. Check component-specific notices before redistribution beyond the official-template use.

`MANIFEST.json` records every retrieved non-.git file size and SHA256. No private reasoning material was accessed.
