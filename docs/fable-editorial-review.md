# Claude Fable 5.1 editorial review — September 21, 2026

The author requested a review and prose revision through Claude. A tool-free
Claude CLI run completed with response model identifier `claude-fable-5-1`.
It received the manuscript, reviewer clarifications, and citation-audit notes.
This was an editorial review, not independent experimental validation or an
additional bibliography audit.

## Findings

The review identified the three-seed matched gold-stage comparison as the main
positive result. It recommended foregrounding that comparison, explaining the
public graph supervision earlier, and separating fresh controls from the
historical submitted checkpoint. It emphasized unequal total compute, one
shared public parent, repeated development selection, prior project exposure,
and the missing long supervised control for GSPO512. These are limitations of
the experiments; rewriting does not resolve them.

The suggested next experiments were public-stage seed replication, a
compute-matched gold-only control, a long supervised control, and a manual
audit of synthetic negative labels. No new experiment was conducted for this
editorial pass. The review's informal publication assessment is not evidence
of acceptance or a substitute for venue peer review.

## Integration

The proposed prose retains the empirical claims and moves the main finding
ahead of exploratory results. The coordinator corrected several overstatements:
missing citation edges *can* create noisy negatives; finding no duplicate
matches does not prove independence; RL coverage papers informed evaluation
rather than predicting that generation would beat classification; unanimous
rewards yield zero group-relative advantage; the class-performance tradeoff
applies to 4k versus 8k, not all increases in public adaptation.

Fable revised the remaining sections while keeping the accepted abstract and
introduction unchanged. A file-based pass used Python to verify a reduction
from 2,281 to 2,049 whitespace words. The coordinator retained four factually
preferable sentences from earlier Fable drafts: citation-edge uncertainty,
artifact hashing/replay, selected/final prediction replay, and private probes.
The integrated sections contain 2,071 whitespace words. Claude Fable 5.1 is
credited specifically for manuscript review and prose revision.

Both PDFs compile with four content pages including acknowledgments and two
reference pages. A page break places references on page five; template fonts
and margins are unchanged. All twelve rendered pages were inspected. Numerical
token sets, cited keys, tables, and figure content are preserved relative to
the pre-review version. The submission abstract matches the manuscript.

The revised figure replaces the generative-score panel with paired gold-only
and public-initialized scores for all three seeds. It reads exact values from
`analysis/matched-gold-controls.json`; the historical duration panel remains
separate. Shared public parent, unequal compute, development selection, and
zoomed axes are explicit.
