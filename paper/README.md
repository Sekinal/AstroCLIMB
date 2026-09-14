# Manuscript

Build both PDFs with `make -C paper all` (TeX Live with latexmk, pdfLaTeX,
BibTeX, and the standard packages imported by `main.tex`).

- `build/main.pdf`: anonymous review draft; author and funding text suppressed.
- `build/author.pdf`: author version with Irving Ernesto Quezada Ramírez,
  Independent Researcher, and the Lium compute-grant acknowledgment.

The official ACL template is pinned and unmodified under `template/`.
See `template/SOURCE_AND_REQUIREMENTS.md` for the official sources,
extended dates, and the unresolved system-track-specific submission details.
The current draft has at most four content pages; references occupy pages five and six.
This fits the general short-paper content limit.

This is a technical draft, not a submitted paper. The public score is a
completed submission score, not a verified leaderboard rank. Development
results were used repeatedly for model selection. Larger-rank, larger-model,
final all-label refit, and further RL proposals must not be presented as
completed experiments.

The source repository includes identifying metadata and is not an anonymous
review supplement. Upload only the review PDF if the track requires anonymity.

The manuscript includes an AI-assistance disclosure. `inspiration-audit.md`
records attribution decisions; `inspiration.bib` is the wider research inventory,
while `references.bib` contains the selectively integrated manuscript entries.

Regenerate the two-panel results figure with `make -C paper figures`.
Its numeric inputs are in `figures/results.json`; the script exports vector
PDF/SVG and a PNG preview. It replaces the transfer and generative-result
tables, keeps CXI and whole-development metrics on separately labeled axes,
and shows single-seed selected scores without invented error bars.

See [submission-fields.md](submission-fields.md) for ready-to-copy fields and
the live portal deadline conflict, verified after the workshop-page audit.
