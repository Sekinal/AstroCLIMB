# Fresh source-archive validation

On September 14, 2026, source commit `056bee4` was archived and extracted
into a new temporary directory, outside the working repository. Archive SHA-256:
`56741cd7216640f10c111a8d8af166a623b2f388ee1c6955d9b3469c1d4f8e70`.
The following checks passed there:

- `uv sync --frozen --extra dev` created a fresh Python 3.12 environment.
- All eight portability and submission-assembly tests passed.
- `uv pip check` found compatible installed dependencies.
- All eight preserved training/export source hashes matched the recipe.
- All four data-preparation CLIs imported and displayed help successfully.
- `make -C paper all` rebuilt both PDFs using the included figure and template.
  Neither final log contained undefined citations/references, overfull boxes,
  or LaTeX errors.

This verifies the source package's CPU environment and document build on the
existing host. It does not independently test another operating system,
re-download the datasets, install the separate CUDA training environment,
retrain models, or verify GPU numerical equivalence. The earlier assembly
check reproduced all 50,000 cells of the saved 10,000-row submission from
its saved probabilities; those predictions are not included in the source ZIP.

Later documentation-only commits may change the ZIP hash. The commit and
hash above identify the source snapshot actually exercised by this check.
