# Development error analysis

`development-errors.json` describes the already selected classifier, without
training, retuning, test labels, or inspecting the reserved historical fold.
The pooled macro-F1 exactly reproduces 0.7562033665434074. Class-level analysis
is post hoc on repeatedly selected DEV; it is not independent validation.

Run `uv run python analysis/development_errors.py --manifest /path/to/inputs.json
--out /path/to/results.json` (as one shell command). The JSON input manifest has
three objects, `cxi`, `ixi`, `cxc`, each containing local paths `labels` and
`probabilities`; `ixi` also requires `reversed_probabilities`. Labels need
`pair_id,y`; probability files need `pair_id` and the four `p_<canonical class>`
columns. Use DEV inputs only. Exact saved artifacts are not distributed here.

The script checks unique and matching IDs, expected modality counts, disjoint
slices, valid probabilities, absent-class masking, and the exact pooled score.
The output records input SHA-256 hashes and aggregate counts; no row-level
labels, identifiers, or private filesystem paths are published.

The related-paper class has the lowest F1 in each modality. In pooled DEV,
same-paper versus related-paper confusions account for 281/549 errors (51.2%).
This identifies a useful research target but does not establish why those cases
fail. Missing citation edges, ambiguous scientific content, and model limitations
cannot be separated by this confusion matrix alone.

## What would strengthen the scientific claim next?

- A matched gold-only CXI control with the same schedule, selection opportunities,
  and inference contract would isolate the benefit of public adaptation better.
- Compare balanced synthetic supervision against a controlled alternative at
  equal exposure; vary one factor at a time and use multiple seeds.
- A matched-duration generative SFT control is needed to attribute the long-RL
  comparison to RL rather than extra training or a different context pool.
- A newly collected, independently labeled paper-disjoint evaluation, inaccessible
  until recipe freeze, would test generalization more credibly. Existing project
  exposure cannot be undone by renaming or reshuffling an old partition.
- Paper-component paired resampling could describe sampling variation of fixed
  saved predictions, but would not remove development-selection bias or measure
  training-seed variability. It is not a substitute for new evaluation.

These are proposed future experiments, not completed work or promised gains.
