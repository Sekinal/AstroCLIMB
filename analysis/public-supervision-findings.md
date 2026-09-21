# Public supervision findings (TRAIN-only, exact-input diagnostics)

Source: `analysis/public-supervision-diagnostics.json`, produced by
`analysis/public_supervision_diagnostics.py`. Descriptive only. No training,
no model selection, and no val/DEV/test use.

## Diagnosis definition

- Scope is TRAIN rows only: 80,000 public TRAIN rows and 2,230 gold TRAIN rows.
- Effective TRAIN input identity is `(img_bytes_md5, case-preserving caption[:2400])`.
  Full input identity is `(img_bytes_md5, full caption)`.
- Image identity is byte MD5, not perceptual or decoded-pixel identity.
- Caption handling is case-preserving with a hard `cap_limit = 2400` characters.
- Caption column actually used: `cap_norm` for public TRAIN; `cap_text` for gold
  TRAIN (preferred when present, matching the trainer; `cap_norm` is the fallback).
- Per identity kind, the script reports duplicate groups (size > 1),
  contradictory groups (> 1 distinct label), affected rows, and
  `sum(group_size - max_class_count)`, a lower bound on errors for a deterministic
  map of that exact identity on these TRAIN rows. That bound is not a Bayes error.
- Truncation is blamed only for verified new mergers: an effective contradictory
  group whose constituent full identities are each label-pure but differ.

## Result

Public TRAIN classes are exactly balanced (20,000 / 20,000 / 20,000 / 20,000).
Gold TRAIN classes are 580 / 570 / 534 / 546.

| Dataset | Rows | Distinct full / effective | Duplicate groups | Contradictory groups / rows | Lower-bound errors | Truncated (len > 2400) |
| --- | --- | --- | --- | --- | --- | --- |
| Public TRAIN | 80,000 | 80,000 / 80,000 | 0 / 0 | 0 / 0 | 0 / 0 | 19 (0.024%) |
| Gold TRAIN | 2,230 | 2,230 / 2,230 | 0 / 0 | 0 / 0 | 0 / 0 | 1 (0.045%) |

Full and effective totals are reported independently; both are zero for conflicts.
Verified truncation mergers are zero in both datasets:

- `n_effective_contradictory_groups_merging_distinct_full = 0`
- `n_new_contradictory_groups_from_truncation_verified = 0`
- `n_new_rows_affected_verified = 0`

Caption length quantiles (0.0 / 0.25 / 0.50 / 0.75 / 1.0) are
40.0 / 260.75 / 424.0 / 648.0 / 3227.0 for public TRAIN and
10.0 / 261.0 / 429.5 / 654.0 / 2571.0 for gold TRAIN, so truncation exposure is
rare in TRAIN under the 2400-character rule. Input SHA-256 hashes for the two
frozen TRAIN files are recorded in the JSON receipt.

## Interpretation

This is a narrow negative finding. It weakens two specific explanations for TRAIN
behavior: frequent TRAIN caption truncation, and exact-input label contradictions
under byte identity. There are no exact duplicate inputs at all in these TRAIN
inputs, hence no exact-input contradictions and no verified truncation-created
merger.

## What this does not establish

- Not semantic correctness: a label-pure exact-input group may still be wrong.
- Not graph or annotation completeness: missing relations or entities are untouched.
- Not absence of near duplicates: byte MD5 misses same-pixel/different-bytes
  images, paraphrased captions, and any additional collapsing by a downstream
  image processor. This diagnostic is byte identity only.
- Not Bayes error: the reported zero is a TRAIN exact-identity lower bound, not
  irreducible error on the task distribution.
- Not DEV truncation: TRAIN exposure (19/80,000 and 1/2,230) says nothing about
  DEV caption lengths or truncation there.

## Reproduce

Script CLI (placeholder paths only; both inputs are local `.parquet` or `.csv`):

```bash
python analysis/public_supervision_diagnostics.py \
  --public-train /path/to/public_train.parquet \
  --gold-train /path/to/gold_train.parquet \
  --out /path/to/public-supervision-diagnostics.json
```

The output contains aggregate counts, quantiles, and input hashes only; it writes
no raw captions, per-row labels, IDs, raw MD5 lists, or filesystem paths.
