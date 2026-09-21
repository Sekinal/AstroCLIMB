"""Cheap TRAIN-only diagnostics for exact prompt/image supervision.

Reads a frozen public TRAIN file and a gold TRAIN file (local parquet or CSV)
and reports, per dataset, whether the canonical effective TRAIN input has
contradictory labels, plus caption truncation exposure. Descriptive only:
no training, no selection, no val/DEV/test access.

Effective input identity is (img_bytes_md5, exact case-preserving
caption[:2400]). Full input identity is (img_bytes_md5, full caption).
For each identity kind this reports duplicate-group counts, contradictory
group counts (>1 distinct label), affected rows, and
sum(group_size - max_class_count), which is a lower bound on errors for a
deterministic classifier that maps that EXACT identity on these TRAIN
examples. That bound is not a Bayes error and does not diagnose
semantically incorrect labels.

Public TRAIN is expected to carry cap_norm, img_bytes_md5, y. Gold TRAIN
carries img_bytes_md5, y, and cap_text when available (preferred, matching
the actual trainer) else cap_norm. Only the two files passed on the command
line are read. No public val, DEV, test/audit, or labels from other files.

Output is aggregate counts, caption length quantiles, and input SHA-256
hashes. No raw captions, per-row labels, IDs, raw MD5 lists, or private
filesystem paths are written.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

CAP_LIMIT = 2400
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
QUANTILE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

SCOPE = (
    "TRAIN-only descriptive diagnostics for exact effective input "
    "(image byte MD5, case-preserving caption[:2400]) label conflicts and "
    "caption truncation exposure; no training, selection, or val/DEV/test use"
)
CAVEATS = [
    "Lower-bound errors are sum(group_size - max_class_count) for a deterministic "
    "mapping of that EXACT identity on these TRAIN rows only; not a Bayes error.",
    "This does not diagnose semantically incorrect labels; label-pure groups may "
    "still be wrong.",
    "Byte-MD5 identity misses visual duplicates (same pixels, different bytes).",
    "A downstream image processor may collapse additional images, so this is a "
    "narrow diagnostic on byte identity only.",
    "Effective and full conflict totals are reported independently; an effective "
    "conflict is attributed to truncation only when a verified new merger is "
    "counted (constituent full identities each label-pure but differing).",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_frame(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"{Path(path).name}: unsupported suffix {suffix!r}; use .parquet or .csv")


def pick_caption_column(frame, kind, basename):
    cols = set(frame.columns)
    if kind == "public":
        if "cap_norm" not in cols:
            raise ValueError(f"{basename}: public TRAIN must contain cap_norm")
        return "cap_norm"
    if "cap_text" in cols:
        return "cap_text"
    if "cap_norm" in cols:
        return "cap_norm"
    raise ValueError(f"{basename}: gold TRAIN must contain cap_text or cap_norm")


def validate_columns(frame, caption_col, basename):
    for col in (caption_col, "img_bytes_md5", "y"):
        if col not in frame.columns:
            raise ValueError(f"{basename}: missing required column {col!r}")
    if frame.height == 0:
        raise ValueError(f"{basename}: empty frame")


def extract_validated_lists(frame, caption_col, basename):
    cap = frame[caption_col]
    md5 = frame["img_bytes_md5"]
    y = frame["y"]
    if cap.is_null().sum() > 0:
        raise ValueError(f"{basename}: null caption in {caption_col!r}")
    if md5.is_null().sum() > 0:
        raise ValueError(f"{basename}: null img_bytes_md5")
    if y.is_null().sum() > 0:
        raise ValueError(f"{basename}: null label in 'y'")
    captions = cap.to_list()
    md5s = md5.to_list()
    labels = y.to_list()
    for i, c in enumerate(captions):
        if not isinstance(c, str):
            raise ValueError(f"{basename}: non-string caption at row {i}")
    for i, m in enumerate(md5s):
        if not isinstance(m, str) or MD5_RE.match(m) is None:
            raise ValueError(f"{basename}: corrupt img_bytes_md5 at row {i}")
    for i, v in enumerate(labels):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > 3:
            raise ValueError(f"{basename}: invalid label at row {i}; expected int 0..3")
    return captions, md5s, labels


def group_stats(keys, labels):
    groups = defaultdict(lambda: [0, 0, 0, 0])
    for k, lab in zip(keys, labels):
        groups[k][lab] += 1
    n_distinct = len(groups)
    n_dup = 0
    n_contra = 0
    n_contra_rows = 0
    lower = 0
    for counts in groups.values():
        size = counts[0] + counts[1] + counts[2] + counts[3]
        distinct = sum(1 for c in counts if c > 0)
        if size > 1:
            n_dup += 1
        if distinct > 1:
            n_contra += 1
            n_contra_rows += size
        lower += size - max(counts)
    return {
        "n_distinct": n_distinct,
        "n_duplicate_groups": n_dup,
        "n_contradictory_groups": n_contra,
        "n_contradictory_rows": n_contra_rows,
        "lower_bound_errors": lower,
    }, groups


def diagnose_lists(captions, md5s, labels):
    n = len(labels)
    counts = [0, 0, 0, 0]
    for v in labels:
        counts[v] += 1
    lengths = [len(c) for c in captions]
    values = np.quantile(
        np.asarray(lengths, dtype=np.float64), np.asarray(QUANTILE_LEVELS, dtype=np.float64)
    ).tolist()
    n_truncated = sum(1 for L in lengths if L > CAP_LIMIT)
    effective = [c[:CAP_LIMIT] for c in captions]
    full_keys = list(zip(md5s, captions))
    eff_keys = list(zip(md5s, effective))
    full_stats, full_groups = group_stats(full_keys, labels)
    eff_stats, eff_groups = group_stats(eff_keys, labels)
    # Verified truncation mergers: effective contradictory groups whose
    # constituent full identities are each label-pure but differ.
    full_labelsets = {}
    for k, counts4 in full_groups.items():
        full_labelsets[k] = frozenset(j for j, c in enumerate(counts4) if c > 0)
    eff_to_fulls = defaultdict(set)
    for fk, ek in zip(full_keys, eff_keys):
        eff_to_fulls[ek].add(fk)
    n_merging = 0
    n_new_verified = 0
    n_new_rows = 0
    for ek, counts4 in eff_groups.items():
        size = counts4[0] + counts4[1] + counts4[2] + counts4[3]
        distinct = sum(1 for c in counts4 if c > 0)
        if distinct <= 1:
            continue
        fulls = eff_to_fulls[ek]
        if len(fulls) <= 1:
            continue
        n_merging += 1
        sets = [full_labelsets[fk] for fk in fulls]
        if all(len(s) == 1 for s in sets) and len(set().union(*sets)) > 1:
            n_new_verified += 1
            n_new_rows += size
    return {
        "n_rows": n,
        "counts_per_class": counts,
        "caption_len_quantiles": {"levels": list(QUANTILE_LEVELS), "values": [float(v) for v in values]},
        "n_truncated": n_truncated,
        "full": full_stats,
        "effective": eff_stats,
        "truncation_merger": {
            "n_effective_contradictory_groups_merging_distinct_full": n_merging,
            "n_new_contradictory_groups_from_truncation_verified": n_new_verified,
            "n_new_rows_affected_verified": n_new_rows,
        },
    }


def summarize_file(path, kind):
    basename = Path(path).name
    frame = read_frame(path)
    caption_col = pick_caption_column(frame, kind, basename)
    validate_columns(frame, caption_col, basename)
    captions, md5s, labels = extract_validated_lists(frame, caption_col, basename)
    result = diagnose_lists(captions, md5s, labels)
    result["caption_column_used"] = caption_col
    return result


def build_receipt(public_train, gold_train):
    pub = summarize_file(str(public_train), "public")
    gold = summarize_file(str(gold_train), "gold")
    return {
        "scope": SCOPE,
        "cap_limit": CAP_LIMIT,
        "input_sha256": {
            "public_train": sha256_file(str(public_train)),
            "gold_train": sha256_file(str(gold_train)),
        },
        "datasets": {"public_train": pub, "gold_train": gold},
        "caveats": list(CAVEATS),
        "note": (
            "Effective and full conflict totals are independent; truncation is "
            "implicated only for the verified new-merger counts."
        ),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public-train", type=Path, required=True)
    ap.add_argument("--gold-train", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    receipt = build_receipt(args.public_train, args.gold_train)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    main()
