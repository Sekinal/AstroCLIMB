"""Assemble the selected probability recipe into the submitted five-column CSV.

The template supplies ID order only. Its placeholder labels and optional Usage
column never enter prediction. Ties use the documented canonical class order.
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
PROBS = ["p_" + label for label in LABELS]


def unique_ids(frame, column):
    series = frame[column]
    if not series.dtype.is_integer() or series.dtype == pl.Boolean:
        raise ValueError(f"{column} must contain integer IDs")
    if not len(frame) or series.null_count() or series.n_unique() != len(frame):
        raise ValueError(f"{column}: empty, null or duplicate IDs")
    return series.to_list()


def probabilities(frame, task):
    if frame.columns != ["pair_id", *PROBS]:
        raise ValueError("Probability columns must have the exact canonical names and order")
    ids = unique_ids(frame, "pair_id")
    if any(not frame[c].dtype.is_numeric() or frame[c].null_count() for c in PROBS):
        raise ValueError("Probabilities must be nonnull numeric values")
    values = frame.select(PROBS).to_numpy().astype(np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    if task in ("ixi", "cxc") and (values[:, 0] != 0).any():
        raise ValueError(f"{task} must retain the zero same-figure probability mask")
    return dict(zip(ids, values))


def assemble(sample, cxi, ixi_original, ixi_reversed, cxc):
    if sample.columns not in (["id", *LABELS], ["id", *LABELS, "Usage"]):
        raise ValueError("Unexpected official template class schema/order")
    order = unique_ids(sample, "id")
    cx = probabilities(cxi, "cxi")
    ia = probabilities(ixi_original, "ixi")
    ib = probabilities(ixi_reversed, "ixi")
    cc = probabilities(cxc, "cxc")
    if ia.keys() != ib.keys():
        raise ValueError("IXI orientations must contain identical IDs")
    tasks = [set(cx), set(ia), set(cc)]
    if any(tasks[i] & tasks[j] for i in range(3) for j in range(i)):
        raise ValueError("Task IDs must be disjoint")
    if set.union(*tasks) != set(order):
        raise ValueError("Predictions must cover every template ID exactly once")
    combined = {**cx, **cc, **{key: 0.5 * ia[key] + 0.5 * ib[key] for key in ia}}
    indices = np.argmax(np.stack([combined[key] for key in order]), axis=1)
    # Emit the actual submitted schema; optional template Usage is discarded.
    return pl.DataFrame({"id": order, **{name: (indices == i).astype(np.int64) for i, name in enumerate(LABELS)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sample", "cxi", "ixi-original", "ixi-reversed", "cxc", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; choose a new output")
    sample = pl.read_csv(args.sample)
    result = assemble(sample, *(pl.read_parquet(path) for path in (args.cxi, args.ixi_original, args.ixi_reversed, args.cxc)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        result.write_csv(stream)


if __name__ == "__main__":
    main()
