"""Prune inherited gold paper groups using label-blind public duplicate bridges."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import polars as pl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--test-objects", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    canonical = pl.read_parquet(args.canonical)
    groups = dict(canonical.select("pair_id", "group_id").iter_rows())
    columns = ["pair_id", "key_1", "key_2"]
    train = pl.read_parquet(args.gold / "gold_train_manifest.parquet", columns=columns)
    dev = pl.read_parquet(args.gold / "gold_dev_manifest.parquet", columns=columns)
    audit = pl.read_parquet(args.gold / "gold_audit_manifest.parquet", columns=columns)
    all_gold = set(pl.read_parquet(args.gold / "objects_train.parquet", columns=["obj_key"])["obj_key"])
    test = set(pl.read_parquet(args.test_objects, columns=["obj_key"])["obj_key"])
    keys = lambda frame: set(frame["key_1"]) | set(frame["key_2"])
    train_keys, dev_keys = keys(train), keys(dev)
    # Include orphan gold objects belonging to already-pruned heldout groups.
    train_forbidden = (all_gold - train_keys) | test
    dev_forbidden = (all_gold - train_keys - dev_keys) | test
    parent = {}
    def root(k):
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[b] = a
    for frame in (train, dev, audit):
        for pid, a, b in frame.iter_rows():
            union("group:" + groups[pid], "obj:" + a)
            union("group:" + groups[pid], "obj:" + b)
    candidates = pl.read_parquet(args.candidates, columns=["obj_key", "public_component"]).unique()
    for obj, component in candidates.iter_rows():
        union("obj:" + obj, "public:" + str(component))
    blocked_train = {root("obj:" + k) for k in train_forbidden}
    blocked_dev = {root("obj:" + k) for k in dev_forbidden}
    allow = {
        "train": {pid for pid in train["pair_id"] if root("group:" + groups[pid]) not in blocked_train},
        "dev": {pid for pid in dev["pair_id"] if root("group:" + groups[pid]) not in blocked_dev},
        "audit": set(audit["pair_id"]),
    }
    assert allow["train"] and allow["dev"], "Public closure leaves no supervised split"
    counts = {}
    for split in ("train", "dev", "audit"):
        for name in (f"gold_{split}_manifest.parquet", f"cxi_{split}.parquet", f"ixi_{split}.parquet", f"cxc_{split}.parquet"):
            frame = pl.read_parquet(args.gold / name).filter(pl.col("pair_id").is_in(allow[split]))
            frame.write_parquet(args.out / name)
            counts[name] = frame.height
    (args.out / "imgcache").symlink_to((args.gold / "imgcache").resolve())
    (args.out / "objects_train.parquet").symlink_to((args.gold / "objects_train.parquet").resolve())
    (args.out / "train.parquet").symlink_to("cxi_train.parquet")
    (args.out / "val.parquet").symlink_to("cxi_dev.parquet")
    report = {"counts": counts, "removed_train": train.height-len(allow["train"]),
              "removed_dev": dev.height-len(allow["dev"]),
              "train_forbidden_objects": len(train_forbidden), "dev_forbidden_objects": len(dev_forbidden),
              "public_candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "policy": "Transitive union of inherited paper groups and full public duplicate components; test objects are exclusion-only",
              "limitations": "Fold0 historically used for selection; public DOI matching inherits supplied candidate map coverage"}
    (args.out / "RECOVERY_PUBLIC_CLOSURE.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
