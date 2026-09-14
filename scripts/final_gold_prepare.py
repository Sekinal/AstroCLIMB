"""Recover a gold-only corpus from explicitly downloaded train.csv.

Never opens test inputs, public synthetic data, or solution files. Audit labels
are not exported. Preserved paper-strict folds are strengthened by current gold
content duplicate closure; this does not recover the missing public closure.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import polars as pl

from astroclimb.utils import image_hashes, normalize_caption

LABELS = ("same_figure", "same_paper", "related_papers", "unrelated_papers")


def cache_image(value, key, cache):
    raw = base64.b64decode(value, validate=True)
    obj = {"obj_key": key, "modality": "image", **image_hashes(raw)}
    path = Path(cache) / (obj["bytes_md5"] + ".img")
    if not path.exists():
        path.write_bytes(raw)
    return obj


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def content_components(objects):
    parent = {k: k for k in objects}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[b] = a

    first = {}
    bands = [defaultdict(set) for _ in range(9)]
    hashes = {}
    bounds = [(i * 256 // 9, (i + 1) * 256 // 9) for i in range(9)]
    for key, obj in objects.items():
        for field in ("cap_norm", "bytes_md5", "px_md5"):
            val = obj.get(field)
            if val:
                identity = (field, val)
                if identity in first:
                    union(key, first[identity])
                else:
                    first[identity] = key
        if obj.get("dhash"):
            h = int(obj["dhash"], 16)
            assert len(obj["dhash"]) == 64
            if h in hashes:
                union(key, hashes[h])
                continue
            candidates = set()
            for band, (lo, hi) in zip(bands, bounds):
                candidates.update(band.get((h >> lo) & ((1 << (hi - lo)) - 1), ()))
            for other in candidates:
                if (h ^ other).bit_count() <= 8:
                    union(key, hashes[other])
            hashes[h] = key
            for band, (lo, hi) in zip(bands, bounds):
                band[(h >> lo) & ((1 << (hi - lo)) - 1)].add(h)
    return {k: root(k) for k in objects}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume-cache", action="store_true")
    args = ap.parse_args()
    if args.train_csv.name != "train.csv":
        raise ValueError("Allowlisted input basename must be train.csv")
    if args.resume_cache and args.out.exists():
        assert {p.name for p in args.out.iterdir()} <= {"imgcache"}, "Only an unfinished image cache can be resumed"
    args.out.mkdir(parents=True, exist_ok=args.resume_cache)
    cache = args.out / "imgcache"
    cache.mkdir(exist_ok=args.resume_cache)
    manifest = pl.read_parquet(args.manifest)
    assert set(manifest["mode"]) == {"paper_strict"}
    assert manifest["pair_id"].n_unique() == manifest.height
    folds = dict(manifest.select("pair_id", "fold").iter_rows())
    assert set(folds.values()) == set(range(5))
    csv.field_size_limit(sys.maxsize)
    objects, pairs = {}, []
    pending = set()
    def drain():
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            obj = future.result()
            objects[obj["obj_key"]] = obj
            pending.remove(future)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.train_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "obj_1", "obj_2", *LABELS}
        assert required <= set(reader.fieldnames or ()), reader.fieldnames
        for row in reader:
            pid = int(row["id"])
            fold = folds[pid]
            keys = []
            for name in ("obj_1", "obj_2"):
                value = row[name]
                key = hashlib.md5(value.encode()).hexdigest()
                keys.append(key)
                if key not in objects:
                    obj = {"obj_key": key}
                    if value.startswith("iVBORw0KGgo"):
                        obj["modality"] = "image"
                        pending.add(pool.submit(cache_image, value, key, str(cache)))
                        if len(pending) >= 4 * args.workers:
                            drain()
                    else:
                        obj.update(modality="caption", cap_text=value,
                                   cap_norm=normalize_caption(value))
                    objects.setdefault(key, obj)
            mods = sorted(objects[k]["modality"][0] for k in keys)
            pair = dict(pair_id=pid, fold=fold, key_1=keys[0], key_2=keys[1],
                        pair_type="x".join(mods))
            # Do not parse the sealed audit's labels even while recovering rows.
            if fold != 0:
                labels = [int(row[k]) for k in LABELS]
                assert all(v in (0, 1) for v in labels) and sum(labels) == 1
                pair.update(y=labels.index(1), label=LABELS[labels.index(1)])
            pairs.append(pair)
            if len(pairs) % 1000 == 0:
                print(json.dumps({"pairs": len(pairs), "objects": len(objects)}), flush=True)
        while pending:
            drain()
    assert len(pairs) == len(folds) == len({r["pair_id"] for r in pairs})
    components = content_components(objects)
    refs = lambda rows: {components[r[k]] for r in rows for k in ("key_1", "key_2")}
    audit = [r for r in pairs if r["fold"] == 0]
    dev_all = [r for r in pairs if r["fold"] == 1]
    tr_all = [r for r in pairs if r["fold"] >= 2]
    audit_refs, held_refs = refs(audit), refs(audit + dev_all)
    clean = lambda rows, forbidden: [r for r in rows if not refs([r]) & forbidden]
    dev, train = clean(dev_all, audit_refs), clean(tr_all, held_refs)
    # A duplicated endpoint can identify its entire inherited paper component.
    # Exclude that whole group, not just the one visibly duplicated pair.
    groups = dict(manifest.select("pair_id", "group_id").iter_rows())
    def expand_group_pruning(original, kept):
        retained_ids = {r["pair_id"] for r in kept}
        blocked = {groups[r["pair_id"]] for r in original if r["pair_id"] not in retained_ids}
        return [r for r in kept if groups[r["pair_id"]] not in blocked]
    dev, train = expand_group_pruning(dev_all, dev), expand_group_pruning(tr_all, train)
    assert not (refs(train) & refs(dev) or refs(train) & refs(audit) or refs(dev) & refs(audit))
    pl.DataFrame(list(objects.values()), infer_schema_length=None).write_parquet(args.out / "objects_train.parquet")
    counts = {}
    for split, rows in (("train", train), ("dev", dev), ("audit", audit)):
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(args.out / f"gold_{split}_manifest.parquet")
        counts[split] = len(rows)
        for ptype in ("cxi", "ixi", "cxc"):
            result = []
            for row in rows:
                if row["pair_type"] != ptype:
                    continue
                a, b = objects[row["key_1"]], objects[row["key_2"]]
                r = {**row, "curriculum": "gold_train_only"}
                if ptype == "cxi":
                    cap, img = (a, b) if a["modality"] == "caption" else (b, a)
                    r.update(cap_norm=cap["cap_norm"], cap_text=cap["cap_text"],
                             img_bytes_md5=img["bytes_md5"], image_path=str(cache / (img["bytes_md5"] + ".img")))
                elif ptype == "ixi":
                    r.update(img_a=a["bytes_md5"], img_b=b["bytes_md5"])
                else:
                    r.update(cap_a=a["cap_text"], cap_b=b["cap_text"])
                result.append(r)
            pl.DataFrame(result, infer_schema_length=None).write_parquet(args.out / f"{ptype}_{split}.parquet")
            counts[f"{ptype}_{split}"] = len(result)
    # Compatibility with VLM corpus convention; only cxi supervised rows.
    for source, dest in (("cxi_train.parquet", "train.parquet"), ("cxi_dev.parquet", "val.parquet")):
        (args.out / dest).symlink_to(source)
    report = {"counts": counts, "removed_train": len(tr_all) - len(train),
              "removed_dev": len(dev_all) - len(dev),
              "manifest_sha256": sha(args.manifest),
              "source": str(args.train_csv), "source_bytes": args.train_csv.stat().st_size,
              "script_sha256": sha(Path(__file__)),
              "policy": "Gold training folds 2-4; selection fold1; audit fold0 inputs only",
              "cross_split_content_components": 0,
              "limitations": ["Inherited paper-strict assignment; original DOI candidate map absent.",
                               "Public transitive duplicate closure is absent: do not add synthetic data.",
                               "Fold0 participated in historical selection; it is not an untouched test."]}
    (args.out / "RECOVERY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
