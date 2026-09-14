"""Recover only id/obj_1/obj_2 from the explicitly named test.csv input."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import polars as pl
from final_gold_prepare import cache_image, normalize_caption


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.test_csv.name != "test.csv":
        raise ValueError("Only explicitly named test.csv allowed")
    args.out.mkdir(parents=True, exist_ok=False)
    cache = args.out.resolve() / "imgcache"
    cache.mkdir()
    objects, pairs, pending = {}, [], set()
    def drain():
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            obj = future.result()
            objects[obj["obj_key"]] = obj
            pending.remove(future)
    csv.field_size_limit(sys.maxsize)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.test_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        assert {"id", "obj_1", "obj_2"} <= set(reader.fieldnames or ())
        for raw_row in reader:
            row = {name: raw_row[name] for name in ("id", "obj_1", "obj_2")}
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
                        obj.update(modality="caption", cap_text=value, cap_norm=normalize_caption(value))
                    objects.setdefault(key, obj)
            mods = sorted(objects[k]["modality"][0] for k in keys)
            pairs.append(dict(pair_id=int(row["id"]), key_1=keys[0], key_2=keys[1], pair_type="x".join(mods)))
            if len(pairs) % 1000 == 0:
                print(json.dumps({"pairs": len(pairs), "objects": len(objects)}), flush=True)
        while pending:
            drain()
    assert len(pairs) == len({r["pair_id"] for r in pairs})
    pl.DataFrame(list(objects.values()), infer_schema_length=None).write_parquet(args.out / "objects_test.parquet")
    pl.DataFrame(pairs).write_parquet(args.out / "pairs_test.parquet")
    for ptype in ("cxi", "ixi", "cxc"):
        rows = []
        for pair in pairs:
            if pair["pair_type"] != ptype:
                continue
            a, b = objects[pair["key_1"]], objects[pair["key_2"]]
            row = dict(pair)
            if ptype == "cxi":
                cap, img = (a, b) if a["modality"] == "caption" else (b, a)
                row.update(cap_text=cap["cap_text"], cap_norm=cap["cap_norm"], img_bytes_md5=img["bytes_md5"],
                           image_path=str(cache / (img["bytes_md5"] + ".img")))
            elif ptype == "ixi":
                row.update(img_a=a["bytes_md5"], img_b=b["bytes_md5"])
            else:
                row.update(cap_a=a["cap_text"], cap_b=b["cap_text"])
            rows.append(row)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(args.out / f"{ptype}_test.parquet")
    report = {"pairs": len(pairs), "objects": len(objects), "input_columns": ["id", "obj_1", "obj_2"],
              "purpose": "Inference inputs and exclusion-only public exposure boundary; no test labels"}
    (args.out / "RECOVERY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
