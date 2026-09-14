"""Full pinned HF public-only cxi corpus with all-gold duplicate closure.

Index and build are separate: indexing can run while gold preparation finishes.
No competition labels or test files are read. Public validation components are
held out before pair generation; no class priors are fitted to competition data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from astroclimb.utils import image_hashes, normalize_caption, normalize_doi


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def index_one(task):
    source, out = map(Path, task)
    index = out / "index" / source.name
    receipt = index.with_suffix(".json")
    if index.exists() and receipt.exists():
        return json.loads(receipt.read_text())
    rows, edges, errors = [], set(), 0
    # One shard per process keeps memory bounded without IPC of image bytes.
    for r in pl.read_parquet(source).iter_rows(named=True):
        paper = normalize_doi(r["Paper DOI"] or "")
        for field in ("References DOIs", "Citing DOIs"):
            for dst in r[field] or []:
                dst = normalize_doi(dst or "")
                if paper and dst and paper != dst:
                    edges.add(tuple(sorted((paper, dst))))
        raw = r["image"]["bytes"] or b""
        try:
            hashes = image_hashes(raw)
            cache = out / "imgcache" / (hashes["bytes_md5"] + ".img")
            valid = cache.exists() and cache.stat().st_size == len(raw)
            if valid:
                valid = hashlib.md5(cache.read_bytes()).hexdigest() == hashes["bytes_md5"]
            if not valid:
                with tempfile.NamedTemporaryFile(
                    dir=cache.parent, prefix=".image-", delete=False
                ) as f:
                    f.write(raw)
                    temp = f.name
                os.replace(temp, cache)
        except Exception:
            errors += 1
            hashes = {"bytes_md5": None, "px_md5": None, "dhash": None}
        rows.append(
            dict(
                uuid=r["UUID"],
                image_id=r["Image ID"],
                doi=paper,
                cap_norm=normalize_caption(r["Image Caption"] or ""),
                cap_text=r["Image Caption"] or "",
                **hashes,
            )
        )
    tmp = index.with_suffix(".tmp")
    pl.DataFrame(rows).write_parquet(tmp)
    tmp.replace(index)
    graph = out / "graph" / source.name
    pl.DataFrame(sorted(edges), schema=["src_doi", "dst_doi"], orient="row").write_parquet(graph)
    result = {
        "shard": source.name,
        "rows": len(rows),
        "hash_errors": errors,
        "source_sha256": sha(source),
        "index_sha256": sha(index),
        "graph_sha256": sha(graph),
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def index_all(a):
    files = sorted(a.source.glob("*.parquet"))
    expected = {
        f"train-{i:05d}-of-{a.expected_shards:05d}.parquet" for i in range(a.expected_shards)
    }
    assert {p.name for p in files} == expected, "Full expected dataset is required"
    for name in ("index", "graph", "imgcache"):
        (a.out / name).mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for future in as_completed([ex.submit(index_one, (str(p), str(a.out))) for p in files]):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(done=len(results), total=len(files), **result)), flush=True)
    manifest = {
        "full_dataset": True,
        "expected_shards": a.expected_shards,
        "rows": sum(r["rows"] for r in results),
        "hash_errors": sum(r["hash_errors"] for r in results),
        "shards": sorted(results, key=lambda r: r["shard"]),
        "source": str(a.source),
    }
    (a.out / "index_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build(a):
    for name in ("corpus_manifest.json", "exposure_map_manifest.json"):
        (a.out / name).unlink(missing_ok=True)
    manifest = json.loads((a.out / "index_manifest.json").read_text())
    assert manifest["full_dataset"] and len(manifest["shards"]) == a.expected_shards
    for receipt in manifest["shards"]:
        for folder, field in [("index", "index_sha256"), ("graph", "graph_sha256")]:
            assert sha(a.out / folder / receipt["shard"]) == receipt[field], (
                "Index/graph changed after indexing"
            )
    rows = pl.read_parquet(str(a.out / "index" / "*.parquet")).to_dicts()
    assert len(rows) == manifest["rows"]
    held = pl.read_parquet(a.gold_objects)
    assert not {"label", "y", "target"} & set(held.columns), "Exposure objects must be label-free"
    assert {"cap_norm", "bytes_md5", "px_md5", "dhash"} <= set(held.columns)
    keys = ("cap_norm", "bytes_md5", "px_md5")
    forbidden_keys = {k: set(held[k].drop_nulls().to_list()) - {""} for k in keys}
    exposure_lookup = {k: defaultdict(set) for k in (*keys, "dhash")}
    for obj in held.iter_rows(named=True):
        for k in exposure_lookup:
            if obj.get(k):
                exposure_lookup[k][obj[k]].add(obj["obj_key"])
    object_dois = defaultdict(set)
    parent = {}

    def root(p):
        parent.setdefault(p, p)
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(p, q):
        p, q = root(p), root(q)
        if p != q:
            parent[max(p, q)] = min(p, q)

    first, seeds, hits = {}, set(), Counter()
    matched = {k: set() for k in keys}
    for r in rows:
        p = r["doi"]
        if not p:
            continue
        root(p)
        for k in (*keys, "uuid"):
            v = r.get(k)
            if not v:
                continue
            ident = (k, v)
            if ident in first:
                union(p, first[ident])
            else:
                first[ident] = p
            if v in forbidden_keys.get(k, set()):
                seeds.add(p)
                hits[k] += 1
                matched[k].add(v)
                for obj_key in exposure_lookup[k][v]:
                    object_dois[obj_key].add(p)
    # Exact Hamming radius 8 via nine disjoint bands on true 256-bit hashes.
    bands = [defaultdict(list) for _ in range(9)]
    bounds = [(i * 256 // 9, (i + 1) * 256 // 9) for i in range(9)]
    hash_paper = {}

    def near(h):
        candidates = set()
        for band, (lo, hi) in zip(bands, bounds, strict=True):
            candidates.update(band.get((h >> lo) & ((1 << (hi - lo)) - 1), ()))
        return [v for v in candidates if (h ^ v).bit_count() <= 8]

    unions = 0
    for r in rows:
        if not r["doi"] or not r["dhash"]:
            continue
        assert len(r["dhash"]) == 64
        h = int(r["dhash"], 16)
        if h in hash_paper:
            union(r["doi"], hash_paper[h])
            continue
        for other in near(h):
            union(r["doi"], hash_paper[other])
            unions += 1
        hash_paper[h] = r["doi"]
        for band, (lo, hi) in zip(bands, bounds, strict=True):
            band[(h >> lo) & ((1 << (hi - lo)) - 1)].append(h)
    held_near = set()
    for v in set(held["dhash"].drop_nulls().to_list()):
        assert len(v) == 64
        for h in near(int(v, 16)):
            seeds.add(hash_paper[h])
            held_near.add(h)
            for obj_key in exposure_lookup["dhash"][v]:
                object_dois[obj_key].add(hash_paper[h])
    candidate_rows = [
        {"obj_key": obj, "doi": p, "public_component": root(p)}
        for obj, papers in sorted(object_dois.items())
        for p in sorted(papers)
    ]
    pl.DataFrame(
        candidate_rows,
        schema={"obj_key": pl.String, "doi": pl.String, "public_component": pl.String},
    ).write_parquet(a.out / "exposure_public_candidates.parquet")
    pl.DataFrame([{"doi": p, "public_component": root(p)} for p in sorted(parent)]).write_parquet(
        a.out / "public_doi_components.parquet"
    )
    exposure_manifest = {
        "full_public_shards": manifest["expected_shards"],
        "public_rows": len(rows),
        "exposure_objects": held.height,
        "matched_objects": len(object_dois),
        "candidate_rows": len(candidate_rows),
        "index_manifest_sha256": sha(a.out / "index_manifest.json"),
        "exposure_objects_sha256": sha(a.gold_objects),
        "candidate_map_sha256": sha(a.out / "exposure_public_candidates.parquet"),
        "doi_component_map_sha256": sha(a.out / "public_doi_components.parquet"),
        "labels_read": False,
    }
    (a.out / "exposure_map_manifest.json").write_text(
        json.dumps(exposure_manifest, indent=2) + "\n"
    )
    print(
        json.dumps({"stage": "full_public_exposure_map_complete", **exposure_manifest}), flush=True
    )
    blocked_roots = {root(p) for p in seeds}
    forbidden = {p for p in list(parent) if root(p) in blocked_roots}
    (a.out / "forbidden_gold_closure_dois.txt").write_text("\n".join(sorted(forbidden)) + "\n")
    clean = [
        r
        for r in rows
        if r["doi"]
        and r["doi"] not in forbidden
        and r["bytes_md5"]
        and r["dhash"]
        and len(r["cap_norm"]) >= 40
    ]
    assert not any(r[k] in forbidden_keys[k] for r in clean for k in keys)
    assert not any(int(r["dhash"], 16) in held_near for r in clean)
    edges = {
        tuple(sorted(v)) for v in pl.read_parquet(str(a.out / "graph" / "*.parquet")).iter_rows()
    }
    clean_papers = {r["doi"] for r in clean}
    components = {root(p) for p in clean_papers}
    component_edges = {
        tuple(sorted((root(p), root(q))))
        for p, q in edges
        if p in clean_papers and q in clean_papers and root(p) != root(q)
    }
    ranked_edges = sorted(
        component_edges,
        key=lambda e: hashlib.sha256((str(a.seed) + "\0" + "\0".join(e)).encode()).hexdigest(),
    )
    dev_components = set()
    seeded_edges = 0
    # Give public REL validation independent paper-pair support, not hundreds
    # of image pairs drawn from only a handful of accidental citation edges.
    for p, q in ranked_edges:
        if p not in dev_components and q not in dev_components:
            dev_components.update((p, q))
            seeded_edges += 1
            if seeded_edges >= min(32, max(1, len(components) // 40)):
                break
    budget = max(len(dev_components), (len(components) + 19) // 20)
    for component in sorted(
        components, key=lambda p: hashlib.sha256((str(a.seed) + p).encode()).hexdigest()
    ):
        if len(dev_components) >= budget:
            break
        dev_components.add(component)
    splits = {p: "public_dev" if root(p) in dev_components else "train" for p in clean_papers}
    outputs = {}
    used_image_hashes = set()
    for split, count in [("train", a.per_class), ("public_dev", a.dev_per_class)]:
        pool = [r for r in clean if splits[r["doi"]] == split]
        requested_count = count
        count = min(count, len({(r["bytes_md5"], r["cap_norm"]) for r in pool}))
        if not count:
            raise ValueError(f"No clean public figures for {split}")
        bypaper = defaultdict(list)
        for r in pool:
            bypaper[r["doi"]].append(r)
        papers = sorted(bypaper)
        sp = [p for p in papers if len(bypaper[p]) > 1]
        rel = sorted((p, q) for p, q in edges if p in bypaper and q in bypaper)
        if not rel or not sp:
            raise ValueError(f"Insufficient public {split} class support")
        journals = defaultdict(list)
        for p in papers:
            journals["/".join(p.split("/")[:2])].append(p)
        journals = [ps for ps in journals.values() if len(ps) > 1]
        neighbors = defaultdict(list)
        for p, q in rel:
            neighbors[p].append(q)
            neighbors[q].append(p)
        connected = sorted(neighbors)
        rng = random.Random(a.seed + (split != "train"))
        output, seen, counts = [], set(), Counter()
        for y in range(4):
            for _attempt in range(count * 300):
                if counts[y] >= count:
                    break
                if y == 0:
                    x = z = rng.choice(pool)
                    policy = "same_figure"
                elif y == 1:
                    x, z = rng.sample(bypaper[rng.choice(sp)], 2)
                    policy = "same_paper"
                elif y == 2:
                    p, q = rng.choice(rel)
                    if rng.random() < 0.5:
                        p, q = q, p
                    x, z = rng.choice(bypaper[p]), rng.choice(bypaper[q])
                    policy = "citation_edge"
                else:
                    mode = counts[y] % 3
                    if mode == 0:
                        p, q = rng.sample(papers, 2)
                        policy = "unrelated_random"
                    elif mode == 1 and journals:
                        p, q = rng.sample(rng.choice(journals), 2)
                        policy = "unrelated_same_journal"
                    else:
                        p = rng.choice(connected)
                        q = rng.choice(neighbors[rng.choice(neighbors[p])])
                        policy = "unrelated_two_hop"
                    if p == q or tuple(sorted((p, q))) in edges:
                        continue
                    x, z = rng.choice(bypaper[p]), rng.choice(bypaper[q])
                if y and (
                    x["cap_norm"] == z["cap_norm"]
                    or x["px_md5"] == z["px_md5"]
                    or (int(x["dhash"], 16) ^ int(z["dhash"], 16)).bit_count() <= 8
                ):
                    continue
                ident = (x["bytes_md5"], z["cap_norm"])
                if ident in seen:
                    continue
                seen.add(ident)
                counts[y] += 1
                output.append(
                    {
                        "pair_id": "public_"
                        + hashlib.sha256(("\0".join(ident)).encode()).hexdigest(),
                        "cap_norm": z["cap_text"],
                        "img_bytes_md5": x["bytes_md5"],
                        "y": y,
                        "label": ("SF", "SP", "REL", "UNR")[y],
                        "doi_a": x["doi"],
                        "doi_b": z["doi"],
                        "image_uuid": x["uuid"],
                        "caption_uuid": z["uuid"],
                        "sampling_policy": policy,
                    }
                )
            if counts[y] != count:
                raise ValueError(f"Cannot fill {split} class {y}: {counts[y]}/{count}")
        for row in output:
            expected = (
                0
                if row["image_uuid"] == row["caption_uuid"]
                else 1
                if row["doi_a"] == row["doi_b"]
                else 2
                if tuple(sorted((row["doi_a"], row["doi_b"]))) in edges
                else 3
            )
            assert row["y"] == expected
            assert row["doi_a"] not in forbidden and row["doi_b"] not in forbidden
        used_image_hashes.update(r["img_bytes_md5"] for r in output)
        rng.shuffle(output)
        result = pl.DataFrame(output)
        result.write_parquet(a.out / (split + ".parquet"))
        if split == "public_dev":
            result.write_parquet(a.out / "val.parquet")
        else:
            for size in (20000, 40000):
                small = pl.concat(
                    [result.filter(pl.col("y") == y).head(size // 4) for y in range(4)]
                ).sample(fraction=1.0, shuffle=True, seed=a.seed)
                small.write_parquet(a.out / f"train_{size}.parquet")
        outputs[split] = {
            "rows": len(output),
            "counts": dict(counts),
            "requested_per_class": requested_count,
            "actual_per_class": count,
            "papers": len(papers),
            "edges": len(rel),
            "policies": dict(Counter(r["sampling_policy"] for r in output)),
            "sha256": sha(a.out / (split + ".parquet")),
        }
    train_roots = {root(p) for p, s in splits.items() if s == "train"}
    dev_roots = {root(p) for p, s in splits.items() if s == "public_dev"}
    assert not train_roots & dev_roots

    def cache_error(digest):
        path = a.out / "imgcache" / (digest + ".img")
        return (
            None
            if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == digest
            else digest
        )

    # This audit also protects artifacts created by older non-atomic writers.
    with ThreadPoolExecutor(max_workers=8) as ex:
        corrupt = [h for h in ex.map(cache_error, sorted(used_image_hashes)) if h]
    if corrupt:
        (a.out / "corrupt_image_hashes.json").write_text(json.dumps(corrupt))
        raise ValueError(
            f"{len(corrupt)} missing/corrupt training images; repair before publishing corpus"
        )
    audit = {
        "source_revision": json.loads((a.source.parent / "revision.json").read_text())
        if (a.source.parent / "revision.json").exists()
        else None,
        "used_image_cache_md5_verified": len(used_image_hashes),
        "used_image_cache_errors": 0,
        "full_public_shards": manifest["expected_shards"],
        "public_rows": len(rows),
        "eligible_rows": len(clean),
        "gold_content_sha256": sha(a.gold_objects),
        "gold_exact_match_counts": dict(hits),
        "gold_unique_contents_matched": {k: len(v) for k, v in matched.items()},
        "gold_unique_contents_total": {k: len(v) for k, v in forbidden_keys.items()},
        "gold_near_hash_matches": len(held_near),
        "public_near_unions": unions,
        "forbidden_papers": len(forbidden),
        "gold_content_hits_after_exclusion": 0,
        "public_train_dev_component_overlap": 0,
        "public_dev_independent_citation_edges_seeded": seeded_edges,
        "public_dev_split_policy": "Public-only citation-edge support seeding, then deterministic fill to 5% duplicate components; no gold priors or scores",
        "gold_labels_read": False,
        "gold_class_priors_used": False,
        "seed": a.seed,
        "outputs": outputs,
        "source_index_manifest_sha256": sha(a.out / "index_manifest.json"),
        "prepare_script_sha256": sha(Path(__file__)),
        "convenience_output_sha256": {p.name: sha(p) for p in a.out.glob("train_*.parquet")}
        | {"val.parquet": sha(a.out / "val.parquet")},
        "caveat": "UNR means no observed direct public citation edge; public graph completeness is assumed.",
    }
    (a.out / "corpus_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["index", "build"])
    p.add_argument("--source", type=Path, default=Path("data/raw/hf_full/data"))
    p.add_argument("--out", type=Path, default=Path("output/final_public_20260913"))
    p.add_argument(
        "--gold-objects",
        type=Path,
        default=Path("output/final_gold_20260913/objects_train.parquet"),
    )
    p.add_argument("--expected-shards", type=int, default=114)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--per-class", type=int, default=20000)
    p.add_argument("--dev-per-class", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260913)
    a = p.parse_args()
    (index_all if a.mode == "index" else build)(a)


if __name__ == "__main__":
    main()
