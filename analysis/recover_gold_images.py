"""Minimal image-cache recovery for a frozen experiment.

Recovers only the required CXI image bytes referenced by frozen selected
``TRAIN``/``DEV`` parquets, streaming an explicitly downloaded ``train.csv``.

Policy:
  - Never opens test inputs, solution files, or secrets; nonselected rows are streamed but their objects and labels are not inspected.
  - Never uses the corpus ``image_path`` column; only ``pair_id`` and
    ``img_bytes_md5`` are read from the frozen corpora.
  - Streams ``train.csv`` one row at a time; rows whose ID is not selected
    are skipped without decoding objects or inspecting label values.
  - Selected rows expose only ``id``, ``obj_1``, ``obj_2``; PNG objects are
    detected by the archived base64 prefix convention (``iVBORw0KGgo``),
    decoded with ``validate=True``, and stored raw as ``<md5>.img`` with an
    atomic rename. No pixel reencoding.
  - Existing cache files are verified by byte MD5, never trusted by name.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import polars as pl

PNG_B64_PREFIX = "iVBORw0KGgo"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_corpus(path: Path) -> tuple[dict[int, str], set[str]]:
    frame = pl.read_parquet(path, columns=["pair_id", "img_bytes_md5"])
    if "pair_id" not in frame.columns or "img_bytes_md5" not in frame.columns:
        raise ValueError(f"Corpus missing required columns: {path.name}")
    pair_ids = frame["pair_id"].to_list()
    hashes = frame["img_bytes_md5"].to_list()
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"Duplicate pair_id in corpus: {path.name}")
    expected: dict[int, str] = {}
    for pid, digest in zip(pair_ids, hashes):
        pid_int = int(pid)
        digest_str = str(digest).lower()
        if pid_int in expected:
            raise ValueError(f"Duplicate pair_id in corpus: {path.name}")
        expected[pid_int] = digest_str
    return expected, {str(h).lower() for h in hashes}


def _verify_existing(path: Path, digest: str) -> bytes | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    actual = hashlib.md5(data).hexdigest()
    if actual != digest:
        raise ValueError(f"Corrupt cached image (MD5 mismatch): {digest}.img")
    return data


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument("--dev-parquet", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.train_csv.name != "train.csv":
        raise ValueError("Allowlisted input basename must be train.csv")

    train_expected, train_hashes = _load_corpus(args.train_parquet)
    dev_expected, dev_hashes = _load_corpus(args.dev_parquet)
    overlap = set(train_expected) & set(dev_expected)
    if overlap:
        raise ValueError("Duplicate selected IDs across train/dev corpora")
    expected_by_id = {**train_expected, **dev_expected}
    allowed_ids = set(expected_by_id)
    required_hashes = set(train_hashes | dev_hashes)

    cache_dir = args.cache
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Verify any pre-existing required files before trusting them.
    for digest in required_hashes:
        _verify_existing(cache_dir / f"{digest}.img", digest)

    csv.field_size_limit(sys.maxsize)
    seen: set[int] = set()
    seen_hashes: set[str] = set()
    selected_rows = 0
    with open(args.train_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or ())
        if not {"id", "obj_1", "obj_2"} <= fieldnames:
            raise ValueError("train.csv missing required columns id/obj_1/obj_2")
        for row in reader:
            # Only the id decides selection; outside rows are skipped
            # without decoding objects or inspecting label values.
            try:
                pid = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if pid not in allowed_ids:
                continue
            if pid in seen:
                raise ValueError("Duplicate selected IDs in train.csv")
            seen.add(pid)
            selected_rows += 1
            # Allowed rows expose only id/obj_1/obj_2.
            v1 = row["obj_1"]
            v2 = row["obj_2"]
            expected = expected_by_id[pid]
            found: set[str] = set()
            for value in (v1, v2):
                if not isinstance(value, str) or not value.startswith(PNG_B64_PREFIX):
                    continue
                try:
                    raw = base64.b64decode(value, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(f"Corrupt base64 image object for selected row: {exc}") from exc
                digest = hashlib.md5(raw).hexdigest()
                found.add(digest)
                if digest in required_hashes and digest not in seen_hashes:
                    seen_hashes.add(digest)
                    target = cache_dir / f"{digest}.img"
                    existing = _verify_existing(target, digest)
                    if existing is None:
                        _atomic_write(target, raw)
            if expected not in found:
                raise ValueError("Expected image MD5 not present among row image objects")

    if seen != allowed_ids:
        missing = len(allowed_ids - seen)
        extra = len(seen - allowed_ids)
        raise ValueError(
            f"Missing selected IDs: {missing} missing, {extra} unexpected"
        )
    missing_hashes = required_hashes - seen_hashes
    if missing_hashes:
        raise ValueError(f"Missing required image hashes: {len(missing_hashes)}")

    manifest = []
    for digest in sorted(required_hashes):
        path = cache_dir / f"{digest}.img"
        data = path.read_bytes()
        if hashlib.md5(data).hexdigest() != digest:
            raise ValueError(f"Corrupt cached image after recovery: {digest}.img")
        manifest.append({"bytes_md5": digest, "size": len(data)})
    digest_input = "\n".join(f"{m['bytes_md5']}:{m['size']}" for m in manifest)
    manifest_digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    receipt = {
        "counts": {
            "dev_pairs": len(dev_expected),
            "recovered_images": len(manifest),
            "required_images": len(required_hashes),
            "selected_pairs": len(allowed_ids),
            "selected_rows_seen": selected_rows,
            "train_pairs": len(train_expected),
        },
        "image_manifest": manifest,
        "image_manifest_sha256": manifest_digest,
        "inputs": {
            "dev_parquet_sha256": sha256_file(args.dev_parquet),
            "train_csv_name": "train.csv",
            "train_parquet_sha256": sha256_file(args.train_parquet),
        },
        "policy": (
            "Gold image-cache recovery only; corpora unaltered; "
            "no test/solution inputs or nonselected-object/label use; no pixel reencoding"
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp_receipt = args.receipt.with_name(args.receipt.name + ".tmp")
    tmp_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_receipt, args.receipt)
    print(json.dumps(receipt["counts"], sort_keys=True), flush=True)
    return receipt


if __name__ == "__main__":
    main()
