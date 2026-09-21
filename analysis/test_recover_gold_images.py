"""Synthetic tests for analysis/recover_gold_images.py (stdlib + polars only)."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import polars as pl

MOD_PATH = Path(__file__).with_name("recover_gold_images.py")
_SPEC = importlib.util.spec_from_file_location("recover_gold_images", MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["recover_gold_images"] = _mod
_SPEC.loader.exec_module(_mod)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_bytes(payload: bytes) -> bytes:
    # Keep the archived base64 prefix detectable: the byte following the
    # 8-byte PNG magic must leave the "iVBORw0KGgo" prefix intact, so
    # prepend a zero byte (as real PNG IHDR length bytes do).
    return PNG_MAGIC + b"\x00" + payload


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def write_corpus(path: Path, pairs: list[tuple[int, str]]) -> None:
    frame = pl.DataFrame(
        {
            "pair_id": [p[0] for p in pairs],
            "img_bytes_md5": [p[1] for p in pairs],
            "key_1": [f"k1-{p[0]}" for p in pairs],
            "key_2": [f"k2-{p[0]}" for p in pairs],
        }
    )
    frame.write_parquet(path)


def write_csv(path: Path, rows: list[dict], extra_fields: list[str] | None = None) -> None:
    fields = ["id", "obj_1", "obj_2"] + list(extra_fields or [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def run_recovery(tmp: Path, csv_rows, train_pairs, dev_pairs, csv_name="train.csv",
                 extra_fields=None, pre_cache: dict[str, bytes] | None = None):
    train_csv = tmp / csv_name
    train_pq = tmp / "train_sel.parquet"
    dev_pq = tmp / "dev_sel.parquet"
    cache = tmp / "imgcache"
    receipt = tmp / "receipt.json"
    write_corpus(train_pq, train_pairs)
    write_corpus(dev_pq, dev_pairs)
    write_csv(train_csv, csv_rows, extra_fields=extra_fields)
    cache.mkdir(parents=True, exist_ok=True)
    for name, data in (pre_cache or {}).items():
        (cache / name).write_bytes(data)
    _mod.main([
        "--train-csv", str(train_csv),
        "--train-parquet", str(train_pq),
        "--dev-parquet", str(dev_pq),
        "--cache", str(cache),
        "--receipt", str(receipt),
    ])
    return cache, receipt


class RecoverGoldImagesTest(unittest.TestCase):
    def test_exact_bytes_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img_a = png_bytes(b"tiny-a")
            img_b = png_bytes(b"tiny-b")
            img_c = png_bytes(b"tiny-c")
            train_pairs = [(11, md5(img_a)), (12, md5(img_b))]
            dev_pairs = [(21, md5(img_c))]
            rows = [
                {"id": "11", "obj_1": b64(img_a), "obj_2": "caption alpha"},
                {"id": "12", "obj_1": "caption beta", "obj_2": b64(img_b)},
                {"id": "21", "obj_1": b64(img_c), "obj_2": "caption gamma"},
            ]
            cache, receipt_path = run_recovery(tmp, rows, train_pairs, dev_pairs)
            for img in (img_a, img_b, img_c):
                self.assertEqual((cache / f"{md5(img)}.img").read_bytes(), img)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["counts"]["train_pairs"], 2)
            self.assertEqual(receipt["counts"]["dev_pairs"], 1)
            self.assertEqual(receipt["counts"]["required_images"], 3)
            manifest = receipt["image_manifest"]
            self.assertEqual([m["bytes_md5"] for m in manifest], sorted(m["bytes_md5"] for m in manifest))
            expect = hashlib.sha256(
                "\n".join(f"{m['bytes_md5']}:{m['size']}" for m in manifest).encode()
            ).hexdigest()
            self.assertEqual(receipt["image_manifest_sha256"], expect)
            blob = receipt_path.read_text(encoding="utf-8")
            self.assertNotIn(str(tmp), blob)
            self.assertNotIn("caption alpha", blob)
            self.assertNotIn(b64(img_a)[:16], blob)

    def test_corrupted_existing_cache_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img = png_bytes(b"good-bytes")
            train_pairs = [(11, md5(img))]
            rows = [{"id": "11", "obj_1": b64(img), "obj_2": "cap"}]
            with self.assertRaises(ValueError):
                run_recovery(tmp, rows, train_pairs, [], pre_cache={f"{md5(img)}.img": b"tampered"})

    def test_missing_required_hash_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img = png_bytes(b"present")
            absent = png_bytes(b"absent")
            train_pairs = [(11, md5(img)), (12, md5(absent))]
            rows = [{"id": "11", "obj_1": b64(img), "obj_2": "cap"}]
            with self.assertRaises(ValueError):
                run_recovery(tmp, rows, train_pairs, [], pre_cache=None)

    def test_wrong_basename_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img = png_bytes(b"x")
            train_pairs = [(11, md5(img))]
            rows = [{"id": "11", "obj_1": b64(img), "obj_2": "cap"}]
            with self.assertRaises(ValueError):
                run_recovery(tmp, rows, train_pairs, [], csv_name="other.csv")

    def test_outside_ids_with_invalid_objects_ignored(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img = png_bytes(b"kept")
            train_pairs = [(11, md5(img))]
            rows = [
                {"id": "11", "obj_1": b64(img), "obj_2": "cap ok"},
                # Outside IDs carry corrupt/lookalike objects and bogus labels;
                # they must be skipped without decoding or label inspection.
                {"id": "999", "obj_1": "!!!not-base64!!!", "obj_2": "junk", "y": "nope"},
                {"id": "1000", "obj_1": "iVBORw0KGgo!!!corrupt!!!", "obj_2": "junk", "y": "nope"},
            ]
            cache, receipt_path = run_recovery(
                tmp, rows, train_pairs, [], extra_fields=["y"])
            self.assertEqual((cache / f"{md5(img)}.img").read_bytes(), img)
            self.assertFalse((cache / f"{hashlib.md5(b'junk').hexdigest()}.img").exists())

    def test_duplicate_selected_ids_fail(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img = png_bytes(b"dup")
            train_pairs = [(11, md5(img))]
            rows = [
                {"id": "11", "obj_1": b64(img), "obj_2": "cap"},
                {"id": "11", "obj_1": b64(img), "obj_2": "cap"},
            ]
            with self.assertRaises(ValueError):
                run_recovery(tmp, rows, train_pairs, [])

    def test_wrong_image_md5_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img = png_bytes(b"expected")
            other = png_bytes(b"other-image")
            train_pairs = [(11, md5(img))]
            rows = [{"id": "11", "obj_1": b64(other), "obj_2": "cap"}]
            with self.assertRaises(ValueError):
                run_recovery(tmp, rows, train_pairs, [])

    def test_corrupt_base64_in_selected_row_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img = png_bytes(b"expected")
            train_pairs = [(11, md5(img))]
            rows = [{"id": "11", "obj_1": "iVBORw0KGgo!!!corrupt!!!", "obj_2": "cap"}]
            with self.assertRaises(ValueError):
                run_recovery(tmp, rows, train_pairs, [])


if __name__ == "__main__":
    unittest.main()
