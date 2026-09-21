"""Tiny-data tests for analysis/public_supervision_diagnostics.py."""
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import polars as pl

MOD_PATH = Path(__file__).with_name("public_supervision_diagnostics.py")
_SPEC = importlib.util.spec_from_file_location("public_supervision_diagnostics", MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["public_supervision_diagnostics"] = _mod
_SPEC.loader.exec_module(_mod)


def md5hex(tag):
    return hashlib.md5(tag.encode("utf-8")).hexdigest()


MD_A = md5hex("img-a")
MD_B = md5hex("img-b")
MD_C = md5hex("img-c")


def write_parquet(path, data):
    pl.DataFrame(data).write_parquet(path)


def run_cli(public_data, gold_data, tmp):
    pub = tmp / "public_train.parquet"
    gold = tmp / "gold_train.parquet"
    out = tmp / "out.json"
    write_parquet(pub, public_data)
    write_parquet(gold, gold_data)
    _mod.main(["--public-train", str(pub), "--gold-train", str(gold), "--out", str(out)])
    return json.loads(out.read_text(encoding="utf-8"))


class DiagnosticsTest(unittest.TestCase):
    def test_duplicate_same_label_not_contradiction(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            public_data = {
                "cap_norm": ["same caption", "same caption", "other caption"],
                "img_bytes_md5": [MD_A, MD_A, MD_B],
                "y": [2, 2, 1],
            }
            gold_data = {
                "cap_text": ["g one", "g two"],
                "img_bytes_md5": [MD_A, MD_B],
                "y": [0, 3],
            }
            receipt = run_cli(public_data, gold_data, tmp)
            pub = receipt["datasets"]["public_train"]
            self.assertEqual(pub["counts_per_class"], [0, 1, 2, 0])
            for kind in ("full", "effective"):
                self.assertEqual(pub[kind]["n_duplicate_groups"], 1)
                self.assertEqual(pub[kind]["n_contradictory_groups"], 0)
                self.assertEqual(pub[kind]["n_contradictory_rows"], 0)
                self.assertEqual(pub[kind]["lower_bound_errors"], 0)
            blob = json.dumps(receipt)
            self.assertNotIn("same caption", blob)
            self.assertNotIn(MD_A, blob)

    def test_identical_effective_different_labels_lower_bound(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            public_data = {
                "cap_norm": ["dup", "dup", "dup"],
                "img_bytes_md5": [MD_A, MD_A, MD_A],
                "y": [0, 0, 1],
            }
            gold_data = {
                "cap_norm": ["solo"],
                "img_bytes_md5": [MD_B],
                "y": [2],
            }
            receipt = run_cli(public_data, gold_data, tmp)
            pub = receipt["datasets"]["public_train"]
            self.assertEqual(pub["full"]["n_contradictory_groups"], 1)
            self.assertEqual(pub["full"]["n_contradictory_rows"], 3)
            self.assertEqual(pub["full"]["lower_bound_errors"], 1)
            self.assertEqual(pub["effective"]["lower_bound_errors"], 1)
            self.assertEqual(pub["counts_per_class"], [2, 1, 0, 0])

    def test_truncation_merges_different_full_captions(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prefix = "x" * 2400
            cap1 = prefix + "AAA-tail-one"
            cap2 = prefix + "BBB-tail-two"
            self.assertEqual(cap1[:2400], cap2[:2400])
            self.assertNotEqual(cap1, cap2)
            public_data = {
                "cap_norm": [cap1, cap2],
                "img_bytes_md5": [MD_C, MD_C],
                "y": [0, 1],
            }
            gold_data = {
                "cap_norm": ["plain"],
                "img_bytes_md5": [MD_A],
                "y": [0],
            }
            receipt = run_cli(public_data, gold_data, tmp)
            pub = receipt["datasets"]["public_train"]
            self.assertEqual(pub["n_truncated"], 2)
            self.assertEqual(pub["full"]["n_contradictory_groups"], 0)
            self.assertEqual(pub["full"]["lower_bound_errors"], 0)
            self.assertEqual(pub["effective"]["n_contradictory_groups"], 1)
            self.assertEqual(pub["effective"]["n_contradictory_rows"], 2)
            self.assertEqual(pub["effective"]["lower_bound_errors"], 1)
            merger = pub["truncation_merger"]
            self.assertEqual(
                merger["n_effective_contradictory_groups_merging_distinct_full"], 1)
            self.assertEqual(
                merger["n_new_contradictory_groups_from_truncation_verified"], 1)
            self.assertEqual(merger["n_new_rows_affected_verified"], 2)
            q = pub["caption_len_quantiles"]
            self.assertEqual(q["levels"], [0.0, 0.25, 0.5, 0.75, 1.0])
            self.assertTrue(q["values"][0] <= q["values"][2] <= q["values"][-1])

    def test_caption_case_retained(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            public_data = {
                "cap_norm": ["Hello World", "hello world"],
                "img_bytes_md5": [MD_A, MD_A],
                "y": [0, 1],
            }
            gold_data = {
                "cap_norm": ["solo"],
                "img_bytes_md5": [MD_B],
                "y": [1],
            }
            receipt = run_cli(public_data, gold_data, tmp)
            pub = receipt["datasets"]["public_train"]
            # Case-sensitive identity keeps these distinct, so no conflict.
            self.assertEqual(pub["full"]["n_contradictory_groups"], 0)
            self.assertEqual(pub["effective"]["n_contradictory_groups"], 0)
            self.assertEqual(pub["full"]["n_distinct"], 2)

    def test_cap_text_preferred(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            public_data = {
                "cap_norm": ["p"],
                "img_bytes_md5": [MD_A],
                "y": [0],
            }
            gold_path = tmp / "gold_train.parquet"
            pub_path = tmp / "public_train.parquet"
            out_path = tmp / "out.json"
            write_parquet(pub_path, public_data)
            write_parquet(gold_path, {
                "cap_text": ["AAA distinct", "BBB distinct"],
                "cap_norm": ["SAME", "SAME"],
                "img_bytes_md5": [MD_A, MD_A],
                "y": [0, 1],
            })
            _mod.main(["--public-train", str(pub_path), "--gold-train", str(gold_path),
                       "--out", str(out_path)])
            receipt = json.loads(out_path.read_text(encoding="utf-8"))
            gold = receipt["datasets"]["gold_train"]
            self.assertEqual(gold["caption_column_used"], "cap_text")
            self.assertEqual(gold["full"]["n_contradictory_groups"], 0)
            # Fallback to cap_norm when cap_text is absent.
            gold_path2 = tmp / "gold2.parquet"
            write_parquet(gold_path2, {
                "cap_norm": ["SAME", "SAME"],
                "img_bytes_md5": [MD_A, MD_A],
                "y": [0, 1],
            })
            out2 = tmp / "out2.json"
            _mod.main(["--public-train", str(pub_path), "--gold-train", str(gold_path2),
                       "--out", str(out2)])
            receipt2 = json.loads(out2.read_text(encoding="utf-8"))
            self.assertEqual(receipt2["datasets"]["gold_train"]["caption_column_used"], "cap_norm")
            self.assertEqual(
                receipt2["datasets"]["gold_train"]["full"]["n_contradictory_groups"], 1)

    def test_corrupt_md5_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pub = tmp / "public_train.parquet"
            gold = tmp / "gold_train.parquet"
            out = tmp / "out.json"
            write_parquet(pub, {
                "cap_norm": ["ok"],
                "img_bytes_md5": ["not-a-md5"],
                "y": [0],
            })
            write_parquet(gold, {
                "cap_norm": ["ok"],
                "img_bytes_md5": [MD_A],
                "y": [0],
            })
            with self.assertRaises(ValueError):
                _mod.main(["--public-train", str(pub), "--gold-train", str(gold),
                           "--out", str(out)])

    def test_invalid_label_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for bad in (4, -1):
                pub = tmp / "public_train.parquet"
                gold = tmp / "gold_train.parquet"
                out = tmp / "out.json"
                write_parquet(pub, {
                    "cap_norm": ["ok"],
                    "img_bytes_md5": [MD_A],
                    "y": [bad],
                })
                write_parquet(gold, {
                    "cap_norm": ["ok"],
                    "img_bytes_md5": [MD_A],
                    "y": [0],
                })
                with self.assertRaises(ValueError):
                    _mod.main(["--public-train", str(pub), "--gold-train", str(gold),
                               "--out", str(out)])

    def test_receipt_hashes_and_no_private_paths(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            public_data = {
                "cap_norm": ["alpha", "beta"],
                "img_bytes_md5": [MD_A, MD_B],
                "y": [1, 2],
            }
            gold_data = {
                "cap_norm": ["gamma"],
                "img_bytes_md5": [MD_C],
                "y": [3],
            }
            receipt = run_cli(public_data, gold_data, tmp)
            self.assertIn("public_train", receipt["input_sha256"])
            self.assertIn("gold_train", receipt["input_sha256"])
            for v in receipt["input_sha256"].values():
                self.assertEqual(len(v), 64)
            blob = json.dumps(receipt)
            self.assertNotIn(str(tmp), blob)
            self.assertNotIn("alpha", blob)
            self.assertNotIn("beta", blob)
            self.assertNotIn("gamma", blob)
            self.assertEqual(receipt["cap_limit"], 2400)


if __name__ == "__main__":
    unittest.main()
