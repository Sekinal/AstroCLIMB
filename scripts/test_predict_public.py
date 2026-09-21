"""CPU-only checks for the public release inference path; no GPU required."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import predict_public


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


class PublicReleaseTests(unittest.TestCase):
    def _checkpoint(self, tmp, name="cxi837", marker=None):
        root = Path(tmp) / name
        files = {}
        files["adapter/adapter_model.safetensors"] = _write(
            root / "adapter" / "adapter_model.safetensors", b"adapter-weights"
        )
        files["adapter/adapter_config.json"] = _write(
            root / "adapter" / "adapter_config.json",
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}).encode(),
        )
        files["head.pt"] = _write(root / "head.pt", b"head-weights")
        args = {
            "model": "Qwen/Qwen3.5-4B",
            "lora_r": 16,
            "max_pixels": 262144,
            "max_cap_chars": 2400,
            "four_bit": False,
        }
        if marker is not None:
            args["task"] = marker
        files["meta.json"] = _write(
            root / "meta.json",
            json.dumps({"args": args, "labels": list(predict_public.LABELS)}).encode(),
        )
        files["processor/tokenizer.json"] = _write(
            root / "processor" / "tokenizer.json", b"tokenizer"
        )
        manifest = {"checkpoints": {name: {"files": files}}}
        manifest_path = Path(tmp) / "public-manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return root, manifest_path, manifest

    def _pinned_model(self, tmp):
        model = Path(tmp) / predict_public.PINNED_REVISION
        model.mkdir()
        (model / "config.json").write_text("{}")
        return model

    def test_clean_release_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path, manifest = self._checkpoint(tmp)
            predict_public.check_model_pin(self._pinned_model(tmp))
            predict_public.verify_release(root, manifest, "cxi837")
            predict_public.check_adapter_config(root)
            predict_public.check_task_checkpoint("cxi", "cxi837", root)

    def test_corrupted_weights_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, manifest = self._checkpoint(tmp)
            target = root / "adapter" / "adapter_model.safetensors"
            target.write_bytes(b"adapter-weightX")
            with self.assertRaises(ValueError):
                predict_public.verify_release(root, manifest, "cxi837")

    def test_missing_manifest_entry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, manifest = self._checkpoint(tmp)
            del manifest["checkpoints"]["cxi837"]["files"]["head.pt"]
            with self.assertRaises(ValueError):
                predict_public.verify_release(root, manifest, "cxi837")

    def test_unknown_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, manifest = self._checkpoint(tmp)
            with self.assertRaises(ValueError):
                predict_public.verify_release(root, manifest, "ixi627")

    def test_wrong_checkpoint_task_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self._checkpoint(tmp, "ixi627", predict_public.IXI_TASK)
            with self.assertRaises(ValueError):
                predict_public.check_task_checkpoint("cxc", "ixi627", root)
            with self.assertRaises(ValueError):
                predict_public.check_task_checkpoint("ixi", "cxi837", root)

    def test_public4000_never_predicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self._checkpoint(tmp, "public4000")
            with self.assertRaises(ValueError):
                predict_public.check_task_checkpoint("cxi", "public4000", root)

    def test_noncanonical_adapter_base_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, manifest = self._checkpoint(tmp)
            (root / "adapter" / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Other/Model"})
            )
            with self.assertRaises(ValueError):
                predict_public.check_adapter_config(root)

    def test_model_pin_checked_before_gpu_import(self):
        before = set(sys.modules)
        with self.assertRaises(ValueError):
            predict_public.check_model_pin(Path("models/wrong-name"))
        added = {name.split(".")[0] for name in set(sys.modules) - before}
        self.assertFalse(added & {"torch", "transformers", "peft"})

    def test_main_rejects_bad_model_pin_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.parquet"
            argv = [
                "--checkpoint", str(Path(tmp) / "cxi837"),
                "--manifest", str(Path(tmp) / "missing.json"),
                "--task", "cxi",
                "--model-path", "models/wrong-name",
                "--pairs", str(Path(tmp) / "pairs.parquet"),
                "--out", str(out),
            ]
            before = set(sys.modules)
            with self.assertRaises(SystemExit):
                predict_public.main(argv)
            added = {name.split(".")[0] for name in set(sys.modules) - before}
            self.assertFalse(added & {"torch", "transformers", "peft"})

    def test_main_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_model(tmp)
            root, manifest_path, _ = self._checkpoint(tmp)
            out = Path(tmp) / "out.parquet"
            out.write_bytes(b"existing")
            argv = [
                "--checkpoint", str(root),
                "--manifest", str(manifest_path),
                "--task", "cxi",
                "--model-path", str(model),
                "--pairs", str(Path(tmp) / "pairs.parquet"),
                "--out", str(out),
            ]
            with self.assertRaises(SystemExit):
                predict_public.main(argv)
            self.assertEqual(out.read_bytes(), b"existing")

    def test_historical_input_validator_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            import polars as pl

            pairs = Path(tmp) / "pairs.parquet"
            pl.DataFrame(
                {
                    "pair_id": [1, 1],
                    "cap_text": ["a caption", "another caption"],
                    "image_path": ["a.png", "b.png"],
                }
            ).write_parquet(pairs)
            with self.assertRaises(ValueError):
                predict_public.read_validated_inputs("cxi", pairs, None)


if __name__ == "__main__":
    unittest.main()
