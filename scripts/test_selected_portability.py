"""CPU-only checks of the location adapter; no model dependencies required."""
import argparse
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from astroclimb import reproduction as runner


class PortabilityTests(unittest.TestCase):
    def test_relocation_preserves_archival_configuration_and_restores_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
            model.mkdir()
            (model / "config.json").write_text("{}")
            archived = argparse.Namespace(model="unavailable-archival-location", batch=1)
            calls = []

            def load(config, *args, **kwargs):
                calls.append((config, args, kwargs))
                return "loaded"

            trainer = SimpleNamespace(load_model=load)
            entry = SimpleNamespace(main=lambda: trainer.load_model(archived, "adapter", train=False))
            png = SimpleNamespace(MAX_TEXT_CHUNK=1024**2, MAX_TEXT_MEMORY=64 * 1024**2)
            argv = ["run_selected.py", "final_vlm_export", "--model-path", str(model), "--", "--checkpoint", "example"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=png)}), patch.object(runner.importlib, "import_module", side_effect=lambda name: trainer if name == "vlm_cxi_peft" else entry):
                runner.main()
                self.assertIs(sys.argv, argv)
            self.assertEqual(archived.model, "unavailable-archival-location")
            self.assertEqual(calls[0][0].model, str(model))
            self.assertEqual(calls[0][1:], (("adapter",), {"train": False}))
            self.assertIs(trainer.load_model, load)
            self.assertEqual(png.MAX_TEXT_CHUNK, 16 * 1024**2)

    def test_wrong_snapshot_rejected_before_import(self):
        with patch.object(sys, "argv", ["run", "final_vlm_export", "--model-path", "missing"]), patch.object(runner.importlib, "import_module") as imports:
            with self.assertRaises(SystemExit):
                runner.main()
            imports.assert_not_called()


if __name__ == "__main__":
    unittest.main()
