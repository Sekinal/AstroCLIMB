"""CPU-only checks of the location adapter; no model dependencies required."""
import argparse
import json
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

    def _pinned_snapshot(self, tmp):
        model = Path(tmp) / "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        model.mkdir()
        (model / "config.json").write_text("{}")
        return model

    def _parent_checkpoint(self, tmp, name, model_value):
        parent = Path(tmp) / name
        parent.mkdir()
        (parent / "meta.json").write_text(json.dumps({"args": {"model": model_value}}))
        return parent

    def _training_stub(self, captured, calls):
        def load(config, *args, **kwargs):
            calls.append((config, args, kwargs))
            return "loaded"

        stub = SimpleNamespace(load_model=load)

        def fake_main():
            captured["argv"] = list(sys.argv)
            cli = argparse.ArgumentParser()
            cli.add_argument("--model", default=None)
            known, _ = cli.parse_known_args(captured["argv"][1:])
            stub.load_model(argparse.Namespace(model=known.model))

        stub.main = fake_main
        return stub

    def _png_stub(self):
        return SimpleNamespace(MAX_TEXT_CHUNK=1024**2, MAX_TEXT_MEMORY=64 * 1024**2)

    def _stub_import(self, stub):
        real = runner.importlib.import_module

        def _fake(name, *args, **kwargs):
            if name in runner.ENTRIES:
                return stub
            return real(name, *args, **kwargs)

        return _fake

    def test_parent_init_passes_logical_model_and_relocates_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_snapshot(tmp)
            physical = str(model.expanduser().resolve())
            parent = self._parent_checkpoint(tmp, "public4000", "Qwen/Qwen3.5-4B")
            before = parent.joinpath("meta.json").read_bytes()
            captured, calls = {}, []
            stub = self._training_stub(captured, calls)
            original_load = stub.load_model
            argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                    "--", "--corpus", "c", "--out", "o",
                    "--init-checkpoint", str(parent)]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=self._stub_import(stub)), patch("builtins.print") as echo:
                runner.main()
                self.assertIs(sys.argv, argv)
            self.assertIn("--model", captured["argv"])
            self.assertEqual(captured["argv"][captured["argv"].index("--model") + 1], "Qwen/Qwen3.5-4B")
            self.assertEqual(calls[0][0].model, physical)
            self.assertEqual(parent.joinpath("meta.json").read_bytes(), before)
            self.assertIs(stub.load_model, original_load)
            echo.assert_called_once()
            receipt = json.loads(echo.call_args[0][0])
            self.assertEqual(receipt["logical_model"], "Qwen/Qwen3.5-4B")
            self.assertEqual(receipt["pinned_revision"], model.name)
            self.assertNotIn(physical, echo.call_args[0][0])

    def test_resume_equals_syntax_preserves_meta_and_relocates_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_snapshot(tmp)
            physical = str(model.expanduser().resolve())
            parent = self._parent_checkpoint(tmp, "ckpt-50", physical)
            before = parent.joinpath("meta.json").read_bytes()
            captured, calls = {}, []
            stub = self._training_stub(captured, calls)
            argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                    "--", "--corpus", "c", "--out", "o", f"--resume={parent}"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=self._stub_import(stub)), patch("builtins.print") as echo:
                runner.main()
            self.assertEqual(captured["argv"][captured["argv"].index("--model") + 1], physical)
            self.assertEqual(calls[0][0].model, physical)
            self.assertEqual(parent.joinpath("meta.json").read_bytes(), before)
            echo.assert_not_called()

    def test_explicit_model_matching_parent_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_snapshot(tmp)
            physical = str(model.expanduser().resolve())
            parent = self._parent_checkpoint(tmp, "public4000", "Qwen/Qwen3.5-4B")
            captured, calls = {}, []
            stub = self._training_stub(captured, calls)
            argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                    "--", "--corpus", "c", "--out", "o",
                    "--init-checkpoint", str(parent), "--model=Qwen/Qwen3.5-4B"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=self._stub_import(stub)), patch("builtins.print"):
                runner.main()
            self.assertEqual(calls[0][0].model, physical)

    def test_wrong_parent_model_rejected_before_trainer_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_snapshot(tmp)
            parent = self._parent_checkpoint(tmp, "public4000", "Other/Model")
            before = parent.joinpath("meta.json").read_bytes()
            calls = []
            stub = SimpleNamespace(load_model=lambda config, *a, **k: calls.append(config) or "loaded",
                                   main=lambda: calls.append("main"))
            argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                    "--", "--corpus", "c", "--out", "o",
                    "--init-checkpoint", str(parent)]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=lambda name: stub):
                with self.assertRaises(SystemExit):
                    runner.main()
                self.assertIs(sys.argv, argv)
            self.assertEqual(calls, [])
            self.assertEqual(parent.joinpath("meta.json").read_bytes(), before)

    def test_explicit_model_contradicting_parent_rejected_before_trainer_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_snapshot(tmp)
            physical = str(model.expanduser().resolve())
            parent = self._parent_checkpoint(tmp, "public4000", "Qwen/Qwen3.5-4B")
            main_calls = []
            stub = SimpleNamespace(load_model=lambda config, *a, **k: "loaded",
                                   main=lambda: main_calls.append("main"))
            argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                    "--", "--corpus", "c", "--out", "o",
                    "--init-checkpoint", str(parent), "--model", physical]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=lambda name: stub):
                with self.assertRaises(SystemExit):
                    runner.main()
            self.assertEqual(main_calls, [])

    def test_training_without_parent_records_physical_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_snapshot(tmp)
            physical = str(model.expanduser().resolve())
            captured, calls = {}, []
            stub = self._training_stub(captured, calls)
            argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                    "--", "--corpus", "c", "--out", "o"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=self._stub_import(stub)), patch("builtins.print") as echo:
                runner.main()
            self.assertEqual(captured["argv"][captured["argv"].index("--model") + 1], physical)
            self.assertEqual(calls[0][0].model, physical)
            echo.assert_not_called()


    def test_relocated_parent_old_path_uses_new_snapshot_without_printing_old_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_snapshot(tmp)
            physical = str(model.expanduser().resolve())
            revision = model.name
            old_abs = f"/root/cache/snapshots/{revision}"
            parent = self._parent_checkpoint(tmp, "public4000", old_abs)
            before = parent.joinpath("meta.json").read_bytes()
            captured, calls = {}, []
            stub = self._training_stub(captured, calls)
            original_load = stub.load_model
            argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                    "--", "--corpus", "c", "--out", "o",
                    "--init-checkpoint", str(parent)]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=self._stub_import(stub)), patch("builtins.print") as echo:
                runner.main()
                self.assertIs(sys.argv, argv)
            self.assertEqual(captured["argv"][captured["argv"].index("--model") + 1], old_abs)
            self.assertEqual(calls[0][0].model, physical)
            self.assertEqual(parent.joinpath("meta.json").read_bytes(), before)
            self.assertIs(stub.load_model, original_load)
            echo.assert_called_once()
            receipt = json.loads(echo.call_args[0][0])
            self.assertEqual(receipt["logical_model"], "Qwen/Qwen3.5-4B")
            self.assertEqual(receipt["pinned_revision"], revision)
            self.assertNotIn("/root/cache", echo.call_args[0][0])

    def test_omitted_delimiter_exact_model_does_not_overwrite_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_snapshot(tmp)
            physical = str(model.expanduser().resolve())
            captured, calls = {}, []
            stub = self._training_stub(captured, calls)
            argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                    "--corpus", "c", "--out", "o",
                    "--max-pixels", "262144", "--model", "Qwen/Qwen3.5-4B"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=self._stub_import(stub)), patch("builtins.print"):
                runner.main()
            self.assertEqual(calls[0][0].model, physical)
            self.assertEqual(captured["argv"][captured["argv"].index("--model") + 1], "Qwen/Qwen3.5-4B")
            self.assertIn("--max-pixels", captured["argv"])

    def test_guarded_abbreviations_rejected_before_trainer_main(self):
        cases = [
            ["--init", str("parent")],
            ["--resum", str("parent")],
            ["--mode", "Qwen/Qwen3.5-4B"],
            ["--init-checkpoint".replace("checkpoint", "check") + "=x"],
        ]
        for extra in cases:
            with tempfile.TemporaryDirectory() as tmp:
                model = self._pinned_snapshot(tmp)
                parent = self._parent_checkpoint(tmp, "public4000", "Qwen/Qwen3.5-4B")
                argv_extra = [a.replace("parent", str(parent)) for a in extra]
                argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                        "--", "--corpus", "c", "--out", "o", *argv_extra]
                main_calls = []
                stub = SimpleNamespace(load_model=lambda config, *a, **k: "loaded",
                                       main=lambda: main_calls.append("main"))
                with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=lambda name: stub):
                    with self.assertRaises(SystemExit):
                        runner.main()
                    self.assertIs(sys.argv, argv)
                self.assertEqual(main_calls, [])

    def test_throwing_main_restores_load_hook_and_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._pinned_snapshot(tmp)
            captured = {}
            calls = []

            def load(config, *args, **kwargs):
                calls.append(config)
                return "loaded"

            stub = SimpleNamespace(load_model=load)

            def boom():
                captured["argv"] = list(sys.argv)
                raise RuntimeError("entry failed")

            stub.main = boom
            original_load = stub.load_model
            argv = ["run_selected.py", "vlm_cxi_peft", "--model-path", str(model),
                    "--", "--corpus", "c", "--out", "o"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"PIL": SimpleNamespace(PngImagePlugin=self._png_stub())}), patch.object(runner.importlib, "import_module", side_effect=self._stub_import(stub)), patch("builtins.print"):
                with self.assertRaises(RuntimeError):
                    runner.main()
                self.assertIs(sys.argv, argv)
            self.assertIs(stub.load_model, original_load)
            self.assertTrue(captured["argv"])


if __name__ == "__main__":
    unittest.main()
