"""Location-only adapter; historical source and checkpoint hashes stay intact."""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = (
    "vlm_cxi_peft", "vlm_ixi_peft", "vlm_cxc_peft",
    "final_vlm_export", "final_ixi_vlm_export", "final_cxc_vlm_export",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint", choices=ENTRIES)
    parser.add_argument("--model-path", required=True, type=Path)
    args, rest = parser.parse_known_args()
    recipe = json.loads((ROOT / "configs/selected-recipe.json").read_text())
    model_path = args.model_path.expanduser().resolve()
    if model_path.name != recipe["model"]["revision"] or not (model_path / "config.json").is_file():
        parser.error("--model-path must point to the pinned snapshot directory named by its revision")
    for name, expected in recipe["verified_source_sha256"].items():
        if hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Selected source changed: {name}")
    sys.path.insert(0, str(ROOT / "scripts"))
    from PIL import PngImagePlugin
    PngImagePlugin.MAX_TEXT_CHUNK = 16 * 1024 * 1024
    assert PngImagePlugin.MAX_TEXT_MEMORY == 64 * 1024 * 1024
    trainer = importlib.import_module("vlm_cxi_peft")
    original = trainer.load_model

    def relocated(config, *positional, **keywords):
        config = copy.copy(config)
        config.model = str(model_path)
        return original(config, *positional, **keywords)

    trainer.load_model = relocated
    old_argv = sys.argv
    rest = rest[1:] if rest[:1] == ["--"] else rest
    if args.entrypoint.startswith("vlm_") and "--model" not in rest:
        rest += ["--model", str(model_path)]
    sys.argv = [args.entrypoint, *rest]
    try:
        importlib.import_module(args.entrypoint).main()
    finally:
        trainer.load_model = original
        sys.argv = old_argv
