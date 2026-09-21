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


def _passthrough_value(tokens, flag, parser):
    """Return (found, value) for an exact flag match in passthrough tokens.

    Only the exact ``--flag value`` and ``--flag=value`` spellings are
    recognized, so abbreviated prefixes never trigger parent or model
    handling. A present flag without a usable value fails like a normal
    argparse error. The last occurrence wins, matching argparse.
    """
    found = False
    value = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == flag:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is None or nxt == "--" or (nxt.startswith("-") and nxt != "-"):
                parser.error(f"argument {flag}: expected one argument")
            found = True
            value = nxt
            i += 2
            continue
        if token.startswith(flag + "="):
            inline = token[len(flag) + 1:]
            if not inline:
                parser.error(f"argument {flag}: expected one argument")
            found = True
            value = inline
        i += 1
    return found, value


GUARDED_FLAGS = ("--model", "--init-checkpoint", "--resume")


def _reject_guarded_abbreviations(tokens, parser):
    """Reject abbreviated spellings of guarded flags before the native parser sees them.

    Only a proper prefix of a guarded flag is rejected, so exact flags,
    ``--flag=value`` spellings and unrelated flags such as ``--max-pixels``
    are unaffected.
    """
    for token in tokens:
        if token == "--" or not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        if name == "--" or name in GUARDED_FLAGS:
            continue
        for guarded in GUARDED_FLAGS:
            if guarded.startswith(name):
                parser.error(f"unsupported abbreviated flag {name!r}; use {guarded}")


def _is_compatible_model(value, recipe):
    """Accept the configured logical ID or a path naming the pinned revision."""
    return value == recipe["model"]["id"] or Path(value).name == recipe["model"]["revision"]


def _parent_logical_model(parent, recipe, parser):
    """Read the parent checkpoint's recorded logical model without modifying it."""
    try:
        meta = json.loads((Path(parent) / "meta.json").read_text())
        logical = meta["args"]["model"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"cannot read parent logical model from {parent}/meta.json: {exc}")
    if not isinstance(logical, str) or not _is_compatible_model(logical, recipe):
        parser.error(
            "parent checkpoint model {!r} is not the configured model {!r} "
            "or pinned revision {!r}".format(
                logical, recipe["model"]["id"], recipe["model"]["revision"]
            )
        )
    return logical


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
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
    rest = rest[1:] if rest[:1] == ["--"] else rest
    _reject_guarded_abbreviations(rest, parser)
    logical = None
    if args.entrypoint.startswith("vlm_"):
        init_found, init_value = _passthrough_value(rest, "--init-checkpoint", parser)
        resume_found, resume_value = _passthrough_value(rest, "--resume", parser)
        if init_found and resume_found:
            parser.error("--resume and --init-checkpoint are mutually exclusive")
        parent = init_value if init_found else (resume_value if resume_found else None)
        model_found, model_value = _passthrough_value(rest, "--model", parser)
        if parent is not None:
            parent_logical = _parent_logical_model(parent, recipe, parser)
            if model_found:
                if not _is_compatible_model(model_value, recipe):
                    parser.error(
                        "explicit --model {!r} is not the configured model {!r} "
                        "or pinned revision {!r}".format(
                            model_value, recipe["model"]["id"], recipe["model"]["revision"]
                        )
                    )
                if model_value != parent_logical:
                    parser.error(
                        "--model {!r} contradicts parent checkpoint model {!r}".format(
                            model_value, parent_logical
                        )
                    )
                logical = model_value
            else:
                logical = parent_logical
                rest = [*rest, "--model", logical]
        elif model_found:
            if not _is_compatible_model(model_value, recipe):
                parser.error(
                    "explicit --model {!r} is not the configured model {!r} "
                    "or pinned revision {!r}".format(
                        model_value, recipe["model"]["id"], recipe["model"]["revision"]
                    )
                )
            logical = model_value
        else:
            logical = str(model_path)
            rest = [*rest, "--model", logical]
    if logical is not None and logical != str(model_path):
        payload = {"logical_model": recipe["model"]["id"], "pinned_revision": recipe["model"]["revision"]}
        if logical != recipe["model"]["id"]:
            payload["archival_path_identity"] = True
        print(json.dumps(payload))
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
    sys.argv = [args.entrypoint, *rest]
    try:
        importlib.import_module(args.entrypoint).main()
    finally:
        trainer.load_model = original
        sys.argv = old_argv
