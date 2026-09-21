"""Public release inference for selected classifier checkpoints.

Disclosure: this utility verifies release file hashes (weights, config,
processor, head) and inference inputs. It does not replay the historical
probe.pt, which serializes TRAIN examples and is not shipped. There is no
GPU numerical validation claim; the output sidecar records
``gpu_replay_verified: false``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

MODEL_ID = "Qwen/Qwen3.5-4B"
PINNED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LABELS = ("same_figure", "same_paper", "related_papers", "unrelated_papers")
PROBS = ["p_" + label for label in LABELS]
EXPECTED_CHECKPOINT = {"cxi": "cxi837", "ixi": "ixi627", "cxc": "cxc414"}
IXI_TASK = "astroclimb_ixi_two_image_v1"
CXC_TASK = "astroclimb_cxc_text_pair_v1"
REQUIRED_FILES = (
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "head.pt",
    "meta.json",
)
DISCLOSURE = (
    "Public inference validates release file hashes and inputs only; "
    "historical probe.pt replay was not performed and no GPU equivalence "
    "with historical scores is claimed."
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_model_pin(model_path):
    """Require the pinned snapshot directory before any GPU import."""
    path = Path(model_path).expanduser()
    if path.name != PINNED_REVISION or not (path / "config.json").is_file():
        raise ValueError(
            "--model-path must point to the pinned snapshot directory "
            "named by its revision and containing config.json"
        )
    return path


def load_manifest(manifest_path):
    try:
        data = json.loads(Path(manifest_path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("checkpoints"), dict):
        raise ValueError("manifest must contain a checkpoints mapping")
    return data


def verify_release(checkpoint_dir, manifest_data, checkpoint_name):
    """Verify every manifest-listed file hash and size before model loading."""
    checkpoint_dir = Path(checkpoint_dir)
    try:
        entry = manifest_data["checkpoints"][checkpoint_name]
        files = entry["files"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"manifest has no files entry for checkpoint {checkpoint_name!r}"
        ) from exc
    if not isinstance(files, dict) or not files:
        raise ValueError(f"manifest files entry for {checkpoint_name!r} is empty")
    for required in REQUIRED_FILES:
        if required not in files:
            raise ValueError(f"manifest entry for {checkpoint_name!r} lacks {required}")
    if not any(key.startswith("processor/") for key in files):
        raise ValueError(
            f"manifest entry for {checkpoint_name!r} lacks processor files"
        )
    for relative, info in files.items():
        if not isinstance(info, dict) or "sha256" not in info or "bytes" not in info:
            raise ValueError(f"manifest file record for {relative!r} needs sha256 and bytes")
        target = checkpoint_dir / relative
        if not target.is_file():
            raise ValueError(f"release file missing: {relative}")
        if target.stat().st_size != info["bytes"]:
            raise ValueError(f"release file size mismatch: {relative}")
        if sha256_file(target) != info["sha256"]:
            raise ValueError(f"release file hash mismatch: {relative}")
    return entry


def check_adapter_config(checkpoint_dir):
    try:
        config = json.loads(
            (Path(checkpoint_dir) / "adapter" / "adapter_config.json").read_text()
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read adapter_config.json: {exc}") from exc
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError(
            "adapter base_model_name_or_path is not the canonical model ID "
            f"{MODEL_ID!r}"
        )
    return config


def check_task_checkpoint(task, checkpoint_name, checkpoint_dir):
    """Bind a release folder to its task; public4000 is never a gold checkpoint."""
    if checkpoint_name == "public4000":
        raise ValueError("public4000 is not a selected gold checkpoint")
    expected = EXPECTED_CHECKPOINT.get(task)
    if expected is None:
        raise ValueError(f"unknown task {task!r}")
    if checkpoint_name != expected:
        raise ValueError(
            f"checkpoint {checkpoint_name!r} does not match task {task!r} "
            f"(expected {expected!r})"
        )
    try:
        meta = json.loads((Path(checkpoint_dir) / "meta.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read meta.json: {exc}") from exc
    if tuple(meta.get("labels", ())) != LABELS:
        raise ValueError("checkpoint label order differs from SF/SP/REL/UNR")
    marker = meta.get("args", {}).get("task") if isinstance(meta.get("args"), dict) else None
    if task == "ixi" and marker != IXI_TASK:
        raise ValueError("IXI checkpoint task marker mismatch")
    if task == "cxc" and marker != CXC_TASK:
        raise ValueError("CXC checkpoint task marker mismatch")
    if task == "cxi" and marker in (IXI_TASK, CXC_TASK):
        raise ValueError("CXI checkpoint carries another task marker")
    return meta


def read_validated_inputs(task, pairs, image_cache):
    """Reuse the historical label-free input readers without modifying them."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if task == "cxi":
        import final_vlm_export as export

        return export.read_inputs(Path(pairs), Path(image_cache) if image_cache else None)
    if task == "ixi":
        import final_ixi_vlm_export as ixi_export

        return ixi_export.read_inputs(Path(pairs), Path(image_cache) if image_cache else None)
    import final_cxc_vlm_export as cxc_export

    return cxc_export.read_inputs(Path(pairs), None)


def run_gpu_inference(task, checkpoint_dir, model_path, ids, rows, meta):
    """Load the verified release and predict one single-sample batch at a time."""
    import torch
    from PIL import PngImagePlugin

    PngImagePlugin.MAX_TEXT_CHUNK = 16 * 1024 * 1024
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import final_vlm_export as export
    import vlm_cxi_peft as trainer

    mask = None
    if task == "ixi":
        import final_ixi_vlm_export as ixi_export
        import vlm_ixi_shared as shared

        encode_inputs = ixi_export.encode_inputs
        mask = shared.mask_same_figure
    elif task == "cxc":
        import final_cxc_vlm_export as cxc_export
        import vlm_cxc_shared as shared

        encode_inputs = cxc_export.encode_inputs
        mask = shared.mask_same_figure
    else:
        encode_inputs = export.encode_inputs
    if not torch.cuda.is_available():
        raise RuntimeError("public GPU inference requires CUDA")
    config = SimpleNamespace(**meta["args"])
    config.model = str(Path(model_path).expanduser().resolve())
    if getattr(config, "four_bit", False):
        raise ValueError("quantized inference is not supported")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    model, head, proc = trainer.load_model(
        config, Path(checkpoint_dir) / "adapter", train=False
    )
    head.load_state_dict(
        torch.load(Path(checkpoint_dir) / "head.pt", weights_only=True, map_location="cuda")
    )
    model.eval()
    head.eval()
    if proc.tokenizer.padding_side != "right":
        raise ValueError("release processor must use right padding")
    probabilities = []
    with torch.inference_mode():
        for row in rows:
            enc = encode_inputs(proc, [row], config)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = trainer.logits(model, head, enc)
                if mask is not None:
                    raw = mask(raw)
                probs = raw.float().softmax(-1)
            probabilities.extend(probs.cpu().tolist())
            del enc, raw, probs
    import polars as pl

    frame = ids.hstack(pl.DataFrame(probabilities, schema=PROBS, orient="row"))
    export.validate_probabilities(frame, ids)
    if task in ("ixi", "cxc") and (frame["p_same_figure"] != 0).any():
        raise ValueError(f"{task} must retain the zero same-figure probability mask")
    return frame


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("cxi", "ixi", "cxc"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("cxi", "ixi", "cxc"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out already exists; choose a new output")
    # Pinned-model and release-hash checks run before any torch import.
    try:
        check_model_pin(args.model_path)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        manifest_data = load_manifest(args.manifest)
        checkpoint_name = Path(args.checkpoint).name
        verify_release(args.checkpoint, manifest_data, checkpoint_name)
        check_adapter_config(args.checkpoint)
        meta = check_task_checkpoint(args.task, checkpoint_name, args.checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        ids, rows, input_hash = read_validated_inputs(
            args.task, args.pairs, args.image_cache
        )
    except Exception as exc:
        parser.error(f"invalid inputs: {exc}")
    try:
        frame = run_gpu_inference(
            args.task, args.checkpoint, args.model_path, ids, rows, meta
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.out)
    sidecar = args.out.with_name(args.out.name + ".sidecar.json")
    sidecar.write_text(
        json.dumps(
            {
                "disclosure": DISCLOSURE,
                "gpu_replay_verified": False,
                "task": args.task,
                "checkpoint": checkpoint_name,
                "input_sha256": input_hash,
                "manifest_sha256": sha256_file(args.manifest),
                "prediction_sha256": sha256_file(args.out),
                "rows": len(frame),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.out),
                "rows": len(frame),
                "gpu_replay_verified": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
