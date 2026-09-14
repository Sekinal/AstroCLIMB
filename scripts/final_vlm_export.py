"""Resume-safe label-free inference for vlm_cxi_peft.py checkpoints.

Example:
  python scripts/final_vlm_export.py --checkpoint RUN/checkpoint-00000100 \
    --pairs DATA/dev_inputs.parquet --image-cache DATA/imgcache \
    --out OUTPUT/dev_probs.parquet --batch-size 2

The checkpoint is explicit: no implicit latest/best selection. Only inference
columns are read. Atomic probability chunks are validated and reused on restart;
copy OUT.parts/ too when migrating a spot instance. Output preserves input order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

LABELS = ("same_figure", "same_paper", "related_papers", "unrelated_papers")
PROBS = [f"p_{label}" for label in LABELS]
INPUT_COLUMNS = ("pair_id", "cap_text", "cap_norm", "img_bytes_md5", "image_path")


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_header(checkpoint):
    meta = json.loads((checkpoint / "meta.json").read_text())
    if tuple(meta.get("labels", ())) != LABELS:
        raise ValueError("Checkpoint classifier label order differs from SF/SP/REL/UNR")
    required = ["meta.json", "head.pt", "probe.pt", "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors"]
    for name in required:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    proc_files = sorted(p for p in (checkpoint / "processor").rglob("*") if p.is_file())
    if not proc_files:
        raise ValueError("Saved checkpoint processor required; no fallback allowed")
    files = {name: digest_file(checkpoint / name) for name in required}
    files.update({str(p.relative_to(checkpoint)): digest_file(p) for p in proc_files})
    header = {"files_sha256": files, "labels": LABELS, "step": meta["step"],
              "training_data_hash": meta["data_hash"]}
    encoded = json.dumps(header, sort_keys=True).encode()
    return meta, header, hashlib.sha256(encoded).hexdigest()


def read_inputs(path, image_cache=None):
    import polars as pl

    schema = pl.read_parquet_schema(path)
    if "pair_id" not in schema:
        raise ValueError("Input needs explicit pair_id")
    if not ({"cap_text", "cap_norm"} & schema.keys()):
        raise ValueError("Input needs cap_text or cap_norm")
    if "image_path" not in schema and ("img_bytes_md5" not in schema or image_cache is None):
        raise ValueError("Input needs image_path or img_bytes_md5 plus --image-cache")
    frame = pl.read_parquet(path, columns=[c for c in INPUT_COLUMNS if c in schema])
    if not len(frame) or frame["pair_id"].null_count() or frame["pair_id"].n_unique() != len(frame):
        raise ValueError("Input pair_id must be nonnull, unique, and nonempty")
    cap_col = "cap_text" if "cap_text" in frame.columns else "cap_norm"
    if frame[cap_col].null_count():
        raise ValueError(f"Null captions in {cap_col}")
    # Hash selected, label-free values. Labels or their encoding never enter inference.
    input_hash = hashlib.sha256(frame.write_ndjson().encode()).hexdigest()
    rows = frame.to_dicts()
    for row in rows:
        if row.get("image_path"):
            path_value = Path(row["image_path"])
            row["_image_path"] = path_value if path_value.is_absolute() else path.parent / path_value
        elif row.get("img_bytes_md5") and image_cache:
            md5 = row["img_bytes_md5"]
            if len(md5) != 32 or any(c not in "0123456789abcdefABCDEF" for c in md5):
                raise ValueError(f"Invalid image MD5 for pair {row['pair_id']}")
            row["_image_path"] = image_cache / (md5 + ".img")
        else:
            raise ValueError(f"Missing image reference for pair {row['pair_id']}")
    return frame.select("pair_id"), rows, input_hash


def validate_probabilities(frame, expected_ids):
    import numpy as np

    if frame.columns != ["pair_id", *PROBS]:
        raise ValueError("Probability table schema mismatch")
    if frame["pair_id"].to_list() != expected_ids["pair_id"].to_list():
        raise ValueError("Probability table pair_id alignment mismatch")
    if frame["pair_id"].n_unique() != len(frame):
        raise ValueError("Duplicate output pair_id")
    values = frame.select(PROBS).to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(values.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError("Probabilities do not sum to one")


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def atomic_parquet(path, frame):
    temp = path.with_name(path.name + ".tmp")
    frame.write_parquet(temp)
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    temp.replace(path)


def encode_inputs(proc, rows, config):
    from PIL import Image
    from vlm_cxi_peft import PROMPT

    images, prompts = [], []
    for row in rows:
        # Verify the content-addressed cache on first use, before any prediction.
        if row.get("img_bytes_md5"):
            image_data = row["_image_path"].read_bytes()
            actual_md5 = hashlib.md5(image_data).hexdigest()
            if actual_md5 != row["img_bytes_md5"].lower():
                raise ValueError(f"Image checksum mismatch for pair {row['pair_id']}")
        with Image.open(row["_image_path"]) as im:
            images.append(im.convert("RGB"))
        cap = row["cap_text"] if "cap_text" in row else row["cap_norm"]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": images[-1]},
            {"type": "text", "text": PROMPT.format(cap=cap[:config.max_cap_chars])},
        ]}]
        prompts.append(proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    enc = proc(text=prompts, images=images, padding=True, return_tensors="pt")
    return {key: value.to("cuda") for key, value in enc.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--check-inputs-only", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.chunk_size) < 1:
        parser.error("batch-size and chunk-size must be positive")
    import polars as pl

    meta, header, header_hash = checkpoint_header(args.checkpoint)
    ids, rows, input_hash = read_inputs(args.pairs, args.image_cache)
    print(json.dumps({"event": "inputs_checked", "rows": len(rows),
                      "checkpoint_header_sha256": header_hash}), flush=True)
    if args.check_inputs_only:
        return
    # Relative locations may change on spot migration; hashes identify the payload.
    import vlm_cxi_peft as trainer

    provenance = {"format_version": 1, "checkpoint_header_sha256": header_hash,
                  "checkpoint_header": header, "input_sha256": input_hash,
                  "rows": len(rows), "labels": list(LABELS), "chunk_size": args.chunk_size,
                  "batch_size": args.batch_size, "trainer_sha256": digest_file(trainer.__file__),
                  "exporter_sha256": digest_file(__file__)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    parts = args.out.with_name(args.out.name + ".parts")
    parts.mkdir(exist_ok=True)
    # Prevent competing jobs from writing the same output on a shared disk.
    import fcntl

    lock = (parts / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = parts / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != json.loads(json.dumps(provenance)):
            raise ValueError("Resume provenance mismatch; use another --out")
    else:
        if any(parts.glob("*.parquet")) or args.out.exists():
            raise ValueError("Existing output without provenance; use another --out")
        atomic_json(manifest, provenance)
    chunks = [(start, min(start + args.chunk_size, len(rows)))
              for start in range(0, len(rows), args.chunk_size)]
    pending = []
    for start, end in chunks:
        chunk = parts / f"{start:08d}-{end:08d}.parquet"
        if chunk.exists():
            validate_probabilities(pl.read_parquet(chunk), ids.slice(start, end-start))
        else:
            pending.append((start, end, chunk))
    if pending:
        import torch
        from safetensors.torch import load_file

        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        config = SimpleNamespace(**meta["args"])
        model, head, proc = trainer.load_model(config, args.checkpoint / "adapter", train=False)
        head.load_state_dict(torch.load(args.checkpoint / "head.pt", weights_only=True, map_location="cuda"))
        model.eval().requires_grad_(False)
        head.eval().requires_grad_(False)
        if proc.tokenizer.padding_side != "right":
            raise ValueError("Trainer processor must use right padding")
        state = load_file(args.checkpoint / "adapter" / "adapter_model.safetensors")
        bs = [v for k, v in state.items() if "lora_B" in k.split(".")]
        if not bs or not any(torch.count_nonzero(v).item() for v in bs):
            raise ValueError("Checkpoint LoRA-B missing or all zero")
        del state, bs
        probe = torch.load(args.checkpoint / "probe.pt", weights_only=True, map_location="cuda")
        with torch.inference_mode():
            replay = trainer.logits(model, head, probe["inputs"])
        torch.testing.assert_close(replay, probe["logits"], atol=0.02, rtol=0.01)
        print(json.dumps({"event": "fresh_replay_passed", "max_logit_diff":
                          (replay-probe["logits"]).abs().max().item()}), flush=True)
        del probe, replay
        start_time = time.monotonic()
        completed = len(rows) - sum(end-start for start, end, _ in pending)
        with torch.inference_mode():
            for start, end, chunk in pending:
                probabilities = []
                for offset in range(start, end, args.batch_size):
                    enc = encode_inputs(proc, rows[offset:min(offset+args.batch_size, end)], config)
                    probs = trainer.logits(model, head, enc).float().softmax(-1)
                    probabilities.extend(probs.cpu().tolist())
                    del enc, probs
                frame = ids.slice(start, end-start).hstack(
                    pl.DataFrame(probabilities, schema=PROBS, orient="row"))
                validate_probabilities(frame, ids.slice(start, end-start))
                atomic_parquet(chunk, frame)
                completed += end-start
                print(json.dumps({"event": "chunk_saved", "completed": completed,
                                  "total": len(rows), "elapsed_s": time.monotonic()-start_time}), flush=True)
    output = pl.concat([pl.read_parquet(parts / f"{start:08d}-{end:08d}.parquet")
                        for start, end in chunks])
    validate_probabilities(output, ids)
    atomic_parquet(args.out, output)
    atomic_json(args.out.with_name(args.out.name + ".provenance.json"),
                {**provenance, "output_sha256": digest_file(args.out)})
    print(json.dumps({"event": "complete", "output": str(args.out), "rows": len(output)}), flush=True)


if __name__ == "__main__":
    main()
