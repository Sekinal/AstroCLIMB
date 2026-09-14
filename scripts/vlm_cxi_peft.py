"""Plain Transformers/PEFT caption-image classifier; no Unsloth dependency.

Use ONLY a separately audited training corpus. This script does not construct
splits or certify upstream data provenance. Schema: train.parquet, val.parquet
with cap_norm, img_bytes_md5, y; imgcache/<md5>.img. Labels are SF/SP/REL/UNR.

Train: python scripts/vlm_cxi_peft.py --corpus CLEAN_CORPUS --out RUN
Resume: same arguments plus --resume RUN/checkpoint-00000050
Transfer: --init-checkpoint PARENT/checkpoint-N loads verified weights only,
with a fresh optimizer/data schedule and recorded parent artifact hashes.
Fresh reload check (separate process, no training): --verify CHECKPOINT
Early guard verifies actual adapter gradients AND updates, optimizer membership,
then saves/reloads adapter tensors and checks logits. Checkpoints include AdamW
state, RNG state and exact next data index. --verify uses a fresh model and the
checkpoint's own processor/input snapshot, never another checkpoint's logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
LABELS = ("same_figure", "same_paper", "related_papers", "unrelated_papers")
PROMPT = "Caption A: {cap}\nQuestion: what is the relation of this figure to the caption?"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--resume", type=Path, help="Restore exact optimizer, RNG and data position"
    )
    source.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Verify and load adapter/head only; fresh optimizer and new corpus",
    )
    p.add_argument("--verify", type=Path)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-pixels", type=int, default=262144)
    p.add_argument("--max-cap-chars", type=int, default=2400)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--guard-steps", type=int, default=5)
    p.add_argument("--deadline-min", type=float, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--four-bit", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    a = p.parse_args()
    if not a.verify and (a.corpus is None or a.out is None):
        p.error("training requires explicit --corpus and --out")
    if min(a.batch, a.accum, a.epochs, a.guard_steps, a.save_every, a.val_every) < 1:
        p.error("batch, accum, epochs, guard/save/val intervals must be positive")
    return a


def load_model(a, adapter=None, train=True):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor

    options = {"dtype": torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if a.four_bit:
        from transformers import BitsAndBytesConfig

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(a.model, **options)
    model.config.use_cache = False
    if a.four_bit and train:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=train)
        model.set_adapter("default")
    else:
        # Full names avoid inadvertently adapting the frozen visual tower.
        suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        targets = [
            n
            for n, m in model.named_modules()
            if n.endswith(suffixes)
            and "visual" not in n
            and "vision" not in n
            and hasattr(m, "weight")
        ]
        if not targets:
            raise RuntimeError("No language LoRA targets found")
        model = get_peft_model(
            model,
            LoraConfig(
                r=a.lora_r,
                lora_alpha=2 * a.lora_r,
                lora_dropout=0.0,
                bias="none",
                target_modules=targets,
            ),
        )
    if train and not a.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    proc_source = adapter.parent / "processor" if adapter else None
    proc = AutoProcessor.from_pretrained(
        str(proc_source) if proc_source and proc_source.exists() else a.model
    )
    proc.tokenizer.padding_side = "right"
    proc.image_processor.size = {"shortest_edge": 65536, "longest_edge": a.max_pixels}
    head = torch.nn.Linear(model.config.text_config.hidden_size, 4).to("cuda", torch.float32)
    return model, head, proc


def logits(model, head, enc):
    import torch

    o = model(**enc, logits_to_keep=1, output_hidden_states=True, use_cache=False)
    h = o.hidden_states[-1]
    idx = enc["attention_mask"].sum(1) - 1
    return head(h[torch.arange(h.shape[0], device=h.device), idx].float())


def encode(proc, rows, a):
    import torch
    from PIL import Image

    images, prompts = [], []
    for r in rows:
        with Image.open(a.corpus / "imgcache" / (r["img_bytes_md5"] + ".img")) as im:
            images.append(im.convert("RGB"))
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[-1]},
                    {
                        "type": "text",
                        "text": PROMPT.format(
                            cap=r.get("cap_text", r["cap_norm"])[: a.max_cap_chars]
                        ),
                    },
                ],
            }
        ]
        prompts.append(
            proc.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        )
    enc = proc(text=prompts, images=images, padding=True, return_tensors="pt")
    return {k: v.to("cuda") for k, v in enc.items()}, torch.tensor(
        [r["y"] for r in rows], device="cuda"
    )


def main():
    a = arguments()
    import polars as pl
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from sklearn.metrics import f1_score

    if a.verify:
        ck = a.verify
        meta = json.loads((ck / "meta.json").read_text())
        for key, value in meta["args"].items():
            setattr(a, key, value)
        model, head, _ = load_model(a, ck / "adapter", train=False)
        head.load_state_dict(torch.load(ck / "head.pt", weights_only=True, map_location="cuda"))
        probe = torch.load(ck / "probe.pt", weights_only=True, map_location="cuda")
        model.eval()
        head.eval()
        with torch.no_grad():
            result = logits(model, head, probe["inputs"])
        torch.testing.assert_close(result, probe["logits"], atol=0.02, rtol=0.01)
        state = load_file(ck / "adapter" / "adapter_model.safetensors")
        bs = [v for k, v in state.items() if "lora_B" in k.split(".")]
        if not bs or not any(torch.count_nonzero(v).item() for v in bs):
            raise RuntimeError("Saved LoRA-B missing or all zero")
        print(
            json.dumps(
                {
                    "fresh_reload_verified": str(ck),
                    "max_logit_diff": (result - probe["logits"]).abs().max().item(),
                }
            ),
            flush=True,
        )
        return
    a.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}

    def log(event, **kw):
        line = json.dumps({"event": event, **kw})
        print(line, flush=True)
        with (a.out / "train.jsonl").open("a") as f:
            f.write(line + "\n")

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    rows = pl.read_parquet(a.corpus / "train.parquet").to_dicts()
    val = pl.read_parquet(a.corpus / "val.parquet").to_dicts()[: a.val_limit]
    random.Random(a.seed).shuffle(rows)
    if a.max_train:
        rows = rows[: a.max_train]
    if not rows or not val:
        raise RuntimeError("Empty training or validation split")
    data_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    source_checkpoint = a.resume or a.init_checkpoint
    source_meta = None
    if source_checkpoint:
        source_meta = json.loads((source_checkpoint / "meta.json").read_text())
        if source_meta["args"]["model"] != a.model:
            raise RuntimeError("Checkpoint base model differs from --model")
        if source_meta["args"]["lora_r"] != a.lora_r:
            raise RuntimeError("Checkpoint LoRA rank differs from --lora-r")
        if tuple(source_meta["labels"]) != LABELS:
            raise RuntimeError("Checkpoint class order differs")
    model, head, proc = load_model(a, source_checkpoint / "adapter" if source_checkpoint else None)
    parent_checkpoint = source_meta.get("parent_checkpoint") if source_meta else None
    if a.init_checkpoint:
        head.load_state_dict(
            torch.load(a.init_checkpoint / "head.pt", weights_only=True, map_location="cuda")
        )
        parent_probe = torch.load(
            a.init_checkpoint / "probe.pt", weights_only=True, map_location="cuda"
        )
        model.eval()
        head.eval()
        with torch.no_grad():
            replay = logits(model, head, parent_probe["inputs"])
        torch.testing.assert_close(replay, parent_probe["logits"], atol=0.02, rtol=0.01)
        hashes = {}
        for relative in (
            "meta.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "head.pt",
        ):
            digest = hashlib.sha256()
            with (a.init_checkpoint / relative).open("rb") as fh:
                for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            hashes[relative] = digest.hexdigest()
        parent_checkpoint = {
            "path": str(a.init_checkpoint.resolve()),
            "files_sha256": hashes,
            "manifest_sha256": hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode()
            ).hexdigest(),
            "probe_max_logit_diff": (replay - parent_probe["logits"]).abs().max().item(),
        }
        log("init_checkpoint_verified", **parent_checkpoint)
        del parent_probe, replay
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any("lora_" not in n for n, p in named):
        raise RuntimeError("Expected only adapter parameters trainable in backbone")
    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in named], "lr": a.lr, "base_lr": a.lr},
            {"params": list(head.parameters()), "lr": a.head_lr, "base_lr": a.head_lr},
        ],
        weight_decay=0.01,
    )
    optimizer_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) in optimizer_ids for n, p in named)
    assert all(id(p) in optimizer_ids for p in head.parameters())
    start_epoch = start_index = step = 0
    best = -1.0
    if a.resume:
        meta = json.loads((a.resume / "meta.json").read_text())
        if meta["data_hash"] != data_hash:
            raise RuntimeError("Resume data fingerprint mismatch")
        for key in (
            "model",
            "batch",
            "accum",
            "seed",
            "max_train",
            "max_pixels",
            "max_cap_chars",
            "four_bit",
            "epochs",
            "max_steps",
        ):
            if config[key] != meta["args"][key]:
                raise RuntimeError(f"Resume config mismatch: {key}")
        head.load_state_dict(
            torch.load(a.resume / "head.pt", weights_only=True, map_location="cuda")
        )
        state = torch.load(a.resume / "trainer.pt", weights_only=False, map_location="cpu")
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        start_epoch, start_index, step, best = (
            meta["epoch"],
            meta["next_index"],
            meta["step"],
            meta["best"],
        )
    guard_initial = {n: p.detach().cpu().clone() for n, p in named if "lora_B" in n.split(".")}
    if not guard_initial:
        raise RuntimeError("No LoRA-B parameters found")
    all_params = [p for n, p in named] + list(head.parameters())
    probe_enc, _ = encode(proc, rows[: a.batch], a)

    def save(epoch, next_index):
        ck = a.out / f"checkpoint-{step:08d}"
        tmp = a.out / (ck.name + ".tmp")
        tmp.mkdir(exist_ok=False)
        model.eval()
        head.eval()
        with torch.no_grad():
            before = logits(model, head, probe_enc).detach().cpu()
        model.save_pretrained(tmp / "adapter", safe_serialization=True)
        proc.save_pretrained(tmp / "processor")
        torch.save(head.state_dict(), tmp / "head.pt")
        saved = load_file(tmp / "adapter" / "adapter_model.safetensors")
        # Serialization removes the adapter name. Match the component, not `.default`.
        b_values = [v for k, v in saved.items() if "lora_B" in k.split(".")]
        if not b_values or not any(torch.count_nonzero(v).item() for v in b_values):
            raise RuntimeError("Checkpoint LoRA-B is missing or zero")
        current = get_peft_model_state_dict(model)
        if saved.keys() != current.keys():
            raise RuntimeError("Serialized adapter key set differs")
        for key in saved:
            torch.testing.assert_close(saved[key], current[key].detach().cpu(), rtol=0, atol=0)
        set_peft_model_state_dict(model, saved, adapter_name="default")
        model.set_adapter("default")
        with torch.no_grad():
            after = logits(model, head, probe_enc).detach().cpu()
        torch.testing.assert_close(after, before, rtol=0.001, atol=0.001)
        torch.save(
            {"inputs": {k: v.cpu() for k, v in probe_enc.items()}, "logits": before},
            tmp / "probe.pt",
        )
        torch.save(
            {
                "optimizer": opt.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
            },
            tmp / "trainer.pt",
        )
        (tmp / "meta.json").write_text(
            json.dumps(
                {
                    "args": config,
                    "step": step,
                    "epoch": epoch,
                    "next_index": next_index,
                    "best": best,
                    "data_hash": data_hash,
                    "parent_checkpoint": parent_checkpoint,
                    "labels": LABELS,
                },
                indent=2,
            )
        )
        tmp.rename(ck)
        (a.out / "latest.txt").write_text(str(ck.resolve()) + "\n")
        model.train()
        head.train()
        log("checkpoint", path=str(ck), step=step, reload_verified=True)
        return ck

    total = math.ceil(len(rows) / (a.batch * a.accum)) * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    if step >= total:
        log("already_complete", step=step, total_steps=total)
        return
    deadline = time.monotonic() + a.deadline_min * 60
    base_step = step
    model.train()
    head.train()
    log(
        "ready",
        train_rows=len(rows),
        val_rows=len(val),
        total_steps=total,
        adapter_tensors=len(named),
        data_hash=data_hash,
        config=config,
    )
    last_saved = -1
    for epoch in range(start_epoch, a.epochs):
        for index in range(
            start_index if epoch == start_epoch else 0, len(rows), a.batch * a.accum
        ):
            batch_started = time.monotonic()
            group = rows[index : index + a.batch * a.accum]
            opt.zero_grad(set_to_none=True)
            loss_value = 0.0
            for j in range(0, len(group), a.batch):
                batch = group[j : j + a.batch]
                enc, y = encode(proc, batch, a)
                loss = torch.nn.functional.cross_entropy(logits(model, head, enc), y)
                (loss * (len(batch) / len(group))).backward()
                loss_value += loss.item() * len(batch) / len(group)
            if step - base_step < a.guard_steps:
                b_grads = [
                    p.grad for n, p in named if "lora_B" in n.split(".") and p.grad is not None
                ]
                if not b_grads or not any(g.count_nonzero().item() for g in b_grads):
                    raise RuntimeError(
                        "Adapter LoRA-B has no nonzero gradients; stopping before long run"
                    )
                if any(
                    not torch.isfinite(p.grad).all().item()
                    for p in all_params
                    if p.grad is not None
                ):
                    raise RuntimeError("Nonfinite training gradient")
            torch.nn.utils.clip_grad_norm_(all_params, 1.0, error_if_nonfinite=True)
            scale = min(1.0, (step + 1) / max(1, min(50, int(total * 0.03))))
            scale *= 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total))))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            opt.step()
            step += 1
            next_index = index + len(group)
            if step - base_step <= a.guard_steps:
                delta = max(
                    (p.detach().cpu() - guard_initial[n]).abs().max().item()
                    for n, p in named
                    if n in guard_initial
                )
                if delta <= 0:
                    raise RuntimeError("Adapter gradients present but LoRA-B did not change")
                log(
                    "adapter_guard",
                    step=step,
                    max_B_update=delta,
                    seconds_per_sample=(time.monotonic() - batch_started) / len(group),
                    peak_GB=torch.cuda.max_memory_allocated() / 2**30,
                )
            if step % 10 == 0:
                log(
                    "train",
                    step=step,
                    loss=loss_value,
                    seconds_per_sample=(time.monotonic() - batch_started) / len(group),
                    sequence_tokens=int(enc["input_ids"].shape[1]),
                    peak_GB=torch.cuda.max_memory_allocated() / 2**30,
                )
            if step % a.val_every == 0:
                model.eval()
                head.eval()
                preds, truth = [], []
                with torch.no_grad():
                    for j in range(0, len(val), a.batch):
                        enc, y = encode(proc, val[j : j + a.batch], a)
                        preds.extend(logits(model, head, enc).argmax(1).cpu().tolist())
                        truth.extend(y.cpu().tolist())
                score = f1_score(
                    truth, preds, labels=list(range(4)), average="macro", zero_division=0
                )
                improved = score > best
                best = max(best, score)
                ck = save(epoch, next_index)
                last_saved = step
                if improved:
                    (a.out / "best.txt").write_text(str(ck.resolve()) + "\n")
                log("validation", step=step, macro=float(score), best=float(best))
            stop = step >= total or time.monotonic() >= deadline
            if last_saved != step and (
                step % a.save_every == 0 or step - base_step == a.guard_steps or stop
            ):
                save(epoch, next_index)
                last_saved = step
            if stop:
                log("done", step=step, best=best)
                return
    log("done", step=step, best=best)


if __name__ == "__main__":
    main()
