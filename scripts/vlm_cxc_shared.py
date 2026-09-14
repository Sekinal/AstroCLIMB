import torch

TASK = "astroclimb_cxc_text_pair_v1"
PROMPT_VERSION = "cxc_ab_v1"


def _check_cap(cap, name):
    if not isinstance(cap, str) or not cap.strip():
        raise ValueError(f"{name} must be a nonblank string")


def encode_cxc(proc, rows, max_cap_chars=2400, device="cpu"):
    if not rows:
        raise ValueError("rows must be non-empty")
    if isinstance(max_cap_chars, bool) or not isinstance(max_cap_chars, int) or max_cap_chars <= 0:
        raise ValueError("max_cap_chars must be a positive integer")
    caps = []
    for r in rows:
        _check_cap(r.get("cap_a"), "cap_a")
        _check_cap(r.get("cap_b"), "cap_b")
        caps.append((r["cap_a"], r["cap_b"]))
    for s in proc.tokenizer.all_special_tokens:
        if s and any(s in a or s in b for a, b in caps):
            raise ValueError("special token literal embedded in caption")
    prompts = []
    for a, b in caps:
        text = (
            f"Caption A:\n{a[:max_cap_chars]}\n\n"
            f"Caption B:\n{b[:max_cap_chars]}\n\n"
            "Are Caption A and Caption B from the same paper, related papers, or unrelated papers?"
        )
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    proc.tokenizer.padding_side = "right"
    enc = proc(text=prompts, padding=True, return_tensors="pt")
    for k in ("pixel_values", "image_grid_thw", "video", "video_grid_thw"):
        if k in enc:
            raise ValueError(f"unexpected {k} in encoded batch")
    if "mm_token_type_ids" in enc and enc["mm_token_type_ids"].any():
        raise ValueError("nonzero mm_token_type_ids")
    if enc["input_ids"].shape[0] != len(prompts):
        raise ValueError("batch size mismatch")
    return enc.to(device)


def train_encode(proc, rows, args):
    ys = []
    for r in rows:
        y = r.get("y")
        if isinstance(y, bool) or not isinstance(y, int) or y not in (1, 2, 3):
            raise ValueError("y must be integer 1, 2, or 3")
        ys.append(y)
    enc = encode_cxc(proc, rows, args.max_cap_chars, device="cuda")
    labels = torch.tensor(ys, dtype=torch.long, device="cuda")
    return enc, labels


def mask_same_figure(logits):
    if not torch.is_floating_point(logits) or logits.dim() == 0 or logits.shape[-1] != 4:
        raise ValueError("expected floating tensor with final dim 4")
    out = logits.clone()
    out[..., 0] = torch.finfo(out.dtype).min
    return out
