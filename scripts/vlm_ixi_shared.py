"""Two-image (Figure A / Figure B) encoder for AstroCLIMB Qwen3.5 IXI classification."""

import hashlib
import re
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

TASK = "astroclimb_ixi_two_image_v1"
PROMPT_VERSION = "ixi_ab_v1"
QUESTION = "Are Figure A and Figure B from the same paper, related papers, or unrelated papers?"
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")


def _decode(image_cache, md5):
    if not isinstance(md5, str) or not _MD5_RE.fullmatch(md5):
        raise ValueError(f"img hash must be a lowercase 32-hex MD5, got {md5!r}")
    path = Path(image_cache) / (md5 + ".img")
    if not path.is_file():
        raise ValueError(f"image cache entry is absent or not a file: {path}")
    data = path.read_bytes()
    if hashlib.md5(data).hexdigest() != md5:
        raise ValueError(f"cached bytes do not match recorded MD5 {md5}")
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except Exception as exc:
        raise ValueError(f"undecodable image for MD5 {md5}") from exc
    return image.convert("RGB")


def _build_prompt(proc):
    content = [
        {"type": "text", "text": "Figure A:"},
        {"type": "image"},
        {"type": "text", "text": "Figure B:"},
        {"type": "image"},
        {"type": "text", "text": QUESTION},
    ]
    return proc.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def encode_ixi(proc, rows, image_cache, max_pixels=262144, device="cpu"):
    if not rows:
        raise ValueError("rows must be a non-empty sequence")
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels < 65536:
        raise ValueError("max_pixels must be an integer >= 65536")
    prompt = _build_prompt(proc)
    flat_images = []
    for row in rows:
        if not row or "img_a" not in row or "img_b" not in row:
            raise ValueError("each row must provide img_a and img_b MD5 hashes")
        flat_images += (
            _decode(image_cache, row["img_a"]),
            _decode(image_cache, row["img_b"]),
        )
    proc.tokenizer.padding_side = "right"
    proc.image_processor.size = {"shortest_edge": 65536, "longest_edge": max_pixels}
    enc = proc(text=[prompt] * len(rows), images=flat_images, padding=True, return_tensors="pt")
    if enc["image_grid_thw"].shape[0] != 2 * len(rows) or enc["input_ids"].shape[0] != len(rows):
        raise ValueError("processor output must hold 2 grid entries and 1 sample per row")
    return {key: value.to(device) for key, value in enc.items()}


def train_encode(proc, rows, args):
    ys = []
    for row in rows:
        y = row["y"]
        if isinstance(y, bool) or not isinstance(y, int) or y not in (1, 2, 3):
            raise ValueError(f"label y must be integer 1, 2 or 3, got {y!r}")
        ys.append(y)
    enc = encode_ixi(proc, rows, Path(args.corpus) / "imgcache", args.max_pixels, device="cuda")
    return enc, torch.tensor(ys, device="cuda")


def mask_same_figure(raw_logits):
    if (
        not isinstance(raw_logits, torch.Tensor)
        or raw_logits.dim() == 0
        or raw_logits.size(-1) != 4
    ):
        raise ValueError("expected a tensor of raw logits with last dimension 4")
    masked = raw_logits.clone()
    masked[..., 0] = torch.finfo(raw_logits.dtype).min
    return masked
