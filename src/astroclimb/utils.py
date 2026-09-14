"""Content hashing and normalization primitives.

Kept free of heavy imports at module level where possible so worker processes
can pickle these functions cheaply.
"""

import base64
import hashlib
import io
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from PIL import PngImagePlugin

# ADS figures can carry huge PNG text chunks; defaults (1MB) reject them.
# Generous limits + per-image exception isolation below keep one weird PNG
# from killing a full index build.
PngImagePlugin.MAX_TEXT_CHUNK = 128 * 1024 * 1024
PngImagePlugin.MAX_TEXT_MEMORY = 2 * 1024 * 1024 * 1024

PNG_B64_MAGIC = "iVBORw0KGgo"
DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi.org/")


def is_image_object(s: str) -> bool:
    return s.startswith(PNG_B64_MAGIC)


def decode_b64_png(s: str) -> bytes:
    return base64.b64decode(s)


def normalize_doi(doi: str) -> str:
    d = doi.strip().lower()
    for p in DOI_PREFIXES:
        if d.startswith(p):
            return d[len(p) :].strip()
    return d.strip()


def normalize_caption(text: str) -> str:
    return " ".join(text.lower().split())


def image_hashes(raw: bytes) -> dict[str, str | None]:
    """Identity keys for a PNG payload.

    bytes_md5 identifies the exact file; px_md5 identifies decoded RGB pixels
    (survives re-encoding with same pixels); dhash is a near-dup fallback.
    """
    import imagehash
    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    px = np.ascontiguousarray(img.convert("RGB"))
    digest = hashlib.md5()
    digest.update(str(px.shape).encode())
    digest.update(px.tobytes())
    return {
        "bytes_md5": hashlib.md5(raw).hexdigest(),
        "px_md5": digest.hexdigest(),
        "dhash": str(imagehash.dhash(img, hash_size=16)),
    }


def safe_hash_batch(raws: list[bytes]) -> list[dict[str, str | None]]:
    """Hash a batch; an undecodable image yields null keys (never matches)."""
    out: list[dict[str, str | None]] = []
    for r in raws:
        try:
            out.append(image_hashes(r))
        except Exception:
            out.append({"bytes_md5": None, "px_md5": None, "dhash": None})
    return out


def parallel_hashes(raws: list[bytes], workers: int) -> pl.DataFrame:
    batches = [raws[i : i + 128] for i in range(0, len(raws), 128)]
    rows: list[dict[str, str | None]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for batch in ex.map(safe_hash_batch, batches):
            rows.extend(batch)
    n_failed = sum(1 for r in rows if r["px_md5"] is None)
    if n_failed:
        logging.getLogger(__name__).warning("%d images failed to hash", n_failed)
    return pl.DataFrame(
        {
            "bytes_md5": [r["bytes_md5"] for r in rows],
            "px_md5": [r["px_md5"] for r in rows],
            "dhash": [r["dhash"] for r in rows],
        }
    )


def md5_key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()
