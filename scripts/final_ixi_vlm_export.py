import hashlib
import json
from pathlib import Path

import final_vlm_export as export
import polars as pl
import vlm_cxi_peft as trainer
import vlm_ixi_peft as ixi
import vlm_ixi_shared as shared

_HEX = set("0123456789abcdef")


def _is_img_hash(value):
    return isinstance(value, str) and len(value) == 32 and set(value) <= _HEX


def read_inputs(path, image_cache):
    """Validate the pair parquet; hash content before any private paths attach."""
    if not image_cache:
        raise ValueError("image_cache must be provided explicitly")
    cache_dir = Path(image_cache)
    if not cache_dir.is_dir():
        raise NotADirectoryError(f"missing image cache dir: {cache_dir}")
    frame = pl.read_parquet(path, columns=["pair_id", "img_a", "img_b"])
    if frame.is_empty():
        raise ValueError(f"no rows in {path}")
    if sum(frame.null_count().row(0)) > 0:
        raise ValueError("null values in pair_id/img_a/img_b")
    if frame["pair_id"].n_unique() != frame.height:
        raise ValueError("duplicate pair_id values")
    for col in ("img_a", "img_b"):
        if not all(_is_img_hash(v) for v in frame[col]):
            raise ValueError(f"{col} values must be lowercase 32-hex hashes")
    input_hash = hashlib.sha256(frame.write_ndjson().encode("utf-8")).hexdigest()
    rows = frame.to_dicts()
    cache = str(cache_dir.resolve())
    for row in rows:
        row["_image_cache"] = cache
    return frame.select("pair_id"), rows, input_hash


def encode_inputs(proc, rows, config):
    """Encode validated rows; every row must share one resolved image cache."""
    if not rows:
        raise ValueError("no rows to encode")
    caches = {row["_image_cache"] for row in rows}
    if len(caches) != 1:
        raise ValueError("rows do not share a single _image_cache")
    return shared.encode_ixi(proc, rows, caches.pop(), config.max_pixels, device="cuda")


def main():
    orig_header = export.checkpoint_header
    orig_read = export.read_inputs
    orig_encode = export.encode_inputs
    orig_validate = export.validate_probabilities
    orig_logits = trainer.logits

    def wrapped_header(path):
        ixi.check_checkpoint(path)
        meta, header, _ = orig_header(path)
        header["ixi_runtime"] = ixi.runtime_provenance()
        header["ixi_exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        newhash = hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()
        return meta, header, newhash

    def wrapped_validate(frame, ids):
        result = orig_validate(frame, ids)
        psf = frame["p_same_figure"]
        bad = (psf != 0).any() if hasattr(psf, "any") else any(psf)
        if bad:
            raise ValueError("nonzero p_same_figure rejected")
        return result

    def wrapped_logits(*args, **kwargs):
        return shared.mask_same_figure(orig_logits(*args, **kwargs))

    try:
        export.checkpoint_header = wrapped_header
        export.read_inputs = read_inputs
        export.encode_inputs = encode_inputs
        export.validate_probabilities = wrapped_validate
        trainer.logits = wrapped_logits
        export.main()
    finally:
        export.checkpoint_header = orig_header
        export.read_inputs = orig_read
        export.encode_inputs = orig_encode
        export.validate_probabilities = orig_validate
        trainer.logits = orig_logits


if __name__ == "__main__":
    main()
