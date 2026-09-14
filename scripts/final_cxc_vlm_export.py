import hashlib
import json
from pathlib import Path

import final_vlm_export as export
import polars as pl
import vlm_cxc_peft as cxc
import vlm_cxc_shared as shared
import vlm_cxi_peft as trainer


def read_inputs(path, image_cache=None):
    """Validate the caption parquet; image_cache is accepted and ignored."""
    del image_cache
    frame = pl.read_parquet(path, columns=["pair_id", "cap_a", "cap_b"])
    if frame.is_empty():
        raise ValueError(f"no rows in {path}")
    if sum(frame.null_count().row(0)) > 0:
        raise ValueError("null values in pair_id/cap_a/cap_b")
    if frame["pair_id"].n_unique() != frame.height:
        raise ValueError("duplicate pair_id values")
    for col in ("cap_a", "cap_b"):
        if not all(isinstance(v, str) and v.strip() for v in frame[col]):
            raise ValueError(f"{col} values must be nonblank strings")
    input_hash = hashlib.sha256(frame.write_ndjson().encode("utf-8")).hexdigest()
    rows = frame.to_dicts()
    return frame.select("pair_id"), rows, input_hash


def encode_inputs(proc, rows, config):
    """Encode validated caption rows on the GPU."""
    if not rows:
        raise ValueError("no rows to encode")
    return shared.encode_cxc(proc, rows, config.max_cap_chars, device="cuda")


def main():
    orig_header = export.checkpoint_header
    orig_read = export.read_inputs
    orig_encode = export.encode_inputs
    orig_validate = export.validate_probabilities
    orig_logits = trainer.logits

    def wrapped_header(path):
        cxc.check_checkpoint(path)
        meta, header, _ = orig_header(path)
        header["cxc_runtime"] = cxc.runtime_provenance()
        header["cxc_exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
