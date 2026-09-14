import hashlib
import json
from pathlib import Path

import vlm_cxi_peft as base
import vlm_ixi_shared as shared


def runtime_provenance() -> dict:
    return {
        "task": shared.TASK,
        "prompt_version": shared.PROMPT_VERSION,
        "shared_sha256": hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }


def check_checkpoint(path):
    meta = json.loads(Path(path).joinpath("meta.json").read_text())
    if tuple(meta.get("labels", [])) != tuple(base.LABELS):
        raise ValueError("labels mismatch base.LABELS")
    args = meta.get("args", {})
    for key, value in runtime_provenance().items():
        if args.get(key) != value:
            raise ValueError(f"args mismatch for {key!r}")
    return meta


def validate_corpus(a):
    import polars as pl

    pairs, shas = {}, {}
    for name in ("train", "val"):
        f = Path(a.corpus) / f"{name}.parquet"
        try:
            df = pl.read_parquet(f, columns=["pair_id", "img_a", "img_b", "y"])
        except Exception as e:
            raise ValueError(f"{name}.parquet unreadable or missing required columns: {e}") from e
        if df.height == 0 or any(df.null_count().row(0)) or df["pair_id"].n_unique() != df.height:
            raise ValueError(f"{name}.parquet: empty, contains nulls, or duplicate pair_id")
        ydt = df.schema["y"]
        if (
            not ydt.is_integer()
            or ydt == pl.Boolean
            or not set(df["y"].unique().to_list()) <= {1, 2, 3}
        ):
            raise ValueError(
                f"{name}.parquet: y must be a non-bool integer column with values in {{1,2,3}}"
            )
        ids = set(df["pair_id"].to_list())
        shas[name] = hashlib.sha256(f.read_bytes()).hexdigest()
        setattr(a, f"{name}_sha", shas[name])
        pairs[name] = ids
    a.dev_count = len(pairs["val"])
    if pairs["train"] & pairs["val"]:
        raise ValueError("train and val pair_id sets intersect")
    if getattr(a, "val_limit", 0) < a.dev_count:
        raise ValueError(f"val_limit {getattr(a, 'val_limit', 0)} < dev_count {a.dev_count}")
    return shas["train"], shas["val"]


def checked_arguments(original):
    """Wrap an argparse entry point, enforcing checkpoint/model/corpus/output invariants."""
    a = original()

    source = a.verify or a.resume or a.init_checkpoint
    meta = check_checkpoint(source) if source else None

    if a.verify:
        return a
    if a.init_checkpoint:
        raise ValueError(f"--init_checkpoint is not supported: {a.init_checkpoint}")
    if Path(a.model).name != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a":
        raise ValueError(f"unexpected --model: {a.model}")

    train_sha, val_sha = validate_corpus(a)

    if a.resume:
        expect = (meta["args"].get("train_file_sha256"), meta["args"].get("val_file_sha256"))
        if (train_sha, val_sha) != expect:
            raise ValueError(
                f"--resume corpus hash mismatch: got {(train_sha, val_sha)}, want {expect}"
            )

    a.train_file_sha256 = train_sha
    a.val_file_sha256 = val_sha
    for key, value in runtime_provenance().items():
        setattr(a, key, value)

    if a.out and Path(a.out).exists() and not a.resume:
        raise ValueError(f"--out already exists: {a.out}")

    return a


def main():
    original_arguments = base.arguments
    original_encode = base.encode
    original_logits = base.logits
    base.arguments = lambda: checked_arguments(original_arguments)
    base.encode = shared.train_encode
    base.logits = lambda model, head, enc: shared.mask_same_figure(
        original_logits(model, head, enc)
    )
    try:
        base.main()
    finally:
        base.arguments = original_arguments
        base.encode = original_encode
        base.logits = original_logits


if __name__ == "__main__":
    main()
