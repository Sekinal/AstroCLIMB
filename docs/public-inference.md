# Public release inference

This page covers inference from the public Hugging Face release only. It does
not describe historical training and it does not replay the historical
`probe.pt`, which serializes TRAIN examples and is not shipped.

## What the release contains

The public repository `Thermostatic/AstroCLIMB-Qwen3.5-4B` contains four release folders. Only the three
named gold checkpoints are supported by `scripts/predict_public.py`:

- `cxi837/` for CXI
- `ixi627/` for IXI
- `cxc414/` for CXC
- `public4000/` is the public pretraining parent and is not a selected gold
  checkpoint; the prediction utility refuses it.

Each folder contains `adapter/adapter_model.safetensors`,
`adapter/adapter_config.json` with canonical
`base_model_name_or_path` equal to `Qwen/Qwen3.5-4B`, `head.pt`, processor
files under `processor/`, and `meta.json` with sanitized paths and otherwise
preserved training arguments. The repository root contains
`public-manifest.json` with a `checkpoints` mapping of release folder name to
`{files: {relative_path: {sha256, bytes}}}`.

Checkpoint folder names include the task prefix so modality is never
ambiguous.

## Download

Download the released artifacts below. Install the GPU environment described in
[reproduction.md](reproduction.md) before running inference.

```bash
hf download Thermostatic/AstroCLIMB-Qwen3.5-4B --repo-type model --include 'cxi837/*' --local-dir release
hf download Thermostatic/AstroCLIMB-Qwen3.5-4B --repo-type model --include 'ixi627/*' --local-dir release
hf download Thermostatic/AstroCLIMB-Qwen3.5-4B --repo-type model --include 'cxc414/*' --local-dir release
hf download Thermostatic/AstroCLIMB-Qwen3.5-4B --repo-type model --include 'public-manifest.json' --local-dir release
```

Obtain the base model snapshot separately and keep the directory named by the
pinned revision:

```bash
MODEL="models/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
```

## Prediction

`scripts/predict_public.py` checks the pinned snapshot name and config, then
verifies every manifest-listed weight, config, processor, and head hash before
any model import. There are no switches that skip hash checks. Inference uses
single-sample batches under `torch.inference_mode` with BF16 autocast and no
quantization. IXI and CXC apply the historical same-figure mask. IXI exports
one orientation per run; average original and reversed probabilities yourself
and finish with `scripts/assemble_selected.py`.

```bash
python scripts/predict_public.py --checkpoint release/cxi837 --manifest release/public-manifest.json --task cxi --model-path "$MODEL" --pairs inputs/cxi.parquet --image-cache data/imgcache --out predictions/cxi.parquet
python scripts/predict_public.py --checkpoint release/ixi627 --manifest release/public-manifest.json --task ixi --model-path "$MODEL" --pairs inputs/ixi.parquet --image-cache data/imgcache --out predictions/ixi_original.parquet
python scripts/predict_public.py --checkpoint release/cxc414 --manifest release/public-manifest.json --task cxc --model-path "$MODEL" --pairs inputs/cxc.parquet --out predictions/cxc.parquet
```

Input schemas reuse the historical exporters: CXI needs `pair_id` plus
`cap_text` or `cap_norm` and an image path or MD5 with `--image-cache`; IXI
needs `pair_id`, `img_a`, `img_b` with `--image-cache`; CXC needs `pair_id`,
`cap_a`, `cap_b`. Outputs use the `pair_id` plus `p_same_figure`,
`p_same_paper`, `p_related_papers`, `p_unrelated_papers` schema and are checked
with the existing validators.

For the selected IXI recipe, build the reversed input by swapping only the two
image columns, run the same command with a distinct `--out`, average the two
probability tables with weight 0.5 each, then assemble:

```bash
python scripts/assemble_selected.py \
  --sample inputs/sample_submission.csv \
  --cxi predictions/cxi.parquet \
  --ixi-original predictions/ixi_original.parquet \
  --ixi-reversed predictions/ixi_reversed.parquet \
  --cxc predictions/cxc.parquet \
  --out predictions/selected.csv
```

## Verification scope and limitations

- The utility validates release file hashes and inputs. It does not perform
  historical probe replay and the historical exporter guard is left untouched.
- Every output writes a sidecar `<out>.sidecar.json` with the disclosure and
  `gpu_replay_verified: false`. No GPU equivalence with historical scores is
  claimed.
- `public4000` is documented as not a selected gold checkpoint and is refused.
- `--out` is never overwritten; choose a fresh path per run.
- CPU tests cover hash corruption, wrong checkpoint-task pairing, missing
  manifest entries, refused overwrite, model pin ordering before GPU import,
  and input validation. GPU numerical equivalence has not been demonstrated
  in this repository.

```bash
python scripts/test_predict_public.py
```
