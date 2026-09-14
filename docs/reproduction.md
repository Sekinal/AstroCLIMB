# Reproducing the selected classifier recipe

This repository contains the source closure for the selected Qwen3.5-4B system: CXI public 4000 → gold 837, IXI gold 627 with equal original/reversed probabilities, and CXC gold 414 in original order. Later public 8000 and generative RL experiments are not this selected recipe. The three task branches use a four-class linear head over the last valid hidden state; IXI and CXC mask the same-figure class.

## What is available

Eight historical training/export files in `scripts/` were copied byte-for-byte. Their SHA-256 values and the selected checkpoint file hashes are in `configs/selected-recipe.json`. Every listed selected checkpoint file was found and hashed in the source workspace during curation. The adapters, heads, optimizer states, probes, processors, exact corpus snapshots and image cache are **not distributed in this checkout**, and no public checkpoint download URL is supplied. This is a runnable source release conditional on obtaining those artifacts, not a demonstrated clean retraining reproduction. The data preparation documentation describes available upstream reconstruction tools; rebuilding a corpus is not proof of matching the recorded corpus fingerprints.

A supplied checkpoint bundle needs `adapter/`, `processor/`, `meta.json`, `head.pt`, and `probe.pt`; exact training resume also needs `trainer.pt`. Retain original checkpoint metadata and saved processor bytes. Expected release locations are `checkpoints/cxi837`, `checkpoints/ixi627` and `checkpoints/cxc414`. Do not substitute a generative LM adapter for one of these classifier checkpoints.

## Environment and model

The recorded working classifier core was Python 3.12, Torch 2.12.0+cu130, Transformers 5.5.0, PEFT 0.20.0, Accelerate 1.15.0 and FLA/core 0.5.2. Ancillary observed versions are recorded in the recipe. Core evidence is the historical receipt `output/gspo_qwen35_prepare_20260913/ENVIRONMENT_CPU_CHECK.json` (SHA in the recipe); this later isolated CPU check preserved the working classifier core, and is not an original-run lock of every package. That operational receipt is not copied into this source release. Data-preparation lock versions are a separate environment and may differ.

Torch's CUDA wheel is obtained from the PyTorch CUDA index, not assumed to be an ordinary PyPI resolution. In a fresh CUDA environment, an example installation is:

```bash
python -m pip install 'torch==2.12.0+cu130' 'torchvision==0.27.0+cu130' --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'transformers==5.5.0' 'peft==0.20.0' 'accelerate==1.15.0' 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' 'polars==1.44.2' 'pillow==12.2.0' 'numpy==2.4.4' 'scikit-learn==1.9.1' 'safetensors==0.8.0' 'tokenizers==0.22.2'
python -m pip check
```

These commands document observed pins; fresh installation and GPU execution have not been validated in this clean repository. FLA kernels and the CUDA wheel require a compatible GPU/driver. Historical execution did not have causal-conv1d installed, so its convolution path used PyTorch. SDPA is explicit; do not silently change attention engines when comparing numerical results.

Obtain the base model from `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, keeping the snapshot directory named by that revision. Set `MODEL` to that local directory. The launcher checks the directory name and config existence; it does not independently certify all base weight hashes.

```bash
MODEL="models/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
python scripts/run_selected.py --help
```

`run_selected.py` is the only portability adaptation: it relocates the base-model loader without rewriting historical checkpoint metadata, preserves the source hashes required by IXI/CXC, and applies the historical Pillow PNG text-chunk allowance of 16 MiB (64 MiB aggregate unchanged). New training records the supplied model path. Exact resume requires keeping that path stable because the native trainer validates its saved model argument. The original eight files remain unchanged. The portability module and CPU tests were necessarily authored during curation after prior preferred-provider failures; they are not attributed to DeepSeek.

## Training inputs and exact flags

Each corpus directory contains `train.parquet` and `val.parquet`. CXI needs `cap_norm`, `img_bytes_md5`, `y`, and `imgcache/<md5>.img`; IXI needs `pair_id`, `img_a`, `img_b`, `y` and that cache; CXC needs `pair_id`, `cap_a`, `cap_b`, `y`. Integer classes are SF=0, SP=1, REL=2, UNR=3. IXI/CXC use only 1–3. The corpus must match the recorded provenance and satisfy source validators. Training is not an upstream leakage audit.

Use the following commands with supplied, verified corpus snapshots. Output directories must be fresh. Do not reduce `--max-steps` merely to select a checkpoint: that changes the cosine learning-rate horizon.


The public pretraining command below retains the original 10,000-update horizon. It will continue past the selected checkpoint 4000 unless stopped externally after its atomic save and `reload_verified: true` checkpoint event. Retain checkpoint 4000 as `checkpoints/public4000` for the following gold command; alternatively allow training to finish and still use that explicit intermediate checkpoint. Do not replace `--max-steps 0` with 4000.

```bash
python scripts/run_selected.py vlm_cxi_peft --model-path "$MODEL" -- \
  --corpus data/public80000 --out runs/public \
  --batch 4 --accum 2 --epochs 1 --max-steps 0 --max-train 0 \
  --val-limit 2000 --lr 0.0001 --head-lr 0.0003 --lora-r 16 \
  --max-pixels 262144 --max-cap-chars 2400 --save-every 500 --val-every 2000 \
  --guard-steps 5 --deadline-min 400 --seed 7 --no-gradient-checkpointing
```

The exact 80,000-row public corpus and its held-out validation snapshot are required. These are upstream artifacts, not created by the command itself. Public run resumptions use the identical arguments with `--resume runs/public/checkpoint-N`; weights, optimizer state, RNG and data cursor are restored.

```bash
python scripts/run_selected.py vlm_cxi_peft --model-path "$MODEL" -- \
  --corpus data/cxi \
  --out runs/cxi \
  --init-checkpoint checkpoints/public4000 \
  --batch 1 \
  --accum 8 \
  --epochs 3 \
  --max-steps 0 \
  --max-train 0 \
  --val-limit 794 \
  --lr 3e-05 \
  --head-lr 0.0001 \
  --lora-r 16 \
  --max-pixels 262144 \
  --max-cap-chars 2400 \
  --save-every 279 \
  --val-every 279 \
  --guard-steps 5 \
  --deadline-min 90.0 \
  --seed 7 \
  --no-gradient-checkpointing
```

```bash
python scripts/run_selected.py vlm_ixi_peft --model-path "$MODEL" -- \
  --corpus data/ixi \
  --out runs/ixi \
  --batch 1 \
  --accum 8 \
  --epochs 3 \
  --max-steps 627 \
  --max-train 0 \
  --val-limit 590 \
  --lr 3e-05 \
  --head-lr 0.0001 \
  --lora-r 16 \
  --max-pixels 262144 \
  --max-cap-chars 2400 \
  --save-every 209 \
  --val-every 209 \
  --guard-steps 5 \
  --deadline-min 120.0 \
  --seed 7
```

```bash
python scripts/run_selected.py vlm_cxc_peft --model-path "$MODEL" -- \
  --corpus data/cxc \
  --out runs/cxc \
  --batch 1 \
  --accum 8 \
  --epochs 3 \
  --max-steps 0 \
  --max-train 0 \
  --val-limit 602 \
  --lr 3e-05 \
  --head-lr 0.0001 \
  --lora-r 16 \
  --max-pixels 262144 \
  --max-cap-chars 2400 \
  --save-every 207 \
  --val-every 207 \
  --guard-steps 5 \
  --deadline-min 120.0 \
  --seed 7
```

CXI gold has 2230 TRAIN/794 DEV rows and 837 updates. IXI has 590 DEV rows and 627 updates. CXC has 1649 TRAIN/602 DEV rows, 621 scheduled updates, and the selected checkpoint is 414. Preserve that 621-step horizon even when inspecting 414. The public parent uses 80000 TRAIN rows, B4/accum2, epochs 1, max-steps 0, LR 1e-4/head 3e-4, seed 7, no gradient checkpointing, save 500/eval 2000, pixels 262144 and caption 2400. Its horizon is 10000; checkpoint 4000 is the selected intermediate state, not a 4000-step schedule. Historical execution resumed that run; bitwise reconstruction from a new run is not claimed.

## Fresh verification and label-free export

Use the corresponding entrypoint to verify each supplied checkpoint in a fresh process:

```bash
python scripts/run_selected.py vlm_cxi_peft --model-path "$MODEL" -- --verify checkpoints/cxi837
python scripts/run_selected.py vlm_ixi_peft --model-path "$MODEL" -- --verify checkpoints/ixi627
python scripts/run_selected.py vlm_cxc_peft --model-path "$MODEL" -- --verify checkpoints/cxc414
```

Inference tables contain explicit unique `pair_id` values and only task inputs. CXI requires `cap_text` or `cap_norm` plus an image path or MD5/cache; IXI requires `img_a`,`img_b`; CXC requires `cap_a`,`cap_b`. Exporters project only inference columns. Use batch 1 and chunk 128 for all selected branches. Chunk caches are tied to checkpoint and input provenance; do not reuse them after changing inputs.

```bash
python scripts/run_selected.py final_vlm_export --model-path "$MODEL" -- --checkpoint checkpoints/cxi837 --pairs inputs/cxi.parquet --image-cache data/imgcache --out predictions/cxi.parquet --batch-size 1 --chunk-size 128
python scripts/run_selected.py final_ixi_vlm_export --model-path "$MODEL" -- --checkpoint checkpoints/ixi627 --pairs inputs/ixi.parquet --image-cache data/imgcache --out predictions/ixi_original.parquet --batch-size 1 --chunk-size 128
python scripts/run_selected.py final_cxc_vlm_export --model-path "$MODEL" -- --checkpoint checkpoints/cxc414 --pairs inputs/cxc.parquet --out predictions/cxc.parquet --batch-size 1 --chunk-size 128
```

Create an IXI reversed input by swapping only the two image columns, retaining pair IDs and row order:

```python
import polars as pl
x = pl.read_parquet("inputs/ixi.parquet", columns=["pair_id", "img_a", "img_b"])
x.select("pair_id", pl.col("img_b").alias("img_a"), pl.col("img_a").alias("img_b")).write_parquet("inputs/ixi_reversed.parquet")
```

Run the identical IXI export command on `inputs/ixi_reversed.parquet` with a distinct output `predictions/ixi_reversed.parquet`. For each pair ID and class, compute `0.5 * original + 0.5 * reversed`; do not average hard labels. CXI/CXC retain their direct probabilities. Concatenate the disjoint task predictions, align by the official sample submission's pair IDs, and choose argmax in the recipe's class order. Check unique IDs, complete coverage and row order before writing the official schema. This release does not include the private submission template or any evaluation labels.

Runnable deterministic assembly:

```bash
python scripts/assemble_selected.py \
  --sample inputs/sample_submission.csv \
  --cxi predictions/cxi.parquet \
  --ixi-original predictions/ixi_original.parquet \
  --ixi-reversed predictions/ixi_reversed.parquet \
  --cxc predictions/cxc.parquet \
  --out predictions/selected.csv
```

The official template has `id` plus the four canonical class columns and may include a trailing `Usage` column. The actual submitted output has five columns: `id,same_figure,same_paper,related_papers,unrelated_papers`. Assembly reads template IDs for ordering, ignores its placeholder values and `Usage`, and emits one-hot argmax predictions. It rejects duplicate IDs, overlapping task IDs, missing/extra coverage, differing IXI orientation IDs, nonfinite/out-of-range/unnormalized probabilities, reordered or renamed class schemas, and a nonzero same-figure probability in IXI/CXC. It does not infer task type from ID ranges, labels, or structural metadata. Canonical class order resolves exact ties deterministically. Column-name validation cannot detect probabilities whose numeric meanings were silently mislabeled upstream; use the supplied exporter provenance checks.


## Validation performed during curation

Historical source files were byte-hash checked; selected checkpoint manifests were checked against actual source-workspace files. CPU tests exercise portable model relocation, preservation of archival config and hook restoration, and rejection of a wrong snapshot before loading imports. No clean-repository GPU fit, export or score reproduction has been performed.

```bash
python scripts/test_selected_portability.py
python scripts/test_assemble_selected.py
python -m compileall -q scripts src
```

Assembly validation: six CPU tests passed, covering reordered rows, probability averaging, missing/duplicate/cross-task IDs, invalid probabilities, class schema swaps and the same-figure mask. An in-memory replay from the historical four probability exports matched every cell of the selected 10,000-row, five-column candidate CSV. No new predictions were submitted.

## Privately staged pretrained weights

`configs/release-artifacts.json` inventories four locally staged, unuploaded weights-only archives: CXI837, IXI627, CXC414 and public4000. Each contains nine original adapter/head/processor/metadata files with verified SHA-256 hashes. Archives live in ignored `release-artifacts/` with private filesystem permissions; no public download URLs exist. The manifest lists machine-path fields by name, without publishing their values. Generic historical cache paths remain in the private archive payloads; original hash-bound files were not rewritten.

These are **not drop-in verified checkpoint bundles**. Optimizer state is omitted, and `probe.pt` is deliberately omitted because the historical probe serializes a TRAIN example's token IDs and sometimes image tensors. The historical exporter still requires that original private probe and performs its existing replay checks. We have not bypassed that guard. Creating a distributable synthetic replay probe and validating it in a fresh model process is a separate future release step. Retraining from authorized data creates a new local probe normally. Consequently source reproduction and privately staged pretrained weights have different readiness levels; neither implies a published checkpoint release.
