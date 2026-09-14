# Reconstructing the selected September 13 data

The release includes the preparation code and a **label-free preserved split
assignment**. It does not include competition images/captions/labels or the
72.36 GB public archive. Reproduction requires obtaining those inputs separately.
These instructions describe data reconstruction; they do not run training,
score the audit, or submit predictions.

The selected CXI recipe starts from **public pretraining checkpoint step 4,000
on an 80,000-pair corpus**, then trains on 2,230 clean gold pairs. “Public4k” does
not mean a 4,000-example dataset. Native IXI and CXC use 1,670 and 1,649 clean
gold training pairs respectively. Their shared development partition contains
794 CXI, 590 IXI and 602 CXC pairs. Canonical classes are SF=0, SP=1, REL=2,
UNR=3; IXI/CXC keep indices 1–3 and mask SF rather than relabeling the corpus.

## Sources, access and license

- Public figures: [adsabs/AstroCLIMB](https://huggingface.co/datasets/adsabs/AstroCLIMB/tree/68e8d0ccb44be6bb86a77ccb90fdc8ba735dcdcb),
  pinned revision `68e8d0ccb44be6bb86a77ccb90fdc8ba735dcdcb`.
  There are 114 Parquet shards, 94,233 figures and 72,360,746,667 download bytes.
  The [pinned dataset card](https://huggingface.co/datasets/adsabs/AstroCLIMB/blob/68e8d0ccb44be6bb86a77ccb90fdc8ba735dcdcb/README.md)
  declares **MIT**. This is the dataset-card declaration, not a separate audit
  of the rights of every source-paper figure.
- Competition data: obtain the explicitly named `train.csv`, `test.csv`, and
  `sample_submission.csv` from the [AstroCLIMB competition](https://www.kaggle.com/competitions/astroclimb/data)
  using your own authorized account. Competition access/terms are separate
  from the HF card license. No credentials are distributed. Do not use the
  historical whole-bundle extraction route, which could include a scoring key.
- Split: `data/splits/folds5_seed7_paper_strict.parquet` contains only
  `pair_id`, `fold`, `group_id`, `mode`: 10,000 IDs, 2,616 inherited groups,
  folds 0–4, mode `paper_strict`. It has no labels, text, image bytes, DOI, or
  source checksum column. Its four columns were verified equal to the preserved
  historical assignment. **The split is supplied, not regenerated from seed 7.**
  Original paper-group construction is not independently reconstructed here.

Exact source/release hashes and limitations are in
[`configs/data-provenance.json`](../configs/data-provenance.json).

## Environment and named downloads

Use Python 3.12 or later. CPU preparation directly needs Polars, NumPy, Pillow
and ImageHash. Downloading additionally uses huggingface_hub and Kaggle.
The historical lock versions are recorded in the JSON; they are not evidence
that every historical remote process had the same environment. Preparation
imports `astroclimb.utils`, so run from the repository root with `PYTHONPATH=src`.
Install the release environment with `uv sync`; if using a preparation-only
virtual environment, install these dependencies there first.

```bash
mkdir -p data/raw/kaggle
uv run kaggle competitions download -c astroclimb -f train.csv -p data/raw/kaggle
uv run kaggle competitions download -c astroclimb -f test.csv -p data/raw/kaggle
uv run kaggle competitions download -c astroclimb -f sample_submission.csv -p data/raw/kaggle
```

If a named download is packaged as a ZIP, extract **only its expected named
CSV member** into `data/raw/kaggle/`; the parsers require the literal basenames
`train.csv` and `test.csv`. Do not feed archives to the CSV parsers.

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="adsabs/AstroCLIMB", repo_type="dataset",
    revision="68e8d0ccb44be6bb86a77ccb90fdc8ba735dcdcb",
    allow_patterns=["data/train-*.parquet", "README.md"],
    local_dir="data/raw/hf_full")
PY
```

Download is substantial; indexing additionally writes an original-image cache
and metadata/graph tables. No GPU is required by the following commands.

## Reconstruction order

Commands assume fresh output directories. Do not overwrite an active run's
corpus. The copied parsers detect PNG objects by their base64 PNG prefix; they
are tailored to the named competition CSV representation, not arbitrary images.

```bash
PYTHONPATH=src uv run python scripts/final_gold_prepare.py \
  --train-csv data/raw/kaggle/train.csv \
  --manifest data/splits/folds5_seed7_paper_strict.parquet \
  --out output/final_gold_20260913 --workers 8

PYTHONPATH=src uv run python scripts/final_test_prepare.py \
  --test-csv data/raw/kaggle/test.csv \
  --out output/final_test_20260913 --workers 8
```

Gold recovery uses folds 2–4 for training and fold 1 for development. Fold 0
labels are not parsed or exported. Test recovery retains only ID/object inputs;
no test label fields are used. The parsers necessarily stream the input CSV,
but only the explicitly handled fields enter generated tables.

Gold recovery strengthens inherited paper groups using transitive exact
normalized caption, image-byte, decoded-pixel and dHash-distance ≤8 matches.
An affected pair excludes its whole inherited group. The dHash is a true
256-bit hash (`hash_size=16`), not a distance between hexadecimal characters.

Build the **label-free union of all gold and test content** for exclusion:

```bash
uv run python - <<'PY'
from pathlib import Path
import polars as pl
cols = ["obj_key", "modality", "cap_text", "cap_norm", "bytes_md5", "px_md5", "dhash"]
paths = ["output/final_gold_20260913/objects_train.parquet",
         "output/final_test_20260913/objects_test.parquet"]
objects = pl.concat([pl.read_parquet(p, columns=cols) for p in paths], how="vertical_relaxed")
objects = objects.unique(subset="obj_key", keep="first", maintain_order=True)
assert objects.height == 29179
objects.write_parquet("output/final_exposure_objects.parquet")
PY

PYTHONPATH=src uv run python scripts/final_public_prepare.py index \
  --source data/raw/hf_full/data --out output/final_public_20260913 \
  --expected-shards 114 --workers 8

PYTHONPATH=src uv run python scripts/final_public_prepare.py build \
  --out output/final_public_20260913 \
  --gold-objects output/final_exposure_objects.parquet \
  --expected-shards 114 --per-class 20000 --dev-per-class 500 --seed 20260913

PYTHONPATH=src uv run python scripts/final_public_prune.py \
  --gold output/final_gold_20260913 \
  --test-objects output/final_test_20260913/objects_test.parquet \
  --canonical data/splits/folds5_seed7_paper_strict.parquet \
  --candidates output/final_public_20260913/exposure_public_candidates.parquet \
  --out output/final_gold_publicclean_20260913
```

The public builder rejects label-bearing exposure tables. It creates full
public duplicate-paper components using exact caption/UUID/image hashes and
near-image matches, excludes every gold/test matching component, and generates
balanced SF/SP/direct-citation-REL/no-observed-edge-UNR pairs. Public-only
component-disjoint validation contains 2,000 pairs across 125 papers and 35
citation edges after the label-free support correction. UNR inherits public graph
completeness as an assumption. Gold pruning then closes inherited paper groups
through these public components and excludes whole groups linked to heldout
or test objects. Test identities are exclusion-only, never label features.

Expected final train/development counts are 5,549/1,986 overall, split as above.
Public receipt targets are 7,500 forbidden papers, 21,258 eligible figures,
zero retained gold-content hits and zero train/public-validation component
intersection. Receipts report the inherited audit's historical selection
exposure: it must not be advertised as a never-used statistical test set.

For native slice trainers, create separate corpus directories with `train.parquet`
and `val.parquet` pointing to the matching `ixi_train/ixi_dev` or
`cxc_train/cxc_dev` files. IXI also needs the shared gold `imgcache`; CXI already
has compatible `train`/`val` symlinks. Keep canonical four-class `y` values.
Paths embedded in recovered CXI tables refer to the reconstruction directory;
keep that layout or explicitly rebind paths and record changed hashes.

## Bridge reconstructed outputs to the model commands

After reconstruction and receipt checks, the following CPU-only command makes
exactly the paths consumed by [reproduction.md](reproduction.md). Training
Parquets are linked without changing any row, original-case caption, class
index, or serialization. Public pretraining uses its own image cache; gold
training uses the gold cache; inference uses the separately reconstructed
unlabeled test cache. Output links may be reused only when they already point
to the same source. Run from the repository root.

```bash
uv run python - <<'PYBRIDGE'
from pathlib import Path
import polars as pl

root = Path.cwd()
gold = root / "output/final_gold_publicclean_20260913"
public = root / "output/final_public_20260913"
test = root / "output/final_test_20260913"

def link(source, destination):
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise ValueError(f"Refuse to replace existing path: {destination}")
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())

for task in ("cxi", "ixi", "cxc"):
    target = root / "data" / task
    for split, alias in (("train", "train"), ("dev", "val")):
        link(gold / f"{task}_{split}.parquet", target / f"{alias}.parquet")
    link(gold / "imgcache", target / "imgcache")
for name in ("train.parquet", "val.parquet", "imgcache"):
    link(public / name, root / "data/public80000" / name)
link(test / "imgcache", root / "data/imgcache")

# Explicit allowlist projections; never copy labels or image_path to inference.
columns = {
    "cxi": ["pair_id", "img_bytes_md5", "cap_text"],
    "ixi": ["pair_id", "img_a", "img_b"],
    "cxc": ["pair_id", "cap_a", "cap_b"],
}
inputs = root / "inputs"
inputs.mkdir(exist_ok=True)
seen = set()
for task, fields in columns.items():
    source = test / f"{task}_test.parquet"
    frame = pl.read_parquet(source, columns=fields)
    if frame.is_empty() or any(frame.null_count().row(0)):
        raise ValueError(f"Missing {task} input content")
    ids = set(frame["pair_id"].to_list())
    if len(ids) != frame.height or seen & ids:
        raise ValueError("Duplicate or cross-task pair IDs")
    seen.update(ids)
    for image_field in ({"cxi": ["img_bytes_md5"], "ixi": ["img_a", "img_b"]}.get(task, [])):
        for key in frame[image_field].unique().to_list():
            if not (test / "imgcache" / f"{key}.img").is_file():
                raise FileNotFoundError(f"Missing image content: {key}")
    target = inputs / f"{task}.parquet"
    if target.exists():
        raise FileExistsError(f"Keep existing inference artifact: {target}")
    frame.write_parquet(target)
link(root / "data/raw/kaggle/sample_submission.csv", inputs / "sample_submission.csv")
print("Training corpus links and label-free input projections ready")
PYBRIDGE
```

The CXI projection intentionally omits `image_path`. The historical exporter
resolves a relative `image_path` against the input Parquet's directory, which
could prepend `inputs/` to an already repository-relative cache path. Supplying
only `img_bytes_md5` and `cap_text`, with `--image-cache data/imgcache` as in
reproduction.md, avoids that ambiguity. Caption text is not normalized or
recased here. Inference projection hashes differ from the original Parquet
hashes by design; retain exporter provenance for these exact projections.

The bridge was exercised with synthetic CPU fixtures matching all six slice
schemas, matching image-cache filenames, and an intentionally unwanted
`image_path`/label field in the source inference fixture. Output projections
excluded both fields and all links resolved. This is a schema/path test, not
an execution against real test rows or a GPU reproduction.

The preparation/download imports are covered by base `pyproject.toml`
dependencies: NumPy, Polars, Pillow, ImageHash, huggingface-hub, and Kaggle;
ImageHash's SciPy/PyWavelets dependencies are transitive. Parquet IO here uses
Polars and does not require a direct PyArrow import. Model training needs the
separate training extras and explicitly installed CUDA/Torch/FLA stack in
reproduction.md; base `uv sync` alone is only the data-preparation environment.

## What is included, and what is not guaranteed

Included byte-identical preparation sources: `scripts/final_gold_prepare.py`,
`final_test_prepare.py`, `final_public_prepare.py`, `final_public_prune.py`, and
`src/astroclimb/utils.py`. A package initializer is also required and included.
Native training/export modules are separately documented by the model pipeline.
The selected native IXI/CXC recipe does not require `final_public_ixi.py`, legacy
feature tables, old structural predictions, or old fine-tuned adapters.

The safe projected fold manifest closes the missing-split-file gap. Raw source
CSVs and HF image shards remain external downloads. Historical image caches
and derived pair tables are not included in this release. The commands have
been checked against executable CLI/data contracts, but **the entire pipeline
has not been rerun in this clean repository**. Parallel object recovery, changed
Parquet serialization, and the deliberately projected split artifact can alter
byte hashes while preserving assignments. Compare row/content invariants and
record fresh receipts; do not silently substitute regenerated tables into a
checkpoint expecting historical corpus hashes.

For an exact historical artifact replay, separately preserve the original
label-free exposure/index/graph/candidate receipts and the authorized generated
TRAIN/development tables with the hashes in the JSON, plus image caches rebuilt
from the pinned sources. Exact checkpoint reproduction also requires the
public step-4000 parent and the selected training/checkpoint recipe; these data
instructions alone do not establish that weight-level reproducibility.

## File hashes versus trainer row hashes

A Parquet SHA-256 fingerprints serialized file bytes: compression, metadata or
row ordering can change it without changing the logical rows. The native CXI
base trainer computes `data_hash = sha256(json.dumps(rows, sort_keys=True).encode())`
after loading **all row fields**, shuffling with `random.Random(seed)` and
applying `max_train`. This row-serialization hash is not the Parquet hash and
is not invariant to row order, extra metadata, or absolute image-path changes.
It also does not independently fingerprint the image bytes. Native IXI/CXC
wrappers additionally bind `train_file_sha256`/`val_file_sha256` and refuse
resume when those file hashes differ. Exporters use a further distinct hash
of their allowlisted input projection serialized as NDJSON. Compare each
receipt to the matching algorithm; never replace one type of hash with another
or bypass a checkpoint mismatch merely because a row-count check passes.
