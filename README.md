# AstroCLIMB: scientific figure and caption relations

Code and manuscript for a Qwen3.5-4B system that predicts same-figure,
same-paper, related-paper, and unrelated-paper relations from captions and
images. Public synthetic caption–image adaptation is followed by separate
LoRA classifiers for each input modality.

The completed submitted recipe scored **0.75149 on the Kaggle public
leaderboard**. Its pooled four-class development macro-F1 is **0.75620**
on 1,986 pairs. These are different evaluation sets; neither establishes a
verified rank. Development data was used repeatedly for selection.

| Branch | Initialization | Selected gold update | Prediction |
|---|---|---:|---|
| Caption–image | Public adaptation, update 4,000 | 837 | Original orientation |
| Image–image | Pristine Qwen3.5-4B | 627 | Mean of original/reversed probabilities |
| Caption–caption | Pristine Qwen3.5-4B | 414 | Original orientation |

A later [matched CXI analysis](analysis/matched-gold-findings.md) compares
public initialization with gold-only training under the same gold schedule.
All three gold-stage seeds improve, with a mean development macro-F1 gain of
**0.06738**, conditional on one fixed public parent. These new controls do not
change the submitted Kaggle result and use additional public-training compute.

Start with [data preparation](docs/data.md), then follow
[training and inference](docs/reproduction.md). Exact recipe settings are
in [configs/selected-recipe.json](configs/selected-recipe.json).
Historical trainer sources retain their provenance checks; the portable
entry point supplies an explicitly pinned local model snapshot.

```bash
uv sync --extra dev
uv run python scripts/test_selected_portability.py
make -C paper all
```

GPU dependencies require the separately documented PyTorch/CUDA installation;
the lightweight environment above does not install a GPU training stack.
Competition data, image caches, and trained weights are not bundled in Git.
Read the artifact availability and reconstruction requirements before starting
a training run. This release has CPU checks and a compiled paper; a fresh
end-to-end retraining of the release has not been performed.

The [paper](paper/README.md) builds anonymous and author PDFs using the
official ACL template. The [history and limitations](docs/history-and-limitations.md)
distinguish the submitted model from earlier excluded experiments.
The final longer public-adaptation run scored 0.75557 on pooled development
and was not adopted. Standalone SFT/GSPO alternatives also did not improve
the submitted classifier. Larger-rank and further capacity experiments remain
unrun proposals.

Author: Irving Ernesto Quezada Ramírez, Independent Researcher.
Contact: research@irvingernesto.com.
This research was supported by an unrestricted US$1,000 GPU compute grant
from Lium. The grant amount is not the amount spent on the experiments.

Original project code is released under MIT; see [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md).

The source ZIP also passed a [fresh-directory installation and build check](docs/release-validation.md).

## Public model artifacts

The four [adapter/head packages](https://huggingface.co/Thermostatic/AstroCLIMB-Qwen3.5-4B) include the submitted
CXI/IXI/CXC models and their public-adaptation parent. Tensor hashes match the
original checkpoints. [Public inference](docs/public-inference.md) validates
release hashes without requiring the private training-example replay probe.
This new packaging and entry point have CPU checks; fresh GPU validation of
the public entry point and full retraining from this release remain unverified.
