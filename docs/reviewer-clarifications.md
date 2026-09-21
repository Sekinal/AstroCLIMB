# Clarifications after simulated workshop review

This note records evidence behind the September 21 manuscript clarifications.
The repository and model weights remain private; local preservation is not public
availability. Source-file hashes are in `analysis/reviewer-clarification-evidence.json`.

## Filtering directions and counts

The public builder maps gold/test objects to candidate public papers using
normalized captions and exact/near image content. It joins public duplicate-paper
components and excludes every component touching any gold split or label-free
test content. The receipt identifies 7,500 forbidden public papers and 21,258
eligible figures. Public training/validation components are disjoint.

Gold pruning joins inherited gold groups through those public candidate
components. Training groups touching any nontraining gold objects or test objects
are removed. Development groups touching reserved gold objects outside train/dev,
including previously exposed fold 0, or test objects are removed. Training versus
DEV overlap is resolved by removing the training group. This final closure pass
removes 450 training and seven development pairs, leaving 5,549 and 1,986.
The removal counts are for this pass, not all historical cleanup combined.
The component audit reports zero retained train/dev, train/fold 0, train/test,
dev/fold 0 and dev/test intersections under these checks. Imperfect matching can
miss dependencies; zero detected overlap is not proof of perfect independence.

Historical overlap and label-dependent synthetic sampling, released scoring
information, a structural shortcut, and misaligned OCR occurred in excluded
branches. The later data pipeline cannot undo prior human/agent exposure. The
reported recipe's inputs and supervision exclude competition test labels.

## Historical versus fresh CXI

The submitted CXI classifier is the historical public 4000-to-gold 837 checkpoint,
with CXI DEV macro-F1 0.7483890086. It combines with IXI627 probability averaging
and original-order CXC414 to yield pooled DEV 0.7562033665 and the historical
Kaggle public score 0.75149. Later controls do not change this submission.

The new matched comparison retrains gold adaptation; it does not merely reload
the submitted checkpoint. Historical training ran on L40S, while the matched
pair used RTX3090. Seed 7's public arm scores 0.7561928105. The public parent file
hashes and training-row hash match exactly across old/new runs. Gold
hyperparameters match; paths and the wall-clock safety deadline differ (90 versus
880 minutes). Both complete 837 updates. Fresh checkpoint adapter weights differ
because these are separate training executions.

We have not isolated the cause of the residual cross-run score difference. It is
not established that GPU type alone causes it. The new inference exports exactly
reproduce each run's own training validation; this verifies internal consistency,
not bitwise cross-hardware reproducibility. Historical batch-size sensitivity is
operationally controlled by canonical batch-one exports; its numerical mechanism
has not been isolated either.

## Short matched SFT control

D2 and C3 start from the same answer-SFT128 parent, use the same 64 TRAIN contexts
(input hash `2ace1eedc79299254c9340904be470faf309f5ceae54ede0fbf272e0d6ecfd90`),
and take 32 optimizer updates with batch 16, accumulation 1, learning rate 1e-6 and seed 7.
D2 uses eight epochs /512 supervised canonical-answer-plus-EOS presentations;
loss is averaged over each example's supervised tokens and then the batch.
C3 uses four completions/context, two uses/rollout, sequence importance sampling,
group-normalized rewards, no reference KL, clipping .0003/.0004. Both use zero
weight decay and max gradient norm 1. Contexts/updates/batch shape are matched;
token populations, FLOPs and wall time are not. This is not the missing long-SFT
control for GSPO512.

Canonical greedy decoding allows 64 tokens and requires a single canonical answer
ending in EOS; malformed answers fall back to the frozen incumbent CXI classifier.
Both short branches produce 794 valid answers and zero fallback. Whole scores
retain IXI/CXC predictions: D2 .7467014569 versus C3 .7449355734 over 1986 pairs.

## Feature baselines

IXI .5405197574 active-class F1 uses .75 visual MLP probabilities plus .25 OCR-topic
logistic probabilities. Frozen SigLIP2-so400m-patch14-384 produces normalized
1152-dimensional embeddings. Symmetric absolute-difference/product/sum features,
cosine/distance and missingness form 3459 features. Train-only standardization
precedes three MLPs, averaged equally: hidden 128/32, ReLU, alpha 5, learning rate 3e-4,
batch 128, 180 epochs, seeds 7/17/27, no early stopping. The OCR head uses word/char
TF-IDF similarities, lexical/shape/quality features and 48-dimensional SVD word
topics, with standardized logistic regression C1/max_iter 1000. Visual weights
.25/.50/.75 were prespecified candidates. Train/dev 1670/590.

CXC .6727706883 uses frozen Qwen3-Embedding-8B/BGE-base-en-v1.5 embedding cosines,
word/char TF-IDF similarities, and lexical/anchor statistics, followed by
train-only StandardScaler and logistic regression C1/max_iter 3000. TF-IDF fits
training captions only; train/dev 1649/602. Qwen embeddings are documented as
normalized 4096-dimensional last-token representations; BGE uses normalized
SentenceTransformer embeddings. The original Qwen embedding producer script was
not preserved locally, so extraction provenance is documentary, not independently
verified encoder execution.

## Uncertainty

`analysis/paired-development.json` contains 10000 paired bootstrap draws of 278
inherited CXI DEV groups, seed 20260921. Each sampled group's multiplicity weights
all member examples; F1 is recomputed from pooled counts, not averaged per group.
The 95% percentile intervals concern CXI-only fixed historical predictions, not
whole-task scores or the three-seed matched mean. See `analysis/paired-findings.md`
for all five comparisons. They do not correct for model selection, training-seed
variation or possibly missing dependencies in inherited groups.

## Peripheral material retained outside the main narrative

The DeepEyes-inspired fixed-view screen used 64 TRAIN contexts. Three-vote
accuracy was 60/64 with repeated full views versus 59/64 with a full view plus two
crops; no learned inspection policy was trained. Dr. GRPO normalization and STaR
explanation bootstrapping were discussed but not implemented. Visual-RFT/VLM-R1
motivated verifiable reward experiments, and RL coverage/reasoning papers informed
the distinction between sampled-answer coverage and single-answer accuracy.
Existing manuscript citations retain these attributions.

Exact classifier prompt text, content order, preprocessing and decision rules
are collected in `configs/inference-manifest.json`, with source pointers and no
private example content. This improves the source package but does not create a
public artifact location. No repository or checkpoint URL has been fabricated.
