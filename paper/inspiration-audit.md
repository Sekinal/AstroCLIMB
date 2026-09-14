# Citation and inspiration audit

Verified 2026-09-14. Working editorial audit; not a claim that every referenced method was implemented. Exact author lists and publication metadata are in `inspiration.bib`; the inventory below separates relevance from bibliographic existence.

## Editorial action: credit what the manuscript actually uses

The present manuscript already cites `astroclimb`, `qwen35`, `lora`, `gspo`, and `dapo`. Its remaining essential omissions are **SigLIP 2, Qwen3 Embedding, the exact BGE model, PP-OCRv6, Transformers, PEFT, TRL, and vLLM**. These are attributable components, including negative baselines, rather than a generic related-work list. Suggested placements:

| Manuscript claim/component | Add citation key(s) | Execution and scope |
|---|---|---|
| Frozen IXI visual/OCR baseline | `siglip2`, `ppocr6` | Actual `google/siglip2-so400m-patch14-384` plus correctly hash-aligned OCR. Baseline was superseded by the native IXI classifier. |
| Frozen CXC caption baseline | `qwen3embedding`, `bgemodel` (optionally `bgecpack`) | Actual Qwen3-Embedding-8B and BAAI/bge-base-en-v1.5. Neither historical bge-en-icl nor BGE-M3 is the final baseline. |
| Corrected OCR feature and OCR-conditioned VLM experiments | `ppocr6` | Actual PP-OCRv6_medium_det and PP-OCRv6_medium_rec. This is a model collection citation, not an invented technical report. |
| Native model loading and LoRA implementation | `wolf-etal-2020-transformers`, `peft`; retain `qwen35`, `lora` | Credit software separately from the LoRA method. |
| Actual GSPO objective and trainer | `gspo`, `trl` | Native and subsequent isolated-worker RL completed real updates; no claim that all GSPO paper experiments were reproduced. |
| Isolated vLLM rollout/throughput implementation | `vllm` | Actual serving software and PagedAttention reference. Speed claims remain benchmark-specific. |
| Mixed-outcome canary selection | retain `dapo` | Offline selection motivated by dynamic sampling; not full online DAPO, and not its token-level objective. |

`inspiration.bib` is a **selection pool**, not a request to cite all entries or use `\nocite{*}`. It repeats `lora`, `gspo`, `dapo` already in `references.bib`; merge those keys rather than loading duplicate definitions. `qwen35card` duplicates the subject of existing `qwen35` and is provided only as an alternate verified record. Software citations use upstream recommended author lists where supplied (PEFT and TRL); collection/model resources use responsible organizations. No fictitious academic-paper authors are assigned to OCR or Unsloth.

## RL inspiration: executed, partially transferred, or discussed only

| Source/key | What the campaign did | Citation boundary |
|---|---|---|
| GSPO / `gspo` | Sequence-level GSPO, native 32 and 512 updates, then vLLM-assisted canaries. | Implemented objective family; include in Methods. |
| DAPO / `dapo` | Measured unanimous groups, selected mixed TRAIN contexts, compared with class-matched random controls. | Cite for sampling motivation. Do not rename the campaign algorithm DAPO. |
| Dr. GRPO / `drgrpo` | Discussed normalization bias and task coverage; retained existing GSPO normalization for these runs. | Discussion only; no Dr. GRPO loss experiment. |
| Visual-RFT / `visualrft`; VLM-R1 / `vlmr1` | Motivated trying verifiable visual correctness rewards and avoiding reward hacking. | Related work if visual RL motivation is included; no reproduction of either full pipeline or their reported gains. |
| STaR / `star` | Considered checked rationales and supervised repair. Actual answer-tag SFT used canonical labels, not bootstrapped rationales. | Proposed rationale/teacher branch remained unrun. Do not call answer-tag SFT STaR. |
| DeepEyes / `deepeyes` | Proposed learned panel/axis inspection; ran a fixed-view prerequisite which was negative. | Partial inspiration only. No learned zoom policy, tool reward, or multi-turn environment was trained. |
| Yue et al. / `rlvrcoverage`; Wen et al. / `rlvrreasoning` | Contrasting interpretations of RLVR reasoning capacity informed caution. | Discussion only, never evidence of a universal ceiling or AstroCLIMB gain. |
| PPO / `ppo` | Background in feasibility notes. | No independent PPO experiment; GSPO credit is more specific for the paper. |
| Unsloth / `unsloth` | Investigated existing vision-RL support and ran separate compatibility/implementation experiments. The successful Qwen3.5 vLLM bridge used separate native and vLLM environments. | Credit in an engineering appendix if discussed. Do not claim successful Unsloth-integrated Qwen3.5 GSPO or bypassed support gates. Historical zero-LoRA claim was not proven. |

The RL plan's exact references are `docs/campaign_20260913/rl_signal_papers_and_plan.md`, especially “Papers and what transfers.” Execution status is cross-checked against `output/wrap_20260914/EXPERIMENT_LEDGER.json`, not inferred from the presence of a citation. The fixed-view and mixed/control results are negative or inconclusive screens, not an RL-quality success story.

## Earlier campaign inspirations: verified references, excluded from final-method claims

The historical literature sweep and prior-competition notes predate the clean final campaign. Their projected gains and structural suggestions are not measurements of the submitted system. Bibliographic verification below confirms paper identity, **not** every historical performance claim.

| Historical family | Verified keys/resources | Actual scope established by records |
|---|---|---|
| Transductive consistency / graph propagation | `tim`, `iclcleaning`, `tpn` | Historical proposal and consistency experiments; no evidence that the selected submitted neural candidate uses these published algorithms. |
| Dense retrieval hard-negative hygiene | `rocketqa`, `ance`, `adore` | Proposed inspiration. Public graph-derived relation sampling is not thereby a RocketQA/ANCE/ADORE implementation. |
| F1 threshold selection / surrogate losses | `f1threshold`, `f1optimization`, `softfbeta`, `sigmoidf1` | Historical calibration or loss proposals. The selected neural classifier's argmax is not a claimed implementation of a binary F1 theorem or surrogate loss. |
| Astronomy text representations / entities | `astrobertsts`, `astroner` | Historical alternatives and identifier motivation, not the selected Qwen3/BGE feature identities. |
| Figure extraction and scientific captioning | `figureextract1`, `figureextract2`, `figureextract3`, `scicap`, `lookreadenrich`, `figurealignment`, `galaxysearch` | Domain/corpus/OCR motivation. Cite selectively for domain context, not as implemented extraction/caption-generation algorithms. |
| TRACS ensembles and task specialization | `ojaswa-varshney-etal-2025-automated`, `khatib-etal-2025-clutch`, `kiet-etal-2025-systematic`, `wu-etal-2025-amc`, `rawat-etal-2025-encoder`, `naidu-2025-efficient` | Prior competition inspiration: ensemble reuse, task decomposition, contextual evidence, sampling/voting. Current system did not reproduce all systems. |
| Earlier WIESP methods | `alkan-etal-2022-majority`, `ghosh-etal-2022-astro`, `huang-2022-domain`, `dai-karimi-2022-detecting`, `ikoma-matsubara-2023-use`, `veeramani-etal-2023-automated` | Historical ensemble, teacher, entity, input-window and paraphrase ideas. No claim of implementing CRF, domain teacher distillation, span-NER or paraphrase pipelines. |

The canonical prior TRACS overview is `grezes-etal-2025-overview`. Its table uses **micro-F1**, so historical 0.89/0.84/0.82 numbers must not be compared directly with AstroCLIMB four-class macro-F1. Claims in old notes that identifier exploitation was “accepted” or a transferable best practice are not organizer permission and are not a final-method rationale. The selected system lineage excludes bundled-solution/structural-prior artifacts; do not rewrite campaign history as a global “never accessed anything” claim.

## Source identity corrections and unresolved items

* **astroBERT STS:** correct identifier is [2212.00744](https://arxiv.org/abs/2212.00744). Earlier 2211.13280 points elsewhere; omit it.
* **Naidu:** use the publisher's 2025 WASP paper, [2025.wasp-main.21](https://aclanthology.org/2025.wasp-main.21/). The same-title arXiv 2609.01647 is a later record; do not manufacture a 2026 competition result from it.
* **Dr. GRPO:** actual paper title is *Understanding R1-Zero-Like Training: A Critical Perspective*. Dr. GRPO is not its title.
* **DeepEyes:** first author Ziwei Zheng, distinct from GSPO's Chujie Zheng.
* **Yue et al.:** primary arXiv metadata lists Yang Yue twice. The BibTeX preserves that list rather than silently editing author identities. Verify the desired version's PDF before typesetting this optional citation.
* **LoRA year:** supplied arXiv record is 2021; its ICLR publication is 2022. Use a consistent edition rather than treating those as conflicting works.
* **BGE:** the exact English v1.5 model card lists the C-Pack technical report. The paper's Chinese-resources title alone does not identify the model used; retain the model-card identifier.
* **OCR:** no recovered proof that the old primary OCR store was PP-OCRv5. A software default changed, and the old cache was misaligned. Do not frame this as a verified v5-to-v6 accuracy improvement.
* **Qwen models:** the submitted main model is the pinned 4B model card, not a 27B report or an invented Qwen3.8 paper. Separate exploratory 27B branches require their own asset receipts if described.
* **TypeNet**, **S4VM**, **COP-KMeans**, and **BiCA** appear in historical suggestions, but that note alone does not establish exact primary metadata. No BibTeX was invented for them. The supplied BiCA link is [an AAAI endpoint](https://ojs.aaai.org/index.php/AAAI/article/view/40583/44544); S4VM/COP-KMeans had only secondary Semantic Scholar URLs in the note. These optional references are quarantined pending exact primary verification.
* **Shopee** [historical repository](https://github.com/jingxuanyang/Shopee-Product-Matching): a secondary winning-solution description does not establish that the repository is the official first-place method. Do not cite the ranking claim as verified.
* Historical Kaggle QDA/pseudo-labeling, QUEST, 30-days-of-ML and Tabular Playground write-ups were analogy sources, not identifiable academic publications or selected algorithm implementations. Do not invent paper citations for them.
* Historical `BAAI/bge-en-icl` and Jina-v4 references are model exploration, not the final frozen CXC baseline. Exact model cards would be appropriate only if those excluded runs are described.
* DeepSeek V4.1 Flash was discussed as a teacher, but no external-teacher distillation experiment ran. Coding-assistant attribution is distinct from scientific model-method credit; provider names do not establish a teacher result.

## Audit provenance and coverage limits

Inputs: `docs/campaign_20260913/rl_signal_papers_and_plan.md`, `rl_qwen35_4b_feasibility.md`, `ppocr6.md`, `cxc_dense.md`, `ixi_extract.md`; historical `docs/intel/lit-sweep.md` and `prior-tricks.md`; current `paper/sections.tex` and `references.bib`; the experiment ledger; collaborator-produced safe visible-message indexes `OWN_HISTORY_CITATION_LINKS_PRIVATE.json`, `ZCODE_CITATION_CANDIDATES_PRIVATE.json`, and `ZCODE_INSPIRATION_LINES_PRIVATE.json` under the original repository's private wrap directory. Those indexes retain source locations and paraphrases; private reasoning and Harness stderr were not read for this audit. Do not publish raw traces.

Primary verification: arXiv abstract-page citation metadata for 34 records; ACL Anthology's official `.bib` endpoints for 14 records; upstream PEFT/TRL README recommended citations; official Hugging Face Qwen, BGE, SigLIP2 and PaddlePaddle resources. Exact authors are reproduced in the companion BibTeX, not inferred from campaign shorthand. Software/model access dates are not asserted publication dates; Unsloth has no invented year. This audit deliberately marks unresolved historical items rather than presenting an exhaustive search result as a fully verified reference list.

For a four-page paper, prioritize attribution for actual dependencies and the GSPO/DAPO experiments. Add a short related-work sentence on verifiable visual RL if space permits; retain discussed-only ideas in an appendix or this audit. A long bibliography without corresponding substantive claims would obscure the system's evidence.


## Verified bibliographic inventory

| Key | Verified title | Year | Primary source |
|---|---|---|---|
| `lora` | LoRA: Low-Rank Adaptation of Large Language Models | 2021 | [arXiv](https://arxiv.org/abs/2106.09685) |
| `gspo` | Group Sequence Policy Optimization | 2025 | [arXiv](https://arxiv.org/abs/2507.18071) |
| `dapo` | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | 2025 | [arXiv](https://arxiv.org/abs/2503.14476) |
| `drgrpo` | Understanding R1-Zero-Like Training: A Critical Perspective | 2025 | [arXiv](https://arxiv.org/abs/2503.20783) |
| `visualrft` | Visual-RFT: Visual Reinforcement Fine-Tuning | 2025 | [arXiv](https://arxiv.org/abs/2503.01785) |
| `vlmr1` | VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model | 2025 | [arXiv](https://arxiv.org/abs/2504.07615) |
| `star` | STaR: Bootstrapping Reasoning With Reasoning | 2022 | [arXiv](https://arxiv.org/abs/2203.14465) |
| `deepeyes` | DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning | 2025 | [arXiv](https://arxiv.org/abs/2505.14362) |
| `rlvrcoverage` | Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | 2025 | [arXiv](https://arxiv.org/abs/2504.13837) |
| `rlvrreasoning` | Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs | 2025 | [arXiv](https://arxiv.org/abs/2506.14245) |
| `siglip2` | SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features | 2025 | [arXiv](https://arxiv.org/abs/2502.14786) |
| `qwen3embedding` | Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models | 2025 | [arXiv](https://arxiv.org/abs/2506.05176) |
| `bgecpack` | C-Pack: Packed Resources For General Chinese Embeddings | 2023 | [arXiv](https://arxiv.org/abs/2309.07597) |
| `vllm` | Efficient Memory Management for Large Language Model Serving with PagedAttention | 2023 | [arXiv](https://arxiv.org/abs/2309.06180) |
| `ppo` | Proximal Policy Optimization Algorithms | 2017 | [arXiv](https://arxiv.org/abs/1707.06347) |
| `tim` | Transductive Information Maximization For Few-Shot Learning | 2020 | [arXiv](https://arxiv.org/abs/2008.11297) |
| `iclcleaning` | Iterative label cleaning for transductive and semi-supervised few-shot learning | 2020 | [arXiv](https://arxiv.org/abs/2012.07962) |
| `tpn` | Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning | 2018 | [arXiv](https://arxiv.org/abs/1805.10002) |
| `rocketqa` | RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering | 2020 | [arXiv](https://arxiv.org/abs/2010.08191) |
| `ance` | Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval | 2020 | [arXiv](https://arxiv.org/abs/2007.00808) |
| `adore` | Optimizing Dense Retrieval Model Training with Hard Negatives | 2021 | [arXiv](https://arxiv.org/abs/2104.08051) |
| `f1threshold` | Thresholding Classifiers to Maximize F1 Score | 2014 | [arXiv](https://arxiv.org/abs/1402.1892) |
| `f1optimization` | Optimizing F-measure: A Tale of Two Approaches | 2012 | [arXiv](https://arxiv.org/abs/1206.4625) |
| `softfbeta` | A surrogate loss function for optimization of $F_\beta$ score in binary classification with imbalanced data | 2021 | [arXiv](https://arxiv.org/abs/2104.01459) |
| `sigmoidf1` | sigmoidF1: A Smooth F1 Score Surrogate Loss for Multilabel Classification | 2021 | [arXiv](https://arxiv.org/abs/2108.10566) |
| `astrobertsts` | Improving astroBERT using Semantic Textual Similarity | 2022 | [arXiv](https://arxiv.org/abs/2212.00744) |
| `astroner` | Astro-NER -- Astronomy Named Entity Recognition: Is GPT a Good Domain Expert Annotator? | 2024 | [arXiv](https://arxiv.org/abs/2405.02602) |
| `figureextract1` | Figure and Figure Caption Extraction for Mixed Raster and Vector PDFs: Digitization of Astronomical Literature with OCR Features | 2022 | [arXiv](https://arxiv.org/abs/2209.04460) |
| `figureextract2` | The Digitization of Historical Astrophysical Literature with Highly-Localized Figures and Figure Captions | 2023 | [arXiv](https://arxiv.org/abs/2302.11583) |
| `figureextract3` | Generalizability in Document Layout Analysis for Scientific Article Figure & Caption Extraction | 2023 | [arXiv](https://arxiv.org/abs/2301.10781) |
| `scicap` | SciCap: Generating Captions for Scientific Figures | 2021 | [arXiv](https://arxiv.org/abs/2110.11624) |
| `lookreadenrich` | Look, Read and Enrich. Learning from Scientific Figures and their Captions | 2019 | [arXiv](https://arxiv.org/abs/1909.09070) |
| `figurealignment` | Enhancing Scientific Figure Captioning Through Cross-modal Learning | 2024 | [arXiv](https://arxiv.org/abs/2406.17047) |
| `galaxysearch` | Semantic search for 100M+ galaxy images using AI-generated captions | 2025 | [arXiv](https://arxiv.org/abs/2512.11982) |
| `wolf-etal-2020-transformers` | Transformers: State-of-the-Art Natural Language Processing | 2020 | [ACL Anthology](https://aclanthology.org/2020.emnlp-demos.6/) |
| `grezes-etal-2025-overview` | Overview of {TRACS}: the Telescope Reference and Astronomy Categorization Dataset {\&} Shared Task | 2025 | [ACL Anthology](https://aclanthology.org/2025.wasp-main.2/) |
| `ojaswa-varshney-etal-2025-automated` | Automated Telescope-Paper Linkage via Multi-Model Ensemble Learning | 2025 | [ACL Anthology](https://aclanthology.org/2025.wasp-main.15/) |
| `kiet-etal-2025-systematic` | Systematic Evaluation of Machine Learning and Transformer-Based Methods for Scientific Telescope Literature Classification | 2025 | [ACL Anthology](https://aclanthology.org/2025.wasp-main.16/) |
| `khatib-etal-2025-clutch` | ``Clutch or Cry'' Team at {TRACS} @ {WASP}2025: A Hybrid Stacking Ensemble for Astrophysical Document Classification | 2025 | [ACL Anthology](https://aclanthology.org/2025.wasp-main.17/) |
| `wu-etal-2025-amc` | amc: The Automated Mission Classifier for Telescope Bibliographies | 2025 | [ACL Anthology](https://aclanthology.org/2025.wasp-main.18/) |
| `naidu-2025-efficient` | Efficient Context-Limited Telescope Bibliography Classification for the {WASP}-2025 Shared Task Using {S}ci{BERT} | 2025 | [ACL Anthology](https://aclanthology.org/2025.wasp-main.21/) |
| `rawat-etal-2025-encoder` | Encoder Fine-tuning with Stochastic Sampling Outperforms Open-weight {GPT} in Astronomy Knowledge Extraction | 2025 | [ACL Anthology](https://aclanthology.org/2025.wasp-main.22/) |
| `dai-karimi-2022-detecting` | Detecting Entities in the Astrophysics Literature: A Comparison of Word-based and Span-based Entity Recognition Methods | 2022 | [ACL Anthology](https://aclanthology.org/2022.wiesp-1.9/) |
| `huang-2022-domain` | Domain Specific Augmentations as Low Cost Teachers for Large Students | 2022 | [ACL Anthology](https://aclanthology.org/2022.wiesp-1.10/) |
| `ghosh-etal-2022-astro` | Astro-m{T}5: Entity Extraction from Astrophysics Literature using m{T}5 Language Model | 2022 | [ACL Anthology](https://aclanthology.org/2022.wiesp-1.12/) |
| `alkan-etal-2022-majority` | A Majority Voting Strategy of a {S}ci{BERT}-based Ensemble Models for Detecting Entities in the Astrophysics Literature (Shared Task) | 2022 | [ACL Anthology](https://aclanthology.org/2022.wiesp-1.17/) |
| `ikoma-matsubara-2023-use` | On the Use of Language Models for Function Identification of Citations in Scholarly Papers | 2023 | [ACL Anthology](https://aclanthology.org/2023.wiesp-1.15/) |
| `veeramani-etal-2023-automated` | Automated Citation Function Classification and Context Extraction in Astrophysics: Leveraging Paraphrasing and Question Answering | 2023 | [ACL Anthology](https://aclanthology.org/2023.wiesp-1.16/) |
