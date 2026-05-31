# TabPFN configuration and calibration policy

> Position in the paper: **Architectures (3.2)**. Reads after
> `xgboost.md`; precedes the Methods docs of Additions 2-6. The
> v2.6 / v3-default / v3-binary within-family tie at the headache
> headline cell is the family-level claim load-bearing for
> `results_findings.md` Headline takeaway 1.

## Why TabPFN, why now

The in-context tabular prior that TabPFN extends to ~50,000-row inputs is
recent: `tabpfn` 8.x shipped TabPFN-v3 as the first release whose
small-sample table classification is competitive with hyperparameter-tuned
tree ensembles on heterogeneous tabular benchmarks
[hollmann2025tabpfn; grinsztajn2025tabpfn25]. The Park 2016 SHD cohort
(62 patients, 4,516 patient-days, 7.2% positive rate) is exactly the
small-N, low-prevalence regime where (a) a foundation-model prior has the
most to add over per-leaf re-tuning, and (b) the hyperparameter-search
budget for a stack would otherwise dominate the variance. No prior published evaluation of TabPFN on the SHD cohort
was located: the only published ML reuse of the SHD
dataset [zhu2020mygraine] used logistic regression and random forest on a
random patient-overlapping split well before TabPFN existed.

On this paper's headline headache full_features/chrono cell the three
TabPFN variants *v2.6*, *v3-default*, and *v3-binary* cluster around
AUROC 0.65 (0.652 / 0.653 / 0.653, within the composite_sorted rule's
0.02 noise-level tier).
The composite picks *TabPFN-v2.6* by the calibration-distance tiebreak.
The within-family variant choice is therefore a calibration-noise call,
not a discrimination-driven one; the defensible cross-architecture
claim is at the family level (TabPFN beats XGBoost on the headache
headline cell), and the paper's figures use the four-token slug
`<ARCH-VAR> / <FEATURE> / <SPLIT> / <RATIO>` to keep that distinction
explicit on every appearance.

## Package

`tabpfn>=8.0.1` (PriorLabs distribution; tested at 8.0.1). Model checkpoints are auto-downloaded by the library on first call. The package default since `tabpfn` 8.0.0 is **TabPFN-v3**; prior model versions are still available via `TabPFNClassifier.create_default_for_version(ModelVersion.V<N>)`. Folder labels in the experiment tree refer to *model checkpoint* versions, not the Python package version.

## Configuration in use

```python
# version_2-6/ leaves (the v2.6 baseline)
TabPFNClassifier.create_default_for_version(
    ModelVersion.V2_6,
    device='cuda',          # or 'auto' on machines without GPU
    random_state=...,       # threaded from caller - see §3
)

# version_3-default/ leaves (TabPFN-v3 default checkpoint)
TabPFNClassifier.create_default_for_version(
    ModelVersion.V3,
    device='cuda',
    random_state=...,
)
```

Both bare TabPFN builders pin the model version explicitly via `create_default_for_version`. Without the pin, a bare `TabPFNClassifier(...)` call resolves to whichever model is the package default at runtime (`tabpfn>=8.0.0` ships TabPFN-v3 as the default), which would silently shift the architecture under the `version_2-6` label on the next sweep.

All other arguments are left at their library defaults. Most relevant defaults (identical for v2.6 and v3):
`n_estimators=8`, `softmax_temperature=0.9`, `eval_metric=None`, `balance_probabilities=False`, `categorical_features_indices=None`, `ignore_pretraining_limits=False`, `tuning_config=None`. With `tuning_config=None`, no runtime temperature recalibration step runs - the calibration property of the model rests on its meta-trained prior [1] combined with the fixed `softmax_temperature=0.9` baked into the architecture defaults (verifiable in [TabPFN v8.0.1 `classifier.py::create_default_for_version`](https://github.com/PriorLabs/TabPFN/blob/v8.0.1/src/tabpfn/classifier.py)).

The fitted base estimator was returned directly; no external calibrator wrapped it. See §4 for the rationale and the asymmetry vs the XGBoost-family architectures.

---

## §1 - Why we do *not* set `eval_metric` or `balance_probabilities`

The target has a positive rate of ~7.2% (see [`docs/dataset.md`](dataset.md) for split-level counts). PriorLabs' [Model Parameters page](https://docs.priorlabs.ai/improving-performance/model-parameters) lists two paths for imbalanced binary problems:

1. `balance_probabilities=True` - applies class-prior reweighting that **directly rewrites the output probability vector** so probabilities reflect a balanced prior rather than the natural prior. This is a probability-shape transform.
2. `eval_metric='f1'` (or `'roc_auc'`, `'balanced_accuracy'`, etc.) - triggers TabPFN's internal **decision-threshold tuning**. The underlying `predict_proba` is unaffected as a function of the input, but the tuning consumes part of the train context as an internal validation split, so probability values shift marginally vs `eval_metric=None`.

Neither was used, for two reasons:

- **`balance_probabilities=True`** would corrupt the Brier Score and ECE10 columns reported in the paper. Both metrics measure squared error / calibration of probability vs label; once probabilities are intentionally shifted off the natural prior, those columns no longer measure what their textbook names imply.
- **`eval_metric='f1'`** would be redundant double-tuning. The experiment pipeline already selects the MCC-optimal operating threshold externally on a held-out calibration set (see [`experiment/_eval/sharedMetricPrinter.py`](../experiment/_eval/sharedMetricPrinter.py) and the per-leaf `evaluate.py` / `evaluate_cv.py` files). Setting `eval_metric='f1'` would have TabPFN tune internally for F1, which we then override with externally-chosen MCC thresholds. The internal tuning would also marginally contaminate `predict_proba`, slightly weakening the calibration analysis for no methodological benefit.

Class imbalance is therefore handled at the **threshold-selection layer** of the pipeline, not at the model layer. This keeps TabPFN's posterior-estimation untouched and lets the comparison against XGBoost-family architectures be evaluated on the same operating-point selection procedure.

## §2 - Pretraining envelope

TabPFN-v2.5 / v2.6 is trained for *n* ≤ ~100,000 rows and *d* ≤ ~2,000 features (with the v2.5 sweet spot at ~50,000 rows). TabPFN-v3 extends the envelope substantially: per the v3 model report [10] §F.1 / Figure 4(a), v3 is benchmarked along a cell-budget frontier of *up to 1M rows at 200 features, 100k rows at 2,000 features, or 1k rows at 20,000 features*. The two `MAX_NUMBER_OF_*` ceilings baked into the v3 checkpoint's `InferenceConfig` reflect this: `MAX_NUMBER_OF_SAMPLES = 1,000,000` (vs `100,000` in v2.6) and `MAX_NUMBER_OF_CLASSES = 160` (vs `10` in v2.6, made possible by the attention-based many-class decoder described in §5.4).

The engineered SHD splits are 3,941 × 52 (full feature set) - comfortably inside both the v2 and v3 envelopes. `ignore_pretraining_limits` is left at its default `False` so the library will warn if data drifts outside the active envelope in future runs.

## §3 - `random_state` thread-through

TabPFN's `TabPFNClassifier.__init__` defaults `random_state=0`, which the library docstring flags as a deliberate departure from sklearn convention. If the seed is not threaded from the caller, every "seed sweep" silently runs the same n=8 ensemble. The wrapper accepts a `random_state` argument and propagates it to `TabPFNClassifier`.

## §4 - Calibration policy (asymmetry note)

> **Platt calibration applied to XGBoost-family models per Platt [2] and Niculescu-Mizil & Caruana [3]; TabPFN used directly per PriorLabs guidance, as it is meta-trained for calibration.**

Methodological context: this paper extends the benchmarking work of Spano [5], whose stacked XGBoost + L1-LR architecture (with Platt/isotonic calibration) is replicated as the Addition-0 baseline (`blended_xgb_lr_spano2026`). Addition 1 introduces the foundational tabular model TabPFN as a comparator. The two architecture families have different probability-calibration properties out of the box, so each is configured per its own published best practice rather than imposing a uniform pipeline:

- **XGBoost-family models** (stacked variants - both the Spano replication and the cleaner `stacked_2xgb_meta_lr` re-implementation) produce raw margin scores that are not calibrated as posteriors. A sigmoid (Platt) calibrator is fit on a held-out calibration set after training the base model. The sigmoid scaling method itself is due to Platt [2], originally for SVMs; the empirical justification for applying it to tree ensembles via a held-out calibration set is established by Niculescu-Mizil & Caruana [3], with the broader supervised-learning context covered in their companion ICML survey [4].
- **TabPFN** is meta-trained against a Bayesian prior that includes class-prior modelling. PriorLabs' overview documents the model as returning calibrated probabilities directly; the TabPFN-2.5 model report [6] states that the models are *"meta-trained to yield strong calibration, without the need for time-consuming and labor-intensive hyperparameter tuning."* The architecture default `softmax_temperature=0.9` is the only temperature transformation applied at inference time; with `tuning_config=None` (the default), no runtime temperature recalibration step runs (see §1). For external probability-shape control, PriorLabs recommends adjusting `softmax_temperature` rather than wrapping the model in a post-hoc calibrator.

Wrapping TabPFN in `CalibratedClassifierCV(..., method='sigmoid')` would have double-calibrated the output: a monotone sigmoid applied to already-calibrated probabilities. On the present study's calibration-set size (~439 rows × ~7.2% positives ≈ 32 positives), the Platt fit is also statistically thin and risks introducing variance rather than reducing it.

The asymmetry is noted in the methodology section of the paper. Because AUROC and AUPRC are calibration-invariant (rank-based), the comparison on those metrics is unaffected by the asymmetry. The Brier Score and ECE10 columns are where this design decision is load-bearing: on those columns, each architecture reported calibration of its own native, recommended output, which is the methodologically clean comparison.

---

## §5 - Variants benchmarked in parallel

### Final-model parameters per leaf (TRIPOD+AI Item 22)

The full prediction-model specification per TabPFN leaf is stored at `<leaf>/model.joblib`, the serialised TabPFN classifier (variant-specific: v2.6 stock, v3-default, v3-binary, Real-TabPFN-v2.5, or AutoTabPFN-v2.5) ready to call via `bundle.predict_proba(X)[:, 1]`. TabPFN variants do not carry a per-leaf hyperparameter JSON because the TabPFN family is used at meta-trained defaults under the no-per-variant-HPO discipline of §1.3 (the only fitted artefact per leaf is the `model.joblib` itself); the AutoTabPFN variant additionally co-locates its AutoGluon predictor at `<leaf>/autogluon_predictor/` per the §5.2 path-pinning configuration. Together with the variant-pinned `tabpfn>=8.0.1` package version and the `tabpfn-extensions` commit pin documented in the manuscript Declarations, the per-leaf `model.joblib` permits re-instantiation of any cell's calibrated probability emitter for a third-party replicator.


Addition 1 evaluates five members of the TabPFN family side-by-side as parallel comparators (each a separate `version_<label>/` branch under `experiment/1/<target>/<feature_set>/tabpfn/`). They never combine outputs - each variant fits independently per leaf and contributes its own column to the comparison table.

| Folder | Variant | Citation status |
|---|---|---|
| `version_2-6/` | TabPFN-v2.6 (pinned via `ModelVersion.V2_6`; see §1–§4 above) | Architecture cite: Hollmann et al. [1] (the v2 architecture paper). The v2.6 checkpoint was the package default in `tabpfn` 7.0.0–7.1.1; no dedicated v2.6 paper exists. |
| `version_3-default/` | TabPFN-v3 default classifier checkpoint - see §5.4 | TabPFN-3 model report [10] (technical report, not peer-reviewed). |
| `version_3-binary/` | TabPFN-v3 binary-specialised classifier checkpoint - see §5.5 | TabPFN-3 model report [10]; specialised checkpoint described on the [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3) HuggingFace card. |
| `version_2-5-real/` | Real-TabPFN-2.5 - see §5.1 | Garg et al. [7], FMSD @ ICML 2025 workshop (peer-reviewed, workshop-tier). |
| `version_2-5-auto/` | AutoTabPFN on the v2.5 base - see §5.2 | **No dedicated peer-reviewed paper** (we disclose this explicitly when we introduce AutoTabPFN). |
| `version_2-5-finetuned/` | Fine-tuned TabPFN-v2.5 - see §5.3 | Rubachev et al. [8], arXiv preprint as of May 2026; no peer-reviewed venue confirmed. |

**Why `2-5-auto` and `2-5-finetuned` (not `2-6-*`):** both `AutoTabPFNClassifier` (`tabpfn_extensions==0.3.0`, `sklearn_interface.py:393`) and `FinetunedTabPFNClassifier` (`tabpfn>=7.1.1`, `finetuning/finetuned_classifier.py:210–217`) use `ModelVersion.V2_5` as their base model - `AutoTabPFNClassifier` only supports V2 and V2_5 at all; `FinetunedTabPFNClassifier` hardcodes V2_5. The captured training output in `<leaf>/_running_output/` confirms this empirically (AutoTabPFN's leaderboard lists `RealTabPFN-v2.5_*_BAG_L1` models and `tabpfn-v2.5-classifier-v2.5_*.ckpt` paths). The folder labels reflect the actual base model so the result aggregator's "Model version" column is accurate.

### §5.1 Real-TabPFN-2.5

**What's different:** uses the `v2.5_real` checkpoint shipped in the same `tabpfn` package. Garg et al. [7] describe Real-TabPFN as the result of a continued pre-training stage applied to the v2.5 architecture on a curated set of real-world tabular datasets, reporting *"substantial performance gains on 29 datasets from the OpenML AutoML Benchmark"* relative to the synthetic-only baseline.

**Configuration:** `TabPFNClassifier(model_path="v2.5_real", device=device, random_state=random_state)`. All other knobs match `build_tabpfn` (see §1).

**Calibration framing - empirical, not pre-asserted:** the Real-TabPFN paper [7] reports performance gains; it does not characterise calibration drift between the synthetic-only baseline and the real-data-continued model. The PriorLabs v2.5 model report [6] likewise discusses Real-TabPFN as a stronger-performance variant rather than a calibration concern. Whether continued pre-training on real data shifts ECE10 / Brier Score relative to the synthetic-only baseline is therefore an **empirical question this study's calibration columns answer**, not a published caveat. The §4 no-external-Platt argument still applies; the model's calibration property derives from its meta-trained prior plus the fixed `softmax_temperature=0.9` baked into the v2.5 architecture defaults - same mechanism as the v2.6 base described in §1.

### §5.2 AutoTabPFN

**What's different:** a post-hoc ensemble - `AutoTabPFNClassifier` from `tabpfn-extensions` evaluates multiple TabPFN configurations and stacks them via AutoGluon as the meta-learner. From the outside, still `fit(X, y) → predict_proba(X)`.

**Configuration:** `AutoTabPFNClassifier(max_time=7200, device=device, random_state=random_state, presets='best', eval_metric='roc_auc', phe_init_args={'path': <leaf>/autogluon_predictor})`. Notes on each non-default choice:

  - **`max_time=7200`**. The post-hoc ensemble runs ~20 configs × 8 bag folds = 160 sub-fits, and a single TabPFN fit on this GPU costs ~13 s. A 7200 s budget gives those 160 sub-fits room to fit before the AutoGluon time limit cuts off remaining configs (which would shrink the ensemble down to whichever subset finished). Under `presets='best'` the budget can be used more fully (full HPO + larger ensemble), but AutoGluon respects the ceiling via early termination; the full 4-leaf AutoTabPFN sweep at this budget lands around 8-15 GPU-hours total.
  - **`presets='best'`**. AutoGluon's library default when `presets` is unset is `'medium'`, which AutoGluon's own documentation describes as "ideal for initial prototyping" - minimal HPO, smaller bagging, no meta-learner stacking. For a benchmark paper this is the wrong default. `'best'` is the standard AutoGluon benchmark configuration (full HPO, large ensembles, meta-learner stacking). The `'extreme'` preset is intentionally avoided: it requires the `autogluon.tabular[tabarena]` extra and injects TabICL / Mitra / TabDPT / TabM as competing base models, which breaks the explicit TabPFN-2.5 scoping that the `version_2-5-auto` folder label commits us to.
  - **`eval_metric='roc_auc'`**. AutoGluon's library default is `'accuracy'`, which is degenerate on imbalanced binary: on the migraine target (positive rate ~7.2%) a constant-negative predictor reaches ~93% accuracy and would "win" the meta-learner's selection. `'roc_auc'` aligns the meta-learner's selection criterion with this benchmark's headline metric (AUROC) and is the AutoGluon convention for imbalanced binary classification. `'average_precision'` (AUPRC) would weight the minority class more aggressively but penalises calibration-only models with compressed score ranges; `'roc_auc'` is the conservative published-benchmark choice and what TabPFN's own published comparisons report.
  - **`phe_init_args={'path': <leaf>/autogluon_predictor}`** pins AutoGluon's `TabularPredictor` save directory to a per-leaf subfolder. Without this, AutoGluon defaults to a project-root `TabPFNModels/m-<timestamp>_<rand>/` directory. With the leaf-local path, the model.joblib and the AutoGluon predictor stay co-located and the train→evaluate contract is preserved across re-runs. The builder additionally `shutil.rmtree`s the predictor directory before each fit so AutoGluon's "path already exists" warning does not fire and so artifacts from a prior fit cannot leak into the new one.
  - **Persistence via `AutoTabPFNHandle`, not direct `joblib.dump` of `AutoTabPFNClassifier`.** Pickling the fitted `AutoTabPFNClassifier` directly causes a `SIGSEGV` under PyTorch 2.11.0+rocm7.2: joblib walks the wrapper's torch tensor references at `getstate`, and the AMD HIP runtime crashes inside the destructor cascade after AutoGluon's `TabularPredictor.save()` has already completed. The builder works around this by returning an `AutoTabPFNHandle` (`autotabpfn_v2_5/_persistence.py`) that pickles only the predictor's on-disk path and lazily reloads via `TabularPredictor.load(path)` at inference time. The handle implements the sklearn `predict_proba` contract, so `evaluate.py`'s generic `joblib.load(MODEL_PATH).predict_proba(X)[:, 1]` pattern works unchanged. The builder also explicitly `del base; gc.collect(); torch.cuda.empty_cache()`s before returning so the wrapper does not survive into interpreter shutdown, where the same segfault would fire.

**Restricted scope:** AutoTabPFN is scaffolded for only **four leaves** (instead of the full 24), selected for being where the bare TabPFN-v2.6 baseline showed both high AUROC AND a non-zero MCC at the calibration-set-optimal threshold (so the post-hoc-ensembling comparison is meaningful for both ranking and operating-point metrics):

  | Leaf | Baseline AUROC | Baseline MCC |
  |---|---|---|
  | migraine / full_features / 70_30 / chrono       | 0.769 | 0.300 |
  | migraine / full_features / 70_15_15 / stratified | 0.738 | 0.196 |
  | headache / full_features / 80_20 / stratified   | 0.704 | 0.240 |
  | headache / full_features / 70_30 / stratified   | 0.690 | 0.248 |

  Selection mechanism: a `leaf_filter` callable on the `Version` namedtuple in `experiment/1/_scaffold_leaves.py`. Other variants (bare TabPFN-v2.6, Real-TabPFN-2.5, Fine-tuned TabPFN-v2.6) cover the full 24-leaf grid; AutoTabPFN's 4-leaf coverage is asymmetric and should be noted in the methodology section as a deliberate compute-vs-coverage trade-off.

**Calibration framing - empirical, not pre-asserted:** the post-hoc ensemble combines outputs from multiple TabPFN base models. Whether the resulting probabilities preserve TabPFN's native calibration, drift toward over-confidence due to ensemble averaging, or drift toward under-confidence due to AutoGluon's stacker is **not characterised in the available documentation** and is treated as an empirical question by this study's ECE10 / Brier columns. The §4 no-external-Platt argument applies in the same form: do not wrap in `CalibratedClassifierCV`; let the calibration metrics report the post-hoc ensemble's native output.

**Dependency:** `tabpfn-extensions[post-hoc-ensembles]` installed from a pinned git commit (the latest PyPI release 0.3.0 has a broken `autogluon.tabular==1.4.0` pin to a phantom version). Pulls AutoGluon ~1.5 GB as transitive. AutoGluon constrains the venv to Python `<3.14`, hence this project's pyenv-managed Python 3.13.

**Citation, disclosed up front:** no dedicated peer-reviewed paper documents AutoTabPFN as a named classifier. We state this explicitly when introducing AutoTabPFN rather than implying a peer-reviewed citation exists. Software citation: [PriorLabs/tabpfn-extensions](https://github.com/PriorLabs/tabpfn-extensions). The underlying TabPFN architecture is cited via Hollmann et al. [1]. Optional methodological backing for post-hoc ensemble selection in AutoML: Purucker & Beel [9].

### §5.3 Fine-tuned TabPFN-v2.6

**What's different:** unlike the other three variants (which use pure in-context learning), this variant performs gradient updates on the training data before inference, specialising the base v2.6 model to the downstream dataset.

**Two-pass training design:** `build_finetunedtabpfn` runs two passes to satisfy both the empirical-rigour constraint (no overfitting) and a methodological transparency requirement (proof that early-stopping was the right choice):

  - **PASS 1 (diagnostic, `epochs=50`, `early_stopping=False`)** - runs the full 50-epoch cycle and captures per-epoch metrics (train mean loss, val log loss, val ROC-AUC) via a custom `FinetuningLogger` implementation. The full curve is written to `<leaf>/diagnostics/finetune_convergence_<timestamp>_<uuid>.txt` along with every logger config knob (learning rate, patience, etc.) for reproducibility. The trained model from this pass is **discarded**.
  - **PASS 2 (production, library defaults)** - `FinetunedTabPFNClassifier(device=device, random_state=random_state)` with `epochs=30`, `early_stopping=True`, `early_stopping_patience=8`, `min_delta=1e-4`, `validation_split_ratio=0.1`, `eval_metric=None` - i.e. lets the library's early-stopping signal decide convergence. This is the model serialised to `model.joblib` and consumed by `evaluate.py`.

  The asymmetry in compute budget vs `version_2-5-auto` (early-stop ≈ 30s per leaf vs AutoTabPFN's 7200s) is methodologically defensible because each variant uses *its own* published best practice: fine-tuning is convergence-bound (Rubachev et al. [8] recommend early stopping for stability), AutoTabPFN is coverage-bound (PriorLabs documents `max_time` as a wall-clock cap on independent fits). The PASS 1 convergence files in `<leaf>/diagnostics/` are there to stop the concern about whether fine-tuning was budget-starved - the curves show the validation optimum was reached, then exceeded.

**Calibration framing - empirical, not pre-asserted:** the §4 calibration argument relies on TabPFN's meta-trained-for-calibration property, which is established for the in-context-learning regime by Hollmann et al. [1]. Rubachev et al. [8] study fine-tuning regimes for TabPFN-v2 but do not characterise post-fine-tuning calibration drift. Whether full fine-tuning preserves, improves, or degrades calibration relative to the zero-shot baseline is therefore an **empirical question this study's ECE10 / Brier columns answer**. If the reliability diagram for `version_2-5-finetuned` shows visible off-diagonal drift, that finding is reported alongside the rank-based metrics (AUROC and AUPRC remain calibration-invariant either way).

### §5.4 TabPFN-v3

**What's different:** TabPFN-v3 is the next-generation model released by PriorLabs in May 2026 and shipped as the package default in `tabpfn==8.0.0`. The TabPFN-3 model report [10] describes v3 as a *"substantially redesigned architecture that scales in-context learning to datasets with one million rows"*. The architectural reworks relative to v2.5 / v2.6 (per the [v8.0.0 release notes](https://github.com/PriorLabs/TabPFN/releases/tag/v8.0.0) and the model report) are:

  - **Return to v1-style ICL on row embeddings.** v2.x alternated row-wise and feature-wise attention layers; v3 returns to v1's approach of in-context-learning over per-row embeddings, made tractable at scale by a two-stage row-compression pipeline (column-wise feature embedding → row-wise feature aggregation → ICL over rows). The TabICLv2 row-compression design (Qu et al.) is referenced as the lineage.
  - **Attention-based many-class decoder.** The fixed-width MLP classification head of v2.x is replaced with an attention-based retrieval decoder that treats class prediction as soft nearest-neighbour retrieval over the in-context training set. Class output positions are no longer tied to fixed indices and the head is non-parametric in the class count; the active `InferenceConfig.MAX_NUMBER_OF_CLASSES` is `160` (vs `10` in v2.6).
  - **RMSNorm replaces LayerNorm** throughout (mean-centering term omitted; reduces compute while preserving training stability).
  - **Native missing-value handling.** For each NaN cell, a binary indicator is concatenated with the cell value before embedding, so the model receives an explicit signal about missingness rather than relying on upstream imputation.
  - **GPU preprocessing pipeline.** Feature transformations such as quantile normalisation and the SVD-based feature augmentation are executed on GPU as part of the forward pass when `InferenceConfig.ENABLE_GPU_PREPROCESSING=True`. This flag is `True` by default in the v3 checkpoint's baked-in `InferenceConfig` (and `False` for v2.6).
  - **Feature-subsampling strategies** (`random`, `balanced`, `gini_feature_importance`, `auto`) for datasets exceeding the per-estimator feature budget; v3 defaults to `auto`, which uses Gini importance for >100k-sample datasets and falls back to balanced subsampling otherwise.
  - **Stratified row subsampling** that preserves class proportions across ensemble members (relevant for imbalanced binary tasks like this study's ~7.2% positive rate).
  - **Auto-scaling of `n_estimators`** at fit time so every feature is covered by at least one ensemble member; the effective count is exposed as `n_estimators_` after fitting.
  - **Opt-in performance features.** `torch.compile` and FlashAttention-3 are described as opt-in performance features (model report §2.4.4); the `PerformanceOptions.attention_backend` default is `AUTO`, which routes to FA3 on **Hopper-class NVIDIA GPUs only (e.g. H100)** and falls back to PyTorch SDPA on other hardware. On this project's AMD Radeon RX 7900 XT (ROCm), FA3 does not activate; SDPA is used.
  - **Larger pre-training corpus and synthetic prior.** v3 is *"pretrained exclusively on synthetic data"* from an updated Structural Causal Model prior (added many-class, temporal, spatial, and out-of-distribution components), with the final v3 model trained on more than 8 trillion tokens.

**Configuration:** `TabPFNClassifier.create_default_for_version(ModelVersion.V3, device=device, random_state=random_state)`. Pin via `ModelVersion.V3` mirrors the `version_2-6` pin: it makes the variant's architecture explicit and stable across future `tabpfn` upgrades. The default checkpoint (`tabpfn-v3-classifier-v3_default.ckpt`) is used; the binary-specialised checkpoint is covered as a separate variant in §5.5.

**Calibration framing - empirical, not pre-asserted:** the model report [10] (page 3) describes v3 as having *"calibrated predictive distributions in a single forward pass"*, and lists *"temperature scaling for probability calibration"* (model report §2.3) as an available post-processing capability that we do not use. However, the report does **not** report classification calibration metrics (ECE, Brier, reliability diagrams) for v3 itself, and does **not** compare v3 calibration against v2.x calibration empirically. Whether v3's architectural reworks (many-class decoder, GPU preprocessing, native NaN handling, return-to-row-ICL) preserve, improve, or degrade calibration relative to v2.6 on this dataset is therefore an **empirical question this study's ECE10 / Brier columns answer** rather than something we pre-assert. The §4 no-external-Platt argument applies in the same form: do not wrap in `CalibratedClassifierCV`; let the calibration metrics report the v3 native output.

**License:** TabPFN-v3 model weights are released under `tabpfn-3-license-v1.0` (per the [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3) HuggingFace card). The license explicitly permits *"testing, evaluation, and internal benchmarking"* but prohibits commercial or production use; this benchmark falls within the permitted scope. The same NONCOMMERCIAL constraint applies to v2.5 / v2.6 / v2.5_real weights (see the README license section).

**Cite:** TabPFN-3 model report [10]; the v3 architecture extends but substantially redesigns the v2 architecture of Hollmann et al. [1].

### §5.5 TabPFN-v3 binary-specialised

**What's different vs `version_3-default/`:** uses the binary-specialised v3 checkpoint, `tabpfn-v3-classifier-v3_20260417_binary.ckpt`, instead of the v3 default. Same v3 architecture as `version_3-default/`. Same model-instance defaults (`n_estimators=8`, `softmax_temperature=0.9`) and the same Hopper-only FA3 caveat (does not activate on AMD ROCm; see §5.4). The checkpoint itself ships with its own baked-in `InferenceConfig` that differs from the v3 default checkpoint - inspection via `clf.get_inference_config()` shows three differences:

  - `FEATURE_SUBSAMPLING_METHOD`: v3-default uses `'auto'` (Gini-importance-based above 100k samples, balanced below); v3-binary uses `'random'`.
  - `FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT`: v3-default `150`; v3-binary `'auto'`.
  - `PREPROCESS_TRANSFORMS`: different per-estimator preprocessor configs (different transform mixes).

`ENABLE_GPU_PREPROCESSING=True` is the same on both.

The [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3) HuggingFace model card describes this checkpoint as *"Specialized for binary classification for datasets with <200k rows"*; the model card does not detail the training distribution beyond that statement, so we cite the card's wording verbatim rather than inferring whether the checkpoint was meta-trained on binary-only tasks.

**Why include both `version_3-default/` and `version_3-binary/`:** the v3 model card recommends starting with the default and explicitly notes the specialised checkpoints as *"useful in ensembling or HPO setups, or tried manually in the regime they were trained for"*. The SHD task is binary classification with n ≈ 3.9k - directly inside the `<200k rows` regime the binary checkpoint is described as specialised for. Reporting both is the empirical answer to whether the binary specialist actually helps in our regime, or whether the default's broader training distribution covers the same ground.

**Configuration:** `TabPFNClassifier.create_default_for_version(ModelVersion.V3, model_path=prepend_cache_path("tabpfn-v3-classifier-v3_20260417_binary.ckpt"), device=device, random_state=random_state)`. Mirrors the pattern used by Real-TabPFN-2.5 in §5.1 - pin the architecture version for its defaults, override `model_path` for the specialised checkpoint. The checkpoint's own baked-in `InferenceConfig` (the three field differences listed above) takes precedence over the v3-default config on the fields it diverges on.

**Calibration framing - empirical, not pre-asserted:** the TabPFN-3 model report [10] does not characterise calibration for the binary-specialised checkpoint (it does not even single it out in the experimental results). Whether the binary specialist preserves v3's claimed calibration property is an **empirical question this study's ECE10 / Brier columns answer**. The §4 no-external-Platt argument applies in the same form.

**Cite:** TabPFN-3 model report [10]; same v3 architecture lineage from Hollmann et al. [1].

### §5.6 Final-model parameters per variant (TRIPOD+AI Item 22)

Each TabPFN variant's final fitted model is serialised to `<leaf>/model.joblib` under `experiment/1/<target>/<feature_set>/tabpfn/version_<label>/<ratio>/<split>/`. Prediction at evaluation time is `model.predict_proba(X)[:, 1]` per the §5.4 calibration policy. Because TabPFN performs no per-leaf hyperparameter tuning (the §1 no-`eval_metric` policy and the §3 deterministic `random_state` thread-through eliminate the search dimensions Optuna manages on the XGBoost side), each variant's configuration is captured fully by (i) the `model.joblib` checkpoint, (ii) the per-variant `build_*` factory in `experiment/1/_model_architecture/` (which records the `ModelVersion` enum value, the calibration flag, and any pre-training-envelope overrides), and (iii) the package version pin in `requirements.txt` (`tabpfn>=8.0.1` constrains the upstream checkpoint resolution). The AutoTabPFN variant additionally carries the AutoGluon post-hoc ensemble state under `<leaf>/autogluon_predictor/` (gitignored, regenerable from `train.py`). Together these artefacts permit re-instantiation of any TabPFN cell's prediction pipeline.

---

## References

Citation keys resolve against [`Sources.bib`](../Sources.bib) at the repository root. Numbering is per-document by order of first appearance, IEEE style.

[1] N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, and F. Hutter, "Accurate predictions on small data with a tabular foundation model," *Nature*, vol. 637, no. 8045, pp. 319–326, Jan. 2025. doi: [10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6). BibTeX key: `hollmann2025tabpfn`.

[2] J. C. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in *Advances in Large Margin Classifiers*, A. J. Smola, P. L. Bartlett, B. Schölkopf, and D. Schuurmans, Eds. Cambridge, MA: MIT Press, 1999, pp. 61–74. BibTeX key: `platt1999svm`.

[3] A. Niculescu-Mizil and R. Caruana, "Obtaining calibrated probabilities from boosting," in *Proceedings of the 21st Conference on Uncertainty in Artificial Intelligence (UAI '05)*, 2005, pp. 413–420. arXiv mirror: [1207.1403](https://arxiv.org/abs/1207.1403). BibTeX key: `niculescuMizilCaruana2005uai`.

[4] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in *Proceedings of the 22nd International Conference on Machine Learning (ICML '05)*, 2005, pp. 625–632. doi: [10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430). BibTeX key: `niculescuMizilCaruana2005icml`.

[5] M. S. Spano, "Stacked ensemble baselines for next-day migraine forecasting on the SHD cohort," Bachelor's thesis, Reutlingen University, Reutlingen, Germany, 2026. BibTeX key: `spano2026thesis`.

[6] L. Grinsztajn, K. Flöge, O. Key, *et al.*, "TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models," 2025, arXiv:2511.08667. Online: [arxiv.org/abs/2511.08667](https://arxiv.org/abs/2511.08667). The §4 quoted calibration sentence appears in the corresponding model report at [priorlabs.ai/technical-reports/tabpfn-2-5-model-report](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report). BibTeX key: `grinsztajn2025tabpfn25`.

[7] A. Garg, M. Ali, N. Hollmann, L. Purucker, S. Müller, and F. Hutter, "Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data," in *Foundation Models for Structured Data Workshop @ ICML 2025*, 2025, arXiv:2507.03971. Online: [arxiv.org/abs/2507.03971](https://arxiv.org/abs/2507.03971). BibTeX key: `garg2025realtabpfn`.

[8] I. Rubachev, A. Kotelnikov, N. Kartashev, and A. Babenko, "On Finetuning Tabular Foundation Models," 2025, arXiv:2506.08982. Online: [arxiv.org/abs/2506.08982](https://arxiv.org/abs/2506.08982). Preprint as of May 2026; no peer-reviewed venue confirmed. BibTeX key: `rubachev2025finetuning`.

[9] L. Purucker and J. Beel, "Q(D)O-ES: Population-based Quality (Diversity) Optimisation for Post Hoc Ensemble Selection in AutoML," in *Proceedings of the Second International Conference on Automated Machine Learning (AutoML 2023)*, in *Proceedings of Machine Learning Research*, vol. 224, PMLR, 2023. Online: [proceedings.mlr.press/v224/purucker23b.html](https://proceedings.mlr.press/v224/purucker23b.html). BibTeX key: `purucker2023qdoes`.

[10] Prior Labs, "TabPFN-3 Model Report," PriorLabs technical report, May 2026. Online: [storage.googleapis.com/prior-labs-tabpfn-public/reports/TabPFN_3_model_report.pdf](https://storage.googleapis.com/prior-labs-tabpfn-public/reports/TabPFN_3_model_report.pdf). Distributed with the model weights at [huggingface.co/Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3). Technical report; not peer-reviewed. BibTeX key: `priorlabs2026tabpfn3report`.

### Software (cited inline, not numbered)

- PriorLabs official repository: [github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)
- PriorLabs documentation overview (calibrated-probabilities claim): [docs.priorlabs.ai/overview](https://docs.priorlabs.ai/overview)
- Best Practices: [docs.priorlabs.ai/best-practices](https://docs.priorlabs.ai/best-practices)
- Model Parameters (`eval_metric`, `balance_probabilities`, `calibrate_temperature`): [docs.priorlabs.ai/improving-performance/model-parameters](https://docs.priorlabs.ai/improving-performance/model-parameters)
- FAQ (imbalance handling): [docs.priorlabs.ai/faq](https://docs.priorlabs.ai/faq)
- AutoTabPFN software: [github.com/PriorLabs/tabpfn-extensions](https://github.com/PriorLabs/tabpfn-extensions)
