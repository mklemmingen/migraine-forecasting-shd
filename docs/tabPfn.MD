# TabPFN — Configuration & Calibration Policy

## Package

`tabpfn==7.1.1` (PriorLabs distribution; the version-7 line corresponds to the TabPFN-v2 model series). Model checkpoints are auto-downloaded by the library on first call. Latest model checkpoint is **TabPFN-v2.6**, the package default since `tabpfn` 7.0.1 (March 2026).

Folder label `version_2-6` in the experiment tree refers to the *model checkpoint* version (TabPFN v2.6), not the Python package version.

## Configuration in use

```python
TabPFNClassifier(
    device='cuda',          # or 'auto' on machines without GPU
    random_state=...,       # threaded from caller — see §3
)
```

All other arguments are left at their library defaults. Most relevant defaults:
`n_estimators=8`, `softmax_temperature=0.9`, `eval_metric=None`, `balance_probabilities=False`, `categorical_features_indices=None`, `ignore_pretraining_limits=False`. The `inference_config` default keeps `calibrate_temperature=True` enabled — this is TabPFN's in-house posterior-calibration step and we deliberately do not disable it.

The fitted base estimator is **returned directly** — no external calibrator wraps it. See §4 for the rationale and the asymmetry vs the XGBoost-family architectures.

---

## §1 — Why we do *not* set `eval_metric` or `balance_probabilities`

The target has a positive rate of ~5.1% (see [`docs/dataset.md`](dataset.md) for split-level counts). PriorLabs' [Model Parameters page](https://docs.priorlabs.ai/improving-performance/model-parameters) lists two paths for imbalanced binary problems:

1. `balance_probabilities=True` — applies class-prior reweighting that **directly rewrites the output probability vector** so probabilities reflect a balanced prior rather than the natural prior. This is a probability-shape transform.
2. `eval_metric='f1'` (or `'roc_auc'`, `'balanced_accuracy'`, etc.) — triggers TabPFN's internal **decision-threshold tuning**. The underlying `predict_proba` is unaffected as a function of the input, but the tuning consumes part of the train context as an internal validation split, so probability values shift marginally vs `eval_metric=None`.

We use **neither**. Two reasons:

- **`balance_probabilities=True`** would corrupt the Brier Score and ECE10 columns reported in the paper. Both metrics measure squared error / calibration of probability vs label; once probabilities are intentionally shifted off the natural prior, those columns no longer measure what their textbook names imply.
- **`eval_metric='f1'`** would be redundant double-tuning. The experiment pipeline already selects the MCC-optimal operating threshold externally on a held-out calibration set (see [`experiment/_eval/sharedMetricPrinter.py`](../experiment/_eval/sharedMetricPrinter.py) and the per-leaf `evaluate.py` / `evaluate_cv.py` files). Setting `eval_metric='f1'` would have TabPFN tune internally for F1, which we then override with externally-chosen MCC thresholds. The internal tuning would also marginally contaminate `predict_proba`, slightly weakening the calibration analysis for no methodological benefit.

Class imbalance is therefore handled at the **threshold-selection layer** of the pipeline, not at the model layer. This keeps TabPFN's posterior-estimation untouched and lets the comparison against XGBoost-family architectures be evaluated on the same operating-point selection procedure.

## §2 — Pretraining envelope

TabPFN-v2 is trained for *n* ≤ ~100,000 rows and *d* ≤ ~2,000 features (with the v2.5 sweet spot at ~50,000 rows). The engineered SHD splits are 3,941 × 52 (full feature set) — comfortably inside the envelope. `ignore_pretraining_limits` is left at its default `False` so the library will warn if data drifts outside this envelope in future runs.

## §3 — `random_state` thread-through

TabPFN's `TabPFNClassifier.__init__` defaults `random_state=0`, which the library docstring flags as a deliberate departure from sklearn convention. If the seed is not threaded from the caller, every "seed sweep" silently runs the same n=8 ensemble. The wrapper accepts a `random_state` argument and propagates it to `TabPFNClassifier`.

## §4 — Calibration policy (asymmetry note)

> **Platt calibration applied to XGBoost-family models per Platt (1999) and Niculescu-Mizil & Caruana (2005); TabPFN used directly per PriorLabs guidance, as it is meta-trained for calibration.**

Methodological context: this paper extends the benchmarking work of Spano (2026, bachelor thesis, Reutlingen University), whose stacked XGBoost + L1-LR architecture (with Platt/isotonic calibration) is replicated as the Addition-0 baseline (`blended_xgb_lr_spano2026`). Addition 1 introduces the foundational tabular model TabPFN as a comparator. The two architecture families have different probability-calibration properties out of the box, so each is configured per its own published best practice rather than imposing a uniform pipeline:

- **XGBoost-family models** (stacked variants — both the Spano replication and the cleaner `stacked_2xgb_meta_lr` re-implementation) produce raw margin scores that are not calibrated as posteriors. A sigmoid (Platt) calibrator is fit on a held-out calibration set after training the base model. The sigmoid scaling method itself is due to Platt (1999), originally for SVMs; the empirical justification for applying it to tree ensembles via a held-out calibration set is established by Niculescu-Mizil & Caruana (2005, UAI), with the broader supervised-learning context covered in their companion ICML 2005 survey.
- **TabPFN** is meta-trained against a Bayesian prior that includes class-prior modelling. PriorLabs' overview documents the model as returning calibrated probabilities directly; the TabPFN-2.5 model report states that the models are *"meta-trained to yield strong calibration, without the need for time-consuming and labor-intensive hyperparameter tuning."* TabPFN's `inference_config` also enables `calibrate_temperature=True` by default — the in-house posterior-calibration mechanism. We leave this default on. For external probability-shape control, PriorLabs recommends `softmax_temperature` rather than post-hoc calibrators.

Wrapping TabPFN in `CalibratedClassifierCV(..., method='sigmoid')` would constitute *double calibration*: a monotone sigmoid applied to already-calibrated probabilities. On the present study's calibration-set size (~439 rows × ~5.1% positives ≈ 22 positives), the Platt fit is also statistically thin and risks introducing variance rather than reducing it.

The asymmetry should be noted in the methodology section of the paper. Because AUROC and AUPRC are calibration-invariant (rank-based), the comparison on those metrics is unaffected by the asymmetry. The Brier Score and ECE10 columns are the ones where this design decision is load-bearing — for those, each architecture is reporting calibration of its own native, recommended output, which is the methodologically clean comparison.

---

## Sources

### TabPFN

- **TabPFN v2 paper:** Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. *Nature* 637, 319–326. https://doi.org/10.1038/s41586-024-08328-6
- **PriorLabs official repository:** https://github.com/PriorLabs/TabPFN
- **PriorLabs documentation:**
  - Overview (calibrated-probabilities claim): https://docs.priorlabs.ai/overview
  - Best Practices: https://docs.priorlabs.ai/best-practices
  - Model Parameters (`eval_metric`, `balance_probabilities`, `calibrate_temperature`): https://docs.priorlabs.ai/improving-performance/model-parameters
  - FAQ (imbalance handling): https://docs.priorlabs.ai/faq
- **TabPFN-2.5 Model Report** (calibration claim, exact wording quoted in §4): https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report

### Calibration of XGBoost-family architectures

- **Platt sigmoid scaling (original method):** Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74.
- **Sigmoid scaling on tree ensembles (most directly on-point empirical justification):** Niculescu-Mizil, A. & Caruana, R. (2005). Obtaining calibrated probabilities from boosting. In *Proceedings of the 21st Conference on Uncertainty in Artificial Intelligence* (UAI '05), 413–420. https://arxiv.org/abs/1207.1403
- **Broader supervised-learning calibration survey:** Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (ICML '05), 625–632. https://doi.org/10.1145/1102351.1102430

### Prior work this paper extends

- **Spano, M. S. (2026).** Bachelor thesis, Reutlingen University. Original `blended_xgb_lr_spano2026` architecture (Addition 0) is replicated from this work; legacy code preserved under `headfree-backend/`. See [`data/cc_MarcoSpano-oldSet/dataset.md`](../data/cc_MarcoSpano-oldSet/dataset.md) and the README "Baseline Model" section for rationale on the rebuild.
