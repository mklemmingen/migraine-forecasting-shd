# shd-migraine-benchmark

Benchmark comparing ML approaches for **next-day migraine forecasting** on the Korean Smartphone Headache Diary cohort (Park et al., 2016, *PLOS ONE* 11(2):e0149577).

---

## Dataset

62 adult patients · 2 Korean neurology clinics · Aug 2014 – Apr 2015  
4,516 (engineered) diary days · 1,099 labelled headache events  
Source: Park et al. (2016) supplementary file S1. Column inventory, row provenance, and transformation notes are documented in `docs/dataset.md`!

## Task

Binary classification of next-day migraine occurrence per patient-day.  
Time-forward chronological 70/15/15 split · train n = 3,941 · val n = 439 · test n = 136.

## Evaluation Protocol

Fixed across all experiments for direct comparability. Model tuning is strictly restricted to the Validation set; the Test set is locked for final unbiased evaluation.

| Metric | Purpose |
|--------|---------|
| AUROC, AUPRC | Discrimination |
| Brier score, ECE10 | Calibration |
| MCC-optimal threshold, sensitivity ≥ 0.50 | Operating point |

Bootstrap confidence intervals reported throughout.

## Experiments

| Stage | Approach | Package | Status |
|-------|----------|---------|--------|
| 0 | Stacked ensemble baseline (XGBoost + L1-LR, isotonic calibration) | [`xgboost`](https://xgboost.readthedocs.io/en/stable/) · [`scikit-learn`](https://scikit-learn.org/stable/) | In progress |
| 1 | Foundational tabular models (TabPFN) | [`tabpfn`](https://priorlabs.ai/tabpfn-documentation/) | Planned |
| 2 | Explainability (SHAP, LIME) | [`shap`](https://shap.readthedocs.io/en/latest/) · [`lime`](https://github.com/marcotcr/lime) | Planned |
| 3 | Medically pretrained models | [`transformers`](https://huggingface.co/docs/transformers/index) | Planned |
| 4 | Sequence models (LSTM) | [`torch`](https://pytorch.org/docs/stable/index.html) | Planned |
| 5 | Per-event disability prediction | TBD | Exploratory |

Each stage is self-contained under `experiments/<stage>/` with code, config, and results.

## Baseline Model

The stacked ensemble baseline (Stage 0) is based on the architecture originally conceived and developed by Marco Samuel Spano as part of his bachelor thesis at Reutlingen University (2026). The implementation in this repository was rebuilt from scratch following analysis of the original thesis and codebase, the latter of which required a full rework by hand prior to scientifically sound re-execution. See `data/cc_MarcoSpano-oldSet/dataset.md` for Mr. Spano's approach details and the rationale for rebuilding.

## Reproducibility

Dependencies pinned per stage in `environment.yml`.  
Random seeds recorded in `config.yaml`.  
Aggregated metrics in `results/` are regenerated from per-experiment outputs — not manually edited.

## Citation

Park, J.-W., Chu, M. K., Kim, J.-M., Park, S.-G., & Cho, S.-J. (2016). Analysis of trigger factors in episodic migraineurs using a smartphone headache diary applications. *PLOS ONE*, 11(2), e0149577. https://doi.org/10.1371/journal.pone.0149577

## License

Code: Full Rights Reserved to HSRT (Reutlingen University). SHD data redistributed under Park et al. (2016) supplementary file terms; consult the original publication before reuse.
