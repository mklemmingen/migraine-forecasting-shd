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

| Nr of Addition | Approach | Package | Status |
|-------|----------|---------|--------|
| 0 | Stacked ensemble baseline (XGBoost + L1-LR, isotonic calibration). Organised under `experiment/0/<feature_set>/<architecture>/`: `full_features/{clean_stack,spano_blend}` and `spano_features/spano_blend`. Clean architecture eval by CV. ; hyperparameter training | [`xgboost`](https://xgboost.readthedocs.io/en/stable/) · [`scikit-learn`](https://scikit-learn.org/stable/) | In progress  |
| 1 | Foundational tabular models (TabPFN), under `experiment/1/full_features/tabpfn/`. Architecture eval by CV. | [`tabpfn`](https://priorlabs.ai/tabpfn-documentation/); hyperparameter training ; multiple tabpfn model types evaluated incl. Real-TabPFN ; Additionally added in-built interpretability Extension: Explain TabPFN predictions with SHAP values and feature selection (TabPFN Paper) | In progress |
| 2 | Explainability (SHAP, LIME) | [`shap`](https://shap.readthedocs.io/en/latest/) · [`lime`](https://github.com/marcotcr/lime) | Planned |
| 3 | Data Analysis and Data Possibilities | per patient, clusters, medians ; Establishing with Bakir & Janosch : do migraines depend on past migraines / is it fully independent between migraines (outcomes), or is it depended on the amount of migraines in the last week / month. : -> yes for time series models ; If Yes, research if other datasets have had research done on it | Planned | 
| 4 | Sequence models (LSTM) | [`torch`](https://pytorch.org/docs/stable/index.html) ; Additionally time series with pretrained | Planned |

<!--- | 3 | Medically pretrained models | [`transformers`](https://huggingface.co/docs/transformers/index) | Planned | --->
<!--- | 5 | Per-event disability prediction | TBD | Exploratory | ---> 

(small) Sparse Data - Research if this dataset counts as "sparse" | or the possibly highly imbalance 

(small) Does it make sense to split time-based against random based if we have / dont have dates or patient id inside one row

----

Each Rough topical Addition is self-contained under `experiment/<stage>/<feature_set>/<architecture>/` with code, model, and results.

## Baseline Model

The stacked ensemble baseline (Stage 0) is based on the architecture originally conceived and developed by Marco Samuel Spano as part of his bachelor thesis at Reutlingen University (2026). The implementation in this repository was rebuilt from scratch following analysis of the original thesis and codebase, the latter of which required a full rework by hand prior to scientifically sound re-execution. See `data/cc_MarcoSpano-oldSet/dataset.md` for Mr. Spano's approach details and the rationale for rebuilding.

## Reproducibility

Aggregated metrics in each stages `results/` are regenerated from per-experiment outputs (through model evaluate and evaluate_cv files) — not manually edited.

## Citation

Park, J.-W., Chu, M. K., Kim, J.-M., Park, S.-G., & Cho, S.-J. (2016). Analysis of trigger factors in episodic migraineurs using a smartphone headache diary applications. *PLOS ONE*, 11(2), e0149577. https://doi.org/10.1371/journal.pone.0149577

## License

Code: Full Rights Reserved to HSRT (Reutlingen University). SHD data redistributed under Park et al. (2016) supplementary file terms; consult the original publication before reuse.
