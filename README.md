# shd-migraine-benchmark

Benchmark comparing ML approaches for **next-day migraine forecasting** on the Korean Smartphone Headache Diary cohort [1].

---

## Dataset

62 adult patients · 2 Korean neurology clinics · Aug 2014 – Apr 2015  
Source: Park et al. [1] supplementary file S1. Column inventory, row provenance, and transformation notes are documented in `docs/dataset.md`.
Current row counts, positive rate, and split statistics are reported in `data/processed/dataset_characterization.pdf`.

## Task

Binary classification of next-day headache/migraine occurrence per patient-day.

Two split categories headache/(subamount) migraines x Three split strategies (chrono, patient, stratified) × three ratios (70/15/15, 80/20, 70/30)

- see `data/processed/` package reports for per-split data statistics.
- see upcoming result aggregator html and pdf for all model results across target, splits and ratios. 

## Evaluation Protocol

Fixed across all experiments for direct comparability. Model tuning is strictly restricted to the Validation set; the Test set is locked for final unbiased evaluation.

| Metric | Purpose |
|--------|---------|
| AUROC, AUPRC | Discrimination |
| Brier score, ECE10 | Calibration |
| MCC, Sensitivity ≥ 0.50 (at respective thresholds) | Operating point |
| Accuracy, Precision, Recall, F1 (at MCC-optimal threshold) | Operating point |

Bootstrap confidence intervals reported throughout.

## Experiments

| Nr of Addition | Approach                                                                                                                                                                                                                                                                                                                        | Package | Status |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|--------|
| 0 | Two-Fold: Stacked ensemble baseline of: (1) XGBoost + L1-LR, isotonic calibration and (2) clean XGBoost. Organised under `experiment/0/<headache/migraine>/<feature_set>/<architecture>/`: `full_features/{stacked_2xgb_meta_lr,blended_xgb_lr_spano2026}` and `spano_features/blended_xgb_lr_spano2026`. `stacked_2xgb_meta_lr` = sklearn StackingClassifier with two XGBoost base learners + L1-LR meta-learner (clean methodology). `blended_xgb_lr_spano2026` = replication of Spano [2] with alpha-grid blend of XGBoost and L1-LR base learners. Clean architecture eval by CV. ; hyperparameter training | [`xgboost`](https://xgboost.readthedocs.io/en/stable/) · [`scikit-learn`](https://scikit-learn.org/stable/) | In progress  |
| 1 | Foundational tabular models (TabPFN family), under `experiment/1/<target>/<feature_set>/tabpfn/version_{2-6, 2-5-real, 2-6-auto, 2-6-finetuned}/`. Four variants benchmarked in parallel: bare TabPFN-v2.6 (current package default), Real-TabPFN-2.5 (the `v2.5_real` checkpoint, continued pre-training on real-world tabular data), AutoTabPFN (post-hoc ensemble via `tabpfn-extensions`), and fine-tuned TabPFN-v2.6 (gradient updates on the train set via `FinetunedTabPFNClassifier`). Architecture eval by 5-fold time-series CV. Per-variant configuration, calibration policy, and primary citations: [`docs/tabPfn.MD`](docs/tabPfn.MD). | [`tabpfn>=7.1.1`](https://docs.priorlabs.ai/) · [`tabpfn-extensions==0.3.0`](https://github.com/PriorLabs/tabpfn-extensions) (AutoTabPFN; pulls AutoGluon) | In progress |
| 2 | Explainability (SHAP, LIME)                                                                                                                                                                                                                                                                                                     | [`shap`](https://shap.readthedocs.io/en/latest/) · [`lime`](https://github.com/marcotcr/lime) | Planned |
| 3 | Data Analysis and Data Possibilities                                                                                                                                                                                                                                                                                            | per patient, clusters, medians ; Establishing with Bakir & Janosch : do migraines depend on past migraines / is it fully independent between migraines (outcomes), or is it depended on the amount of migraines in the last week / month. : -> yes for time series models ; If Yes, research if other datasets have had research done on it | Planned | 
| 4 | Sequence models (LSTM)                                                                                                                                                                                                                                                                                                          | [`torch`](https://pytorch.org/docs/stable/index.html) ; Additionally time series with pretrained | Planned |

<!--- | 3 | Medically pretrained models | [`transformers`](https://huggingface.co/docs/transformers/index) | Planned | --->
<!--- | 5 | Per-event disability prediction | TBD | Exploratory | ---> 

(small) Sparse Data - Research if this dataset counts as "sparse" | or the possibly highly imbalance 

(small) Does it make sense to split time-based against random based if we have / dont have dates or patient id inside one row

----

Each Rough topical Addition is self-contained under (*where applicable) `experiment/<NrAddition>/<headache/migraine>/<feature_set>/<architecture>/<*modelVersion>/<*dataSplit>/<*SplitType>/<*HyperparameterTuned>` with code, model, and results.

## Baseline Model

The stacked ensemble XGBoost with L1 with isotonic calib [sic] baseline (Addition 0) is based on the architecture conceived and developed by Marco Samuel Spano as part of his bachelor thesis at Reutlingen University [2]. 
The implementation in this repository was rebuilt from scratch following analysis of the original thesis and codebase, the latter of which required a full rework by hand prior to scientifically sound re-execution. 
See `data/cc_MarcoSpano-oldSet/dataset.md` for Mr. Spano's approach details and the rationale for rebuilding.

## Reproducibility

Aggregated metrics in each sub-additions `results/` are regenerated from per-experiment outputs (through model evaluate and evaluate_cv files) - not manually edited.

## References

Citation keys resolve against [`Sources.bib`](Sources.bib) at the repository root. Numbering is per-document by order of first appearance, IEEE style.

[1] J.-W. Park, M. K. Chu, J.-M. Kim, S.-G. Park, and S.-J. Cho, "Analysis of trigger factors in episodic migraineurs using a smartphone headache diary applications," *PLOS ONE*, vol. 11, no. 2, p. e0149577, Feb. 2016. doi: [10.1371/journal.pone.0149577](https://doi.org/10.1371/journal.pone.0149577). BibTeX key: `park2016shd`.

[2] M. S. Spano, "Stacked ensemble baselines for next-day migraine forecasting on the SHD cohort," Bachelor's thesis, Reutlingen University, Reutlingen, Germany, 2026. BibTeX key: `spano2026thesis`.

## License

Code: Full Rights Reserved to HSRT (Reutlingen University). SHD data redistributed under Park et al. [1] supplementary file terms; consult the original publication before reuse.

**TabPFN model weights (Addition 1) are non-commercial.** The TabPFN-v2.5 / v2.6 / v2.5_real checkpoints distributed by PriorLabs (`tabpfn`, `tabpfn-extensions`) and any derivative artefacts produced under `experiment/1/` are licensed for non-commercial use only. AutoGluon emits this warning at every fit: *"TabPFN-2.5 is a NONCOMMERCIAL model. Usage of this artifact (including through AutoGluon) is not permitted for commercial tasks unless granted explicit permission by the model authors (PriorLabs)."* Results in this repository are scientific replications under that licence; commercial deployment of the trained models requires a separate agreement with PriorLabs (`sales@priorlabs.ai`).
