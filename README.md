# shd-migraine-benchmark

Benchmark comparing ML approaches for **next-day migraine forecasting** on the Korean Smartphone Headache Diary cohort (Park et al., 2016, *PLOS ONE* 11(2):e0149577).

---

## Dataset

62 adults with migraine · 2 Korean neurology clinics · Aug 2014 – Jan 2015  
4,591 diary days · 1,099 labelled headache events · 5.3% positive prevalence  
Source: Park et al. (2016) supplementary file S1. Column inventory and transformation notes in `docs/dataset.md`.

## Task

Binary classification of next-day migraine occurrence per patient-day.  
Time-forward 85/15 split · cutoff 2015-01-24 · train n = 3,717 · val n = 674

## Evaluation Protocol

Fixed across all experiments for direct comparability.

| Metric | Purpose |
|--------|---------|
| AUROC, AUPRC | Discrimination |
| Brier score, ECE10 | Calibration |
| MCC-optimal threshold, sensitivity ≥ 0.50 | Operating point |

Bootstrap confidence intervals reported throughout.

## Experiments

| Stage | Approach | Status |
|-------|----------|--------|
| 0 | Stacked ensemble baseline (XGBoost + L1-LR, isotonic calibration) | In progress |
| 1 | Foundational tabular models (TabPFN) | Planned |
| 2 | Explainability (SHAP, LIME) | Planned |
| 3 | Medically pretrained models | Planned |
| 4 | Sequence models (LSTM) | Planned |
| 5 | Per-event disability prediction | Exploratory |

Each stage is self-contained under `experiments/<stage>/` with code, config, and results.

## Reproducibility

Dependencies pinned per stage in `environment.yml`.  
Random seeds recorded in `config.yaml`.  
Aggregated metrics in `results/` are regenerated from per-experiment outputs — not manually edited.

## Citation

Park, J.-W., Chu, M. K., Kim, J.-M., Park, S.-G., & Cho, S.-J. (2016). Analysis of trigger factors in episodic migraineurs using a smartphone headache diary applications. *PLOS ONE*, 11(2), e0149577. https://doi.org/10.1371/journal.pone.0149577

## License

Code: Full Rights Reserved to HSRT (Reutlingen University). SHD data redistributed under Park et al. (2016) supplementary file terms; consult the original publication before reuse.
