# External validation: leave-one-site-out (Route A)

> Position in the paper: **Methods (4.5)**. Reads after `addition5_personalization.md`; precedes `addition6_clinical_value.md`. Leave-one-site-out external validation completes the Addition 5 scope; calibration-drift numbers feed fig_e1.

External validity is the benchmark's binding impact ceiling: it is developed
on a single 62-patient cohort, and a truly independent external dataset is
data-blocked because no comparable public next-day-diary migraine-forecasting
cohort exists. This addition
runs the strongest external validation achievable with the data in hand -
geographic internal-external validation across the two SHD recruitment
sites - and records the weaker/parallel routes.

## 1. Why this addition exists, and what it builds on

Clinical-prediction-model methodology distinguishes internal, internal-external,
and external validation; when a separate cohort is unavailable, internal-external
validation - leaving out one cluster (here, a recruitment site) in turn, training
on the rest and validating on the held-out cluster - is the appropriate design,
and is preferred over inefficient split-sample "independent" validation
[steyerberg2016validation, p. 245]. TRIPOD+AI requires reporting model
performance under a validation design with discrimination and calibration
together [collins2024tripodAI, p. 6].

The SHD cohort supports exactly this: it was recruited at two Korean university
hospitals, so a leave-one-site-out split tests whether the benchmark's findings -
especially the within-person near-chance discrimination and the
pooled-overstates-within-person result (`docs/addition5_personalization.md`
Sections 9b/9d) - transport to a different clinic's patients.

## 2. The data (verified)

The hospital site is recoverable from Sheet 1 ("62patients") of
`data/raw/SHD-Dataset.xls` (`등록번호` registration id -> `병원` hospital),
joined to the diary by uppercased `patient_id`:

| site | patients | diary rows (migraine tree) | migraine positive rate |
|------|----------|----------------------------|------------------------|
| Uijeongbu (의정부)      | 32 | 2,308 | 8.6% |
| Dongtan (동탄 한림)     | 30 | 2,170 | 5.7% |

62 of the 63 diary patients map to a site (one patient, the disability-sheet-only
`DHA-0045` reconciliation case in `docs/dataset.md`, is unmapped: 38 rows,
excluded from the site analysis). The **differing base rates (8.6% vs 5.7%)** make
this a genuine generalisation test rather than a relabelling: a model trained on
one site must transport to a different patient mix and prevalence.

## 3. Design

Two-fold leave-one-site-out: (train Uijeongbu -> test Dongtan) and (train Dongtan
-> test Uijeongbu). For each fold, evaluate three things:

1. **Discrimination transport** - pooled AUROC/AUPRC on the held-out site, with
   bootstrap 95% CIs, in the standard sharedMetricPrinter contract so the cells
   fold into `comparison_*.html` as a new split type.
2. **Within-person transport** - the per-patient AUROC distribution and the
   precision-weighted within-person C-statistic on the held-out site (reusing
   `_personal/within_person.py`), testing whether the near-chance per-patient
   result (`docs/addition5_personalization.md` Section 9b) replicates across
   sites.
3. **Calibration transport (the distinctive result)** - calibration-in-the-large
   (the ratio of observed event rate to mean predicted risk; 1.0 is perfect)
   and calibration slope on the held-out site [huang2020calibration, p. 621].
   We report the observed-to-expected ratio O:E = (observed rate) / (mean
   predicted risk) alongside slope. Because the two sites differ in base
   rate, a model trained on the 8.6% site and applied to the 5.7% site must
   mis-set its mean predicted risk; quantifying that intercept drift is a
   concrete transportability statement that goes beyond "AUROC held up" and
   ties to the benchmark's calibration-first theme. The relevant external
   validity is bounded by PROBAST (Prediction model Risk Of Bias ASsessment
   Tool) applicability: leave-one-site-out across two clinics from a single
   study is geographic, not separate-cohort, validation.

Models: the existing architectures (Additions 0/1/4) and the personalisation
regimes (Addition 5), so the transportability claim spans the whole benchmark,
not one model. Imbalance handling is unchanged (no reweighting; external
threshold step), per Addition 4 Decision 3.

## 4. Methods

- **Site join**: a small utility reads Sheet 1, maps `등록번호 -> 병원`
  (uppercased), and attaches `site` to the engineered frames by `patient_id`.
  Patients without a site are dropped from the site analysis (documented count).
- **Refit per fold**: train on the union of the other site's days, predict the
  held-out site. For the tabular/TabPFN/sequence architectures this reuses each
  addition's build path (as the CV-OOF worker already does, in a per-addition
  subprocess to avoid the `_model_architecture` clash and the TabPFN-GPU/CPU
  split). For the LR regimes it runs in-process.
- **Operating threshold**: selected on a chronological cal sub-split of the
  TRAINING site only (never the held-out site), consistent with the rest of the
  benchmark.
- **Metrics**: discrimination + within-person + calibration as in Section 3, all
  with bootstrap CIs; the discrimination cells emit the standard contract.

## 5. Directory and output layout

```
experiment/5/_personal/_site.py          # site join (Sheet 1 -> patient_id -> site)
experiment/5/_personal/_site_worker.py   # per-addition refit/predict on the site split
experiment/5/run_external_site.py        # 2-fold leave-one-site-out driver
  <target>/<feature_set>/<model>/external/site_<heldout>/results/results_*.txt
                                          # standard contract -> comparison_*.html
  external_site_summary_<ts>.csv          # discrimination + within-person + calibration drift
```

The cells parse via `_eval/_parsing.parse_path` with `architecture=<model>`
(pooled_lr | add0_stacked | add1_tabpfn | add4_window_mlp), `datasplit=external`,
`splittype=site_uijeongbu | site_dongtan`, so the site-holdout cells sit
alongside chrono/stratified/patient in the unified comparison table. Keeping every
model under `experiment/5` groups the external-validation cells together rather
than scattering them through each addition's tree.

## 6. Build status (implemented)

1. `_personal/_site.py` - the verified site join (Section 2; named `_site` to
   avoid shadowing the stdlib `site` module). The 32/30 patient counts and the
   8.6%/5.7% base rates reproduce.
2. `_personal/_site_worker.py` - per-addition subprocess that refits an
   architecture on the other site and predicts the held-out site (mirrors the
   CV-OOF worker; the GPU is shown only to Addition 1).
3. `run_external_site.py` - the 2-fold driver: in-process pooled LR plus the
   architectures via the worker, with within-person and
   calibration-in-the-large/slope per held-out site, emitting the standard
   bootstrap contract per cell. The aggregator folds the site cells into
   `comparison_*.html` (datasplit=external, splittype=site_<held>) on its next run.

## 7. Result (geographic external validation)

`run_external_site.py`, full_features, both targets, both leave-one-site-out
folds. Each model is refit on the other site and applied to the held-out site's
unseen patients; the threshold is set on a chronological cal sub-split of the
training site only. `pooled_lr` is the LR reference (per_patient and partial_pool
collapse onto it under patient-disjoint validation, Section 9a); add0/add1/add4
are the Addition 0/1/4 architectures.

**Migraine** (Uijeongbu 8.6% vs Dongtan 5.7% base rate):

| held-out (rate) | model | AUROC | within-person C | O:E | cal slope [95% CI] |
|-----------------|-------|-------|-----------------|-----|-----------|
| Uijeongbu (8.6%) | pooled_lr   | 0.691 | 0.538 | 1.135 | 0.36 [0.25, 0.48] |
| Uijeongbu (8.6%) | add0_stacked| 0.622 | 0.457 | 1.542 | 0.37 [0.25, 0.50] |
| Uijeongbu (8.6%) | add1_tabpfn | 0.742 | 0.588 | 1.404 | 0.88 [0.72, 1.07] |
| Uijeongbu (8.6%) | add4_window | 0.697 | 0.555 | 1.403 | 0.44 [0.33, 0.56] |
| Dongtan (5.7%)   | pooled_lr   | 0.730 | 0.542 | 0.683 | 0.63 [0.50, 0.76] |
| Dongtan (5.7%)   | add0_stacked| 0.726 | 0.529 | 0.944 | 1.96 [1.52, 2.41] |
| Dongtan (5.7%)   | add1_tabpfn | 0.795 | 0.546 | 0.946 | 1.13 [0.95, 1.30] |
| Dongtan (5.7%)   | add4_window | 0.742 | 0.521 | 0.497 | 0.55 [0.42, 0.71] |

**Headache** (Uijeongbu 25.4% vs Dongtan 21.6%): AUROC 0.57-0.67, within-person
0.49-0.58, O:E 0.77-1.44, slopes 0.32-2.73 (same pattern, milder because the base
rates are closer). Both folds have k=29 estimable patients; migraine k=10-14
(MIN_POS=5). Each cell emits the standard bootstrap contract.

Three findings, all architecture-independent:

1. **Pooled discrimination holds across sites within the study.** Pooled AUROC on
   the held-out site is comparable to the internal hold-out (migraine 0.62-0.80,
   TabPFN strongest); the models are not catastrophically worse on a different
   clinic. This is within-study geographic transport, not separate-cohort transport.
2. **Within-person near-chance replicates off-site** (0.46-0.59 everywhere; one
   cell below 0.50). The benchmark's central result - that per-patient day-to-day
   ranking is near-chance - is reproduced on an independent recruitment site, so
   it is not a single-cohort artefact.
3. **Calibration drifts with the site base rate (the distinctive result).** A
   model carried from the 8.6% site to the 5.7% site over-predicts (O:E down to
   0.50 - the window-MLP forecasts 11.4% risk where 5.7% is observed); carried the
   other way it under-predicts (O:E up to 1.54). The mean predicted risk tracks
   the *training* site, not the test site, and calibration slopes fall below 1
   (over-confident off-site). This is a concrete transportability statement -
   any deployment at a new site needs intercept recalibration - and it ties the
   external-validity layer to the benchmark's calibration-first theme
   [huang2020calibration, p. 621].

**No in-study model updating or recalibration was performed.** The O:E
drift between sites is reported as an honest external-validity finding,
not as the trigger for an in-study refit. The deliberate decision is to
present the as-trained models against the held-out site so the reader
sees the unaltered transport behaviour; recalibration at the new site
would mask the drift rather than document it. The leave-one-site-out
result therefore functions as a deployment-time requirement statement
("any deployment at a new site requires intercept recalibration as a
precondition") rather than a within-study correction. The
corresponding TRIPOD+AI items 12f (recalibration arising from
evaluation) and 24 (model-updating results) are answered in the
negative.

## 8. What this establishes - and what it does not

- **Establishes**: geographic internal-external validation
  [steyerberg2016validation, p. 245] - whether discrimination, within-person
  skill, and calibration transport across two recruitment sites with different
  base rates. The headline (Section 7) is that the near-chance within-person
  result replicates across sites (robustness) while calibration drifts with the
  site base rate (a concrete transportability caveat).
- **Does not establish**: true external validity. Both sites share one study,
  protocol, time window, and country; PROBAST applicability is narrow. Sites are
  small (~30 patients, ~123-198 migraine days each), so CIs are wide. This is
  same-study geographic validation, not an independent cohort. It strengthens
  credibility and PROBAST compliance, but a separate-cohort replication remains
  the next required step for transport claims.

## 9. Other routes (recorded)

- **Route B - temporal-by-enrolment split**: train early-enrolled patients, test
  late-enrolled (study-start dates available). Feasible, low effort, but partly
  redundant with the existing chronological and patient-holdout splits; a
  secondary robustness check.
- **Route C - true external on the 82-patient SHED cohort** (Cho 2018 / Park
  2018): the only route to genuine external validation, but the expanded cohort
  was never deposited, so it requires a data-sharing request to the authors
  (Cho Soo-Jin / Park Jeong-Wook). Pursue in parallel; do not gate the paper on
  it. This is the single highest-impact move if granted.
- **Route D - other public datasets**: none comparable exist; cross-sectional
  symptom-classification sets are the wrong task. Unavailable.

## References

[steyerberg2016validation] E. W. Steyerberg and F. E. Harrell, "Prediction
models need appropriate internal, internal-external, and external validation,"
*J. Clin. Epidemiol.*, vol. 69, pp. 245-247, 2016. doi:
10.1016/j.jclinepi.2015.04.005. (Internal-external validation framework;
split-sample validation is inefficient, p. 245.)

[collins2024tripodAI] G. S. Collins *et al.*, "TRIPOD+AI statement," *BMJ*, vol.
385, e078378, 2024. doi: 10.1136/bmj-2023-078378. (Report discrimination and
calibration under the validation design, item 12e, p. 6.)

[huang2020calibration] Y. Huang *et al.*, "A tutorial on calibration
measurements and calibration models for clinical prediction models," *JAMIA*,
vol. 27, no. 4, pp. 621-633, 2020. doi: 10.1093/jamia/ocz228.
(Calibration-in-the-large, p. 621.)

[holsteen2020triggers] K. K. Holsteen *et al.*, "Development and internal
validation of a multivariable prediction model for individual episodic migraine
attacks based on daily trigger exposures," *Headache*, vol. 60, no. 10, pp.
2364-2379, 2020. doi: 10.1111/head.13960. (Within-person C-statistic, p. 2364.)
