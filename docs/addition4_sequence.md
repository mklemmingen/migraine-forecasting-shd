# Addition 4: sequence model for next-day forecasting

> Position in the long-form record: **Methods**. Reads after `addition3_temporal.md`; precedes `addition5_personalization.md`. Sequence-baseline result feeds back into Addition 3's IID-assumption verdict; cross-architecture comparison appears in results_findings.md.

The sequence-model layer (`experiment/4/`) asks the question Additions 0
(XGBoost stacking) and 1 (TabPFN) cannot: does an explicit sequence model,
given the raw per-patient day sequence, extract more next-day signal than
the engineered history features already capture? It is framed deliberately
as a falsification test, not a leaderboard entry.

## 1. Why this addition exists, and what it builds on

Addition 3 settled that the daily attack series is serially dependent: an attack
today raises tomorrow's probability from 5.0% to 31.5% in migraine (Markov risk
ratio 6.3), and the trigger-controlled lag-1 self-excitation odds ratio is 5.43
(likelihood-ratio vs triggers-only chi2 = 155, p = 2.5e-33) (`docs/addition3_results.md`,
RQ1-RQ2). That is the empirical licence for a sequence model, and it matches the
foundational diary finding that headache days cluster, with day 1 a good
predictor of day 2 for both tension-type and migraine [houle2005timeseries, p. 445].

But the same analysis found the dependence is **short-range**: the pooled ACF
decays from lag-1 r = 0.267 to near zero by day 3-7, and the Goh-Barabasi
burstiness is near-Poisson (median B approx -0.05), i.e. there is no long-tailed
bursty memory for a deep recurrence to exploit (`docs/addition3_results.md`, RQ2).
The engineered sets in Additions 0-1 already encode exactly this short window:
`migraine_yesterday`, `migraine_rate_last3`, `migraine_rate_last7`,
`headache_free_streak`, `days_since_last_migraine` (`docs/dataset.md`, migraine
history features). So the scientific hypothesis is the **null**: a sequence model
should not materially beat the tabular models on the honest chronological split,
because the recoverable signal is short-range and already represented. Confirming
or refuting that null is the contribution and closes the loop on the temporal
plan's open question about whether explicit sequence modelling adds value
(`docs/addition3_temporal.md`, Section 10).

The state-of-the-art context bounds the expectation. The current high-water mark,
a time-series ML model on mobile-health data, reaches a next-day test AUC of 0.84
(95% CI 0.82-0.85), but on 21,550 headache days with wearable signals
(trapezius EMG, HRV, skin temperature), and its most predictive features were
headache intensity, headache duration, and heart-rate scores
[faisal2026forecasting, p. 1; p. 5], none of which the diary-only SHD cohort
carries. The small-cohort diary+wearable comparator reached only hold-out AUROC
0.62 with a random forest over support-vector-machine, random-forest, and
gradient-boosting candidates [stubberud2023forecasting, p. 1; p. 3]. Addition 4
therefore targets the diary-only, ~4.5k-row regime, where a heavy sequence model
is most likely to overfit rather than to win.

## 2. Research questions

1. **Incremental skill.** On the honest chronological 70/15/15 split, does a
   sequence model (consuming the raw per-patient day sequence) achieve higher
   test AUROC/AUPRC than the Addition 0/1 tabular models on the *same* cells,
   with non-overlapping bootstrap 95% CIs?
2. **Representation redundancy.** If a sequence model is fed only same-day
   features (no engineered lags), does it recover the performance of the
   tabular model that *was* given the engineered lags? That isolates whether the
   sequence architecture re-derives the short-window history the lag features
   encode.
3. **Window length.** Does forecast skill saturate at a short look-back (3-7
   days), as the Addition 3 ACF/burstiness result predicts, or keep improving
   with longer context?
4. **Calibration.** Does the sequence model's calibration slope sit in the same
   fragile band as the tabular models (median 0.64 across the grid,
   `docs/results_findings.md`, Section 6), or worse, given the smaller effective
   sample after sequence windowing?

## 3. Methods (with rationale and literature anchor)

### 3.1 Architectures, scaled to the data

Three models of increasing capacity, all kept small by design because the
migraine target has only 287 positive days on the train split and the
`full_features` cell sits at events-per-variable 5.5, the high-risk
overfitting band (`docs/dataset.md`, EPV section). Penalisation and capacity
control are the discipline the sample-size literature requires for small
development sets [martin2025samplesize, p. 2].

- **GRU/LSTM** with a single small hidden layer (8-32 units), dropout, and heavy
  weight decay. The default recurrent baseline.
- **1D-CNN** (project-internal module name `tcn`) with a receptive field capped at the
  look-back window via same-padded 1D convolutions, not a Bai-Kolter-Koltun
  Temporal Convolutional Network (no dilation stack, no causal padding, no
  residual connections). A convolutional alternative that is cheaper and often
  more stable than recurrence on short sequences.
- **N-day-window MLP**: the last N days of features were flattened and passed to a small
  dense network. This is the bridge model: it is a sequence model only in that
  it sees N days at once, and it is the cleanest test of RQ2 (does seeing the raw
  window beat seeing the engineered summary of the window?).

All three were trained without imbalance reweighting (`_pos_weight = 1.0`,
matching Decision 3 in Section 9); reweighting and resampling both
miscalibrated the minority class on the small SHD cohort, and imbalance was
handled downstream by the external operating-threshold step. The reported
probabilities are the network's sigmoid output; the validation split is used
to select the operating threshold (the external threshold step shared with
the tree and TabPFN leaves), and no separate post-hoc calibrator is fitted by
default (matching the Addition 1 no-external-calibrator policy). If the
calibration slope on a cell is poor, post-hoc Platt/isotonic recalibration on
the validation split is the documented fallback, applied identically across
architectures so the comparison stays like-for-like.

### 3.2 Sequence construction without leakage

The single largest methodological risk is constructing the input-output
sequences before the train/test partition, which leaks future information into
the evaluation and inflates apparent skill; this is documented specifically for
LSTM time-series evaluation, where pre-split sequence generation produces a
measurable RMSE inflation [albelali2025leakage, p. 2]. The protocol therefore
windows **within** each split, never across the split boundary: a test-set
target day may only draw its look-back window from days already inside the test
period (or be dropped if the window would cross into train/val). This mirrors the
feature-channel-leakage finding for the tabular models, where random shuffling
scattered neighbouring days across the boundary (`docs/results_findings.md`,
Section 1).

### 3.3 Gap-aware sequencing

The diary has 208 day-transitions with gaps > 1 day (max 37), 9.55% of calendar
days missing (`docs/addition3_results.md`). `_seq/windowing.py` is the
forecasting analogue of Addition 3's descriptive series builder
(`experiment/3/_temporal/series.py`): it windows the full engineered feature
matrix per patient in date order, masks padded steps, and segments at gaps
longer than a threshold N (default 7 days) so the model never recurs across a
month-long gap as if it were continuous. The Addition 3 gap features
(`days_since_last_record`,
`recording_gap_flag`, `docs/dataset.md`, Gap Awareness) are passed through as
inputs so the model can also learn to discount a stale window.

### 3.4 Evaluation: layered, same contract as Additions 0-1

Discrimination and calibration use the identical evaluator contract as the
tabular sweep (the shared `_eval.metrics_lib` / `sharedMetricPrinter` routines):
AUROC and AUPRC, MCC at the validation-/cal-derived threshold, and the full
calibration panel (slope, intercept, ECE10, Brier). Reporting discrimination and
calibration together is the TRIPOD+AI requirement [collins2024tripodAI, p. 6].
AUPRC is read per-target only, never compared across the migraine (~7%) and
headache (~24%) cells, because AUPRC's baseline depends on prevalence
[mcdermott2024aurocAuprc, p. 1].

The estimate is layered, because on this cohort a single hold-out is the weakest
link: the chronological hold-out splits 439 rows / 33 positive migraine days
into the val block and 136 rows / 5 positive migraine days into the test block,
so a single-split point estimate is high-variance and depends on whichever late
time-window happens to fall last - exactly the situation in which the
prediction-model sample-size
literature prescribes resampling over a single split [martin2025samplesize,
p. 2]. Addition 4 therefore reports two estimates per chronological cell:

1. **Primary - expanding-window time-series cross-validation** (`evaluate_cv.py`,
   scikit-learn `TimeSeriesSplit`, the same 5-fold scheme Additions 0/1 use on
   the 70/15/15 chrono cells). Each fold fits on dates up to fold k, selects the
   operating threshold on a chronological cal sub-split of the training fold, and
   scores fold k - never the reverse, so temporal order is preserved within every
   fold (consistent with the within-split windowing of Section 3.2). Reported as
   the across-fold mean and standard deviation, which is the honest uncertainty
   over multiple time origins rather than one window's bootstrap. The
   sequence-vs-tabular headline comparison leads on these numbers.
2. **Secondary - single chronological 70/15/15 hold-out** (`evaluate.py`,
   1000-iteration bootstrap 95% CIs), as the pre-registered final lock that the
   CV result was not inflated by model selection.

Both feed the existing aggregator (`run_aggregate_results.py`) unchanged, so the
sequence cells sit at the identical (target, feature_set, ratio, split)
coordinates as the Additions 0/1 cells and any difference is attributable to the
architecture, not the data.

Two further estimands are deliberately out of scope here and cross-referenced
rather than duplicated: generalisation to an unseen patient (the whole-patient
hold-out split of Additions 0/1) and within-person discrimination (the
per-patient AUROC distribution of Addition 5). They answer different questions
and belong to their own additions; mixing them in here would conflate estimands.

## 4. Honest-comparison and multiple-comparisons discipline

The three architectures x two targets x look-back windows is a modest grid; the
"best in row" marker fires only when one cell's bootstrap CI is strictly disjoint
from every comparator's, the same rule as the tabular comparison
(`docs/results_findings.md`, headline). The pre-registered confirmatory test is
RQ1 (sequence vs best tabular at the canonical 70/15/15 chronological
`full_features` cell, per target); the window-length sweep (RQ3) is exploratory
and reported as a curve, not a battery of significance tests.

## 5. Directory and output layout

Addition 4 is a model-benchmarking addition, so it follows the per-leaf scaffold
pattern of Additions 0 and 1 (not the driver pattern of Addition 3, which fits no
models). Each config is one fitted model emitting a `results_*.txt` the existing
aggregator reads; the architecture is the version axis.

```
experiment/4/
  _scaffold_leaves.py            # generates train.py/evaluate.py per leaf
  run_all_avaliable_leaves.py    # sequential train-then-evaluate runner
  _templates/
    train.py.tpl  evaluate.py.tpl  evaluate_cv.py.tpl
  _seq/
    windowing.py                 # gap-aware, within-split, row-aligned windows
    sklearn_wrapper.py           # SequenceClassifier: self-windowing estimator,
                                 #   .predict_proba(X) -> (n, 2) row-aligned
    dataread.py                  # load_seq: keeps patient_id + date on X
  _model_architecture/
    gru/model.py                 # build_gru        (+ _GRUNet)
    tcn/model.py                 # build_tcn        (+ _1D-CNNNet)
    window_mlp/model.py          # build_window_mlp (+ _WindowMLPNet)
  <target>/<feature_set>/sequence/version_<arch>/<ratio>/<split>/
    train.py  evaluate.py        # generated (all cells)
    evaluate_cv.py               # generated on chrono cells only (5-fold TimeSeriesSplit)
    model.joblib                 # fitted SequenceClassifier (CPU-resident)
    results/results_<ts>_<uuid>.txt      # hold-out, sharedMetricPrinter contract
    results/results_cv_<ts>_<uuid>.txt   # CV, METRICS_CV contract
```

The windowing lives inside the sklearn-wrapped estimator (decision in Section 9),
so the generated `evaluate.py` is line-for-line the Additions 0/1 evaluator
except it loads the id-bearing frame via `load_seq`; `model.predict_proba(X)`
builds the windows internally and returns one probability per input row.

The `results_*.txt` emitter is the same `sharedMetricPrinter` contract the
aggregator (`experiment/run_aggregate_results.py`) already parses, so Addition 4
cells fold into the existing `comparison_*.html` without aggregator changes.

## 6. Build status

The full chain is implemented and runs end-to-end. All three sequence variants
are trained on the `full_features` chronological cells for both targets (val+test
AUROC via the shared prediction worker, supplementary Figure B7): window-MLP 0.60 / GRU 0.64 /
1D-CNN 0.63 on headache, and 0.77 / 0.74 / 0.73 on migraine. No sequence variant
beats the XGBoost stack on the headline AUROC; on migraine the window-MLP
narrowly beats the vanilla TabPFN baseline (0.77 vs 0.76), so the verdict is
"competitive but not dominant" rather than uniformly behind (supplementary Figure B7). The
broader pattern still holds: explicit temporal modelling adds nothing decisive
once the rolling/lag history features are present, consistent with
the Addition 3 finding that day-to-day dependence decays within a few days.

Calibration slope (RQ4) is emitted alongside discrimination in each leaf's
`results/results_*.txt` via the shared bootstrap evaluator and is rendered
on the cohort-level reliability panel (Figure C3); the sequence cells do
not yet flow into the main `comparison_*.csv` aggregator and are read
directly from the per-leaf results for the calibration audit. The
sequence-cell slopes sit within the same fragile band documented for the
tabular cells in `docs/results_findings.md` Section 6, confirming the RQ4
prediction that windowing does not rescue calibration on the small SHD
cohort.

The build chain:

1. `_seq/windowing.py` `make_windows` - left-pad + mask edge policy, the
   per-step gap (time-interval) channel, and gap segmentation (Section 3.2-3.3,
   Decisions 1-2).
2. `_seq/sklearn_wrapper.py` `_pos_weight` (returns 1.0, Decision 3), `_train`
   (unweighted BCE, AdamW, chronological early stopping on a train tail,
   Decision 4), `_forward_probs`.
3. `_model_architecture/{gru,tcn,window_mlp}/model.py` - the three nn.Modules and
   their forward passes (last-timestep readout; the right-aligned window makes
   the last step the observed current day).

`_scaffold_leaves.py` generates the 30 leaves (chrono cells also get
`evaluate_cv.py`); `run_all_avaliable_leaves.py` trains every leaf, then runs
both `evaluate.py` (hold-out lock) and `evaluate_cv.py` (the primary CV
estimate). Each module stays under the 300-line script budget. Remaining
follow-ups are the gap-threshold sensitivity sweep (Section 9) and optional
ratio-parity with Additions 0/1 (`70_30`/`80_20`).

## 7. Dependencies

- PyTorch (already present for the TabPFN ROCm stack, `docs/tabPfn.MD`); no new
  deep-learning framework is introduced.
- No new data dependency: inputs are the existing engineered split parquets,
  loaded via `_seq/dataread.load_seq` (which keeps patient_id and date so the
  windower can build per-patient sequences).

## 8. Compute budget

Small models on ~4.5k rows train in seconds to low minutes on the existing GPU;
the look-back sweep is the only multiplier. No contention with the tabular sweep.

## 9. Resolved decisions and the open one

- **Leaf-scaffold style, not a driver**: Addition 4 follows the per-leaf
  train/evaluate scaffold of Additions 0/1 (each config emits a `results_*.txt`
  the aggregator folds into `comparison_*.html`), not the Addition 3 driver,
  because it fits and benchmarks models.
- **Windowing lives inside a sklearn-wrapped estimator**: `SequenceClassifier`
  builds the gap-aware windows internally and returns row-aligned
  probabilities, so the generated `evaluate.py` stays identical to Additions 0/1
  and the comparison table needs no special-casing. The alternative -
  windowing in the leaf templates - was rejected to keep the leaf scripts and
  the aggregator contract unchanged.
- **Layered evaluation (CV primary, hold-out lock)**: the headline estimate is
  expanding-window `TimeSeriesSplit` CV, not the single 70/15/15 hold-out,
  because a ~30-positive test window is too unstable to carry the
  sequence-vs-tabular comparison on this cohort [martin2025samplesize, p. 2]; the
  hold-out is retained as the pre-registered final lock (Section 3.4).
- **Recurrent vs convolutional vs windowed-MLP**: all three are run, because the
  point is the architecture comparison itself, not picking one a priori.
- **Continuous-time point process**: rejected for the same reason as Addition 3
  (`docs/addition3_temporal.md`, Section 3.3): day-resolution data makes a
  continuous model inappropriate; this addition stays discrete.

The four implementation decisions, each source-grounded:

- **Decision 1 - window semantics: recorded days + an explicit gap channel.**
  The window indexes the patient's recorded days (compact, trainable on ~4.5k
  rows) rather than a fully calendar-regular grid, but a final feature channel
  carries the per-step time interval since the previous retained day, normalised
  to [0, 1] - the GRU-D "time interval" representation of informative
  missingness [che2018grud, p. 1; pp. 1-2, 4]. So the model sees the calendar
  gaps Addition 3 found matter, without paying the masked-timestep cost of full
  reindexing. (`days_since_last_record` already carries the current-day gap for
  `full_features`; the channel generalises it per step and to all feature sets.)
- **Decision 2 - gap-segment threshold = lookback (7 days).** A window was not
  allowed to bridge a gap longer than the dependence horizon it exploited: Addition 3's ACF
  decays to ~0 by day 3-7 (`docs/addition3_results.md`), and day-1 is the strong
  predictor of day-2 [houle2005timeseries, p. 445]. Treated as a sensitivity
  parameter, not a silent constant, and the sweep confirms robustness: on the
  migraine/full_features chronological cell the 5-fold CV AUROC is flat across
  {3, 7, 14} (window-MLP 0.65, GRU 0.67-0.69, 1D-CNN 0.70-0.71; within-architecture
  spread <= 0.02, far inside the +/-0.05-0.10 fold-to-fold noise), so the default
  of 7 stands and the conclusion does not depend on the threshold. Reproduce with
  `experiment/4/_gap_sensitivity.py`.
- **Decision 3 - no imbalance correction (`_pos_weight = 1.0`).** Reweighting or
  resampling does not improve discrimination and strongly miscalibrates the
  minority-class probability [vandengoorbergh2022imbalance, p. 1525; p. 1530];
  this benchmark is calibration-first, and imbalance is handled downstream by the
  external operating-threshold step (the TabPFN/Addition 1 policy). Disclose the
  asymmetry that XGBoost (Addition 0) uses `scale_pos_weight`; the threshold-free
  discrimination metrics keep the comparison fair. Clamped reweighting is the
  sensitivity fallback if a cell degenerates, never the default.
- **Decision 4 - small fixed architectures, strong regularisation, early stop,
  no per-leaf HP search.** Deep nets underperform trees on tabular data even
  after HP search [grinsztajn2022tabular, p. 1; p. 2], and shrinkage/penalisation
  is the small-sample overfitting defence [martin2025samplesize, p. 2] at the
  EPV-5.5 migraine full_features cell (`docs/dataset.md`). Hyperparameters are
  fixed across cells to avoid the
  selection inflation documented for the tuned XGBoost sweep
  (`docs/results_findings.md`, Section 5); early stopping uses a chronological
  tail of the train windows only.

- **De-scoped**: a per-patient sequence model (one network per patient) was
  considered and not implemented. Addition 5 carries the personalisation
  evaluation, but at the logistic-regression level (pooled / per-patient /
  partial-pool LR regimes), not at the sequence level. The combination of a
  sequence backbone with per-patient training would require ~62 fits on
  per-patient day counts that are mostly too small to support a sequence
  model (median ~70 calendar days per patient; min < 30). The pooled
  sequence verdict (above) plus the LR-level personalisation result in
  Addition 5 § 9d together cover the per-patient question; a sequence
  per-patient extension remains a future-work item.

## 10. How this addition changes the rest of the paper

If RQ1 is null (sequence model does not beat the tabular models on honest cells),
the paper gains a clean, defensible statement: at the Park 2016 cohort scale,
with diary-only inputs, the short-range serial dependence Addition 3 measured is
already captured by the engineered lag features, and deep sequence modelling adds
no discrimination, consistent with the short-range, near-Poisson structure and
with the high-water-mark result requiring wearable signals the diary lacks
[faisal2026forecasting, p. 1]. If RQ1 is positive, it localises *which* window
length and architecture buys the gain, and RQ2 says whether the gain is the raw
sequence or merely a better-learned version of the same history summary. Either
outcome is publishable and neither requires the model to win.

## References

Citation keys resolve against `Sources.bib`. Page numbers are taken from
the cited primary source.

[houle2005timeseries] T. T. Houle, D. B. Penzien, and J. C. Rains, "Time-series
features of headache: individual distributions, patterns, and predictability of
pain," *Headache*, vol. 45, no. 5, pp. 445-458, 2005. doi:
10.1111/j.1526-4610.2005.05096.x. (Day-1 predicts day-2 clustering, p. 445.)

[faisal2026forecasting] F. Faisal *et al.*, "Forecasting migraine with
time-series machine learning from mobile health data," *J. Headache Pain*, vol.
27, no. 91, 2026. doi: 10.1186/s10194-026-02346-7. (Test AUC 0.84 and top
features, p. 1; 21,550 days, p. 5.)

[stubberud2023forecasting] A. Stubberud *et al.*, "Forecasting migraine with
machine learning based on mobile phone diary and wearable data," *Cephalalgia*,
vol. 43, no. 5, pp. 1-10, 2023. doi: 10.1177/03331024231169244. (18 patients, 388
diary entries, RF hold-out AUROC 0.62, p. 1; model candidates, p. 3.)

[albelali2025leakage] S. Albelali and M. Ahmed, "Hidden leaks in time series
forecasting: how data leakage affects LSTM evaluation across configurations and
validation strategies," arXiv:2512.06932, 2025. (Pre-split sequence construction
leaks future information, p. 2.)

[martin2025samplesize] G. P. Martin, R. D. Riley, J. Ensor, and S. W. Grant,
"Statistical primer: sample size considerations for developing and validating
clinical prediction models," *Eur. J. Cardiothorac. Surg.*, vol. 67, no. 5, p.
ezaf142, 2025. doi: 10.1093/ejcts/ezaf142. (Shrinkage/penalisation combats
overfitting in small samples, p. 2.)

[collins2024tripodAI] G. S. Collins *et al.*, "TRIPOD+AI statement," *BMJ*, vol.
385, e078378, 2024. doi: 10.1136/bmj-2023-078378. (Report discrimination and
calibration, item 12e, p. 6.)

[mcdermott2024aurocAuprc] M. B. A. McDermott *et al.*, "A closer look at AUROC and
AUPRC under class imbalance," arXiv:2401.06091, 2024. doi:
10.48550/arXiv.2401.06091. (AUPRC depends on prevalence, p. 1.)

[che2018grud] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, "Recurrent
neural networks for multivariate time series with missing values," *Scientific
Reports*, vol. 8, no. 6085, 2018. doi: 10.1038/s41598-018-24271-9. (Informative
missingness, p. 1; masking + time-interval representation with trainable decay,
pp. 1-2, 4.)

[vandengoorbergh2022imbalance] R. van den Goorbergh, M. van Smeden, D. Timmerman,
and B. Van Calster, "The harm of class imbalance corrections for risk prediction
models: illustration and simulation using logistic regression," *J. Am. Med.
Inform. Assoc.*, vol. 29, no. 9, pp. 1525-1534, 2022. doi: 10.1093/jamia/ocac093.
(Imbalance corrections overestimate the minority class and do not improve
discrimination, p. 1525; no better AUROC than uncorrected data, p. 1530.)

[grinsztajn2022tabular] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do
tree-based models still outperform deep learning on tabular data?," in *Advances
in Neural Information Processing Systems 35 (NeurIPS 2022)*, 2022.
arXiv:2207.08815. (Trees state-of-the-art on medium tabular data, p. 1; MLP
rotation-invariance/uninformative-feature weakness, p. 2.)
