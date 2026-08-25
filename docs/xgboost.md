# XGBoost in this study: architecture, tuning protocol, and findings

> Position in the long-form record: **Architectures**. Reads after
> `park_features.md`; precedes `tabPfn.MD`. The composite migraine
> headline cell that `results_findings.md` reports is one of the leaves
> trained under the HP-search protocol documented here.

This document covers the XGBoost-based learner family in Addition 0 of
the comparison: the stacking architecture we use in the NonHP baselines,
the hyperparameter-tuning protocol we apply to it for the
`HyperparameterTuned/` branch, and the findings the tuning sweep
surfaced. It is the source of truth for the XGBoost-related material in
the paper's methods section.

## 1. Architecture (NonHP baseline)

The `stacked_2xgb_meta_lr` cells used a depth-diverse stacking ensemble
of two XGBoost classifiers (`max_depth=3` and `max_depth=6`) with an
L1-penalised logistic-regression meta-learner. The stack is wrapped in
sklearn's `StackingClassifier` with `KFold(n_splits=5, shuffle=False)`
to generate the out-of-fold base-learner probabilities the meta-learner
trains on. A Platt sigmoid (LogisticRegression at `C=1e10`, unpenalised)
calibrates the meta-output on the calibration parquet (3-way ratios)
or on the chronological `cal_sub` (2-way ratios).

The choice of `KFold(shuffle=False)` for the **inner** stacking CV is a
deliberate disclosure: on a chronological cell the training parquet is
already in time order, so a non-shuffled KFold partitions the training
window into five contiguous time-ordered chunks, and the base-learner
out-of-fold predictions for chunk k are produced from a model fit on
the other four chunks. This is *approximately* time-respecting (each
fold's training set is contiguous-in-time relative to its held-out
chunk) but is not strictly an expanding-window TimeSeriesSplit (the
held-out chunk is not always the latest). The outer CV, which drives
the headline benchmark, is expanding-window TimeSeriesSplit.
KFold(shuffle=False) is chosen for the inner stack because it keeps
the out-of-fold size per fold balanced (which matters for the L1
meta's fit stability) whereas a strict TimeSeriesSplit would feed it
progressively smaller earlier-fold OOF samples; the inner-vs-outer
asymmetry is therefore a fit-strategy detail of the stacker, not part
of the evaluation protocol.

The diverse-depth design intent is to give the meta-learner access to
both a "main-effect" view of the trigger panel (the shallow tree) and
a "higher-order-interaction" view (the deep tree), so that the L1
meta-learner can pick whichever decomposition is more predictive per
patient subgroup. The hyperparameter sweep keeps the `(shallow=3,
deep=6)` structural pair fixed and tunes only the regularisation and
budget parameters that are shared across both base learners.

The `blended_xgb_lr_spano2026` cells are the canonical Spano anchor
(one XGBoost + one logistic regression, blended on `cal_sub` rather
than stacked) and are not part of the HP-tuned branch.

### Final-model parameters (TRIPOD+AI Item 22)

The full prediction-model specification per leaf is stored at two locations: (i) `<leaf>/model.joblib`, the serialised calibrated stack (shallow-depth-3 XGBoost + deep-depth-6 XGBoost + logistic-regression meta-learner + Platt-calibration sigmoid) ready to call via `calibrated_proba(bundle, X)` from `experiment/0/_model_architecture/stacked_2xgb_meta_lr_hp/model.py`; (ii) for the HP-tuned branch, the full Optuna search history (every trial) is in `<leaf>/HyperparameterTuned/<strategy>/trajectory.jsonl`, from which the selected per-trial hyperparameters (learning-rate, max-depth, subsample, colsample, gamma, min-child-weight, `n_estimators`) are recoverable via the variant-selection rule in `experiment/0/_model_architecture/stacked_2xgb_meta_lr_hp/search_lib.py`. Together these two artefacts permit re-instantiation of any cell's model and inspection of how its hyperparameters were chosen.

## 2. Search infrastructure

Two Optuna [1] search families per HP cell, sharing one hyperparameter
space:

| Family | Sampler | Trials | Used to derive |
|---|---|---|---|
| `single_AUROC` | `RandomSampler(seed=42)` | 500 | HP020 / HP050 / HP100 / HP200 / HP500 tier variants (best AUROC among trials 0..N-1) |
| `pareto_AUROC_slope` | `NSGAIISampler(seed=42, population_size=50)` | 500 | auroc_max / knee / slope_closest operating points on the (AUROC, \|slope-1\|) Pareto frontier |
| `pareto_AUPRC_slope` (migraine cells only) | `NSGAIISampler(seed=42, population_size=50)` | 500 | auprc_max / knee / slope_closest operating points on the (AUPRC, \|slope-1\|) Pareto frontier |

`RandomSampler` is deterministic given a fixed seed when run
sequentially, which guarantees `HP020 ⊂ HP050 ⊂ HP100 ⊂ HP200 ⊂ HP500`
is a strict subset relation: the first 20 trials of a 500-trial run
are bit-identical to the first 20 trials of a 20-trial run. NSGA-II at
`population_size=50` and 500 trials completes 10 generations, which is
the standard convergence target for two-objective smooth problems at
the search-space dimensionality used here.

The budget-tier ladder is applied only to the single-objective track.
NSGA-II is a stateful sampler (trial k+1 is generated from the
population of trials 1..k via selection, crossover, and mutation), so
a prefix of a 500-trial NSGA-II run is NOT equivalent to a stand-alone
NSGA-II run at that shorter budget: trials 1-49 are random
initialisation before the first generation completes, and selection
pressure only begins to operate from trial 51 onwards. Reporting
"NSGA-II Pareto at trial 20" would therefore conflate
random-initialisation samples with converged-frontier samples and
would not represent a budget-sensitivity result. The Pareto cells
therefore report the single converged 10-generation frontier at three
operating points, and the bias-from-search-size story is told
exclusively by the single-objective ladder.

## 3. Search space

Five shared dimensions across the two XGBoost base learners. The
`max_depth` structural difference (`shallow=3`, `deep=6`) is held
fixed so the stacking ensemble retains its diverse-depth design intent.

| Parameter | Range | Type | Notes |
|---|---|---|---|
| `n_estimators` | [50, 500] | int | Per-tree budget reachable on a ~3.9k-row dataset |
| `learning_rate` | [0.01, 0.30] | float (log-uniform) | Standard XGBoost range |
| `subsample` | [0.6, 1.0] | float | Row bagging |
| `colsample_bytree` | [0.6, 1.0] | float | Column bagging |
| `min_child_weight` | [1, 10] | int | Per-leaf regularisation |
| `max_depth_shallow` | 3 | fixed | Structural |
| `max_depth_deep` | 6 | fixed | Structural |
| `scale_pos_weight` | inverse class frequency | fixed | Mirrors NonHP baseline |

`reg_alpha` and `reg_lambda` are intentionally held at XGBoost
defaults: at 6 search dimensions and 500 trials random-search density
is already ~83 trials/dim; adding two more would dilute the search to
~63 trials/dim with no clear methodological gain, since
`min_child_weight` already provides the primary regularisation lever
on this sample size.

## 4. HP-scoring protocol (the val-CV decision)

The validation parquet (3-way cells) or `cal_sub` from
`chronological_subsplit(train, cal_ratio=0.20)` (2-way cells) plays
two roles during HP search: Platt-calibrator fitting and metric
scoring. Using the same data for both makes the post-Platt
calibration slope mechanically collapse toward 1.0 (Platt's role is to
enforce slope=1 on its fitting data, so any measurement on the same
set returns a near-degenerate value). This would reduce the Pareto
frontier's y-axis to a flat line and make NSGA-II behave like
single-objective AUROC search [3, calibration-on-the-same-set warning].

The HP-scoring protocol used **5-fold cross-validation** on the
calibration parquet: each trial fitted the stacker once on `X_train`,
then for each fold fitted Platt on 4/5 of the calibration parquet and
scored post-Platt metrics on the held-out 1/5. The trial's reported
metric was the per-fold average.

Per-fold metrics are preserved in the trajectory file under the
`per_fold` key so the per-trial average can be checked against
per-fold variance.

### 4.1 Aggregation rule for the per-trial metric dictionary

Each per-fold metric is averaged across the five folds independently:

    stored_calibration_slope  =  mean(  per_fold_slope_k  for k in 1..5 )
    stored_slope_dist_to_1    =  mean(  |per_fold_slope_k - 1|  for k in 1..5 )
    stored_auroc              =  mean(  per_fold_auroc_k  for k in 1..5 )
    stored_brier              =  mean(  per_fold_brier_k  for k in 1..5 )
    ...

The `slope_dist_to_1` field is therefore the K-fold-averaged
per-fold absolute deviation of slope from 1.0, not the absolute
deviation of the K-fold-averaged slope. These are different
quantities: `mean(|slope_k - 1|)` is bounded below by
`|mean(slope_k) - 1|` (triangle inequality), with equality only when
every fold's slope sits on the same side of 1.0. A trial whose
per-fold slopes are `[0.5, 1.5, 0.5, 1.5, 1.0]` has
`mean(slope) = 1.0` (perfect on average) but `mean(|slope - 1|) = 0.4`
(substantial per-fold deviation, penalised by the metric).

The choice of averaging absolute deviations rather than deviating the
average is deliberate. On the 80-90-row score-fold sizes used here
(~18 positives per fold on migraine), individual-fold slope estimates
have substantial standard error (~0.2 from logistic-regression theory
at N=20 positives). The averaged absolute-deviation metric penalises
this per-fold variance directly, biasing the search toward
configurations whose calibration is stable across folds rather than
just balanced on average. The metric is strictly stronger than the
conventional `|mean(slope) - 1|` criterion: a trial passing the
stored `slope_dist_to_1` threshold also passes the conventional one,
but not vice versa.

| Property | Single 50/50 split | 5-fold CV (chosen) |
|---|---|---|
| Migraine positives used for slope | ~19 | all ~39 in cal_sub |
| Slope-estimate standard error (1/√N) | ~0.23 | ~0.16 |
| Compute cost per trial | 1 Platt fit + 1 score | 5 Platt fits + 5 scores |
| Pareto frontier stability | medium | higher |

The choice follows the standard "use a held-out set with K-fold CV
for HP tuning, evaluate on an untouched test set" pattern; this
pattern is endorsed by TRIPOD+AI item 12c (HP tuning on a separate
set from final evaluation) [2] and addresses PROBAST+AI Analysis
signalling question 4.5 (whether HP tuning was conducted on data
separate from the evaluation set) [4].

K=5 is the conventional default in clinical prediction modelling
literature for sample sizes in the 400-1000 range. 10-fold gives ~2x
compute for marginal variance reduction; LOOCV is unbiased but
high-variance.

## 5. Evaluation-time protocol (variant-rebuild)

At evaluate time the variant's selected hyperparameters are refit
using the NonHP-matching protocol:

1. Stacker fits on `X_train` (3-way) or `train_sub` (2-way) with the
   chosen hyperparameters.
2. Platt calibrator fits on the FULL `X_val` (3-way) or full `cal_sub`
   (2-way), not the K-fold subset used during search.
3. Operating thresholds (MCC-optimal, Sensitivity >= 0.50) are
   selected on the same calibration set.
4. The locked test set is bootstrap-resampled 1000 times with the
   per-leaf protocol shared across NonHP and HP cells.

The asymmetry between HP-search Platt (K-fold subsets) and
evaluation-time Platt (full calibration set) is a deliberate choice.
The HP search produces a hyperparameter recommendation; the
evaluation applies the NonHP-matching protocol so cross-architecture
comparisons in the comparison HTML stay fair. Platt is a 2-parameter
model and its fitted coefficients are not sensitive to
calibration-set size at the scale used here.

## 6. Operating-point selection on Pareto cells

For Pareto cells, three points are reported from the non-dominated
subset of the search trajectory in (x_objective, \|slope-1\|) space:

- `auroc_max` (or `auprc_max` on AUPRC-Pareto): frontier point with
  the highest discrimination metric. Useful as the "we tuned for
  AUROC and accepted the calibration cost" reference.
- `slope_closest`: frontier point with the lowest \|slope-1\|. Useful
  as the "we tuned for calibration and accepted the discrimination
  cost" reference.
- `knee`: frontier point with the maximum perpendicular distance to
  the chord connecting the two extremes, after normalising both axes
  to [0, 1]. Useful as the "balanced trade-off" reference. The
  geometric core of the Kneedle algorithm [5]; full Kneedle adds a
  spline-smoothing pass that we skip because at 500 NSGA-II trials the
  frontier is already smoother than the smoothing window would
  resolve.

## 7. What is NOT tuned

By design, the following were held at their NonHP-baseline values
across all HP variants:

- The L1 logistic meta-learner (`penalty='l1'`, `solver='saga'`,
  `C=1.0`, `class_weight='balanced'`)
- The Stacking CV folds inside the StackingClassifier
  (`KFold(n_splits=5, shuffle=False)`)
- `scale_pos_weight` (inverse class frequency, computed at fit time)
- `reg_alpha`, `reg_lambda` (XGBoost defaults)
- The Platt calibration sigmoid (`LogisticRegression(C=1e10)`,
  unpenalised)

These choices preserve the NonHP architecture's structural identity,
so the HP variants remain a "same architecture, tuned base learners"
comparison rather than introducing de-facto different architectures.

## 8. Findings from the May 2026 sweep

The HP grid spans 12 (target, ratio, split) cells. The single-objective
budget ladder produces 5 variants per cell (HP020/050/100/200/500), the
AUROC-Pareto track produces 3 variants per cell (auroc_max / knee /
slope_closest), and the AUPRC-Pareto track adds another 3 variants on
migraine cells only. Three results are reported below that bear on
the paper's methods discussion.

### 8.1 Budget convergence at approximately 100 trials

Across every chronological cell, the single-objective tier ladder
saturated between HP100 and HP200, and HP500 was bit-identical to HP200
or HP100 on most cells. Two representative examples:

| Cell | HP020 AUROC | HP100 AUROC | HP500 AUROC |
|---|---|---|---|
| migraine/70_30/chrono | 0.793 | 0.788 | 0.788 |
| headache/70_30/chrono | 0.657 | 0.638 | 0.646 |

This is consistent with a 6-dimensional search space at ~3.9k rows:
once each dimension is sampled ~15 times, additional random samples
mostly probe noise. For the paper this supports the claim that the
HP budget is sufficient and that the sweep is not budget-starved.

### 8.2 Pareto produces different trade-offs, not a strict dominator

On the best chronological migraine cell (70/30 chrono), the two
search tracks pick different operating points:

| Variant | AUROC | AUPRC | CalSlope | MCC |
|---|---|---|---|---|
| single_AUROC / HP100 | 0.788 | 0.323 | 1.385 | 0.368 |
| pareto_AUROC_slope / knee | 0.761 | 0.294 | 1.277 | 0.341 |
| pareto_AUROC_slope / slope_closest | 0.762 | 0.287 | 1.283 | 0.250 |
| NonHP | 0.777 | 0.320 | 1.599 | 0.292 |

The Pareto knee trades 0.027 AUROC for a 0.108 closer-to-1
calibration slope. It is a valid alternative operating point but does
not dominate the single-objective AUROC pick on this cell. We report
both tracks in the comparison HTML so the trade-off is visible rather
than collapsed into one number.

### 8.3 Stratified cells reveal anti-predictive Platt fits under HP search

The stratified-split cells showed a failure mode informative for
the paper's split-design discussion. Headache/70_30/stratified is the
clearest example:

| Variant | AUROC | CalSlope | MCC |
|---|---|---|---|
| NonHP | 0.662 | 7.891 | 0.200 |
| single_AUROC / HP020 | 0.377 | -3.116 | -0.156 |
| pareto_AUROC_slope / knee | 0.381 | -0.926 | -0.153 |

A calibrated AUROC below 0.5 with a negative calibration slope
indicates Platt's fitted coefficient on `cal_sub` came out negative,
inverting the calibrated probabilities relative to the test set's
class structure. Because AUROC is invariant under monotonic
transforms but anti-invariant under sign flips of the rank order,
AUROC(1 - p) = 1 - AUROC(p): a model returning 0.377 has the same
discriminative information as a model returning 0.623, with the
labels mapped through Platt in the wrong direction.

The chronological cells do not show this. Every chronological HP cell
returns CalSlope in the [1.2, 1.4] range (over-gradient but the
correct sign). The asymmetry is consistent with patient_id leakage:
in stratified splits the same patient can appear in `X_train`,
`cal_sub`, and `X_test`. HP search picks XGBoost configurations that
maximise per-fold AUROC inside `cal_sub`'s 5-fold CV; on stratified
splits this overfits to per-patient signal in `cal_sub` that does not
generalise to the held-out test set. Platt then fits the relationship
on `cal_sub` (positive slope, well-calibrated there) but applying
that Platt to test produces inverted rankings because the model's
logits are anti-correlated with the test labels.

NonHP is partly insulated from this because XGBoost defaults produce
relatively well-spread logits that Platt can fit with a positive slope
(NonHP's CalSlope of 7.891 on the same cell shows it is still
miscalibrated, just not inverted). HP tuning explicitly searches for
configurations that maximise per-fold AUROC inside `cal_sub`, which
amplifies the latent overfitting the NonHP baseline only partly hides.

The methodological implication on this dataset is that chronological
splits avoided the calibration-inversion failure that stratified splits
exhibited under HP tuning, even when the HP-scoring protocol already
used an internal 5-fold CV on `cal_sub`. The
internal CV reduces variance of the per-trial metric estimate but does
not prevent the calibration overfit that the stratified split enables
through patient_id leakage. This strengthens the case for the
chronological-split branch as the methods-section main result, with
the stratified branch reported as a negative-finding ablation.

### 8.4 Small-holdout noise on headache/70_15_15

The 70/15/15 chronological cell on the headache target has the
smallest absolute test holdout (~155 rows, ~30% positive rate, ~46
positive events). Every HP variant on this cell clusters AUROC near
0.49 with NonHP also at 0.507. We report this as a holdout-size
ceiling, not an HP failure: the cell does not have enough events to
discriminate between HP configurations, and the bootstrap confidence
intervals on its metrics are wider than the across-variant spread.

## References

[1] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A
Next-generation Hyperparameter Optimization Framework," in
*Proceedings of the 25th ACM SIGKDD International Conference on
Knowledge Discovery & Data Mining*, 2019, pp. 2623-2631.
doi: 10.1145/3292500.3330701.

[2] G. S. Collins *et al.*, "TRIPOD+AI statement: updated guidance for
reporting clinical prediction models that use regression or machine
learning methods," *BMJ*, vol. 385, p. e078378, 2024.
doi: 10.1136/bmj-2023-078378.

[3] Y. Huang, W. Li, F. Macheret, R. A. Gabriel, and L. Ohno-Machado,
"A tutorial on calibration measurements and calibration models for
clinical prediction models," *Journal of the American Medical
Informatics Association*, vol. 27, no. 4, pp. 621-633, 2020.
doi: 10.1093/jamia/ocz228.

[4] K. G. M. Moons *et al.*, "PROBAST+AI: an updated quality, risk of
bias, and applicability assessment tool for prediction models using
regression or artificial intelligence methods," *BMJ*, vol. 388,
p. e082505, 2025. doi: 10.1136/bmj-2024-082505.

[5] V. Satopaa, J. Albrecht, D. Irwin, and B. Raghavan, "Finding a
'Kneedle' in a Haystack: Detecting Knee Points in System Behavior,"
in *31st International Conference on Distributed Computing Systems
Workshops (ICDCSW)*, 2011, pp. 166-171.
doi: 10.1109/ICDCSW.2011.20.
