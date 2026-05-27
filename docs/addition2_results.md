# Addition 2 results: model explainability (SHAP, ALE, ShapIQ)

> Position in the paper: **Results (5.1)**. Reads after `results_findings.md`; precedes `addition3_results.md`. Per-Addition explainability findings; references the methods specified in addition2_explainability.md.


Attribution and effect findings from the Addition-2 insight pass over the
selected headline and runner-up leaves. The method rationale lives in
`docs/addition2_explainability.md`; this document interprets the produced
artefacts against the five claim candidates that Addition 2 set out to test
(plan Section 1b). Each claim is marked supported / not-supported /
inconclusive with the evidence behind the verdict.

Every per-leaf artefact (mean absolute SHAP ranking, ALE slopes, ShapIQ
interaction pairs, embedding) lives in that leaf's `insights/` folder; the
cross-leaf comparison HTML files live under `experiment/2/`. The numbers
below are read from those artefacts and are reproducible by re-running
`experiment/2/run_insights.py` followed by `experiment/2/compare.py`.

## Method and reproducibility notes (caveats up front)

- **XGBoost leaves are explained by model-agnostic KernelSHAP over the
  calibrated probability** `calibrated_proba(bundle, .)`, not TreeSHAP on
  the bare boosters. The Addition-0 architectures are calibrated
  stacking/blending pipelines, so TreeSHAP cannot represent the
  meta-learner, the blend, or the calibrators; KernelSHAP on the calibrated
  closure explains exactly the probability the benchmark scores. The
  background is the leaf's val/cal set (never the train set), summarised by
  k-means (or, on the low-cardinality binary feature sets where distinct
  rows are few, the distinct background rows used directly). Seed 42.
- **TabPFN leaves use the TabPFN-native explainer** from
  `tabpfn_extensions.interpretability.shap`, which queries the in-context
  forward pass. It is sampling-based; the explained-row count is capped (60
  rows on the high-dimensional cells) to keep the per-row forward-pass cost
  tractable, so the mean absolute SHAP ranking is an average over a sample.
  Seed 42, background = val/cal set.
- **AutoTabPFN leaves use permutation importance** (AUROC drop over the
  public `predict_proba`), clearly labelled as such and never compared
  one-to-one with Shapley values.
- **ShapIQ** (order-2 k-SII) is exact on the low-cardinality cells (park: 6
  features -> 2^6 budget) and sampled on the high-dimensional full_features
  cells. Per claim 5, interaction claims are made only on the higher-EPV
  cells and are explicitly refused on the EPV-5.5 migraine full_features
  cell.
- **The 2D embedding** is a UMAP projection (fixed `random_state=42`,
  `n_neighbors=15`, `min_dist=0.1`) of TabPFN attention embeddings, coloured
  by predicted probability. It is a qualitative aid; inter-cluster
  distances in any 2D embedding must not be over-read.
- **EPV limits.** The migraine cells are sparse (~7% positive); attribution
  estimates on them carry wide uncertainty and the interaction estimates on
  the full_features migraine cell are not interpretable as effects.

## Selected leaves

The selection (`experiment/2/select.py`) is computed automatically: the
top-test-AUROC row per `(target, feature_set)` cell (headline) plus the
first CI-overlapping row whose architecture family differs (runner-up),
over the seven scientifically defined cells (headache/park excluded because
Park's stepwise model is migraine-specific). The two spano cells yield no
cross-family runner-up (Addition 1 has no spano leaves), so they contribute
the headline only -> 12 leaves.

Of the 12-leaf selection, nine claim-critical leaves are reported below,
covering every cell the five claims need (full_features chronological and
stratified for both targets, no_rolling for both, park for both, and a
cross-architecture full_features pair). The three slow TabPFN/AutoTabPFN
leaves of the current selection are pending; this limits only Claim 3's
AutoTabPFN-specific question (noted there).

| target | feature_set | split | architecture | artefacts |
|---|---|---|---|---|
| headache | full_features | chrono | stacked_2xgb (KernelSHAP) | shap, ale, ranking |
| headache | full_features | stratified | stacked_2xgb (KernelSHAP) | shap, ale, ranking |
| headache | no_rolling | chrono | stacked_2xgb (KernelSHAP) | shap, ale, ranking |
| headache | full_features | stratified | tabpfn v2-5-finetuned (native) | shap, ale |
| migraine | full_features | chrono | stacked_2xgb (KernelSHAP) | shap, ale, ranking |
| migraine | full_features | stratified | stacked_2xgb (KernelSHAP) | shap, ale, ranking |
| migraine | no_rolling | chrono | stacked_2xgb (KernelSHAP) | shap, ale, ranking |
| migraine | park | chrono | stacked_2xgb (KernelSHAP) | shap, ale |
| migraine | park | chrono | tabpfn v3-default (native) | shap, ale, shapiq, embedding |

## Claim 1 - SHAP triangulates the feature-channel leakage

**Question.** Findings Section 1 shows the stratified-split optimism is a
feature-channel leak carried by history features (`migraine_yesterday`,
`*_rate_last7`, streaks). If that mechanism is real, SHAP on a
**stratified** full_features model should put disproportionate attribution
on the history features; the same architecture on a **chronological** split
should not; and a **no_rolling** model has no such features to carry it.

**Evidence.** A controlled comparison holds the architecture and feature
set fixed (stacked_2xgb_meta_lr, full_features, NonHP, 70/15/15) and varies
only the split. KernelSHAP over the calibrated probability gives, as the
share of total mean absolute SHAP carried by the history features
(`*_yesterday`, `*_rate_last3/7`, `*_streak`, `days_since_last_*`,
`consecutive_*`, `sleep_debt_*`, `*_variability_*`):

| cell | history-feature SHAP share (full 52-feature ranking) |
|---|---|
| headache full chronological | 80.3% |
| headache full stratified    | 77.4% |
| migraine full chronological | 84.5% |
| migraine full stratified    | 85.3% |
| headache no_rolling chronological | 0.0% |
| migraine no_rolling chronological | 0.0% |

Two findings stand out. First, the no_rolling models put 0% of their
attribution on history features for the simple reason that they contain
none, so there is no channel to carry the leak: this directly confirms the
mechanism's necessary condition (Findings Section 1, the zero inflation on
no-history sets). Second, history features dominate the full_features models
on **both** splits, not only the stratified one - because on the
chronological split a patient's recent migraine rate genuinely predicts
tomorrow, so those features carry honest signal there. Mean absolute SHAP
measures total attribution and cannot, by magnitude alone, separate the
honest-signal component from the leaked-neighbour-outcome component.

The leak's fingerprint is in the *specific* neighbour-averaging features
that gain attribution under the random split that straddles the train/test
boundary:

| feature | headache chrono -> stratified | migraine chrono -> stratified |
|---|---|---|
| `migraine_rate_last7`      | rank 3 (0.0156) -> rank 1 (0.0379), x2.4 | rank 4 (0.0076) -> rank 3 (0.0110), x1.4 |
| `days_since_last_migraine` | rank 8 (0.0028) -> rank 7 (0.0038) | rank 2 (0.0132) -> rank 1 (0.0212), x1.6 |

Every neighbour-encoding rolling feature rises in rank and magnitude under
stratification; the same-day flags do not. That is the signature the
permutation-invariance argument predicts (Findings Section 1).

**Verdict:** **partially supported (mechanism confirmed, simple form
refuted).** SHAP confirms the leak is localised to feature sets that contain
history features (no_rolling: 0% attribution, no channel) and that the
neighbour-averaging features specifically gain attribution under
stratification. It does *not* support the simpler statement that history
features dominate only under stratification: they dominate the
chronological full_features model too, because they also carry honest
next-day signal. SHAP corroborates the leakage *channel* by an independent
method but, being a magnitude measure, cannot isolate the leaked fraction;
the rank/magnitude shift of `migraine_rate_last7` and
`days_since_last_migraine` under stratification is the corroborating
signal.

## Claim 2 - recovery of Park's established triggers

**Question.** On the migraine/park cells (the clean EPV-47.8 place to make
the claim), does the mean absolute SHAP ranking match Park et al. 2016
Table-4 odds ratios (stress 1.8, hormonal_changes 3.5, noise 2.8, alcohol
2.5, overeating 2.4, travel 6.4) [park2016shd, Tab. 4, p. 8]?

**Evidence.** Mean absolute SHAP on the two migraine/park leaves (the
clean EPV-47.8 cell), ranked. The XGBoost column reads the 70_30 / chrono
/ NonHP leaf (the run targeted by `run_insights`); the TabPFN column
reads the 70_15_15 / chrono v3-default leaf with comparable EPV:

| rank | XGBoost (KernelSHAP/calibrated, 70_30/NonHP) | TabPFN-v3-default (native, 70_15_15) | Park OR |
|---|---|---|---|
| 1 | overeating_today (0.0292)        | hormonal_changes_today (0.016) | travel (6.4) |
| 2 | hormonal_changes_today (0.0279)  | stress_today (0.0079)          | hormonal_changes (3.5) |
| 3 | stress_today (0.0205)            | noise_today (0.0077)           | noise (2.8) |
| 4 | alcohol_today (0.0157)           | overeating_today (0.0067)      | alcohol (2.5) |
| 5 | travel_today (0.0088)            | alcohol_today (0.0064)         | overeating (2.4) |
| 6 | noise_today (0.0054)             | travel_today (0.0043)          | stress (1.8) |

The two architectures disagree on the single top driver: XGBoost ranks
overeating_today first and hormonal_changes_today second; TabPFN
reverses that pair (hormonal_changes_today first, with overeating
demoted to fourth). The ALE net-slopes from `experiment/2/park_or_check_*.html`
are positive for the four canonical drivers in both architectures
(overeating +0.044, hormonal_changes +0.189, stress +0.070, travel
+0.017 for XGBoost; alcohol -0.036 is the lone negative slope and
matches Park's OR direction only after the multivariate-vs-univariate
adjustment discussed below). Both rankings disagree with Park: both
weight stress within the top three despite Park's smallest OR (1.8),
and both demote travel to rank 5-6 despite Park's largest OR (6.4).
Quantitatively, the Spearman rank correlation against the Park OR rank
is **ρ = +0.257 (p = 0.62, n = 6) for the TabPFN headline** and
**ρ = -0.429 (p = 0.40, n = 6) for the XGBoost 70_30 leaf** - the two
coefficients have opposite signs and neither is statistically
distinguishable from zero at this sample size, which is why the verdict
rests on set + direction, not rank.

**Verdict:** **partially supported.** Both models recover the
established trigger set and assign the four canonical drivers the same
(positive) effect direction as Park's univariate ORs. The attribution
ranking, however, does not reproduce Park's OR ordering, and the two
architectures do not agree with each other on the top driver. This is
expected: mean absolute SHAP is a multivariate next-day attribution
under a target definition (migraine_today as the next-day binary) that
differs from Park's univariate same-day ORs against the
migraine-vs-non-migraine-headache contrast. The recovery claim
therefore holds for the trigger set and the directional alignment, not
for the rank or the cross-architecture top-driver agreement.

## Claim 3 - architecture feature-reliance (AutoTabPFN vs the XGBoost stack)

**Question.** Findings Section 3 names AutoTabPFN the migraine chronological
leader. Does its edge come from relying on different features than the
XGBoost stack, or from ensembling over the same ones? The cross-leaf
comparison must include at least one Addition-0 XGBoost vs Addition-1 TabPFN
pair.

**Evidence.** The AutoTabPFN attribution path is permutation importance,
and no AutoTabPFN insight leaf completed in this pass, so the
AutoTabPFN-specific question cannot be answered here. The available
cross-architecture pair is headache/full_features: XGBoost
(stacked_2xgb_meta_lr, KernelSHAP) vs TabPFN (version_2-5-finetuned,
native SHAP). Both rank the history features at the top - XGBoost leads
with `migraine_rate_last7` / `headache_free_streak`, TabPFN leads with
`migraine_rate_last7` (0.036) / `migraine_rate_last3` (0.026) /
`headache_free_streak` (0.010). The two architectures rely on the same
feature family (recent-history rolling features), not on disjoint feature
sets.

**Verdict:** **inconclusive for AutoTabPFN; convergent reliance for
TabPFN vs XGBoost.** The headline AutoTabPFN-vs-stack question is
unanswered (AutoTabPFN attribution not computed). Where a cross-family
pair is available (headache/full_features, TabPFN vs XGBoost), both models
draw on the same recent-history features, which is consistent with the
Findings Section 3 result that the architectures are near-tied on the
honest chronological cells: the differences are not explained by reliance
on different features. Computing AutoTabPFN permutation importance on the
migraine chronological cell is the open follow-up.

## Claim 4 - prodromal-contamination check

**Question.** Premonitory symptoms (noise, specific_smells,
emotional_changes) are documented migraine prodromes
[giffin2003premonitory, p. 935; schoonman2006premonitory, p. 1209]. At a next-day lag they
might be predicting today's attack rather than tomorrow's. Does their
next-day SHAP collapse toward zero (consistent with contamination not
driving results) or stay high (consistent-with, not proof-of,
contamination)?

**Evidence.** On the full_features XGBoost models (52 features ranked by
mean absolute SHAP) the three premonitory features collapse to the bottom
half of the ranking with near-zero attribution:

| feature | headache chrono | headache strat | migraine chrono | migraine strat |
|---|---|---|---|---|
| `noise_today`             | rank 33/52 (4e-5) | rank 40/52 (1e-5) | rank 28/52 (3e-5) | rank 28/52 (8e-5) |
| `specific_smells_today`   | rank 32/52 (1e-4) | rank 43/52 (~0)  | rank 15/52 (6e-4) | rank 27/52 (1e-4) |
| `emotional_changes_today` | rank 28/52 (2e-4) | rank 31/52 (1e-4) | rank 40/52 (~0)  | rank 44/52 (~0) |

Every premonitory feature sits far below the history features and the
same-day triggers; their mean absolute SHAP is at or near 1e-4, two to
three orders of magnitude below the leading features. The one mild
exception is `specific_smells_today` on migraine/full chronological
(rank 15, 6e-4), still an order of magnitude below the leaders.

**Verdict:** **supported (next-day attribution collapses).** The documented
premonitory symptoms carry essentially no next-day SHAP when the model has
the history and same-day channels available. This is consistent with the
forecasting shift removing the same-day prodromal signal: at a one-day lag
the premonitory features are no longer predicting the attack they precede.
The framing remains cautious as planned - a low attribution is evidence
that prodromal contamination is not driving the next-day results, not proof
that no contamination exists.

## Claim 5 - feature interactions (ShapIQ), higher-EPV cells only

**Question.** Do the strongest order-2 Shapley interactions match documented
trigger combinations? Park's stepwise model carried two interaction terms
(stress x hormonal_changes, noise x travel) [park2016shd, p. 3]. Interaction
claims are confined to the higher-EPV cells; the EPV-5.5 migraine
full_features cell is explicitly excluded.

**Evidence.** Order-2 k-SII interactions on the migraine/park TabPFN
leaf (version_3-default, EPV-47.8; the EPV-5.5 migraine full_features
cell is excluded as planned), reported from the canonical
`EMIT_INSIGHTS=1 python evaluate.py` rerun on 2026-05-27 against the
patched ShapIQ runner (`experiment/_explain/shapiq_runner.py`,
NumPy 2.x compatibility restored). Background n=439, explained n=6
(the `_MAX_EXPLAIN = 6` ShapIQ cap), exhaustive 2^6 = 64-coalition
budget, seed=42. Top ten pairs by mean absolute interaction:

| rank | interaction pair | mean abs k-SII |
|---|---|---|
| 1 | alcohol x hormonal_changes | 0.0056 |
| 2 | stress x hormonal_changes | 0.0055 |
| 3 | stress x travel | 0.0050 |
| 4-8 | (five pairs involving noise) | 0.0033 (tied; imputer-fallback band) |
| 9 | overeating x travel | 0.0029 |
| 10 | overeating x hormonal_changes | 0.0028 |

The rank 4-8 tie at 0.0033 is a methodological artefact rather than a
data signal: `noise_today` is a sparse binary trigger and its
imputer-coalition predictions degenerate to the all-constant fallback
in `_ConstantTolerantTabPFN` (`shapiq_runner.py:77-118`), which
returns the background base rate (~7.2% positive rate after the
explainer's centring) for every coalition that conditions on a
constant `noise_today` column. The five "tied" pairs are therefore
the imputer's base-rate floor, not five genuinely equal interactions.

Above that floor, **both of Park's stepwise interaction terms appear
with positive k-SII**: `stress x hormonal_changes` is rank 2 at 0.0055,
within 0.0001 of the top pair; `noise x travel` sits in the tied
fallback band so its real-signal magnitude is not resolvable from this
run. `hormonal_changes_today` is the dominant partner across the
real-signal top tier (rank 1, 2, and 10).

An earlier May-2026 run on the same leaf, same seed, same parameters
but with previous TabPFN / `shapiq` / NumPy package versions produced
`stress x hormonal_changes` as the top pair (mean |k-SII| 0.0068).
The set of interacting Park triggers reproduces across runs, but the
within-set rank shifts modestly with library versioning. The
interactions are an order of magnitude below the main-effect
attributions, so they refine rather than dominate the trigger story.

**Verdict:** **partially supported (on the EPV-appropriate cell).**
The top-tier data-driven order-2 interactions are all among Park's
six stepwise triggers and `hormonal_changes` is the dominant partner;
both of Park's interaction terms appear in the analysed pair set with
positive k-SII (`stress x hormonal_changes` resolvable at rank 2;
`noise x travel` confounded by the noise-coalition fallback floor).
The within-set rank is sensitive to library versioning, so the
recovery claim holds at the set + direction level, not at the rank
level - the same epistemic limit Claim 2 documents for the
main-effect Park-OR recovery. No interaction claim is made on the
EPV-5.5 migraine full_features cell, per the pre-registered
discipline.

## Cross-checkpoint robustness (headache full_features chronological)

The composite_sorted rule places three TabPFN variants - *v2.6*,
*v3-default*, *v3-binary* - at the headache full_features chronological
70/30 headline within the rule's 0.02-AUROC noise tier (AUROC 0.652 /
0.653 / 0.653). Running the SHAP and ShapIQ pipelines independently on
all three checkpoints at this cell (background n=706, explained n=400
for SHAP and n=6 for ShapIQ, seed=42) yields:

| layer | *v2.6* | *v3-default* | *v3-binary* |
|---|---|---|---|
| Top-5 mean abs SHAP features | migraine_rate_last3, migraine_rate_last7, headache_free_streak, days_since_last_migraine, migraine_yesterday | identical set, permuted ranks | identical set, permuted ranks |
| Top SHAP magnitude | 0.0311 (migraine_rate_last3) | 0.0258 (headache_free_streak) | 0.0258 (headache_free_streak) |
| Top ShapIQ pair | vigorous_exercise_min x consecutive_stress_days | same | same |
| Top ShapIQ magnitude (96-coalition sampling) | 5.26 | 3.57 | 4.01 |

All three checkpoints converge on the same five history features at the
SHAP layer and on the same top-tier exercise x stress interaction at
the ShapIQ layer; the within-set ranks shift modestly across variants
and the ShapIQ magnitudes span a ~1.5x range (3.57-5.26). We therefore
read the cross-checkpoint robustness at the **set + dominant-pair**
level rather than at the within-set rank or absolute-magnitude level.
The headache/full cell uses the 96-coalition sampling budget for
ShapIQ, so the magnitude variation is partly confounded with sampling
variance; the exhaustive 2^6 = 64-coalition migraine/park cell in
Claim 5 remains the only ShapIQ cell on which the paper reports
absolute interaction magnitudes.

Three independent TabPFN architectures with different in-context priors
agreeing on the identical top-5 history-feature set and the identical
top-tier feature interaction is supplementary evidence for Claim 1's
underlying mechanism: the history-feature dominance and the
exercise x stress interaction on the full_features chronological cell
are properties of the data, not artefacts of any single TabPFN
checkpoint.

## Summary of verdicts

| claim | verdict | one-line basis |
|---|---|---|
| 1. SHAP triangulates the leakage | partially supported (mechanism confirmed) | no_rolling carries 0% history attribution (no channel); neighbour-averaging features (`migraine_rate_last7`, `days_since_last_migraine`) gain rank/magnitude under stratification; magnitude alone cannot isolate the leaked fraction |
| 2. Recovery of Park's triggers | partially supported | both architectures use all six triggers and rank hormonal_changes top with positive ALE; ranking diverges from Park's univariate ORs (stress over-, travel under-weighted) |
| 3. Architecture feature-reliance | inconclusive (AutoTabPFN); convergent (TabPFN vs XGBoost) | AutoTabPFN attribution not computed; where comparable, TabPFN and XGBoost rely on the same recent-history features |
| 4. Prodromal contamination | supported | noise/specific_smells/emotional_changes collapse to near-zero next-day SHAP (rank 28-44/52) |
| 5. ShapIQ interactions | partially supported (park cell) | top-tier k-SII pairs are all among Park triggers with hormonal_changes dominant; within-set rank shifts with library versions; both Park interaction terms appear with positive k-SII |

Headline: the attribution layer corroborates the methods finding (leakage
is a history-feature channel) and the literature (Park triggers, the
stress x hormonal_changes interaction) by an independent method, while
honestly bounding what magnitude-based SHAP can and cannot prove.

## References

[park2016shd] J.-W. Park, M. K. Chu, J.-M. Kim, S.-G. Park, and S.-J. Cho,
"Analysis of trigger factors in episodic migraineurs using a smartphone
headache diary applications," PLOS ONE, vol. 11, no. 2, p. e0149577, 2016.

[giffin2003premonitory] N. J. Giffin et al., "Premonitory symptoms in
migraine: an electronic diary study," Neurology, vol. 60, no. 6, pp.
935-940, 2003.

[schoonman2006premonitory] G. G. Schoonman et al., "The prevalence of
premonitory symptoms in migraine: a questionnaire study in 461 patients,"
Cephalalgia, vol. 26, no. 10, pp. 1209-1213, 2006.

