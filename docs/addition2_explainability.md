# Addition 2: model explainability (SHAP, ALE, ShapIQ)

> Position in the paper: **Methods (4.1)**. Reads after `tabPfn.MD`; precedes `addition3_temporal.md`. The SHAP / ALE / ShapIQ pipeline this doc specifies produces the per-leaf insight artefacts whose findings are summarised in addition2_results.md.

The explainability layer trains no models. It computes feature-attribution
and feature-effect artefacts on a small set of selected Addition-0 and
Addition-1 leaves and writes a per-leaf figure pack into the leaf plus a
cross-leaf comparison under `experiment/2/`. Leaf-selection criteria live
in `insights_leaf_selection.md`.

## 1. Architecture: a shared _explain module + an optional evaluate flag

The attribution runs as an optional, off-by-default block at the end of
the evaluate templates, backed by a shared `experiment/_explain/` module
that holds the heavy logic as testable sub-300-line files. This reuses
the model and splits `evaluate.py` already loaded:

```python
# tail of evaluate.py, gated so the main sweep never pays the cost
if EMIT_INSIGHTS:                         # set only for selected leaves
    from _explain import emit_insights
    emit_insights(
        predict_fn=_insight_predict_fn,   # calibrated_proba(bundle, .) [add 0]
                                          # or model.predict_proba(.)[:,1] [add 1]
        X_background=X_val,               # the val/cal set already in scope;
                                          # 2-way leaves pass X_cal instead
        X_explain=X_test, y_explain=y_test,
        leaf_dir=EXPERIMENT_DIR, arch_family=ARCH_FAMILY,
        raw_model=model_or_bundle,        # for the TabPFN ShapIQ/embedding paths
    )
```

The signature deliberately takes a `predict_fn` closure and a
background frame rather than an `X_train` argument: the evaluate
templates never load the training set (3-way leaves hold `X_val` +
`X_test`, 2-way leaves hold `X_cal` + `X_test`), so the SHAP background
is the val/cal set already in scope. Each template builds the closure
appropriate to its architecture (`calibrated_proba(bundle, .)` for
Addition 0, `model.predict_proba(.)[:, 1]` for Addition 1), and
`ARCH_FAMILY` is derived from the existing `{addition}` / `{arch_module}`
substitutions, so `emit_insights` never has to introspect the raw model.

The post-sweep re-run reuses the existing `experiment/_eval_only_runner.py`
(extended with a leaf-subset filter and an extra-env injection) rather
than a new runner: that script already re-runs evaluate-only against the
existing `model.joblib` with the ROCm retry/cooldown logic the TabPFN
SHAP path needs. A thin `experiment/2/run_insights.py` computes the 14
selected leaves and invokes the extended runner with `EMIT_INSIGHTS=1`.
This design:

1. **Reuses evaluate's loaded model and splits** - zero re-load, zero
   re-split duplication.
2. **Writes `insights/` into the source leaf**, the chosen output
   location (Section 5).
3. **Keeps the main sweep fast**: `EMIT_INSIGHTS` defaults to false, so
   a normal `run_all_avaliable_leaves.py` pass produces no SHAP cost.
4. **Fails safe**: the metrics contract is written on the normal clean
   run; a SHAP failure on a later insight-only run cannot corrupt the
   metrics the aggregator already parsed.

`experiment/2/` itself holds only the post-sweep runner and the
cross-leaf comparison outputs (the headline-vs-runner-up SHAP diff and
the Park-OR check) - the artefacts the paper's discussion cites that do
not belong to any single leaf.

## 1b. Claims tested

Five claims test the Addition-2 design; their verdicts are reported in
`addition2_results.md` (Claims 1-5):

1. SHAP triangulates the feature-channel leakage (corroborates the
   Findings §1 mechanism by an independent method).
2. Recovery of Park's established triggers on the migraine/park cells
   (EPV 47.8; six shared triggers — set + direction expected, rank not
   statistically distinguishable at n=6).
3. Architecture feature-reliance: does AutoTabPFN win on different
   features than the XGBoost stack, or by ensembling over the same ones?
4. Prodromal contamination: do premonitory features (noise,
   specific_smells, emotional_changes) carry high next-day SHAP?
5. Pairwise feature interactions (ShapIQ) on the EPV-appropriate cells;
   the EPV-5.5 migraine full_features cell is excluded.

## 2. Research questions

1. **Which features drive each headline model, and do the foundation
   model (TabPFN) and the tree ensemble (XGBoost) rely on the same
   ones?** If a TabPFN headline and an XGBoost runner-up produce
   near-identical SHAP rankings, the win is not feature-driven and the
   paper must not over-claim a feature-interpretability narrative. If
   the rankings diverge, the win is informative beyond its scalar AUROC.
2. **Do the data-driven attributions agree with Park et al.'s
   same-day odds ratios** [park2016shd, Tab. 4]? The park_features
   migraine leaves let us compare each feature's mean absolute SHAP
   against Park's published OR. With six shared triggers the test has
   little power; the realistic claim is that the trigger set and the
   direction of attribution match Park's, while a strict
   rank-correlation test on within-set ordering is not statistically
   distinguishable from chance.
3. **Does the prodromal-contamination risk show up empirically?**
   Noise, specific smells, and emotional changes are documented
   premonitory symptoms [giffin2003premonitory, p. 935; schoonman2006premonitory, p. 1209].
   At a next-day lag they may be predicting today's attack rather than
   tomorrow's. If their attribution collapses toward zero under the
   forecasting shift, that is the evidence the paper needs to state the
   contamination risk is not driving results.

## 3. Method selection (with rationale)

### 3.1 Feature attribution: SHAP on the calibrated probability

SHAP [lundberg2017shap, p. 2] assigns each feature a Shapley value, the
game-theoretic average marginal contribution to a single prediction. A
critical correctness point drives the estimator choice: the quantity
the paper compares across architectures is the final calibrated
probability, and neither Addition-0 architecture is a single tree.
`stacked_2xgb_meta_lr` is a `StackingClassifier` (two XGBoost bases ->
L1-LR meta) wrapped by a Platt calibrator; `blended_xgb_lr_spano2026`
is an XGB + LR-pipeline convex blend with isotonic/Platt calibrators.
TreeSHAP [lundberg2020treeshap, p. 3] explains one tree ensemble's margin and
cannot represent the meta-learner, the blend, or the calibrators, so it
is the wrong tool for the calibrated output. We therefore use:

- **XGBoost leaves (Addition 0)**: model-agnostic **KernelSHAP**
  (the Shapley-value estimator that learns a local linear surrogate
  from coalitional resamples and is therefore agnostic to the
  underlying model class) over the leaf's `calibrated_proba(bundle, .)`
  function (a callable, with a background sample from the val/cal
  set). This explains exactly the probability the benchmark scores,
  and it makes the cross-architecture comparison fair because every
  leaf is then explained by the same estimator class applied to the
  same quantity. **TreeSHAP** (the polynomial-time Shapley estimator
  exact for tree ensembles) on the bare XGBoost base learners is
  available as a secondary, explicitly labelled "uncalibrated
  base-model" view only, never compared 1:1 with the TabPFN SHAP.
- **TabPFN leaves (Addition 1)**: the TabPFN-native explainer in
  `tabpfn_extensions.interpretability.shap` -
  `get_tabpfn_explainer()` builds a SHAP explainer optimised for the
  in-context forward pass, and `parallel_permutation_shap()` computes
  the values efficiently. This is the model-matched path (it knows how
  to query TabPFN's forward pass) rather than a generic KernelSHAP
  wrapper. It is sampling-based, so the seed is fixed and the
  background-sample count is reported for reproducibility. The module is already
  present in the environment (pulled by the pinned tabpfn-extensions
  commit); the `interpretability` extra only needs confirming in
  requirements.
- **AutoTabPFN leaves**: AutoGluon's native `feature_importance`
  (permutation-based) rather than SHAP, because the post-hoc ensemble
  has no single differentiable forward pass. A different code path,
  same per-leaf figure folder, clearly labelled as permutation
  importance so it is not silently compared to Shapley values.

Aggregated to mean absolute SHAP per feature for the ranked-importance
bar chart; retained per-row for the beeswarm summary (omitted when
n > 1500 rows to stay readable).

**Sample-size choices.** Both backends subsample the test frame
before SHAP to keep wall-time tractable on a single workstation, and
both summarise the background set with k-means so the explainer's
coalition refits are bounded:

- *Explain-set cap.* KernelSHAP on XGBoost is run on the first
  400 rows of the test set when the test set is longer; the TabPFN
  native explainer is run on 40 rows because its forward pass is much
  more expensive (constants `_MAX_EXPLAIN = 400`, `_MAX_EXPLAIN_TABPFN
  = 40` in `experiment/_explain/shap_runners.py`). Below the cap the
  full test set is used. Subsampling is a seeded random draw without
  replacement (`np.random.default_rng(seed).choice(len(X), size=n)` at
  `shap_runners.py:62-67`) whose indices are then returned in their
  original chronological order, so the choice is deterministic across
  reruns at fixed seed but does not correspond specifically to the
  earliest patient-days.
- *Background summary.* KernelSHAP background is k-means-summarised
  to 50 clusters before being passed to the explainer (constant
  `_BACKGROUND_K = 50`). The fallback when the val/cal set has fewer
  than 50 distinct rows is to pass the unique rows directly. The
  TabPFN-native path uses the same val/cal background frame without a
  k-means pass, because the explainer conditions on the full
  in-context set internally.

These constants are recorded so the cross-architecture comparison
remains faithful to the same explanation budget; changing them would
shift the SHAP variance, not the rank ordering.

### 3.2 Feature effects: ALE, not PDP

Accumulated Local Effects (ALE) [apley2020ale, p. 3] is the primary
effect plot. The feature sets here are heavily correlated by construction
(rolling/lag/streak features are deterministic functions of the same-day
flags). PDP averages model predictions over the marginal distribution,
fabricating physically impossible feature combinations when features are
correlated and biasing the estimated effect. ALE averages local prediction
differences over the conditional distribution within small feature bins,
so it stays unbiased under correlation. We keep a PDP overlay only where
ALE and PDP agree (a "no correlation artefact here" signal); where they
diverge ALE is reported and the divergence is noted.

### 3.3 LIME considered and excluded

LIME [ribeiro2016lime, p. 3] is not used as a local cross-check, on
best-practice grounds for this paper:

- **It estimates a different quantity.** SHAP returns Shapley
  attributions (additive, axiomatically consistent); LIME returns
  local linear-surrogate coefficients. They are not the same estimand,
  so LIME cannot corroborate SHAP the way a second measurement
  corroborates the first; "do they agree" is ill-posed.
- **It is unstable.** The clinical-XAI literature documents LIME's
  sensitivity to the perturbation kernel and neighbourhood size and
  SHAP-LIME disagreement on identical cases. Reporting an unstable
  method forces a "which do we believe" paragraph that weakens the
  narrative.
- **LIME plus TabPFN is especially fragile**: LIME perturbs inputs and
  re-queries the model, but perturbed rows fall off TabPFN's in-context
  training manifold, so the local surrogate fits noise.
- **SHAP + ALE already answers every interpretability question the
  paper asks** (which features drive the model; in which direction,
  robust to correlation; do they match Park's ORs). LIME adds a third
  method without a new question, against the TRIPOD+AI-era preference
  for one principled attribution method reported well.


### 3.4 Embedding projection (TabPFN only)

For TabPFN leaves the in-context-learning attention embedding is projected
of each row to 2D and colour by predicted probability. This shows
whether the model has learned a neighbourhood structure that separates
migraine-day clusters. It is descriptive, not inferential, and is
omitted for the XGBoost leaves (no comparable embedding). The embedding
is extracted with `tabpfn_extensions.embedding.TabPFNEmbedding`, the
package's own attention-embedding accessor, so we do not reach into
private model internals.

We use **UMAP** [mcinnes2018umap, p. 1] for the projection rather than t-SNE.
The figure's question is cluster separation, and UMAP preserves
global and inter-cluster structure better than t-SNE, whose
inter-cluster distances are not interpretable. UMAP with a fixed
`random_state` is also far more layout-stable than t-SNE across
re-runs, which a regenerable paper figure requires; `n_neighbors`
and `min_dist` are reported so the layout is reproducible. The
dependency cost is incremental because numba (UMAP's heavy transitive
requirement) is already in the environment via `kditransform`. t-SNE
(available in scikit-learn, no new dependency) is the documented
fallback if a `umap-learn` pin conflicts with the AutoGluon/torch
stack at install time.

Either projection distorts distance, so the figure caption carries an
explicit caveat that it is a qualitative aid and that inter-cluster
distances in any 2D embedding should not be over-read
[wattenberg2016tsne, §"Distances between clusters might not mean anything"; chari2023specious, p. 3]. That guard is mandatory for
either method and is the reason t-SNE's local-faithfulness edge does
not change the choice for this descriptive use.

### 3.5 Feature interactions: ShapIQ (TabPFN leaves)

The TabPFN interpretability extension also ships
`tabpfn_extensions.interpretability.shapiq.get_tabpfn_explainer()`,
which returns a ShapIQ explainer [muschalik2024shapiq, p. 4] for the
in-context forward pass. ShapIQ computes Shapley *interaction* indices,
the principled extension of Shapley values to pairs (and higher orders)
of features. A standard SHAP bar chart shows main-effect attribution
only; it cannot tell whether two triggers reinforce each other. The
clinical literature suggests migraine triggers combine non-additively
(for example weather change on a high-stress day), so the pairwise
interaction question is substantively interesting, not decorative.

ShapIQ pairwise (order-2) interactions are run on the TabPFN headline
leaves only, where the foundation model's flexible function class makes
interactions plausible and the extension is native. The intent is to
report the top interaction pairs as a small heatmap and check whether
the strongest pairs match documented trigger combinations.

**Runtime status (2026-05-27).** The TabPFN-native ShapIQ explainer
requires a NumPy 2.x compatibility patch on the project's TabPFN
leaves; without it, calls fail with `TypeError: only 0-dimensional
arrays can be converted to Python scalars`. Root cause: upstream `shapiq.imputer.tabpfn_imputer.
TabPFNImputer.value_function` calls `float(self.predict(...))`; the
wrapped predict returns a 1-element 1-D ndarray, which NumPy 1.x
silently unwrapped but NumPy 2.x rejects. The fix is a one-line
substitution to `float(np.asarray(...).flat[0])`, applied as an
idempotent import-time monkey-patch in
`experiment/_explain/shapiq_runner.py`
(`_patch_shapiq_tabpfn_imputer_for_numpy_2x`). The patch survives
until upstream `shapiq` ships a NumPy 2.x compatible release.

The 2026-05-27 canonical rerun on the migraine/park
`tabpfn/version_3-default/70_15_15/chrono` leaf reproduces the k-SII
computation in 255 s (background n=439, explain n=8, exhaustive
budget 2^6 = 64, seed=42). The output is interpretation-reportable:
the top-tier pairs are all among Park's six stepwise triggers, with
`hormonal_changes_today` the dominant partner. The within-set rank
shifts relative to the earlier May-22 run (which had `stress x
hormonal_changes` at rank 1; the current run has `alcohol x
hormonal_changes` at rank 1); the cause is library-version drift
across TabPFN / `shapiq` / NumPy between sessions. Claim 5 in
`addition2_results.md` is therefore reported at the **set + direction**
level, not the rank level, mirroring the discipline of Claim 2's
Park-OR recovery.

The patched runner additionally produces ShapIQ artefacts on the
headache full_features chronological 70/30 cell across all three
within-tier TabPFN variants (*v2.6*, *v3-default*, *v3-binary*). The
three checkpoints converge on the same top-5 SHAP feature set and the
same top-tier `vigorous_exercise_min x consecutive_stress_days`
interaction at the ShapIQ layer; the cross-checkpoint robustness
section in `addition2_results.md` reports the comparison and reads
the agreement at the set + dominant-pair level. The 96-coalition
sampling budget on this 52-feature cell makes the absolute k-SII
magnitudes scale-sensitive to library version and sampling variance,
so absolute interaction magnitudes the paper loads remain confined to
the exhaustive-budget migraine/park cell in Claim 5.

## 4. Leaf selection (scope control)

Running all artefacts on every leaf produces hundreds of figures, too many
to read and too expensive to compute. The headline-plus-runner-up rules are applied
rules from `insights_leaf_selection.md`, implemented in
`experiment/2/select.py`:

- **Rule A (headline)**: the top leaf per `(target, feature_set, split_type)`
  group under `composite_sorted`. Discrimination is primary but bucketed at a
  noise-level tolerance (AUROC and AUPRC bucketed at 0.02), so a sub-0.02
  edge does not decide the winner. Leaves with degenerate calibration (slope
  non-positive or above 5) sort last; among discrimination-tied leaves the
  one with calibration slope closest to 1.0 wins; exact AUROC is the final
  tiebreak.
- **Rule B (runner-up)**: walking down the composite-ranked list, the first
  leaf whose architecture family differs from the headline's and whose 95%
  AUROC CI overlaps the headline's. The runner-up answers "what almost won,
  and did it win for the same reasons?".

Coverage is the seven scientifically defined `(target, feature_set)` cells
(headache/park_features is excluded because Park's stepwise regression is
migraine-specific) times three split types (chronological, stratified,
patient) = 21 groups, each contributing up to 2 leaves = up to 42 figure
packs. The selection is computed automatically from the aggregator's per-leaf
parsed metrics, not hand-picked, so it updates when the sweep does.

## 5. Directory and output layout

Per-leaf figures land inside the source leaf (the chosen output
location). Cross-leaf comparison artefacts live under `experiment/2/`.

```
experiment/_explain/                # shared module, called by evaluate.py when flagged
    __init__.py                     # emit_insights(predict_fn, X_background, X_explain, y_explain, leaf_dir, arch_family, raw_model)
    shap_runners.py                 # per-architecture SHAP / importance dispatch
    shapiq_runner.py                # ShapIQ pairwise interactions (TabPFN headlines)
    ale.py                          # ALE computation + plot (correlation-robust)
    embedding.py                    # UMAP of TabPFN attention embeddings
    summarise.py                    # per-leaf explain_<ts>.txt writer

experiment/0|1/<...>/<ratio>/<split>/[NonHP|...]/
    insights/                       # written in-place when EMIT_INSIGHTS is set
        shap_bar_<ts>.png
        shap_beeswarm_<ts>.png      # omitted for n>1500
        shapiq_interactions_<ts>.png  # TabPFN headline leaves only
        ale_<feature>_<ts>.png      # top-5 SHAP features
        embedding_<ts>.png          # TabPFN leaves only
        explain_<ts>.txt            # top-10 features, mean |SHAP|, ALE slopes, top interaction pairs

experiment/2/
    run_insights.py                 # selects 14 leaves, calls the extended _eval_only_runner with EMIT_INSIGHTS=1
    select.py                       # headline/runner-up selection (reuses _eval/_parsing + a CI-overlap predicate)
    compare.py                      # cross-leaf SHAP-ranking diff + Park-OR check
    comparison_shap_<ts>.html       # headline-vs-runner-up ranking diffs across the 7 cells
    park_or_check_<ts>.html         # SHAP rank vs Park 2016 Table-4 OR, migraine/park cells
```

## 6. Build order

1. `experiment/_explain/shap_runners.py`: one dispatch per architecture
   family - KernelSHAP over `calibrated_proba` for the XGBoost bundle
   (NOT TreeSHAP; the bundle is a calibrated stacking/blending pipeline,
   not a single tree), the TabPFN-native explainer from
   `tabpfn_extensions.interpretability.shap` for TabPFN, and a permutation
   path for AutoTabPFN (via the handle's public `predict_proba`, not
   AutoGluon internals) - returning a uniform `(feature_names,
   shap_matrix)` so plotting is architecture-agnostic. Each dispatch is
   handed a `predict_fn` closure and a background frame by the template,
   so it never introspects the raw model object.
2. `experiment/_explain/ale.py`: ALE with conditional-bin accumulation;
   unit-test against a known correlated synthetic so the
   correlation-robustness is demonstrable.
3. `experiment/_explain/shapiq_runner.py`: pairwise ShapIQ interactions
   via `tabpfn_extensions.interpretability.shapiq` for the TabPFN
   headline leaves; emits the interaction heatmap and the top-pair list.
4. `experiment/_explain/embedding.py` (via
   `tabpfn_extensions.embedding.TabPFNEmbedding`), `summarise.py`,
   `__init__.py::emit_insights`: the leaf-local artefacts and the single
   entry point evaluate.py calls.
5. Evaluate-template change: add the gated `if EMIT_INSIGHTS:` tail to
   `experiment/0/_templates/evaluate_*.py.tpl` and
   `experiment/1/_templates/evaluate_*.py.tpl`, plumb an
   `EMIT_INSIGHTS` env-var or `--insights` arg, re-scaffold. The flag
   defaults off so a normal sweep is unchanged.
6. `experiment/2/select.py`: build the (headline, runner-up) leaf list
   from the pure parsing layer - `find_results_dirs`, `parse_path`,
   `find_latest_files`, `parse_file`, `postprocess_cv` from
   `_eval/_parsing.py` plus `parse_mean_ci` from `_eval/_metric_palette.py`
   (NOT the aggregator's script-local `_collect_entries`, which has
   module-level side effects). Rule B needs a CI-OVERLAP predicate;
   `_comparison_html._strict_ci_winner` implements the inverse
   (separation), so write a small overlap test rather than reusing it.
7. Extend `experiment/_eval_only_runner.py` with a leaf-subset filter
   and an extra-env dict; `experiment/2/run_insights.py` computes the 14
   leaves and calls it with `EMIT_INSIGHTS=1`, inheriting the ROCm retry
   logic. Collect the written `insights/` folders.
8. `experiment/2/compare.py`: the cross-leaf SHAP-ranking diff (must
   include at least one Addition-0 XGBoost vs Addition-1 TabPFN pair, the
   benchmark's core question) and the Park-OR check.

Each `_explain/*` module and `experiment/2/*` script stays under the
300-line budget; the runner is a thin orchestrator.

## 7. Dependencies to add

- `shap` (TreeSHAP for the XGBoost leaves)
- `tabpfn-extensions[interpretability]` extra: provides
  `interpretability.shap` (TabPFN-native SHAP), `interpretability.shapiq`
  (Shapley interactions), `interpretability.pdp`, and the top-level
  `embedding.TabPFNEmbedding`. The modules are already present in the
  venv via the pinned post-hoc-ensembles commit; the requirements line
  should declare the `interpretability` extra explicitly so a fresh
  install pulls `shapiq` and `shap` rather than relying on the
  transitive presence.
- `shapiq` (pulled by the interpretability extra; pin it directly if the
  extra under-specifies the version).
- `umap-learn` (TabPFN embedding projection; t-SNE via scikit-learn is
  a fallback if a UMAP/numba pin conflicts with the AutoGluon stack).

LIME is intentionally not added (Section 3.3) and is removed from the
README additions table (Section 6, step 0). `statsmodels` and `seaborn`
are already pinned; `seaborn`'s Okabe-Ito theme in
`data/pipeline/analytics/style.py` is the palette these figures reuse
for visual consistency with the rest of the paper.

## 8. Compute budget

Per leaf: KernelSHAP over `calibrated_proba` for the XGBoost leaves is
~1-3 minutes for n≈4000 x ~50 features with a modest background sample
(it is model-agnostic, so slower than the TreeSHAP-seconds figure an
earlier draft assumed); KernelSHAP on TabPFN is ~2-4 minutes on GPU;
ALE top-5 ~1 minute; UMAP ~30 seconds. ~4-7 minutes per leaf, ~60-100
minutes for all 14 selected leaves, run after the sweep via the
selective insight re-run so it does not contend with training.

## References

[apley2020ale] D. W. Apley and J. Zhu, "Visualizing the effects of
predictor variables in black box supervised learning models," *J. R.
Stat. Soc. B*, vol. 82, no. 4, pp. 1059-1086, 2020.

[lundberg2017shap] S. M. Lundberg and S.-I. Lee, "A unified approach
to interpreting model predictions," in *Advances in Neural Information
Processing Systems 30*, 2017.

[lundberg2020treeshap] S. M. Lundberg *et al.*, "From local
explanations to global understanding with explainable AI for trees,"
*Nature Machine Intelligence*, vol. 2, pp. 56-67, 2020.

[ribeiro2016lime] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why
Should I Trust You?': Explaining the predictions of any classifier,"
in *Proc. 22nd ACM SIGKDD*, 2016, pp. 1135-1144.

[muschalik2024shapiq] M. Muschalik *et al.*, "shapiq: Shapley
interactions for machine learning," in *Advances in Neural Information
Processing Systems 37*, 2024 (authors/pages to be confirmed against
source).

[mcinnes2018umap] L. McInnes, J. Healy, and J. Melville, "UMAP:
Uniform Manifold Approximation and Projection for dimension
reduction," *arXiv:1802.03426*, 2018.

[wattenberg2016tsne] M. Wattenberg, F. Viegas, and I. Johnson, "How
to use t-SNE effectively," *Distill*, 2016. doi: 10.23915/distill.00002.

[chari2023specious] T. Chari and L. Pachter, "The specious art of
single-cell genomics," *PLOS Computational Biology*, vol. 19, no. 8,
p. e1011288, 2023 (cited for the general 2D-embedding
over-interpretation caveat, not the genomics content).

[park2016shd] J.-W. Park, M. K. Chu, J.-M. Kim, S.-G. Park, and
S.-J. Cho, "Analysis of trigger factors in episodic migraineurs using
a smartphone headache diary applications," *PLOS ONE*, vol. 11, no. 2,
p. e0149577, 2016.

[giffin2003premonitory] N. J. Giffin *et al.*, "Premonitory symptoms
in migraine: an electronic diary study," *Neurology*, vol. 60, no. 6,
pp. 935-940, 2003.

[schoonman2006premonitory] G. G. Schoonman *et al.*, "The prevalence
of premonitory symptoms in migraine: a questionnaire study in 461
patients," *Cephalalgia*, vol. 26, no. 10, pp. 1209-1213, 2006.
