# Post-sweep insights: leaf-selection criteria

> Position in the paper: **Supplementary (6.1)**. Reads after `addition3_results.md`; precedes `paper_rigor_checklist.md`. Composite_sorted selection rule, its parameters, and the within-family fragility sensitivity sweep results.


This document records which leaves get model-interpretability attention (feature attribution, embedding visualisation, partial-dependence) after the Addition 1 sweep finishes, and why.

The starting observation is that running interpretability on all 154 leaves of Addition 1 (plus the parallel set under Addition 0) produces ~300 figures, too many to read, too expensive to compute. The selection below trades breadth for depth: the cells that drive the paper's discussion get figures; the rest are summarised numerically only.

## What "insights" means here

Three artefact kinds, all available via `tabpfn-extensions[interpretability]` on every `TabPFNClassifier` (and via the inner classifier for `FinetunedTabPFNClassifier`):

1. **Feature attribution**: SHAP-style values per feature per row; aggregated to mean-absolute-SHAP per feature for the "ranked feature importance" bar chart.
2. **Embedding projection**: the in-context-learning attention embedding for each row, projected to 2D (UMAP or t-SNE) and coloured by predicted probability. Tells whether the model has learned a meaningful neighbourhood structure for migraine-day clusters.
3. **Partial-dependence**: per-feature marginal effect on `predict_proba`, useful for the Park-feature leaves where the comparison against Park et al.'s same-day OR estimates [1, Tab. 4, p. 8] is direct.

`AutoTabPFNClassifier` exposes feature importance via AutoGluon's `feature_importance` API instead of SHAP; a different code path but it lands in the same per-leaf figure folder.

## What "runner-up" means here

For each `(target, feature_set, split_type)` group, the sweep produces one
results row per version × ratio combination. After the sweep finishes, the
selection logic in `experiment/2/select.py` ranks those rows by a
multi-metric composite and applies two rules.

A **headline** result is the top-composite row per group (the one the
paper's main results table cites). A **runner-up** is the best-composite row
of a *different* architecture family whose 95% AUROC CI overlaps the
headline's (e.g., headline is TabPFN, runner-up is the XGBoost stack).

## Selection rules

Selection runs over 21 groups: the seven scientifically defined
`(target, feature_set)` cells times three split types (chronological,
stratified, patient).

**Rule A (headline)**: the top leaf per group under `composite_sorted`
(`experiment/2/select.py`), a multi-metric ranking that prevents
noise-level AUROC differences from deciding the winner:

1. Degenerate calibration sorts last. A leaf whose calibration slope is
   missing, non-positive (inverted), or above 5.0 (wildly mis-scaled) ranks
   after all non-degenerate leaves; SHAP on an inverted probability surface
   would explain a meaningless quantity.
2. AUROC is the primary discriminator, bucketed at a noise-level tolerance
   of 0.02. Leaves differing by less than 0.02 in AUROC are treated as tied.
3. Within an AUROC tier, AUPRC is the secondary discriminator, also bucketed
   at 0.02.
4. Among discrimination-tied leaves, the one with calibration slope closest
   to 1.0 wins (most reliable probability estimates).
5. Exact mean AUROC is the final tiebreak.

Calibration cannot override a real discrimination gap, and a sub-tolerance
AUROC edge cannot override calibration.

**Rule B (runner-up)**: walking down the composite-ranked list, the first
leaf whose architecture family (xgboost / tabpfn / autotabpfn) differs from
the headline's and whose 95% AUROC CI overlaps the headline's. The runner-up
answers "what almost won, and did it win for the same reasons?". Absent a
qualifying leaf, the group contributes only the headline.

Both the headline and runner-up receive the full figure pack.

## Why insights go to runner-ups specifically

A headline result alone tells the reader *what won*. A headline plus a tightly-matched runner-up tells them *what almost won and why it didn't*. The second story is the one that constrains the methodology recommendation: if the headline beats the runner-up only at the meta-learner stage but the runner-up's SHAP bar chart is nearly identical, then the win is not feature-driven and the paper should not over-claim a feature-interpretability narrative. If the SHAP charts diverge, the headline's win is informative beyond its scalar AUROC and the paper should say so.

Without runner-up figures the reader has no way to perform this check.

## Cell coverage

Selection covers 21 groups: 7 cells × 3 split types (chronological,
stratified, patient). Each group contributes up to 2 leaves (headline +
runner-up when a qualifying cross-family candidate exists).

| target   | feature_set         | chrono | stratified | patient |
|----------|---------------------|--------|------------|---------|
| headache | full_features       | yes    | yes        | yes     |
| headache | no_rolling_features | yes    | yes        | yes     |
| headache | spano_features      | yes    | yes        | yes     |
| migraine | full_features       | yes    | yes        | yes     |
| migraine | no_rolling_features | yes    | yes        | yes     |
| migraine | spano_features      | yes    | yes        | yes     |
| migraine | park_features       | yes    | yes        | yes     |

7 cells × 3 split types × up to 2 leaves = up to 42 figure packs. At ~3
figures per pack (SHAP bar / beeswarm / ALE), that is at most ~126 figures;
in practice fewer, as not every group yields a qualifying runner-up.

The `headache/park_features` cell is intentionally absent: Park et al.'s
Table 4 stepwise regression discriminates migraine-vs-non-migraine headache,
not any-headache-vs-no-headache, so the feature set is not scientifically
defined for the headache target. See `docs/park_features.md` for the
migraine-only justification.

## Sensitivity of the composite_sorted rule

The composite rule has four free constants: `AUROC_TOL` (default 0.02),
`AUPRC_TOL` (default 0.02), `CALIB_MIN` (default 0.0), and `CALIB_MAX`
(default 5.0). A robust rule is one whose headline leaf does not change
under reasonable perturbation. The sweep at
`experiment/2/sensitivity_composite.py` tries six perturbations
(`AUROC_TOL` and `AUPRC_TOL` at ±0.01 around the default; `CALIB_MIN`
loosened to -0.5; `CALIB_MAX` tightened to 3.0 and loosened to 7.0) and
records the resulting full_features/chrono headline leaf per target.

The verdict on the canonical full_features/chrono headline cell:

- **Cross-architecture choice is stable.** Headache headline stays
  in the TabPFN family across all six perturbations; migraine headline
  stays in the `stacked_2xgb_meta_lr` family across all six.
- **Within-family choice is fragile.** On headache, the TabPFN variant
  chosen by the composite flips between *v2.6* (default tier) and
  *v2.5-finetuned* (when `AUROC_TOL` tightens to 0.01); both leaves
  sit inside the AUROC-tier band the composite considers a noise-level
  tie, so the calibration-distance tiebreak picks the winner. On
  migraine, the HP variant flips between *HP020* (default) and *HP050*
  (at `AUROC_TOL` = 0.01 or 0.03). All four calibration-guard
  perturbations leave the within-family choice on the default.

This is the empirical basis for the paper's family-level framing of the
headline TabPFN claim: across the 0.02 AUROC noise-level tier the
headache TabPFN variants *v2.6*, *v3-default*, and *v3-binary* sit
together at AUROC 0.652 / 0.653 / 0.653, and the specific within-family
choice is a calibration-distance tiebreak rather than a
discrimination-driven decision. The defensible cross-architecture claim
is at the family level (TabPFN beats XGBoost on the headache headline
cell); the within-family variant should be reported as a calibration
tiebreak with the AUROC values disclosed.

## Output destinations

Each figure pack lands under the headline / runner-up leaf's own folder:

```
experiment/1/<target>/<feature_set>/<arch>/<version>/<ratio>/<split>/
    insights/
        shap_bar_<ts>.png
        shap_summary_<ts>.png        # per-row beeswarm; omitted for n>1500 rows to stay readable
        embedding_<ts>.png           # UMAP of attention embedding
        pdp_<feature>_<ts>.png       # one per top-5 SHAP feature
        insights_<ts>.txt            # textual summary: top-10 features, mean |SHAP|, top-3 PDP slopes
```

The `insights/` subfolder is created on demand; the aggregator does not parse it as evaluation metrics.

## Compute budget

Per leaf:
- SHAP via `tabpfn_extensions.interpretability.shap`: ~2-4 minutes on GPU for n=4000 rows × 50 features. Scales with `n_samples × n_features × n_estimators`.
- UMAP projection of attention embeddings: ~30 seconds.
- PDP for top-5 features: ~1 minute total.

Total per leaf ~4-6 minutes. For 14 leaves: ~60-90 minutes added compute after the sweep. Run on the same GPU; non-overlapping with sweep so no contention.

## Why this isn't done inside `evaluate.py`

`evaluate.py` is a deterministic metrics-only contract that the aggregator reads; injecting interpretability into it would (i) couple the runtime of every leaf to the SHAP cost, (ii) require the aggregator to know how to skip the insight figures, and (iii) commit us to producing figures for cells the paper does not discuss. A separate post-sweep `experiment/2/run_insights.py` driver loads the headline/runner-up leaves' `model.joblib`, regenerates the same train/val/test splits, and writes the figure packs, all without touching `evaluate.py`.

## Citation

[1] J.-W. Park, M. K. Chu, J.-M. Kim, S.-G. Park, and S.-J. Cho, "Analysis of trigger factors in episodic migraineurs using a smartphone headache diary applications," *PLOS ONE*, vol. 11, no. 2, p. e0149577, Feb. 2016. doi: 10.1371/journal.pone.0149577. BibTeX key: `park2016shd`.
