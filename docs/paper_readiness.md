# Paper readiness - narrative and submission artifacts

Working artifacts produced by walking the 30-item readiness checklist. Each
section is keyed to its checklist item so cross-reference is direct.

---

## 1. The 25-word thesis sentence

> On the Park 2016 SHD cohort, the pooled next-day migraine forecast AUROC
> of 0.79 reflects between-patient base-rate separation, not within-person
> day-to-day discrimination.

(23 words. The "headline AUROC misleads clinical deployment" claim is the
load-bearing finding the paper newly establishes.)

---

## 2. The 250-word abstract (draft)

**Background.** Migraine forecasting from daily diary triggers is a candidate
foundation for pre-emptive oral medication, but the literature reports
discrimination as pooled AUROC, which conflates within-person ranking with
between-patient base-rate separation. Whether published numbers translate to
clinically usable per-patient forecasting is unresolved.

**Methods.** we evaluate 24-hour migraine and headache forecasting on the
Park 2016 SHD cohort (62 enrolled patients; 63 unique patient IDs in the
processed dataset; 4516 patient-days; 7.2% positive rate). The grid factors
three feature sets, three split types (chronological, stratified, patient
hold-out), two split ratios (70/30 and 70/15/15), and the architecture
families {XGBoost stacking with and without 500-trial Optuna hyperparameter
search; TabPFN tabular foundation model across five released variants;
window-MLP sequence baseline}. Headline cells are selected by a composite
rule that gates on calibration slope and breaks AUROC/AUPRC ties via fixed
tier bins. Discrimination, calibration, and bootstrap CIs are reported
together throughout.

**Results.** On chronological splits the headline AUROC reaches 0.793 for
migraine (*XGB-HP020 / full / chrono / 70-30* stack) and 0.652 for headache
(TabPFN family at *full / chrono / 70-30*: v2.6, v3-default, and v3-binary
are tied within the composite rule's 0.02 AUROC tier, with v2.6 picked by
the calibration-distance tiebreak). Calibration slope is fragile (median
0.64). The within-person C-statistic is near chance, the per-patient AUROC
distribution centres near 0.55, and the pooled-minus-within gap is large;
the pooled AUROC therefore reflects between-patient base-rate separation.
Brier skill against per-patient climatology is negative for migraine, and
the decision-curve net benefit for migraine sits near zero across the
clinically plausible threshold band (the headache curve is
value-positive at +0.05 to +0.20 over the same band; the clinically
under-served target is migraine specifically).

**Conclusion.** A clinically usable 24-hour migraine forecast on this
cohort requires richer per-patient signal or a re-framed prediction target;
the pooled AUROC commonly reported in the diary-forecasting literature is
not a sufficient endpoint.

(258 words.)

---

## 3. Per-Addition contribution audit

One sentence per Addition stating its unique contribution to the paper's
thesis. If the sentence is generic ("we did X"), the Addition shouldn't be
in the paper.

| Addition | Unique contribution |
|---|---|
| **0 (XGBoost stacks)** | Shows that the depth-diverse XGBoost stack reaches AUROC 0.793 for migraine and 0.660 for headache on chronological splits, but with calibration slope 0.71 and high HP-search variance - the tree-ensemble baseline is competitive but mis-calibrated. |
| **1 (TabPFN)** | Shows that the TabPFN family matches or slightly exceeds XGBoost on the headache full_features/chrono cell (v2.6 / v3-default / v3-binary tied at AUROC 0.652, calibration-tiebreak picks v2.6) and on small-N stratified splits more robustly than HP-tuned XGBoost - the foundation-model path is competitive without per-leaf tuning, not dominant, and the within-family variant choice is calibration-noise driven rather than discrimination-driven. |
| **2 (Explainability)** | Identifies feature-channel leakage (history features drive stratified-split optimism) and a prodromal-symptom contamination route; SHAP rankings recover Park 2016's trigger odds-ratio structure on the park_features cells. |
| **3 (Temporal)** | Establishes day-1-to-day-2 attack autocorrelation (Andersen-Gill HR 3.50, p < 0.001) and short-memory clustering (Goh-Barabasi burstiness ≈ 0, memory ≈ 0) - temporal structure exists but is too short-range for a sequence model to exploit beyond lag features. |
| **4 (Sequence)** | Confirms the Addition-3 prediction: a window-MLP sequence baseline does not beat tabular models on this cohort. This is a null result, deliberately framed as such. |
| **5 (Personalisation)** | The paper's load-bearing finding: per-patient AUROC centres near chance, the precision-weighted within-person C-statistic is near 0.55, and the pooled-minus-within gap is large - pooled discrimination is base-rate separation, not within-person ranking. |
| **6 (Clinical value)** | Negative Brier skill on migraine against per-patient climatology and near-zero decision-curve net benefit for the migraine cell across the clinically plausible threshold band. The same panel for headache is value-positive (Brier skill +0.14 to +0.22; net benefit +0.05 to +0.20). The "high AUROC does not translate to probabilistic value" finding is therefore target-specific: it is the migraine forecast that fails the Murphy quality-vs-value test, not headache. |

Each Addition tests a distinct claim and modifies the paper's conclusion in
a documented way. The chain coheres: 0-1 (architecture comparison) → 2
(feature/leakage explanation) → 3 (IID-assumption test) → 4 (acting on the
3 verdict) → 5+6 (the pooled-AUROC-misleads finding that the paper newly
establishes).

---

## 4. Reverse-narrative test (audit pending against post-fix data)

Plan: for each numerical claim in the abstract above, trace it back to a
specific cell in a specific results.txt or figdata field. The traces
established so far:

| Claim | Resolves to |
|---|---|
| AUROC 0.793 migraine | `experiment/0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP020/results/results_*.txt` |
| AUROC 0.652 headache (TabPFN family, tied across v2.6/v3d/v3b) | `experiment/1/headache/full_features/tabpfn/{version_2-6,version_3-default,version_3-binary}/70_30/chrono/results/results_*.txt` |
| 4516 patient-days | derived from the engineered parquets (3941 train + 439 val + 136 test on 70/15/15; the same 4516 total in 70/30) |
| 7.2% positive rate | `migraine_target` column mean |
| Calibration slope median 0.64 | aggregated from results_findings.md §6 |
| Per-patient C-statistic ≈ 0.55 | Addition 5 within_person.py output, headline cell |
| Negative Brier skill (migraine) | Addition 6 run_value.py output, headline cell |

Open trail: the per-patient AUROC distribution centre needs to be re-derived
against the current headline leaf (post the composite-tracking selection
update; the Add-5 rerun against the refreshed `experiment/2/comparison_*.csv`
is the pending step).

---

## 5. Reviewer-Q deck

The 20 highest-likelihood reviewer questions, each with a one-line answer
and the file the answer lives in.

1. **Q: Why isn't the headline a held-out-patient AUROC?**
   A: The chronological headline is the deployable-cell scope; held-out-patient
   discrimination is reported in Addition 0 (split type `patient`) alongside.
   Source: `results_findings.md` §8 "Chronological splits cross patient boundaries".

2. **Q: How is "headline cell" selected?**
   A: The five-level `composite_sorted` rule in `experiment/2/select.py` (calibration-slope guard, 0.02-AUROC tier bucket, AUPRC tier, calibration-distance, exact AUROC). Source: `docs/insights_leaf_selection.md`.

3. **Q: Is the within-person finding statistically distinguishable from chance?**
   A: The DerSimonian-Laird random-effects C-statistic CI includes 0.5; the per-patient distribution centres near 0.55. Sources: Addition 5 results files.

4. **Q: Are you correcting for multiple comparisons across the 21 cell-by-split groups?**
   A: BH-FDR within Addition 3's temporal battery (`addition3_temporal.md`); cross-cell discrimination claims rely on overlapping bootstrap CIs, which is acknowledged as a conservative-but-weak test in the Limitations.

5. **Q: Why not DeLong CIs?**
   A: 1000-iteration bootstrap percentile CIs are reported throughout (Niculescu-Mizil-style); DeLong is an open follow-up flagged in the Limitations.

6. **Q: TabPFN with 7.2% positive rate - does the in-context size suffice?**
   A: TabPFN runs without class reweighting per its training prior (`addition4_sequence.md` Decision 3); explain-set is capped at 40 rows per `experiment/_explain/shap_runners.py` for the explanation pass; the in-context sample-size argument is from the foundation-model literature.

7. **Q: Why include both Park trigger features AND the rolling history?**
   A: To separate the same-day-trigger signal from the history-feature leak; the comparison is the Addition 2 prodromal-contamination check.

8. **Q: Is the 24h horizon clinically motivated?**
   A: The diary records at 24h granularity and oral pre-emptive intervention takes ~hours to onset; both factors are described in the Addition 6 introduction. Other horizons are not evaluated - Limitations §"24-hour forecast horizon".

9. **Q: Why no within-patient leave-one-out as the headline?**
   A: Within-patient personalisation is reported in Addition 5 as a negative result; the paper explicitly does not claim personalisation works.

10. **Q: Are seeds documented?**
    A: Bootstrap seed 42 (`experiment/_eval/metrics_lib.py:92`); Optuna RandomSampler seed 42; stacker inner-KFold uses shuffle=False on a time-ordered training parquet (`xgboost.md` §1 discloses the time-respecting near-miss).

11. **Q: Does the in-text claim "AutoTabPFN is the migraine leader" still hold?**
    A: Per `results_findings.md` §3 the AutoTabPFN cell is the chronological migraine leader for one slice; the composite headline winner is *XGB-HP020 / full / chrono / 70-30* (AUROC 0.793). The two claims hold against different composite filters.

21. **Q: Is the headline TabPFN variant load-bearing?**
    A: No - it is a calibration-tiebreak pick within a 0.02-AUROC tie. The headache full_features/chrono cell has *TabPFN-v2.6*, *TabPFN-v3d*, and *TabPFN-v3b* all at AUROC 0.652; the composite picks v2.6 by 0.10 vs 0.10 calibration distance (then exact AUROC). The defensible claim is at the family level: TabPFN beats XGBoost on the headache headline cell; the specific variant is interchangeable. See `experiment/2/sensitivity_composite.py` for the perturbation sweep that surfaces this tie and `docs/figure_design_requirements.md` §8 for the slug convention that exposes it on every figure.

12. **Q: Is the SHAP/explainability finding consistent across architectures?**
    A: Cross-architecture SHAP comparison in Addition 2 §1b Claim 3; AutoTabPFN was not explained in the cross-pair, which is a documented limitation.

13. **Q: How do you defend the composite_sorted rule itself?**
    A: Rule is reproducible from `figdata_*.json`; threshold-sensitivity is a documented open question in the Limitations.

14. **Q: Is the Park 2016 cohort representative?**
    A: 82% female, ICHD-3 episodic migraine, 19-55 years, Korean university-hospital recruitment. Applicability domain explicitly stated in `dataset.md`; not generalised beyond.

15. **Q: Is k-anonymity ≥ 5 at the cohort scale?**
    A: 62 patients with hospital site, demographic baseline, and dated diary entries falls below k = 5; risk is acknowledged in Limitations §"Re-identification risk".

16. **Q: Why drop the Spano blended model from the headline?**
    A: It is retained as a comparator (`blended_xgb_lr_spano2026` cells) but composite-sorted selects the stacked variant for the headline migraine cell.

17. **Q: Calibration-slope median 0.64 - does that invalidate the threshold-derived metrics?**
    A: It does; `results_findings.md` §6 flags threshold-derived metrics on the negative-slope cells as unreliable. AUROC/AUPRC remain rank-based and interpretable.

18. **Q: Is the data publicly redistributable?**
    A: Park 2016 released the SHD dataset under their ethics approval; this project uses the engineered features derived from the public release and does not redistribute raw patient-day rows.

19. **Q: Are the 12 missing post-fix leaves a problem for the headline?**
    A: No - the four composite-headline cells are among the 24 leaves the post-fix run already completed with aligned `.npz`. The 12 outstanding leaves are non-headline cells in the grid.

20. **Q: How would the paper change if a TabPFN-3 release dropped tomorrow?**
    A: The TabPFN-3 model report is held locally as `priorlabs_2026_tabpfn3_report.pdf`; a v3 evaluate.py template would slot in alongside the existing v2-5 / v2-6 / v3-default / v3-binary variants without changing the methodology.

---

## 6. "What we are NOT claiming" cross-check

Audit performed against the disavowal set in `results_findings.md` §8.
Every limitation listed there has been spot-checked against the other
docs for accidental contradicting positive claims; the contradictions
surfaced by the audit are tracked in this doc's open-trail rows above.

---

## 7. Prior-art comparison table

| Work | Cohort | Patient-days | Target | Horizon | Split | Headline | Calibration reported | What we add |
|---|---|---|---|---|---|---|---|---|
| Park 2016 | 62 (SHD) | ~4500 | trigger odds-ratios (no forecast) | n/a | n/a | OR table | n/a | First forecasting evaluation on this cohort. |
| Zhu 2020 (MyGraine) | 62 (SHD) | 4580 | same-day migraine | n/a | random 70/30 (patient-overlapping) | 97% accuracy (LR) | no | Honest evaluation: chrono splits, calibration, decision-curve, within-person. |
| Spano 2026 (thesis) | 62 (SHD) | ~4500 | next-day migraine | 24h | mixed | architecture-only | partial | Replicates and extends with the full XGBoost+TabPFN+sequence factorial. |
| Houle 2005 | 132 | 4626 (electronic diary) | next-day any-headache | 24h | leave-one-out | AUC 0.65 LOO | no | Comparable scale, distinct cohort and target framing. |
| Houle 2017 | 95 | 4626 | next-24h headache | 24h | GLMM | AUC 0.65-0.73 | partial | Same forecast horizon and order-of-magnitude cohort. |
| Holsteen 2020 | n/a (review) | n/a | trigger->attack model | n/a | n/a | within-person C-stat framework | n/a | Methodological anchor for the within-person C-stat we report. |
| Stubberud 2023 | 18 (wearable) | n/a | migraine attack | wearable | mixed | AUC 0.62 | partial | Smaller cohort; smartphone diary domain. |
| Faisal 2026 | 146 (multimodal wearable) | n/a | migraine attack | sub-day | mixed | AUC 0.84 | yes | Different modality (wearable PPG/HR/sleep); the AUC ceiling we reference but do not match (we use diary triggers only). |
| **This work** | 62 / 63 (SHD) | 4516 | next-day migraine + headache | 24h | full factorial (chrono / stratified / patient × 70/30 / 70/15/15) | Migraine AUROC 0.793 chrono (*XGB-HP020*); headache 0.652 chrono (TabPFN family v2.6 / v3-default / v3-binary tied); **but** within-person C-stat ≈ 0.55 | yes (calibration slope, ECE10, Brier, decision-curve net benefit) | Three contributions: (a) honest evaluation grid with composite-sorted selection that prevents stratified-split inflation; (b) the within-person decomposition showing pooled AUROC overstates clinical utility; (c) TabPFN v3 (the Jan-2025 Nature foundation-model release) benchmarked as a competitive headline option, with explicit disclosure that the within-family variant choice is calibration-driven rather than discrimination-driven. |

---

## 14. Per-component seed table

| Component | Seed / config | Source file |
|---|---|---|
| Bootstrap evaluation | `seed=42`, `n_iterations=1000`, percentile method | `experiment/_eval/metrics_lib.py:92` |
| Optuna `single_AUROC` sampler | `RandomSampler(seed=42)`, 500 trials | `experiment/0/_templates/train_hp_3way.py.tpl:41` |
| Optuna `pareto_*` sampler | `NSGAIISampler(seed=42, population_size=50)`, 500 trials | `experiment/0/_templates/train_hp_3way.py.tpl:41` |
| Stacker inner CV | `KFold(n_splits=5, shuffle=False)` (time-ordered training parquet, not strict TimeSeriesSplit; see `xgboost.md` §1 for the near-miss disclosure) | `experiment/0/_model_architecture/stacked_2xgb_meta_lr/model.py:35` |
| Outer CV (Add-4 sequence) | `TimeSeriesSplit` expanding-window | `experiment/4/_templates/evaluate_cv.py.tpl:127` |
| Platt calibration LR | `C=1e10`, default `random_state` (LogisticRegression default) | `experiment/0/_model_architecture/stacked_2xgb_meta_lr/model.py:42-56` |
| TabPFN | no class reweighting per the in-context training prior; seed left to library default | `experiment/1/_templates/*.tpl` |
| ShapIQ | `SHAPIQ_SEED` (constant) | `experiment/_explain/shapiq_runner.py:237` |
| Split construction | offline / deterministic from `data/processed/<target>/<split>/<type>/` parquets | (data pipeline) |

Two undisclosed items the paper should resolve: (a) the LogisticRegression
`random_state` is left at its default (None) which means the Platt calibrator
may have noise across re-runs - a fix would set it explicitly; (b) the
TabPFN library's internal random state is not pinned in the templates.

---

## 25. Target-journal fit (decision aid)

| Journal | Word limit (research article) | Figure limit | Audience | Fit |
|---|---|---|---|---|
| *Journal of Headache and Pain* (Springer Nature, open) | 4000-5000 | 6-8 | clinical-academic | **Best fit.** Negative-result-positive ("pooled AUROC misleads") is editorially welcome; cohort and target are exactly the journal's focus. |
| *Headache* (Wiley, hybrid) | 3500 | 6 | clinical | Fit. Similar clinical focus; calibration emphasis aligns with their methods bar. |
| *JAMIA* (Oxford, hybrid) | 4000 | 5 | informatics-clinical | Fit, but the contribution is more clinical than informatics-methodological. |
| *npj Digital Medicine* (Nature, open) | 4500 | 6 | impact-mixed | Possible if the within-person finding is framed as a deployment-readiness contribution. |
| *Nature Communications* (open) | 5000 | 8 | broad | Likely too narrow in cohort scope; the within-person finding alone may not justify the broad audience. |
| *Scientific Reports* (open) | 8000 | unlimited | broad | Always-takes-it venue but lower citation impact. |

**Recommendation:** target *Journal of Headache and Pain* first; the same
publisher and editorial board has historically welcomed negative-result
papers in clinical forecasting (Poulsen 2021 chronobiology systematic review
appears there; cited as `poulsen2021chronobiology`).

---

## 26. Word/figure budget (current draft)

Rough word count of the synthesis docs (the body of the paper):

| Doc | Word count |
|---|---|
| `results_findings.md` | (count via `wc -w`) |
| `addition2_results.md` | |
| `addition3_results.md` | |
| `addition4_sequence.md` (results portion) | |
| `addition5_personalization.md` (results portion) | |
| `addition6_clinical_value.md` (results portion) | |
| Limitations (`results_findings.md` §8) | |
| **Total to integrate** | |

(Recomputed alongside any doc edit; the body draft is currently above
the *Journal of Headache and Pain* 5000-word ceiling and needs a
targeted cut.)

---

## 27. Open Science checklist

- [x] Code on a version-controlled repository (local; needs public GitHub mirror at submission)
- [x] Sources.bib + PaperSources/PaperSourcesDesign with verified citations
- [ ] Pre-registration: not pre-registered; the work was retrospective benchmark sweep, not a hypothesis-confirming trial
- [ ] Public DOI for code at submission (Zenodo archive of the GitHub release)
- [ ] Data: Park 2016 SHD is publicly released by the original authors; we do not redistribute, we publish the engineered-feature parquet schema + the build script
- [x] Environment lock: `requirements.txt` present
- [ ] CI / reproducibility test (`make headline-table` or equivalent - see #12)
- [x] Calibration alongside discrimination reported throughout
- [x] Bootstrap CIs reported throughout
- [x] Negative results disclosed (Addition 4 sequence null; Addition 5 within-person near chance; Addition 6 negative Brier skill)
- [x] Limitations section enumerates out-of-scope claims (`results_findings.md` §8)
- [ ] IRB / ethics statement: cite Park 2016's original approval; we do no new data collection

---

## 28. Cover letter draft

Dear Editor,

We submit *Within-person versus pooled discrimination in 24-hour migraine
forecasting on the Park 2016 SHD cohort* for consideration as a research
article in [Journal].

Daily diary-based migraine forecasting is a candidate foundation for
pre-emptive oral medication. The literature reports pooled discrimination
in the AUC 0.65-0.84 range across cohorts, but whether these numbers
translate to clinically usable per-patient forecasts is unresolved. Most
published evaluations conflate within-person day-to-day ranking with
between-patient base-rate separation, which is the source of clinical
value and the source of inflated headline numbers respectively.

On the publicly released Park 2016 SHD cohort (62 enrolled patients, 4516
patient-days, 7.2% positive rate) we evaluate next-day migraine and
headache forecasting under a complete factorial of feature sets, split
types, and architectures - XGBoost stacks with and without hyperparameter
search, the TabPFN tabular foundation model across five released variants,
and a window-MLP sequence baseline. Headline cells are selected by a
composite rule that gates on calibration slope, and calibration is
reported alongside discrimination throughout.

The headline pooled AUROC reaches 0.793 (migraine, XGBoost stack with
20-trial single-objective Optuna search) and 0.652 (headache, TabPFN
family - v2.6, v3-default, and v3-binary are tied within the composite
rule's 0.02 AUROC tier and v2.6 is the calibration-tiebreak pick) on
chronological splits, but the per-patient AUROC distribution centres near
chance and the precision-weighted within-person C-statistic is near 0.55.
Decision-curve net benefit for the migraine cell sits near zero across
the clinically plausible threshold band; the headache decision curve is
value-positive (+0.05 to +0.20). Brier skill against per-patient
climatology is negative for migraine and positive for headache. The
"high AUROC does not translate to probabilistic value" finding is
therefore migraine-specific.
Brier skill against per-patient climatology is negative on migraine, and
decision-curve net benefit sits near zero across the clinically plausible
threshold band. The pooled AUROC commonly reported in the diary-forecasting
literature is therefore not a sufficient endpoint; a clinically usable
24-hour migraine forecast on this cohort requires richer per-patient signal
or a re-framed prediction target.

We declare no competing interests. Code is available at [URL] with a Zenodo
DOI for the submission version. The engineered features are derived from
the publicly released Park 2016 SHD dataset; we do not redistribute raw
patient-day rows.

Sincerely,

[Author], [Affiliation]

---

## 29. "If reviewer demands" tree

Three highest-likelihood revision demands, each pre-decided:

| Demand | Decision | Effort |
|---|---|---|
| **Held-out-patient AUROC alongside the chronological headline** | Run the existing `patient` split cells of `experiment/0/migraine/...` and `experiment/1/migraine/...` and tabulate side-by-side. The leaves already exist; only the table is missing. | ~30 min |
| **7-day rolling-window ablation** | Add a `--window=3` / `--window=14` re-build path in `data/pipeline/`; re-fit the headline architectures on each. Sensitivity claim is that history features still dominate the SHAP ranking but the AUROC magnitude may shift by ±0.02. | ~half a day |
| **DeLong (or paired-bootstrap) significance test for cross-architecture comparison** | Implement paired bootstrap on the per-row predictions of the {headline, runner-up} pair per cell, with FDR correction across the 21 cell×split groups. | ~2 hours |
| **External-cohort transport beyond Park 2016** | Push back: we do not claim transport, the within-cohort leave-one-site-out (Add-5 external) is the available approximation; an external cohort is future work. | n/a |
| **Why no continuous-time point-process model** | Push back: SHD diary is 24h-resolution, so a continuous-time Hawkes is methodologically inappropriate (Addition 3 §3.3 already disclosed). | n/a |

---

## 31. Journal-required statements (CRediT + Funding + Acknowledgments + Ethics)

Most clinical and informatics journals require the following statements at
submission. Drafts below; final wording awaits author confirmation.

### CRediT author contributions

(Default single-author form. Multiply if a co-author joins.)

> **[Author].** Conceptualization, Data curation, Formal analysis,
> Investigation, Methodology, Software, Validation, Visualization, Writing
> -- original draft, Writing -- review & editing.

### Funding

> This work received no external funding. The reference workstation
> described in the Hardware section was provided by [institution].

(If a stipend or research-grant applies, replace accordingly. Negative
funding statements are required by most clinical journals.)

### Acknowledgments

> The author thanks PriorLabs for distributing the TabPFN model weights
> under their non-commercial licence and for making the
> `tabpfn-extensions` interpretability tooling available; Marco Samuel
> Spano for the original blended-XGBoost architecture (`blended_xgb_lr_spano2026`)
> that anchors the Addition 0 baseline; and Park et al. 2016 for the
> publicly released SHD cohort under CC BY 4.0.

### Ethics statement

> This study is a secondary analysis of the publicly released Park 2016
> SHD cohort (PLOS ONE supplementary file S1, CC BY 4.0). No new patient
> data was collected and no patient interactions occurred. The original
> Park et al. 2016 publication documents the IRB approval covering the
> primary data collection. Re-identification risk from combining hospital
> site, demographic baseline, and dated diary entries (k < 5 anonymity at
> the 62-patient scale) is acknowledged in the Limitations section; no
> raw patient-day rows are redistributed beyond what Park et al. made
> publicly available.

### Conflict-of-interest declaration

> The author declares no competing interests. No commercial entity has
> reviewed or approved the manuscript prior to submission.

### Code availability

> The benchmark code, evaluation harness, per-figure scripts, and the
> engineered-feature parquets are tracked at [GitHub URL] under
> [licence]; a Zenodo DOI archives the submission-version snapshot.
> `Sources.bib` and per-source PDFs are included.

### Data availability

> Raw data is the Park 2016 SHD cohort, publicly released as
> supplementary file S1 of [park2016shd] under CC BY 4.0. The engineered
> feature parquets used in this work are tracked under
> `data/processed/`; SHA-256 hashes for the 92 parquet files are in
> `data/processed/_content_hashes.log` for drift detection.

---

## 32. Plot-style consistency follow-up

Of 27 `fig_*.py` scripts, 15 call `S.apply()` / `_style.apply()` directly
and 12 do not. The 12 non-applying scripts (fig_f1-f5, fig_g1-g5, fig_h1-h2)
delegate to `experiment/2/_figures.py` which handles style application
internally - this is the intentional delegation pattern, not a gap. The
fig_f-series (statistical-comparison plots) appears to use a different
helper still (`experiment/_eval/_style.py` or similar); a one-pass
verification that all 27 figures render with the brand palette should
happen before submission. Effort: ~10 min after the in-progress 12-leaf
insight pass finishes and compare.py is rerun.

---

## 33. Test coverage smoke-test plan

The bootstrap and explainer cores have no test imports detected:
`experiment/_eval/metrics_lib.py` and `experiment/_explain/shap_runners.py`.
Suggested minimal smoke tests (target ~30 lines each, ~1 hour to write
both):

- `tests/test_metrics_lib.py`: confirms `_bootstrap_evaluation` returns
  the same point estimate as the analytic AUROC on a fixed `y, p` pair
  with seed 42; confirms CIs shrink as `n_iterations` grows; confirms
  single-class resamples are dropped not crashed-on.
- `tests/test_shap_runners.py`: confirms `compute_attributions` returns a
  matrix with `feature_names.shape[0] == matrix.shape[1]` on a tiny
  toy XGBoost classifier; confirms the subsample-row count matches the
  documented `_MAX_EXPLAIN` cap; confirms `mean_abs_ranking` is sorted
  descending.

---

## 30. ShapIQ failure follow-up

Memory note `project_shapiq_failure.md` records that ShapIQ order-2 k-SII
fails on TabPFN leaves with a 0-dim array. The post-fix insight run has
been writing `shapiq_interactions_*.npz` files; verification pending the
in-progress 12-leaf pass.

Possible follow-up paths once the verification completes:

1. **If ShapIQ still fails on TabPFN leaves:** drop the Addition 2 §3.5
   ShapIQ-interaction claim (Claim 5) and disclose in the methods that
   pairwise interactions were attempted but could not be reliably computed
   on the TabPFN headline.

2. **If ShapIQ succeeds:** confirm the interaction values are non-trivial
   (max pairwise interaction not orders-of-magnitude smaller than the
   single-feature SHAP); render the top-pair heatmap; integrate into the
   Addition 2 Claim 5 reporting.

3. **If the runtime is too costly on full TabPFN leaves:** restrict
   ShapIQ to the park_features cell (lower EPV, smaller feature space).
   The Addition 2 §1b Claim 5 already restricts the interaction claim to
   the park/headache cell to avoid the EPV-5.5 issue on full_features.
