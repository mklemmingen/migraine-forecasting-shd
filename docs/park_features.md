# Park Feature Set

> Position in the paper: **Data and features (2.2)**. Reads after
> `dataset.md`; precedes the architecture docs (`xgboost.md`,
> `tabPfn.MD`). The six Park-trigger features defined here anchor the
> SHAP-attribution recovery check in `addition2_results.md`.

A `migraine`-target-only feature set whose columns are exactly the trigger factors Park et al. (2016) selected by their stepwise multiple logistic regression analysis.

This document covers what the set contains, why those particular features, why this set is migraine-only, and which design choices were taken at the column-mapping layer.

## Source

The set is defined by Table 4 of Park et al. [1, Tab. 4, p. 8]. Park et al. ran a stepwise multiple logistic regression with the 18 candidate triggers recorded in their Smartphone Headache Diary baseline survey [1, Methods, p. 3-4], including 153 pairwise (two-trigger) interaction terms (= C(18, 2)) as candidate predictors [1, p. 6 and p. 14]. The model selected six individual triggers as significantly associated with migraine (as opposed to non-migraine) headache:

| Trigger          | Odds ratio | 95% CI       | p-value  |
|------------------|------------|--------------|----------|
| Traveling        | 6.4        | 1.2 - 10.2   | 0.003    |
| Hormonal changes | 3.5        | 2.3 - 5.2    | <0.001   |
| Noise            | 2.8        | 1.4 - 4.9    | 0.002    |
| Alcohol          | 2.5        | 1.3 - 5.0    | 0.009    |
| Overeating       | 2.4        | 1.1 - 5.7    | 0.009    |
| Stress           | 1.8        | 1.4 - 2.4    | <0.001   |

The other 12 triggers analysed by Park et al. (excessive sleep, sleep deprivation, exercise, fatigue, emotional changes, weather changes, sunlight, odors, fasting, caffeine, smoking, cheese/chocolate) appear in Table 4 with "NA, not available due lack of inclusion in the stepwise multiple regression analysis" in the regression column [1, Tab. 4, p. 8].

The Discussion confirms this as the paper's headline finding:

> "traveling, hormonal changes, noise, alcohol, overeating and stress increased the risk of migraines" [1, p. 8, main finding #4]

The Park feature set materialised here keeps exactly these six features, with column-mapping notes below.

## Mapping to engineered columns

Five of Park's six triggers map one-to-one onto an engineered column in `data/processed/*/<ratio>/<split>/diary_*.parquet`:

| Park label  | Engineered column     | Note |
|-------------|-----------------------|------|
| Stress      | `stress_today`        | direct |
| Noise       | `noise_today`         | direct |
| Alcohol     | `alcohol_today`       | direct |
| Overeating  | `overeating_today`    | direct |
| Traveling   | `travel_today`        | direct |

The remaining trigger, "hormonal changes", requires reconstruction. Park et al.'s baseline 18-trigger survey [1, Methods, p. 3] lists "hormonal changes" as a single binary trigger. The Korean SHD source, by contrast, records the trigger as two distinct columns: 월경기 (menstruation) and 배란기 (ovulation), preserved in our pipeline as `menstruation_today` and `ovulation_today`. To match Park's single-feature representation, the Park filter combines them inside `select_park_features`:

```python
df["hormonal_changes_today"] = (
    df["menstruation_today"].astype(int) | df["ovulation_today"].astype(int)
).astype(int)
df = df.drop(columns=["menstruation_today", "ovulation_today"])
```

That is, `hormonal_changes_today` is **1** if either menstruation or ovulation is reported on that day, **0** otherwise. The two source columns are dropped after the derivation so the final feature matrix exactly mirrors Park's stepwise input.

## Final feature inventory

After the filter runs, the model receives six features plus the standard structural columns:

| Type           | Columns                                                                                       |
|----------------|-----------------------------------------------------------------------------------------------|
| Structural     | `patient_id`, `date`, `migraine_target` (+ `entry_id`, `cv_fold` when present)                |
| Park features  | `stress_today`, `hormonal_changes_today`, `noise_today`, `alcohol_today`, `overeating_today`, `travel_today` |

## Migraine target only

The Park feature set is scaffolded **only** under `experiment/<addition>/migraine/park_features/...` - it is not generated for the `headache/` target tree.

Reasoning: Park et al.'s stepwise multiple logistic regression in Table 4 [1, Tab. 4, p. 8] is specifically a **migraine vs non-migraine headache discriminator**, run on the 1,099 headache days in the SHD dataset, of which 336 were migraines and 763 non-migraine headaches. Park did **not** run an analogous stepwise regression for any-headache-vs-no-headache (the comparison would be 1,099 vs 3,480 days, a different statistical setup with different baseline rates). The Park feature set therefore has direct scientific grounding for the migraine target and weaker grounding for the headache target. Scaffolding it for both targets would invite the reader to compare cells whose underlying analytic justifications differ - cleaner to scope the feature set to the target Park's regression actually addresses.

This choice is also reflected in `experiment/<addition>/_scaffold_leaves.py` `enumerate_leaves()`, where `park_features` is added in a migraine-only branch separate from the cross-product over `("headache", "migraine")` × `("full_features", "no_rolling_features", "spano_features")`.

## Two design points that warrant disclosure

**1. Same-day vs next-day prediction framing.** Park et al.'s analysis is for same-day trigger-migraine association: the question is *given today's triggers, is today's headache a migraine*. This benchmark predicts *next-day* migraine state: `migraine_target = migraine_today.shift(-1)` per patient. Using Park's stepwise-selected triggers at a one-day lag is a defensible scientific choice (the carry-over hypothesis - triggers that statistically discriminate same-day migraines are plausible candidates for next-day prediction too) but it is not exactly what Park studied. The methodology section should disclose this.

**2. Preventive medication is absent.** Park et al. uses `preventive_medication` as a stratifier in Table 5 [1, Tab. 5, p. 9], demonstrating that it modifies several trigger-migraine associations. Including it as a covariate in our Park feature set would let the model learn the interaction effects Park documents. The current engineered parquets in this benchmark, however, do **not** carry the `preventive_medication` column. This is a pre-existing translation-stage gap unrelated to the Park feature set definition: `data/pipeline/translate.py` declares the column mapping at line 62, but the output translated parquet does not actually contain the column. The Park feature set therefore excludes `preventive_medication` as a data-availability matter, not a scientific choice. If the translation pipeline is fixed in a later iteration, `preventive_medication` can be added to `_PARK_REQUIRED_COLUMNS` in `experiment/_dataRead/filter_to_park_features.py` and the filter will pick it up on the next sweep.

## Interaction terms

Park's stepwise model also selected two interaction terms: `stress × hormonal_changes` and `noise × travel` [1, Tab. 4, p. 8]. These are **not** materialised as explicit features by the filter. Both gradient-boosted trees (Addition 0) and TabPFN (Addition 1) can learn pairwise interactions from the constituent features themselves, so explicit interaction columns would add no model expressiveness in either architecture family.

## Citation

[1] J.-W. Park, M. K. Chu, J.-M. Kim, S.-G. Park, and S.-J. Cho, "Analysis of trigger factors in episodic migraineurs using a smartphone headache diary applications," *PLOS ONE*, vol. 11, no. 2, p. e0149577, Feb. 2016. doi: [10.1371/journal.pone.0149577](https://doi.org/10.1371/journal.pone.0149577). BibTeX key: `park2016shd`.
