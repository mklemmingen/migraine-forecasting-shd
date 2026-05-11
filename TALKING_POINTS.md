# Talking Points on Scientific Soundness of the SHD Next-Day Prediction Task

Source: Park et al. 2016 → shd-migraine-benchmark engineering pipeline.

---

## 1. Target definition mismatch (material)

The target `migraine_today = (headache_free == 0)` captures **any headache** (migraine + non-migraine). 
Park reports 1,099 headache days of which only 336 (30.6%) met ICHD-3 migraine criteria. 
The migraine-only rate is ~7–8%. Headache rate (migraine included) is 23.47%.

The README says "next-day migraine forecasting" but the model predicts any headache. 
Feature selection uses Park's migraine-specific ORs (Table 4), which may not hold for non-migraine headaches. 
The column name `migraine_today` is misleading.

**Impact:** Results are valid for headache forecasting but cannot be claimed as
migraine-specific without a migraine-only target variant. 

Solution: Split data into one more headache & migraine targets. Report results for both. 

@Deepu: Maybe even cross-precict? running migraine only val on the headache all model

---

## 2. Prodromal symptoms as features (methodological)

Park measured same-day trigger–headache co-occurrence. 

@Bakir:

Some high-OR triggers 
-
noise (OR 2.8), 
specific smells (71.8% headache likelihood), 
emotional changes (68.8%) 
- 
are known therefore migraine prodromal symptoms (noise sensitivity, osmophobia,
mood changes begin hours before pain onset).

For same-day analysis this is irrelevant?
For next-day prediction, prodromal symptoms predict the **current** migraine, not tomorrow's. 
These features may lose predictive power under the shift(-1) target.

**Impact:** SHAP (Addition 2) will reveal this empirically. If noise/smell
importance drops to near zero, prodrome contamination is the explanation.
Document as a known limitation; do not remove features preemptively.

Solution for now: Not removing any features! until we ran full data analysis and can scientifically justify removing them.
This is especially revelant for later deep learning additions where feature importance may be less transparent.

---

## 3. Rolling windows during diary gaps (data quality)

208 of ~4,453 day-transitions have gaps > 1 day (max 37 days)!
Rolling features use `min_periods=1`, so after a long gap, `migraine_rate_last7` computes on 1 day of data. 
The model cannot distinguish "7 well-recorded headache-free days" from "1 day of data after a month-long gap."

**Impact:** Affects rolling feature reliability for ~5% of rows. Particularly
problematic for LSTM (Addition 4) where sequence continuity is assumed.

Solution: Calculating a column with gap length to last recorded entry, so models can learn this. 

@Deepu: Is this scientifically sound for all of our numbers of additions?

---

## 4. Patient count discrepancy (minor)

Park reports 62 patients. The translated diary has 63 after uppercasing
resolves the CM-004/cm-004 case artifact. The disability sheet has 64. Either
Park excluded 1–2 patients from analysis while leaving their data in the
supplementary file, or the case-resolution logic has an edge case.

**Impact:** Minor. Does not affect modelling. Needs a one-line note in any
paper.

Solution: Will Investigate and document result.

---

## 5. Feature pre-filtering removes potential signal (methodological) (linked to #2)

Features were excluded based on Park's same-day p-values (exercise p=0.78,
sunlight p=0.73, smoking p=0.73) and low prevalence. These p-values measure
same-day association, not next-day predictive power. A feature non-significant
for same-day co-occurrence could theoretically predict at a 1-day lag (e.g.,
exercise → delayed inflammatory response; cheese/chocolate → tyramine pathway).

XGBoost and TabPFN both handle irrelevant sparse features through
regularization and meta-learning. Pre-filtering preempts the model's ability
to discover signal and prevents SHAP (Addition 2) from assessing these
features empirically. (@Deepu: Is my assumption here sound?)

**Resolution:** Retain all 17 of Park's 18 trigger factors (exclude only
`other_trigger`, an unstructured catch-all). Let the model and SHAP be the
judges of feature relevance, not pre-filtering based on a different study's
same-day analysis.
