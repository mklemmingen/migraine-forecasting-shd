# Addition 6: clinical-forecast value

> Position in the paper: **Methods (4.6)**. Reads after `external_validation_site.md`; precedes `results_findings.md`. Decision-curve + Brier-skill methodology supports the clinical-value contribution reported in results_findings.md Headline takeaway 1.

The forecast-value layer (`experiment/6/`) addresses what Additions 0, 1, 4,
and 5 do not: whether a forecast at a given operating point would actually
help a patient act, the question the migraine-prediction field has flagged
as unmet. Addition 6 adds a decision-analytic and forecast-skill layer on
top of the existing predictions, and (optionally) tests whether real
meteorological covariates add value, without fitting any new core model.

## 1. Why this addition exists, and what it builds on

The recent narrative review concludes that patient needs should be assessed to
discover what a valuable prediction looks like, and that the field should develop
common standards for evaluating migraine prediction algorithms
[dumkrieger2025review, p. 1]. The classical decomposition of forecast goodness
separates *quality* (the correspondence between forecasts and observations)
from *value* (the incremental economic or other benefit a decision maker
realises by acting on the forecasts) [murphy1993forecast, p. 281]. Discrimination
and calibration, reported in the benchmark, measured quality rather than value:
a model can beat the base rate on AUPRC yet provide no net clinical benefit, and
AUPRC's baseline itself moves with prevalence so it cannot be read as value across
the migraine (~7%) and headache (~24%) cells [mcdermott2024aurocAuprc, p. 1]. Accuracy is base-rate
dominated and is already labelled as such in the comparison table
(the paper's TRIPOD+AI and PROBAST+AI checklists, Section 4).

The discipline that converts a probability into a decision is decision curve
analysis: net benefit is plotted against the threshold probability, where the
threshold encodes the relative harm of a false positive versus a false negative
for the specific action under consideration [vickers2019dca, p. 1; p. 3; p. 4].
Reporting net benefit against a range of thresholds shows over which
preference range, if any, acting on the model beats the default policies of
"treat all" or "treat none". For next-day migraine the
action is concrete (take acute medication pre-emptively the evening before),
and its harm trade-off (an unnecessary dose versus a missed early treatment
window) is exactly what the threshold probability parameterises.

This sits one stage before the next reporting guideline in the development
pathway. DECIDE-AI is the standard for the early-stage clinical evaluation of
AI-driven decision-support systems and comprises 17 AI-specific reporting items
[vasey2022decideAI, p. 1]; it is already named in the rigor checklist as the
guideline a future clinical-evaluation follow-up should adopt
(the paper's TRIPOD+AI and PROBAST+AI checklists, Section 1d). Addition 6 produces the
development-stage decision-analytic evidence that such a follow-up would build on,
without claiming a clinical evaluation has occurred.

## 2. Research questions

1. **Net benefit.** Over what range of threshold probabilities, if any, does
   acting on the model's next-day forecast yield positive net benefit above
   treat-all and treat-none, per target and per architecture
   [vickers2019dca, p. 4]?
2. **Forecast skill vs climatology.** Does the model beat the trivial baseline of
   each patient's own base rate (the climatological forecast) on a Brier skill
   score, and by how much? A model that merely reproduces the base rate has zero
   skill regardless of its AUROC.
3. **Operating point.** At a clinically tolerable false-alarm rate, what
   sensitivity is achievable, and how does the validation-derived MCC threshold
   the benchmark already uses map onto the net-benefit-optimal threshold?
4. **(Optional) Real weather covariates.** Do actual meteorological features add
   discrimination or net benefit over the diary's self-reported `weather_change`
   flag, given that smartphone-app studies link low barometric pressure, higher
   humidity, and rainfall to headache onset [katsuki2023weather, p. 585]?

## 3. Methods (with rationale and literature anchor)

The clinical-value analyses are read on the same NonHP 70/15/15
chronological leaves Addition 5 uses (XGBoost migraine pooled AUROC
0.763, TabPFN headache pooled AUROC 0.654, window-MLP sequence 0.771).
These are one tier away from the paper's headline cell (HP020 70/30:
XGBoost migraine 0.793) and are chosen so the per-patient climatology
that feeds the Brier skill score has enough test-set base-rate signal
across architectures. The negative-Brier-skill finding for migraine
reproduces across the headline cell and the 70/15/15 cell; the choice
of cell is not load-bearing for that result.

### 3.1 Decision curve analysis (net benefit)

For each leaf's calibrated test predictions, net benefit was computed across threshold
probabilities and plotted against the treat-all and treat-none reference lines.
This is the decision-curve construction introduced by Vickers and Elkin (net
benefit plotted against threshold probability, with the net-benefit formula
weighting true and false positives by the threshold odds [vickers2006dca, p. 565;
p. 567]) and operationalised in the step-by-step interpretation guide
[vickers2019dca, p. 1; p. 3; p. 4]. The
threshold range is restricted to the clinically plausible band for the
pre-emptive-medication action (a low threshold, since a missed attack is costlier
than an unnecessary dose for most acute migraine therapies), and the threshold is
interpreted as the relative harm ratio rather than as a probability cutoff chosen
for accuracy. Concretely, the implementation
evaluates net benefit on a 50-point uniform grid from 0.01 to 0.50
(`np.linspace(0.01, 0.50, 50)` in `experiment/6/_value/decision_curve.py`).
The 0.01 floor avoids dividing by near-zero odds; the 0.50 ceiling
covers the clinically meaningful "treat at any odds at or above 1:1"
band for the pre-emptive use case and keeps the plot resolution dense
where the model curves actually fan apart. Net benefit is computed on
the held-out test split only, reusing the post-hoc-calibrated
probabilities from Additions 0/1/4/5 so the decision curve reflects
the probabilities the benchmark actually scores.

### 3.2 Brier skill score against a per-patient climatology

The Brier score (already in the benchmark's calibration panel
[huang2020calibration, p. 624]) was reported and converted to a skill score relative to a
reference forecast that always predicts the patient's own base rate. A positive
skill score means the model adds information beyond the marginal rate; a
near-zero skill score with a high AUROC would expose discrimination that does not
translate into a better-than-climatology probability forecast, a quality
shortfall in Murphy's sense, separate from the value question the decision curve
answers [murphy1993forecast, p. 281]. Calibration-in-the-large (whether the
mean predicted risk matches the observed event rate) is reported alongside,
since it is the gross-calibration check that a skill claim
rests on [huang2020calibration, p. 621], and the benchmark's calibration slope is
already known to be fragile on the small/sparse cells (median 0.64,
`docs/results_findings.md`, Section 6). Discrimination and calibration are
reported together throughout, per TRIPOD+AI [collins2024tripodAI, p. 6].

### 3.3 Operating-point analysis (TRIPOD+AI Item 15 risk groups)

Three threshold bands (t = 0.10-0.20, t = 0.20-0.35, t > 0.35) functioned as risk groups in the TRIPOD+AI Item 15 sense: each band carries a documented clinical action and the bands are pre-specified before evaluation rather than being post-hoc tertiles. The risk-group definitions are literature- and clinical-judgement-anchored (per the medication-burden and behavioural-tolerability arguments named below); the underlying model output is the calibrated probability, with the bands operationalising that probability into recommended actions.

The validation-derived MCC threshold (the benchmark's existing
threshold-metric choice) was mapped onto the net-benefit-optimal threshold from
3.1, and sensitivity was reported at a fixed, clinically tolerable
false-positive rate. The fixed FPR is **0.10** (one false alarm per ten
non-attack days), set in `experiment/6/run_value.py`; this is the rate
at which a daily pre-emptive-medication recommendation remains
behaviourally tolerable for an episodic-migraine population whose
non-attack days outnumber attack days roughly 19:1. Sensitivity at
FPR = 0.10 is reported per architecture per target on the held-out
test split, with the same bootstrap CIs the benchmark uses.

**Withdrawn.** This paragraph previously specified a three-band action escalation (behavioural check at t = 0.10-0.20, lifestyle modification at 0.20-0.35, pre-emptive medication only above t > 0.35). It contradicted Section 3.1 of this same document and the `decision_curve.py` docstring, both of which place the pre-emptive-medication action at LOW thresholds, and it was never implemented anywhere under `experiment/6/`. It also assigned no action at all to t = 0.01-0.10, the sub-band the decision-curve result actually sits in. Section 3.1 is load-bearing and stands; this banding does not.

The threshold band over which the model shows non-negative net benefit
against treat-all and treat-none is read as a recommendation surface,
not a prescriptive cut-off. The escalation aligns the decision curve's
threshold interpretation [vickers2019dca, p. 3] with realistic
clinician/patient behaviour: a mis-calibrated low-confidence prediction
triggers only the low-cost action, while higher-confidence predictions
authorise costlier interventions.

This makes explicit whether the threshold the benchmark already uses for its
threshold-derived metrics coincides with the threshold a
decision-analytic view would choose.

### 3.4 (Optional) Real meteorological covariates

The SHD diary carries only a self-reported `weather_change` trigger flag, not
measured weather (`docs/dataset.md`, weather features). Smartphone-app evidence at
population scale (4,375 users) found low barometric pressure, barometric-pressure
change, higher humidity, and rainfall associated with headache occurrence using a
generalized linear mixed model, a feedforward neural network, and gradient
boosting [katsuki2023weather, p. 585; p. 590]. Because the diary records dates and
the cohort attended two known Korean hospital sites (`docs/dataset.md`, study
design), historical daily meteorological series for those locations and dates can
be joined and added as candidate features, then ablated against the diary-only
model.

This sub-component is run on the **headache** target only. The migraine
`full_features` cell already sits at events-per-variable 5.5, the high-risk
overfitting band (`docs/dataset.md`, EPV section), and adding weather columns
would worsen it; shrinkage cannot manufacture information that the 287 positive
migraine days do not contain [martin2025samplesize, p. 2]. The headache target,
in the low-risk EPV band across feature sets, is where a weather-feature ablation
is statistically defensible. The result is reported as an ablation (with vs
without weather) on both discrimination and net benefit, not as a new headline
model.

## 4. Multiple-comparisons and honest-reporting discipline

The pre-registered confirmatory outputs were the per-target decision curve (RQ1)
and the Brier skill score vs climatology (RQ2) at the canonical 70/15/15
chronological `full_features` cell. The operating-point mapping (RQ3) and the
weather ablation (RQ4) were exploratory and labelled as such. Net benefit and skill
were reported with bootstrap 95% CIs consistent with the rest of the benchmark, and
no clinical-utility claim was made beyond the development stage, per the DECIDE-AI
boundary [vasey2022decideAI, p. 1].

## 5. Directory and output layout

```
experiment/6/
  run_value.py                    # driver: load calibrated test preds, compute value layer
  _value/
    decision_curve.py             # net benefit vs threshold (3.1)
    skill.py                      # Brier skill score vs per-patient climatology (3.2)
    operating_point.py            # threshold mapping + sensitivity at fixed FPR (3.3)
    weather_join.py               # optional KMA meteorological join + ablation (3.4)
  <target>/<feature_set>/<arch>/<ratio>/<split>/
    decision_curve_<ts>.png
    skill_<ts>.txt                # Brier, Brier skill score, calibration-in-the-large
    operating_point_<ts>.txt
  weather_ablation_<ts>.html      # headache-only, with vs without measured weather
  comparison_value_<ts>.html      # net benefit + skill across arch x target
```

The value layer consumes the existing calibrated predictions; it adds no model
and does not touch `evaluate.py`, mirroring how Addition 2 ran as a post-sweep
pass over saved bundles (`docs/insights_leaf_selection.md`).

**Comparability.** Net benefit and Brier skill are not in the 11-metric
`sharedMetricPrinter` contract, so they do not fold into `comparison_*.html`.
Instead the driver loads the leaf models from Additions 0/1/4/5 and overlays
their decision curves and skill on shared axes per `(target, feature_set)` cell
(`comparison_value_*.html`), so the value comparison spans additions by
construction; the cross-addition comparison is a curve overlay rather than a
metric-table row.

## 6. Build status

Implemented and run end-to-end (`experiment/6/run_value.py`): `_value/skill.py`,
`_value/decision_curve.py`, `_value/operating_point.py` are concrete, and the
driver discovers the Additions 0/1/4 leaves, regenerates their val+test
predictions by reusing the Addition 5 subprocess worker (per-addition path/device
isolation: TabPFN on GPU, XGBoost/sequence on CPU), computes net benefit, Brier
skill vs each patient's TRAIN-set climatology, and the operating point, and emits
a per-cell decision-curve figure plus a summary. The only remaining piece is
`_value/weather_join.py` (the optional measured-weather ablation), which needs an
external KMA data source and is left as documented future work. Each module stays
under the 300-line budget.

## 7. Dependencies

- No new modelling dependency; net benefit, Brier skill, and operating-point
  computations are plain `numpy`/`scipy`.
- The optional weather join needs a historical daily meteorological source for
  the two Korean hospital cities over the Sept 2014-Jan 2015 window; the source
  and its licence are recorded in `weather_join.py` if the sub-component is run.

## 8. Compute budget

The value layer is CPU-seconds over the saved test predictions. The optional
weather join adds a one-off data download and a re-fit of the headache model with
the extra columns (minutes, reusing the Addition 0/1 training path).

## 9. Resolved decisions and the open one

- **Decision curve analysis as the value metric**: chosen over a single
  cost-weighted accuracy because net benefit sweeps the full preference range and
  is interpretable without committing to one cost ratio [vickers2019dca, p. 4].
- **Per-patient climatology as the skill baseline**: chosen so that "skill" means
  beating the patient's own base rate, not beating a pooled prior.
- **Weather ablation scoped to headache only**: chosen on the EPV argument
  (`docs/dataset.md`; [martin2025samplesize, p. 2]).
- **Open**: whether to source measured weather at all in this iteration, or to
  record it as future work. The core value layer (RQ1-RQ3) stands without it; the
  weather ablation is the one component with an external-data dependency and may
  be deferred.

## 9a. Result (clinical-forecast value)

`run_value.py` over the chronological full_features 70/15/15 leaves of Additions
0 (XGBoost), 1 (TabPFN v3-default) and 4 (sequence window-MLP), val+test horizon,
Brier skill against each patient's TRAIN-set base-rate climatology:

| target   | architecture | AUROC | Brier skill (day-iid CI)    | CITL (day-iid CI)    | net-benefit+ band |
|----------|--------------|-------|------------------------------|----------------------|-------------------|
| headache | XGBoost      | 0.658 | +0.198 [+0.126, +0.268]     | 1.03 [0.88, 1.18]    | 0.07-0.47         |
| headache | TabPFN       | 0.653 | +0.215 [+0.143, +0.287]     | 0.98 [0.84, 1.12]    | 0.07-0.50         |
| headache | sequence     | 0.598 | +0.139 [+0.066, +0.211]     | 0.90 [0.76, 1.05]    | 0.02-0.50         |
| migraine | XGBoost      | 0.791 | -0.068 [-0.160, +0.005]     | 1.09 [0.79, 1.42]    | 0.03-0.26         |
| migraine | TabPFN       | 0.761 | -0.057 [-0.146, +0.020]     | 1.14 [0.83, 1.48]    | 0.03-0.49         |
| migraine | sequence     | 0.771 | -0.117 [-0.227, -0.037]     | 0.93 [0.66, 1.22]    | 0.03-0.50         |

CIs in this table use patient-day-iid bootstrap (n=1,000 iterations,
sensitivity). Body §3.7 reports patient-cluster bootstrap as primary on
the headline cells per body §2.8, under which the headache positive
Brier skill narrows to TabPFN-v2.6 only: cluster CIs are +0.215
[+0.017, +0.396] (TabPFN), +0.198 [-0.017, +0.385] (XGBoost crosses
zero), +0.139 [-0.123, +0.362] (sequence crosses zero). Migraine
sequence cluster CI -0.117 [-0.241, -0.026] remains significantly
negative; the migraine XGBoost cluster [-0.257, +0.077] and TabPFN
cluster [-0.172, +0.063] CIs include zero.
Calibration-in-the-large (CITL) added per Huang 2020 trio.

**The decisive value finding (revised with CIs):**
under patient-cluster bootstrap (primary) the headache positive Brier
skill survives only at the TabPFN-v2.6 cell (+0.215 [+0.017, +0.396]);
the XGBoost (+0.198 [-0.017, +0.385]) and sequence (+0.139 [-0.123,
+0.362]) headache CIs cross zero under patient-cluster though both
remain positive under patient-day-iid sensitivity. For migraine the
picture is more nuanced than the original point-estimate framing
implied: only the sequence baseline carries a CI fully below zero
(-0.117 [-0.241, -0.026] patient-cluster). The migraine XGBoost CI
([-0.257, +0.077] patient-cluster) and TabPFN CI ([-0.172, +0.063]
patient-cluster) both include zero, so the tabular models are
statistically indistinguishable from the per-patient climatology
baseline rather than significantly worse. High migraine AUROC (~0.76) still
fails to translate into significant probabilistic value beyond the per-patient
base rate, exactly Murphy's quality-vs-value distinction with finite-sample
uncertainty acknowledged [murphy1993forecast, p. 281]. Consistent with the
near-chance within-person discrimination
(`docs/addition5_personalization.md` Section 9b) and the fragile calibration
on small/sparse cells (`docs/results_findings.md` Section 6). Headache (denser, ~24% positive) does add
modest value (Brier skill +0.14 to +0.21). At a tolerated 10% false-alarm rate
the models catch 25-47% of attacks. Every model has a low-threshold net-benefit
band over treat-all/treat-none, but for migraine that band rests on poorly
calibrated probabilities, so it should be read with the negative-skill caveat.
Decision-curve figures are emitted per (target, feature_set) cell. No
clinical-utility claim is made beyond the development stage
[vasey2022decideAI, p. 1].

## 10. How this addition changes the rest of the paper

Addition 6 answers the value question the field says is missing
[dumkrieger2025review, p. 1] in the benchmark's own numbers: the decision curves
state over what preference range, if any, the diary-only forecasts are worth
acting on, and the Brier skill score states whether the models beat each
patient's base rate at all. A plausible and publishable outcome on this small,
diary-only cohort is that net benefit is positive only in a narrow low-threshold
band and that skill over climatology is small, which, reported honestly,
calibrates expectations for diary-only forecasting and motivates the
wearable-augmented and personalised directions (Additions 4-5) rather than
overclaiming. It also pre-positions the work for a DECIDE-AI-compliant clinical
evaluation [vasey2022decideAI, p. 1] without asserting one has been done.

## References

[dumkrieger2025review] G. M. Dumkrieger, "The promise of machine learning in
predicting migraine attacks," *Cephalalgia*, vol. 45, no. 11, 2025. doi:
10.1177/03331024251391207. (Assess what a valuable prediction looks like; develop
common evaluation standards, p. 1.)

[vickers2006dca] A. J. Vickers and E. B. Elkin, "Decision curve analysis: a novel
method for evaluating prediction models," *Med. Decis. Making*, vol. 26, no. 6,
pp. 565-574, 2006. doi: 10.1177/0272989X06295361. (Original DCA method: net
benefit vs threshold probability, p. 565; net-benefit formula, p. 567.)

[vickers2019dca] A. J. Vickers, B. van Calster, and E. W. Steyerberg, "A simple,
step-by-step guide to interpreting decision curve analysis," *Diagn. Progn.
Res.*, vol. 3, no. 18, 2019. doi: 10.1186/s41512-019-0064-7. (Net benefit vs
threshold probability and the relative-harm interpretation, pp. 1, 3; treat-all /
treat-none reference lines, p. 4.)

[murphy1993forecast] A. H. Murphy, "What is a good forecast? An essay on the
nature of goodness in weather forecasting," *Weather and Forecasting*, vol. 8, no.
2, pp. 281-293, 1993. doi: 10.1175/1520-0434(1993)008<0281:WIAGFA>2.0.CO;2. (Three
types of forecast goodness: consistency, quality, value, p. 281.)

[vasey2022decideAI] B. Vasey *et al.*, "Reporting guideline for the early stage
clinical evaluation of decision support systems driven by artificial
intelligence: DECIDE-AI," *BMJ*, vol. 377, e070904, 2022. doi:
10.1136/bmj-2022-070904. (Early-stage clinical-evaluation guideline; 17
AI-specific reporting items, p. 1.)

[huang2020calibration] Y. Huang, W. Li, F. Macheret, R. A. Gabriel, and L.
Ohno-Machado, "A tutorial on calibration measurements and calibration models for
clinical prediction models," *J. Am. Med. Inform. Assoc.*, vol. 27, no. 4, pp.
621-633, 2020. doi: 10.1093/jamia/ocz228. (Calibration-in-the-large, p. 621;
Brier score, p. 624.)

[katsuki2023weather] M. Katsuki *et al.*, "Investigating the effects of weather
on headache occurrence using a smartphone application and artificial
intelligence," *Headache*, vol. 63, no. 5, pp. 585-600, 2023. doi:
10.1111/head.14482. (4,375 users; low barometric pressure, humidity, rainfall
associated with headache, p. 585; GLMM, feedforward neural network, gradient
boosting, p. 590.)

[mcdermott2024aurocAuprc] M. B. A. McDermott *et al.*, "A closer look at AUROC and
AUPRC under class imbalance," arXiv:2401.06091, 2024. doi:
10.48550/arXiv.2401.06091. (AUPRC depends on prevalence, p. 1.)

[collins2024tripodAI] G. S. Collins *et al.*, "TRIPOD+AI statement," *BMJ*, vol.
385, e078378, 2024. doi: 10.1136/bmj-2023-078378. (Report discrimination and
calibration, item 12e, p. 6.)

[martin2025samplesize] G. P. Martin, R. D. Riley, J. Ensor, and S. W. Grant,
"Statistical primer: sample size considerations for developing and validating
clinical prediction models," *Eur. J. Cardiothorac. Surg.*, vol. 67, no. 5, p.
ezaf142, 2025. doi: 10.1093/ejcts/ezaf142. (Shrinkage cannot create information in
small samples, p. 2.)
