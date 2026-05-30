# Conclusion

> Position in the paper: **Conclusion (8)**. Reads after `discussion.md`; precedes the References list (`Sources.bib`). Single-paragraph wrap of the load-bearing finding and its disciplinary implication.

We benchmarked 24-hour migraine and headache forecasting on the Park
2016 SHD cohort across seven layered contributions: XGBoost stacking,
TabPFN, explainability, temporal-dependence analysis, sequence
modelling, personalisation with leave-one-site-out external check, and
a clinical-value layer. Each contribution reported bootstrap
discrimination CIs, calibration slope, decision-curve net benefit, and
within-person evaluation together. On the chronological headline cell
the migraine AUROC reaches 0.793 and the headache AUROC 0.652. Under
within-person evaluation, however, the per-patient AUROC distribution
centres near 0.55 on the out-of-fold cross-validation pass over the
non-hyperparameter-tuned 70/30 chronological leaves (addition5 §9b);
the precision-weighted within-person C-statistic clusters in the same
band, and the Brier skill against per-patient climatology under
patient-cluster bootstrap (primary) is significantly positive for
headache only on TabPFN-v2.6 (+0.215 [+0.017, +0.396]); the headache
XGBoost (+0.198 [-0.017, +0.385]) and sequence (+0.139 [-0.123, +0.362])
CIs cross zero under patient-cluster though both are positive under
patient-day-iid sensitivity, and Brier skill is significantly negative
for migraine only on the sequence baseline (-0.117 [-0.241, -0.026]
patient-cluster); the migraine XGBoost and TabPFN Brier-skill CIs
include zero, so the tabular migraine forecasts are statistically
indistinguishable from per-patient climatology rather than significantly
worse. The pooled discrimination commonly reported
in the next-day-diary migraine-forecasting literature therefore
overstates within-person forecasting skill on this cohort: it reflects
between-patient base-rate separation more than within-patient
day-to-day ranking. This reproduces the within-person Holsteen 2020
pattern on an independent cohort [holsteen2020triggers, p. 2364] and
clarifies why the diary-only forecasting regime sits in the AUC
0.56-0.73 band that the field reports under within-person evaluation,
well below the wearable-augmented high-water mark. We do not recommend
any of the benchmarked models for clinical deployment on the basis of
the present internal-validation evidence alone. Future diary-only
migraine-forecasting work in this subfield will need to report
calibration slope, decision-curve net benefit, and a within-person
C-statistic alongside pooled discrimination, so that papers can be
compared on the metric that matches the clinical use case rather than
on pooled AUROC alone.
