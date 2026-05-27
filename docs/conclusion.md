# Conclusion

> Position in the paper: **Conclusion (8)**. Reads after `discussion.md`; precedes the References list (`Sources.bib`). Single-paragraph wrap of the load-bearing finding and its disciplinary implication.

We benchmarked 24-hour migraine and headache forecasting on the Park
2016 SHD cohort across seven layered contributions (XGBoost stacking,
TabPFN, explainability, temporal-dependence analysis, sequence
modelling, personalisation with leave-one-site-out external check, and
clinical-value layer), with bootstrap discrimination CIs, calibration
slope, decision-curve net benefit, and within-person evaluation
reported together. On the chronological headline cell the migraine
AUROC reaches 0.793 and the headache AUROC 0.652, but the per-patient
AUROC distribution centres near 0.55, the precision-weighted
within-person C-statistic clusters in the same band, and the Brier
skill against per-patient climatology is negative for migraine. The
pooled discrimination commonly reported in the next-day-diary
migraine-forecasting literature therefore overstates within-person
forecasting skill on this cohort: it reflects between-patient
base-rate separation more than within-patient day-to-day ranking. This
reproduces the within-person Holsteen 2020 pattern on an independent
cohort [holsteen2020triggers, p. 2364] and clarifies why the
diary-only forecasting regime sits in the AUC 0.56-0.73 band that the
field reports under within-person evaluation, well below the
wearable-augmented high-water mark. The
discipline going forward — for this subfield specifically — is to
report calibration slope, decision-curve net benefit, and a
within-person C-statistic alongside pooled discrimination, so that
papers can be compared on the metric that matches the clinical use
case rather than on pooled AUROC alone.
