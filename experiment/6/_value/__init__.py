"""Clinical-forecast-value layer for Addition 6.

Driver+package style (like Addition 2): a post-hoc pass over saved leaf models
that adds a decision-analytic and forecast-skill layer on top of the existing
predictions, fitting no new core model.

Cross-addition comparability
----------------------------
The value metrics (net benefit, Brier skill) are NOT in the 11-metric
sharedMetricPrinter contract, so they do not fold into comparison_*.html. They
are instead overlaid across architectures and additions on their own axes: the
driver loads the leaf models from Additions 0/1/4/5 and plots their decision
curves / skill on shared figures (comparison_value_*.html), so the value
comparison spans additions by construction.

Modules:
  decision_curve.py  - net benefit vs threshold probability (Vickers)
  skill.py           - Brier skill score vs per-patient climatology (Murphy/Brier)
  operating_point.py - val-threshold vs net-benefit-optimal; sensitivity at FPR
  weather_join.py    - optional measured-weather covariates (headache only)

Design rationale and decisions (with citations): docs/addition6_clinical_value.md.
"""
