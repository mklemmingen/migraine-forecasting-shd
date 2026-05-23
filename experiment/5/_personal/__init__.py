"""Personalisation + within-person evaluation layer for Addition 5.

Driver+package style (like Additions 2/3), not the leaf-scaffold of 0/1/4,
because the contribution is an evaluation re-framing plus per-patient model
regimes, not a per-config train sweep.

Comparability (preserved by design)
-----------------------------------
Each regime that produces per-row test probabilities ALSO emits the standard
``sharedMetricPrinter`` ``results/results_*.txt`` under a parse_path-compatible
path:

    experiment/5/<target>/<feature_set>/<regime>/<ratio>/<split>/results/

so ``run_aggregate_results.py`` folds the regimes into the SAME
``comparison_*.html`` as Additions 0/1/4 (architecture = the regime name). The
within-person C-statistic is a different estimand (not in the 11-metric
contract), so it is emitted as its own per-patient artifact and visualised
separately; the pooled-vs-within-person gap (RQ2) is read across the two.

Modules:
  within_person.py  - per-patient AUROC/AUPRC distribution + meta-analytic
                      within-person C-statistic (runs on EXISTING predictions)
  regimes.py        - pooled / per-patient / partial-pooling / TabPFN-in-context,
                      plus the standard-contract emitter (the comparability bridge)
  walkforward.py    - per-patient expanding-window cold-start curve

Design rationale and the decisions (with citations) are in
docs/addition5_personalization.md.
"""
