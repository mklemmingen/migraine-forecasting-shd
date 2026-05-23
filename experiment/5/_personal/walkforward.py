"""Per-patient cold-start curve for Addition 5 (RQ4).

How many of a patient's own diary days does an individual/updating model need
before it beats the population model for that patient - the question the
continuous-updating work was built around [houle2021bayesian, p. 1264].
"""
import numpy as np


def cold_start_curve(train_df, test_df, fit_fn, pooled_p, patient_col="patient_id"):
    """Per-patient expanding-window within-person AUROC vs n own-days seen.

    TODO (decision: protocol, docs Section 9): for each patient, walk the test
    days in date order; predict day t using only that patient's days < t (plus,
    for the partially-pooled regime, the cohort prior), refit/update as t grows;
    record the running within-person AUROC as a function of the number of the
    patient's own days seen. Compare against ``pooled_p`` (the population model's
    prediction for the same rows) to find the crossover point. Never touch
    val/test of other patients. Return a tidy DataFrame
    (patient, n_own_days, auroc_individual, auroc_pooled). ~25-35 lines.

    Expanding-window walk-forward captures the cold-start question without
    committing to a full Bayesian posterior in the first pass; the full Bayesian
    treatment [houle2021bayesian, p. 1264] is the recorded follow-up.
    """
    raise NotImplementedError("cold-start walk-forward - see TODO")
