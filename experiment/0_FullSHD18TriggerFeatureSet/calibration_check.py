"""
Reliability diagram comparison: Platt vs Beta calibration on the stacked ensemble.

Run this AFTER train.py has produced stage0_model.joblib.

Decision rule
-------------
Beta calibration (Kull et al. 2017) generalises Platt scaling by fitting two
independent slopes on the log-probability tails:

    p_cal = σ(a·log(p) − b·log(1−p) + c)

When a == b this collapses exactly to Platt scaling (a single logit slope).
The empirical question is whether a ≈ b on this dataset. Two outputs drive the
decision:

  1. Printed a/b coefficients — if |a−b| is small relative to their magnitude,
     Platt is adequate and adding a third parameter is not justified given only
     93 val positives.

  2. Reliability diagrams — six panels (val + test) × (raw / Platt / beta).
     The honest comparison is the bottom row (test): if Beta-Test ECE is not
     meaningfully lower than Platt-Test ECE, do not adopt beta calibration.

Outputs
-------
  calibration_check.png   — saved next to this script
  ECE and Brier summary   — printed to stdout
"""
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR       = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_PATH       = os.path.join(DATA_DIR, "val_engineered.parquet")
TEST_PATH      = os.path.join(DATA_DIR, "test_engineered.parquet")
MODEL_PATH     = os.path.join(EXPERIMENT_DIR, "stage0_model.joblib")
OUTPUT_PATH    = os.path.join(EXPERIMENT_DIR, "calibration_check.png")
N_BINS         = 10


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_prep_data(filepath):
    df = pd.read_parquet(filepath)
    X = df.drop(columns=["entry_id", "patient_id", "date", "migraine_target"])
    y = df["migraine_target"]
    return X, y


# ---------------------------------------------------------------------------
# Beta calibrator
# ---------------------------------------------------------------------------

class BetaCalibrator:
    """Beta calibration (Kull et al. 2017, ECML-PKDD).

    Fits p_cal = σ(a·log(p) − b·log(1−p) + c) as a two-feature logistic
    regression on [log(p), −log(1−p)].  When the fitted a ≈ b the result is
    equivalent to Platt scaling; when they differ the asymmetric slope captures
    miscalibration that Platt cannot represent.

    Three parameters total (a, b, c) vs. Platt's two (slope, intercept).
    """

    def __init__(self):
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)

    def fit(self, proba, y):
        p = np.clip(proba, 1e-8, 1 - 1e-8)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        self._lr.fit(X, y)
        return self

    def predict(self, proba):
        p = np.clip(proba, 1e-8, 1 - 1e-8)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        return self._lr.predict_proba(X)[:, 1]

    @property
    def a(self):
        return float(self._lr.coef_[0][0])

    @property
    def b(self):
        return float(self._lr.coef_[0][1])

    @property
    def c(self):
        return float(self._lr.intercept_[0])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins=N_BINS):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binned = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    for i in range(n_bins):
        mask = binned == i
        if mask.sum() > 0:
            ece += np.abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.sum()
    return ece / len(y_true)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def reliability_diagram(ax, y_true, y_prob, title, n_bins=N_BINS):
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned      = np.digitize(y_prob, bin_edges[1:-1])

    frac_pos, mean_pred, counts = [], [], []
    for i in range(n_bins):
        mask = binned == i
        if mask.sum() > 0:
            frac_pos.append(float(y_true[mask].mean()))
            mean_pred.append(float(y_prob[mask].mean()))
        else:
            frac_pos.append(np.nan)
            mean_pred.append(bin_centers[i])
        counts.append(int(mask.sum()))

    frac_pos  = np.array(frac_pos)
    mean_pred = np.array(mean_pred)
    counts    = np.array(counts)

    ece   = expected_calibration_error(y_true, y_prob, n_bins)
    brier = brier_score_loss(y_true, y_prob)

    # Histogram (scaled so bars occupy at most the bottom quarter of the axes)
    ax2 = ax.twinx()
    ax2.bar(bin_centers, counts, width=0.085, alpha=0.25, color="steelblue", zorder=1)
    ax2.set_ylim(0, counts.max() * 5 if counts.max() > 0 else 1)
    ax2.set_yticks([])

    # Perfect-calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect", zorder=2)

    # Calibration curve — only non-empty bins
    valid = ~np.isnan(frac_pos)
    ax.plot(mean_pred[valid], frac_pos[valid], "o-", lw=2, ms=5,
            color="tab:orange", label="Model", zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability", fontsize=8)
    ax.set_ylabel("Fraction of positives", fontsize=8)
    ax.set_title(f"{title}\nECE={ece:.3f}   Brier={brier:.3f}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    print("Loading data and model...")
    X_val,  y_val  = load_and_prep_data(VAL_PATH)
    X_test, y_test = load_and_prep_data(TEST_PATH)

    bundle  = joblib.load(MODEL_PATH)
    stacker = bundle["stacker"]
    platt   = bundle["calibrator"]   # LogisticRegression fitted in train.py

    # Raw stacker probabilities
    print("Computing stacker probabilities...")
    p_val_raw  = stacker.predict_proba(X_val)[:, 1]
    p_test_raw = stacker.predict_proba(X_test)[:, 1]

    # Platt: already fitted on val in train.py; apply to both splits
    p_val_platt  = platt.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]
    p_test_platt = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

    # Beta: fit on val raw probs, evaluate on test
    print("Fitting beta calibrator on val raw probabilities...")
    beta = BetaCalibrator().fit(p_val_raw, y_val.values)
    p_val_beta  = beta.predict(p_val_raw)
    p_test_beta = beta.predict(p_test_raw)

    # ------------------------------------------------------------------
    # Asymmetry diagnosis
    # ------------------------------------------------------------------
    print()
    print("Beta calibrator parameters (fitted on val raw probs):")
    print(f"  a = {beta.a:.4f}   (slope on log p)")
    print(f"  b = {beta.b:.4f}   (slope on log(1−p))")
    print(f"  c = {beta.c:.4f}   (intercept)")
    asymmetry = abs(beta.a - beta.b)
    scale     = 0.5 * (abs(beta.a) + abs(beta.b)) + 1e-9
    print(f"  |a−b| = {asymmetry:.4f}   relative asymmetry |a−b|/mean(|a|,|b|) = {asymmetry/scale:.3f}")
    if asymmetry / scale < 0.10:
        print("  → a ≈ b: miscalibration is SYMMETRIC. Platt scaling is sufficient.")
    else:
        print("  → |a−b| is notable: miscalibration is ASYMMETRIC. Beta may help.")

    # ------------------------------------------------------------------
    # ECE / Brier summary table
    # ------------------------------------------------------------------
    print()
    print(f"{'':30s}  {'ECE10':>7}  {'Brier':>7}")
    print("-" * 50)
    configs = [
        ("Raw stacker — Val",    y_val.values,  p_val_raw),
        ("Platt       — Val",    y_val.values,  p_val_platt),
        ("Beta        — Val",    y_val.values,  p_val_beta),
        ("Raw stacker — Test ★", y_test.values, p_test_raw),
        ("Platt       — Test ★", y_test.values, p_test_platt),
        ("Beta        — Test ★", y_test.values, p_test_beta),
    ]
    for label, y, p in configs:
        ece   = expected_calibration_error(y, p)
        brier = brier_score_loss(y, p)
        print(f"  {label:<28}  {ece:7.4f}  {brier:7.4f}")
    print()
    print("★ Test rows are the honest comparison (out-of-sample for all calibrators).")
    print("  If Beta-Test ECE ≈ Platt-Test ECE, Platt is sufficient.")

    # ------------------------------------------------------------------
    # Reliability diagram figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "Reliability diagrams — Stacked Ensemble (stage0_model)\n"
        "Top row: val (in-sample for calibrators)   "
        "Bottom row: test (out-of-sample — honest comparison ★)",
        fontsize=11,
    )

    reliability_diagram(axes[0, 0], y_val.values,  p_val_raw,   "Raw stacker — Val")
    reliability_diagram(axes[0, 1], y_val.values,  p_val_platt, "Platt — Val (in-sample)")
    reliability_diagram(axes[0, 2], y_val.values,  p_val_beta,  "Beta — Val (in-sample)")
    reliability_diagram(axes[1, 0], y_test.values, p_test_raw,  "Raw stacker — Test ★")
    reliability_diagram(axes[1, 1], y_test.values, p_test_platt,"Platt — Test ★")
    reliability_diagram(axes[1, 2], y_test.values, p_test_beta, "Beta — Test ★")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
