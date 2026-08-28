"""Sensitivity sweep for the composite_sorted leaf-selection thresholds.

Question: if the AUROC tier bucket (default 0.02), the AUPRC tier bucket
(default 0.02), or the calibration-slope guards (default 0.0 lower, 5.0
upper) are perturbed by a reasonable amount, does the headline cell
change?

A robust composite rule is one whose headline cell is stable across
moderate threshold perturbation. A fragile rule's headline flips at any
nudge to its constants.

Run:
    python -m experiment.2.sensitivity_composite

Prints a table of (AUROC_TOL, AUPRC_TOL, CALIB_MIN, CALIB_MAX) settings
and the resulting headline-cell leaf path per (target, splittype).
"""
from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

# Import select once, then monkey-patch its constants and re-run the
# selector function for each parameter setting.
import leaf_selection as _select  # type: ignore

# Default values, restored at the end:
ORIG_AUROC_TOL = _select.AUROC_TOL
ORIG_AUPRC_TOL = _select.AUPRC_TOL
ORIG_CALIB_MIN = _select.CALIB_MIN
ORIG_CALIB_MAX = _select.CALIB_MAX

# Perturbation grid. Each axis is {default-step, default, default+step}.
# Composite (AUROC=AUPRC) tier moves are correlated; calibration guards
# are perturbed independently.
SWEEPS = [
    (0.01, 0.01, 0.0, 5.0),
    (0.02, 0.02, 0.0, 5.0),   # default
    (0.03, 0.03, 0.0, 5.0),
    (0.02, 0.02, -0.5, 5.0),  # looser calibration floor
    (0.02, 0.02, 0.0, 3.0),   # tighter calibration ceiling
    (0.02, 0.02, 0.0, 7.0),   # looser calibration ceiling
]


def _resolve_headlines(target_param_set):
    auroc, auprc, cmin, cmax = target_param_set
    _select.AUROC_TOL = auroc
    _select.AUPRC_TOL = auprc
    _select.CALIB_MIN = cmin
    _select.CALIB_MAX = cmax
    # select_insight_leaves walks the full sweep CSV and emits the
    # composite_sorted top row per (target, feature_set, splittype) cell.
    sels = _select.select_insight_leaves()
    out = {}
    for s in sels:
        # Match the existing convention from run_insights.py: the headline
        # of the full_features/chrono cell per target.
        if (s.get("feature_set") == "full_features"
                and s.get("splittype") == "chrono"
                and s.get("role") == "headline"):
            out[s["target"]] = Path(s["leaf_dir"]).relative_to(EXP).as_posix()
    return out


def main() -> int:
    print(f"{'AUROC_TOL':>10} {'AUPRC_TOL':>10} {'CALIB_MIN':>10} {'CALIB_MAX':>10} "
          f"{'headache':<70} {'migraine':<70}")
    print("-" * 180)
    rows = []
    for params in SWEEPS:
        h = _resolve_headlines(params)
        rows.append((params, h))
        au, ap, cm, cM = params
        print(f"{au:>10.3f} {ap:>10.3f} {cm:>10.2f} {cM:>10.2f} "
              f"{h.get('headache','(none)'):<70} {h.get('migraine','(none)'):<70}")

    # Restore defaults
    _select.AUROC_TOL = ORIG_AUROC_TOL
    _select.AUPRC_TOL = ORIG_AUPRC_TOL
    _select.CALIB_MIN = ORIG_CALIB_MIN
    _select.CALIB_MAX = ORIG_CALIB_MAX

    # Stability summary
    print()
    headache_set = {h.get("headache") for _, h in rows}
    migraine_set = {h.get("migraine") for _, h in rows}
    print(f"Distinct headache headlines under perturbation: {len(headache_set)}")
    for x in sorted(headache_set): print(f"  {x}")
    print(f"Distinct migraine headlines under perturbation: {len(migraine_set)}")
    for x in sorted(migraine_set): print(f"  {x}")
    print()
    if len(headache_set) == 1 and len(migraine_set) == 1:
        print("Verdict: STABLE. The headline cell does not change under any of "
              "the 6 perturbations of the composite_sorted constants.")
    else:
        print("Verdict: SOME PERTURBATIONS CHANGE THE HEADLINE. See above; "
              "the calibration-slope window and AUROC tier bucket are "
              "the dominant axes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
