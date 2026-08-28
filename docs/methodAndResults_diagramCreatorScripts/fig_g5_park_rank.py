"""Figure G5 - Park-2016 odds-ratio rank vs model SHAP rank (Addition 2).

Scatter of Park et al. 2016 Table-4 odds-ratio rank (x) against the model's mean
|SHAP| rank (y) for the shared migraine triggers, with the agreement diagonal and
the Spearman rho in the title. A convergence check, not a validation: same-day
population odds ratios vs a next-day individual forecast. One figure per
migraine/park cell (role) present in the frozen data.

Data source: the latest experiment/2/figdata_*.json frozen by
experiment/2/compare.py.

Usage: python fig_g5_park_rank.py
"""
# §11 compliance: Spearman rho between Park 2016 OR rank and SHAP rank.
#   §11.1: Fisher-z 95% CI on rho printed in title by F.park_scatter_figure()
#   §11.3 caption: Park 2016 cohort + n in rendered title
#   §11.7 N/A (Park feature set; not migraine `full_features`)
# Paper caption (for LaTeX):
#   Park et al. 2016 Table-4 odds-ratio rank (x) vs the model's mean |SHAP| rank (y)
#   for the shared migraine triggers, with the perfect-agreement diagonal and the
#   Spearman rho (reported with its p-value and n in the title). This is a convergence
#   check, not a validation: Park's same-day population odds ratios and the model's
#   next-day individual attribution answer different questions, and with only ~6 shared
#   triggers any correlation is highly uncertain (p is large), so the coefficient is
#   descriptive.
import sys as _sys
import importlib.util as _ilu
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
# experiment/ on the path for _style; _figures is loaded by file path rather
# than by adding experiment/2 to sys.path. The module that once shadowed the
# stdlib select from there is now named leaf_selection, so this is hygiene
# rather than a workaround.
_sys.path.insert(0, str(_EXP))
_spec = _ilu.spec_from_file_location("_exp2_figures", _EXP / "2" / "_figures.py")
F = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(F)

HERE = _Path(__file__).resolve().parent
ADDITION2 = _EXP / "2"


def main():
    figdata_path = F.latest_figdata(ADDITION2)
    if figdata_path is None:
        raise SystemExit("no figdata_*.json in experiment/2/ - run experiment/2/compare.py first")
    data = F.load_figdata(figdata_path)
    print(f"  source {figdata_path.name} ({len(data['park'])} migraine/park cells)")
    if not data["park"]:
        print("  skip: no Park cells with >= 3 shared triggers in the figure data")
        return
    for p in data["park"]:
        out = HERE / "figures" / f"fig_g5_park_rank_{p['role']}_{p['splittype']}"
        if F.park_scatter_figure(p["shared"], p["or_rank"], p["shap_rank"],
                                 p["rho"], p, out) is not None:
            print("saved", out)


if __name__ == "__main__":
    main()
