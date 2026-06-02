"""Figure G2 - precision-recall skill by split type, per cell (Addition 2).

AUPRC lift = AUPRC / test-set positive prevalence, one bar per split, reference
line at 1.0 (no skill). Lift puts the two targets on a common scale despite very
different base rates; each headline carries its own leaf's test prevalence in the
frozen data, so the plotter reads no parquet.

Data source: the latest experiment/2/figdata_*.json frozen by
experiment/2/compare.py.

Usage: python fig_g2_auprc_lift.py
"""
# §11 compliance: AUPRC lift bars with 95% CI whiskers (already present).
#   §11.1 PASS; §11.3 caption: cohort+n in rendered title
# Paper caption (for LaTeX):
#   AUPRC lift = AUPRC / test-set positive prevalence of the headline model per cell,
#   one bar per split with 95% CIs. The dotted line at 1.0 is no skill (no better than
#   predicting the base rate); lift puts headache and migraine on a common scale
#   despite very different positive rates. A hatched bar's 95% CI reaches the no-skill
#   line (not significantly above the base rate).
import sys as _sys
import importlib.util as _ilu
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
# experiment/ on the path for _style; load _figures by file path rather than
# adding experiment/2 to sys.path, where its select.py shadows the stdlib select.
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
    print(f"  source {figdata_path.name}")
    out = HERE / "figures" / "fig_g2_auprc_lift"
    if F.auprc_lift_figure(data["headlines"], out) is None:
        print("  skip: no headline cells with AUPRC + prevalence in the figure data")
    else:
        print("saved", out)


if __name__ == "__main__":
    main()
