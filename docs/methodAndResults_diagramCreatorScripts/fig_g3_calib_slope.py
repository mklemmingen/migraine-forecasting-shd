"""Figure G3 - calibration slope of the headline model, by split type (Addition 2).

One marker per split per (target, feature-set) cell against the perfect-calibration
line at 1.0, with the degenerate zones (<= 0 inverted, > 5 mis-scaled) shaded.
Calibration is the second axis of forecast quality: discrimination ranks days,
calibration scales the probabilities.

Data source: the latest experiment/2/figdata_*.json frozen by
experiment/2/compare.py.

Usage: python fig_g3_calib_slope.py
"""
# §11 compliance: calibration slope markers per cell.
#   §11.1 CIs: whiskers from calib_slope_lo/hi (figdata) via F.calib_slope_figure()
#   §11.3 caption: cohort+n in rendered title
# Paper caption (for LaTeX):
#   Calibration slope of the headline model per (target, feature-set) cell, one marker
#   per split, against the perfect-calibration line at 1.0. The shaded zones (<= 0
#   inverted, > 5 mis-scaled) are excluded from model selection. Calibration is the
#   second axis of forecast quality: discrimination ranks days, calibration scales the
#   probabilities, so a cell far from 1.0 discriminates without trustworthy
#   probabilities.
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
    print(f"  source {figdata_path.name}")
    # spano_features is the legacy 4th (Spano-2026 replication) set; it is cut
    # from the manuscript (three named sets: full / no_rolling / park), so drop
    # its rows here to keep the figure consistent with the text.
    rows = [h for h in data["headlines"] if h.get("feature_set") != "spano_features"]
    out = HERE / "figures" / "fig_g3_calib_slope"
    if F.calib_slope_figure(rows, out) is None:
        print("  skip: no headline cells with a calibration slope in the figure data")
    else:
        print("saved", out)


if __name__ == "__main__":
    main()
