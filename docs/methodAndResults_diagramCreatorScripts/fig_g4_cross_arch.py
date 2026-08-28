"""Figure G4 - cross-architecture SHAP attribution, headline vs runner-up (Addition 2).

One grouped horizontal bar per cross-family cell (XGBoost stack vs TabPFN),
comparing each model's top features as a share of its own total mean |SHAP|.
Normalised because the two explainers' absolute magnitudes are not comparable.
This script emits one figure per cross-family cell present in the frozen data.

Data source: the latest experiment/2/figdata_*.json frozen by
experiment/2/compare.py.

Usage: python fig_g4_cross_arch.py
"""
# §11 compliance: SHAP attribution shares (single-fit caveat noted in renderer).
#   §11.1 N/A (no probabilistic CI on attribution shares from one fit)
#   §11.3 caption: cohort+n in rendered title
# Paper caption (for LaTeX):
#   Cross-architecture feature attribution for one cell: the headline model vs a
#   near-tied runner-up of a different family (XGBoost stack green, TabPFN purple).
#   Each bar is a feature's share of that model's total mean |SHAP| (%), normalised
#   because the two explainers' absolute magnitudes are not comparable. These are
#   single-fit attribution shares on a small, heavily imbalanced dataset, so small
#   bar-length differences are not meaningful; rank agreement is reported separately
#   in the comparison report's overlap table.
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
    print(f"  source {figdata_path.name} ({len(data['cross_arch'])} cross-family cells)")
    if not data["cross_arch"]:
        print("  skip: no cross-family pairs in the figure data")
        return
    for c in data["cross_arch"]:
        out = (HERE / "figures"
               / f"fig_g4_cross_arch_{c['target']}_{c['feature_set']}_{c['splittype']}")
        if F.cross_arch_figure(c["headline"], c["runner"], c, out) is not None:
            print("saved", out)


if __name__ == "__main__":
    main()
