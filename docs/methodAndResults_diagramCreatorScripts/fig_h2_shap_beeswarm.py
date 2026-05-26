"""Figure H2 - SHAP beeswarm of the headline model (Addition 2 / explainability).

Per-row SHAP beeswarm for the headline model's top features on the deployable cell
(full_features / chronological) of each target: each point is one patient-day, its
x-position the feature's SHAP value (impact on next-day positive-class probability)
and its colour the feature's own value (low blue -> high vermillion), so the
direction of each feature's effect is visible, not just its magnitude.

Data source: the leaf's frozen insights/shap_matrix_*.npz (the per-row SHAP matrix
and feature values), located via the headline leaf_dir in the latest
experiment/2/figdata_*.json. The matrix is stashed by the insight pass
(_explain.emit_insights); leaves insighted before that stash was added carry only
the rendered PNG, so this script skips them until the insight pass is re-run.

Usage: python fig_h2_shap_beeswarm.py
"""
# Paper caption (for LaTeX):
#   SHAP beeswarm of the headline model on the deployable cell (full_features,
#   chronological split). Each point is one hold-out patient-day; its x-position is
#   the feature's SHAP value (impact on next-day positive-class probability) and its
#   colour is the feature's own value (low blue, high vermillion), showing the
#   direction of each effect. A title flag marks any cell whose AUROC CI reaches
#   chance. Single test-split attributions; spread reflects per-day variability.
import sys as _sys
import importlib.util as _ilu
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
_sys.path.insert(0, str(_EXP))
import numpy as np
from _explain import _plots
_spec = _ilu.spec_from_file_location("_exp2_figures", _EXP / "2" / "_figures.py")
_F = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_F)

HERE = _Path(__file__).resolve().parent
ADDITION2 = _EXP / "2"


def main():
    figdata_path = _F.latest_figdata(ADDITION2)
    if figdata_path is None:
        raise SystemExit("no figdata_*.json in experiment/2/ - run experiment/2/compare.py first")
    data = _F.load_figdata(figdata_path)
    headlines = data.get("headline_explain", [])
    if not headlines:
        print("  no headline_explain in figdata - run compare.py after this update")
        return
    print(f"  source {figdata_path.name} ({len(headlines)} headline cells)")
    for h in headlines:
        npzs = sorted((_Path(h["leaf_dir"]) / "insights").glob("shap_matrix_*.npz"))
        if not npzs:
            print(f"  skip {h['target']}: no shap_matrix_*.npz yet "
                  "(re-run the insight pass for this leaf to stash the SHAP matrix)")
            continue
        z = np.load(npzs[-1], allow_pickle=True)
        feature_names = [str(f) for f in z["feature_names"]]
        lo = h.get("auroc_lo")
        title = f"{h['target']} headline ({h['family']}) - per-row SHAP"
        if lo is not None and lo <= 0.5:
            title += "  [AUROC CI reaches chance: descriptive only]"
        out = HERE / "figures" / f"fig_h2_shap_beeswarm_{h['target']}"
        _plots.plot_beeswarm(feature_names, z["shap"], z["values"], title, out)
        print("saved", out)


if __name__ == "__main__":
    main()
