"""Figure H1 - feature attribution of the headline model (Addition 2 / explainability).

Horizontal bar of the headline model's mean |SHAP| per feature, for the deployable
cell (full_features / chronological) of each target. The headline is the model the
benchmark reports as best for that cell; XGBoost-stack attributions are KernelSHAP
over the calibrated probability, TabPFN's are its native explainer. When the cell's
AUROC 95% CI reaches chance the title flags it: the attribution then describes a
near-chance decision surface and is descriptive only.

Data source: the latest experiment/2/figdata_*.json frozen by compare.py (its
headline_explain section carries the ranking + AUROC CI), so no SHAP recompute is
needed for this figure.

Usage: python fig_h1_feature_attribution.py
"""
# Paper caption (for LaTeX):
#   Mean |SHAP| feature attribution of the headline model on the deployable cell
#   (full_features, chronological split). Bars are the average absolute SHAP value
#   per feature on the hold-out test set (KernelSHAP for the XGBoost stack, the
#   TabPFN-native explainer for TabPFN). A title flag marks any cell whose AUROC CI
#   reaches chance, where the attribution explains a near-random model.
import sys as _sys
import importlib.util as _ilu
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
_sys.path.insert(0, str(_EXP))
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
    from _style import leaf_slug  # noqa: E402
    for h in headlines:
        ranking = [tuple(t) for t in h["ranking"]]
        lo = h.get("auroc_lo")
        title = f"{h['target']} headline ({leaf_slug(h['leaf_dir'])}) - feature attribution"
        if lo is not None and lo <= 0.5:
            title += "  [AUROC CI reaches chance: descriptive only]"
        out = HERE / "figures" / f"fig_h1_feature_attribution_{h['target']}"
        _plots.plot_attribution_bar(ranking, h.get("metric", "mean |SHAP|"), title, out)
        print("saved", out)


if __name__ == "__main__":
    main()
