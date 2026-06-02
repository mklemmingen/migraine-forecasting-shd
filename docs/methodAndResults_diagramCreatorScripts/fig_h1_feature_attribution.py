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
# §11 compliance: SHAP feature attribution bars for the headline cells.
#   §11.1 CIs: per-bar row-bootstrap 95% CI from the leaf's frozen shap_matrix npz
#   §11.3 caption: cohort+n in rendered title
#   §11.10 no banned adjectives; §11.11 self-check this block
# Paper caption (for LaTeX):
#   Mean |SHAP| feature attribution of the headline model on the deployable cell
#   (full_features, chronological split). Bars are the average absolute SHAP value
#   per feature on the hold-out test set (KernelSHAP for the XGBoost stack, the
#   TabPFN-native explainer for TabPFN). A title flag marks any cell whose AUROC CI
#   reaches chance, where the attribution explains a near-random model.
import sys as _sys
import importlib.util as _ilu
from pathlib import Path as _Path
import numpy as _np
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
_sys.path.insert(0, str(_EXP))
from _explain import _plots
_spec = _ilu.spec_from_file_location("_exp2_figures", _EXP / "2" / "_figures.py")
_F = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_F)

HERE = _Path(__file__).resolve().parent
ADDITION2 = _EXP / "2"


def _latest_shap_matrix(leaf_dir):
    insights = _Path(leaf_dir) / "insights"
    if not insights.is_dir():
        return None
    candidates = sorted(insights.glob("shap_matrix_*.npz"))
    return candidates[-1] if candidates else None


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
        npz_path = _latest_shap_matrix(h["leaf_dir"])
        ci = None
        if npz_path is not None:
            z = _np.load(npz_path, allow_pickle=True)
            ci = _plots.attribution_row_bootstrap_ci(
                z["shap"], z["feature_names"], n_boot=500, seed=42)
            title += f"\nrow-bootstrap 95% CI (n_boot=500, test rows n={z['shap'].shape[0]})"
        else:
            print(f"  WARN no shap_matrix_*.npz under {h['leaf_dir']}/insights/ - rendering without CI")
        out = HERE / "figures" / f"fig_h1_feature_attribution_{h['target']}"
        _plots.plot_attribution_bar(ranking, h.get("metric", "mean |SHAP|"), title, out, ci=ci)
        print("saved", out)


if __name__ == "__main__":
    main()
