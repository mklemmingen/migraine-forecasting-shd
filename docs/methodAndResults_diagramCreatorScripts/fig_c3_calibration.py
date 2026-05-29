"""Figure C3 - calibration reliability diagrams.

Observed vs predicted next-day probability by quantile bin, overlaying the
composite-best XGBoost (Add-0), TabPFN (Add-1), and a window-MLP sequence
representative (Add-4) per target, on full_features/chrono. The Add-0 and Add-1
leaves are resolved from the latest experiment/2/figdata_*.json so they track
the composite_sorted selection that drives the headline table and g/h figures.
Add-4 is not part of composite selection (which covers Add-0 and Add-1
only); it is pinned to the ``version_window-mlp / 70_15_15 / chrono``
leaf as a deliberate cross-addition contrast. Reuses the Addition 5
prediction worker (per-addition subprocess; GPU only for TabPFN), so no
model code is re-implemented here. Points sagging below the diagonal at
high predicted risk are the over-confidence fingerprint (slope < 1).

Caveat for the §6 cohort-median-0.64 framing: the headline-leaf curves
plotted here are NOT representative of the 488-cell cohort summary.
At the headline leaves the XGB stack actually shows slope >1
(under-confident at the top) and TabPFN shows slope near 1; the
over-confidence sag the §6 grid-median captures lives in the small/
sparse non-headline cells (98 of 488 have negative slope). This figure
visualises the headline leaves; §6 visualises the cohort.

Usage: python fig_c3_calibration.py
"""
# §11 compliance (figure_design_requirements.md, reviewer-derived 2026-05-28):
#   §11.1 CIs on headline metric:        bootstrap CI on slope in suptitle
#   §11.3 self-contained caption:        cohort name + n + CI method in suptitle
#   §11.6 CC BY 4.0 footer:              auto-applied by S.save()
#   §11.7 EPV-5.5 annotation:            S.epv_annotation() on migraine panel
#   §11.10 no "substantial"/"large":     verified
#   §11.11 self-check:                   this block
#   §6 shared legend:                    fig.legend frameless below panels
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"
sys.path.insert(0, str(EXP))
from _eval.metrics_lib import calibration_slope  # noqa: E402

# Figdata loader from experiment/2 (same one g/h scripts use).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
_F = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_F)

WORKER = EXP / "5" / "_personal" / "_predict_worker.py"
N_BINS = 8


def _resolve_leaf(headlines, target, family):
    """Return the composite-tracked leaf for (target, family) in the
    full_features/chrono headline cell. Prefer headline-role, fall back to
    runner-up if the headline is the other family. Returns None when no
    matching entry has a leaf_dir (figdata predates the field)."""
    for role in ("headline", "runner_up"):
        for e in headlines:
            if (e.get("target") == target and e.get("feature_set") == "full_features"
                    and e.get("splittype") == "chrono" and e.get("role") == role
                    and e.get("family") == family and e.get("leaf_dir")):
                return Path(e["leaf_dir"])
    return None


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path):
    add = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"c3_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(WORKER), str(leaf), str(out)],
                           capture_output=True, text=True, timeout=1800, env=_env(add))
        if r.returncode != 0 or not out.exists():
            print("  skip", leaf.name, (r.stderr.strip().splitlines() or ["?"])[-1])
            return None
        z = np.load(out)
        return z["y"].astype(float), z["p"].astype(float)
    finally:
        out.unlink(missing_ok=True)


def _bootstrap_slope_ci(y, p, n_boot=500, seed=42):
    """Patient-day bootstrap CI on the calibration slope. Matches the
    body §2.8 patient-day resampling unit."""
    rng = np.random.default_rng(seed)
    n = len(y)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]; ps = p[idx]
        if 0 < ys.sum() < n:
            try:
                slopes.append(calibration_slope(ys, ps))
            except Exception:
                pass
    if not slopes:
        return float("nan"), float("nan")
    slopes = np.array(slopes)
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def _reliability(y, p, nbins=N_BINS):
    edges = np.quantile(p, np.linspace(0, 1, nbins + 1))
    edges[0] -= 1e-9
    xs, ys, ns = [], [], []
    for i in range(nbins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def _discover(headlines):
    leaves = {}
    for tgt in ("headache", "migraine"):
        ls = []
        d0 = _resolve_leaf(headlines, tgt, "xgboost")
        if d0 is not None and (d0 / "model.joblib").exists():
            ls.append(("XGBoost stack", d0))
        d1 = _resolve_leaf(headlines, tgt, "tabpfn")
        if d1 is not None and (d1 / "model.joblib").exists():
            ls.append(("TabPFN", d1))
        # Add-4 is not part of composite selection; pin a documented sequence
        # representative for the cross-architecture contrast.
        d4 = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
        if (d4 / "model.joblib").exists():
            ls.append(("window-MLP", d4))
        leaves[tgt] = ls
    return leaves


def main():
    S.apply()
    figdata_path = _F.latest_figdata(EXP / "2")
    if figdata_path is None:
        raise SystemExit("no figdata_*.json in experiment/2/ - run experiment/2/compare.py first")
    headlines = _F.load_figdata(figdata_path).get("headlines", [])
    print(f"  source {figdata_path.name} ({len(headlines)} headline rows)")
    leaves = _discover(headlines)
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.2))
    for _ax, _lt in zip(axes.ravel(), "abcdefgh"):
        S.panel_label(_ax, _lt)
    mark = {"XGBoost stack": "o", "TabPFN": "s", "window-MLP": "^"}  # CVD/grayscale reinforcement
    abbr = {"XGBoost stack": "XGB", "TabPFN": "TabP", "window-MLP": "wMLP"}
    slopes_by_target: dict[str, list[tuple[str, float, float, float]]] = {"headache": [], "migraine": []}
    for ax, tgt in zip(axes, ("headache", "migraine")):
        ax.plot([0, 1], [0, 1], color=S.REF_COLOR, lw=S.REF_LW, ls="--", label="perfect")
        hi = 0.0
        for label, leaf in leaves[tgt]:
            r = _predict(leaf)
            if r is None:
                continue
            y, p = r
            xs, ys, ns = _reliability(y, p)
            slope = calibration_slope(y, p)
            slope_lo, slope_hi = _bootstrap_slope_ci(y, p)
            sizes = 12 + 120 * ns / ns.max()
            col = S.arch_color(label)
            mk = mark.get(label, "o")
            ax.plot(xs, ys, "-", color=col, lw=1.5, marker=mk, markersize=4)
            ax.scatter(xs, ys, s=sizes, color=col, marker=mk, label=label)
            slopes_by_target[tgt].append((label, slope, slope_lo, slope_hi))
            hi = max(hi, xs.max(), ys.max())
            print(f"  {tgt:<9} {label:<13} slope {slope:+.2f} [{slope_lo:+.2f}-{slope_hi:+.2f}]  bins {len(xs)}")
        lim = min(1.0, hi * 1.1 + 0.02)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")   # honest 45-degree perfect line
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(tgt)
        S.epv_annotation(ax, tgt, cell="full_features", loc="upper left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    def _slope_line(tgt: str) -> str:
        parts = [f"{abbr.get(lab, lab)} {sl:.2f} [{lo:.2f}-{hi:.2f}]"
                 for lab, sl, lo, hi in slopes_by_target[tgt]]
        return f"{tgt}: " + "  ".join(parts)

    fig.suptitle("Reliability diagrams - Park 2016 SHD, n=62\n"
                 "bootstrap 95% CI on slope; 1:1 = perfect",
                 y=1.12, fontsize=10)
    fig.text(0.5, 1.00,
             _slope_line("headache") + "\n" + _slope_line("migraine"),
             ha="center", va="bottom", fontsize=8, color=S.GREY)
    print("saved", S.save(fig, HERE / "figures" / "fig_c3_calibration"))


if __name__ == "__main__":
    main()
