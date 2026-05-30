"""Figure C4 - flexible (loess) calibration curves at the headline cells.

Adds the moderate-calibration-level diagnostic of the Van Calster 2016
calibration hierarchy on top of the weak-calibration (slope + CITL) reporting
already in body §3.4 and the discrete-bin reliability diagram in fig_c3. The
loess smooth is fit to the per-leaf out-of-sample (y, p) pairs at the
composite-tracked migraine and headache headline cells; the patient-day
bootstrap 95% band is drawn around the smooth so a reader can see at which
predicted-probability regions the calibration is supported by data.

The predicted-probability density is overlaid on a twinned y-axis per the
pmcalplot convention so a sparse-data region does not get over-interpreted as
"calibrated" from a smooth that only had a few points to fit through.

Architecture set mirrors fig_c3 (XGBoost stack, TabPFN, window-MLP) so the
two calibration figures are read in parallel: fig_c3 shows discrete-bin
reliability with calibration slope CIs in the title, fig_c4 shows the
continuous moderate-calibration smooth that complements the discrete-bin
reliability plot.

Usage: python fig_c4_calibration_flexible.py
"""
# §11 compliance (figure_design_requirements.md, reviewer-derived 2026-05-28):
#   §11.1 CIs on headline metric:        bootstrap CI band on smooth
#   §11.3 self-contained caption:        cohort name + n + CI method in suptitle
#   §11.6 CC BY 4.0 footer:              auto-applied by S.save()
#   §11.7 EPV-5.5 annotation:            S.epv_annotation() on migraine panel
#   §11.10 no "substantial"/"large":     verified
#   §11.11 self-check:                   this block
#   §6 shared legend:                    fig.legend frameless below panels
import importlib.util as _ilu
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"
sys.path.insert(0, str(EXP))
import _style as S  # noqa: E402

WORKER = EXP / "5" / "_personal" / "_predict_worker.py"

# Loess + bootstrap parameters
LOWESS_FRAC = 0.5      # smoothness fraction; smaller -> more flex
LOWESS_IT = 0          # robust-fitting iterations; MUST be 0 for binary y,
                       # because non-zero iterations treat the 1s as outliers
                       # and downweight them, collapsing the smooth to 0
N_BOOT = 300           # bootstrap resamples for the CI band
GRID_POINTS = 80       # evaluation grid for the smooth + band


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path):
    addition = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"c4_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run(
            [sys.executable, str(WORKER), str(leaf), str(out)],
            capture_output=True, text=True, timeout=1800, env=_env(addition),
        )
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr.strip().splitlines() or ["worker failed"])[-1]
            print(f"  SKIP {leaf.name}: {tail}")
            return None
        z = np.load(out, allow_pickle=True)
        return z["y"].astype(float), z["p"].astype(float)
    finally:
        out.unlink(missing_ok=True)


def _loess_with_band(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> dict:
    """Return the loess point-estimate fit on grid + the patient-day bootstrap
    2.5/97.5 band evaluated at the same grid points."""
    smoothed = lowess(y, p, frac=LOWESS_FRAC, it=LOWESS_IT, return_sorted=True)
    smooth_grid = np.interp(grid, smoothed[:, 0], smoothed[:, 1])

    n = len(p)
    rng = np.random.default_rng(42)
    boot_curves = np.empty((N_BOOT, len(grid)))
    valid = 0
    for k in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        ps = p[idx]
        if ys.sum() == 0:
            continue
        try:
            s = lowess(ys, ps, frac=LOWESS_FRAC, it=LOWESS_IT, return_sorted=True)
            boot_curves[valid] = np.interp(grid, s[:, 0], s[:, 1])
            valid += 1
        except Exception:
            continue
    if valid == 0:
        lo = np.full_like(grid, np.nan)
        hi = np.full_like(grid, np.nan)
    else:
        lo = np.percentile(boot_curves[:valid], 2.5, axis=0)
        hi = np.percentile(boot_curves[:valid], 97.5, axis=0)
    return {"smooth": smooth_grid, "lo": lo, "hi": hi, "n_boot_valid": valid}


def _resolve_headlines() -> dict[str, list[tuple[str, Path]]]:
    """Mirror fig_c3 + run_no_aura_sensitivity: composite-tracked migraine /
    headache headline cells from the latest figdata, plus the pinned Add-4
    window-MLP leaf as the cross-architecture sequence representative."""
    spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    fp = mod.latest_figdata(EXP / "2")
    if fp is None:
        raise SystemExit("no figdata_*.json in experiment/2/; run experiment/2/compare.py first")
    print(f"  source {fp.name}")
    headlines = mod.load_figdata(fp).get("headlines", [])

    def find(target: str, family: str) -> Path | None:
        for role in ("headline", "runner_up"):
            for e in headlines:
                if (e.get("target") == target and e.get("family") == family
                        and e.get("feature_set") == "full_features"
                        and e.get("splittype") == "chrono"
                        and e.get("role") == role and e.get("leaf_dir")):
                    return Path(e["leaf_dir"])
        return None

    out: dict[str, list[tuple[str, Path]]] = {}
    for tgt in ("headache", "migraine"):
        leaves: list[tuple[str, Path]] = []
        for label, family in (("XGBoost stack", "xgboost"), ("TabPFN", "tabpfn")):
            d = find(tgt, family)
            if d is not None and (d / "model.joblib").exists():
                leaves.append((label, d))
        # Add-4 window-MLP: pinned cross-addition contrast per fig_c3.
        d4 = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
        if (d4 / "model.joblib").exists():
            leaves.append(("window-MLP", d4))
        out[tgt] = leaves
    return out


def main() -> None:
    S.apply()
    leaves = _resolve_headlines()
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.2))
    for ax, ltr in zip(axes.ravel(), "ab"):
        S.panel_label(ax, ltr)
    mark = {"XGBoost stack": "o", "TabPFN": "s", "window-MLP": "^"}

    for ax, tgt in zip(axes, ("headache", "migraine")):
        # Cache predictions once per (target, architecture) to avoid
        # double-spawning the subprocess predict worker per leaf.
        cached: list[tuple[str, np.ndarray, np.ndarray]] = []
        hi_lim = 0.0
        for label, leaf in leaves[tgt]:
            r = _predict(leaf)
            if r is None:
                continue
            y, p = r
            cached.append((label, y, p))
            p_max = float(p.max())
            grid = np.linspace(0.0, p_max, GRID_POINTS)
            band = _loess_with_band(y, p, grid)
            col = S.arch_color(label)
            ax.fill_between(grid, band["lo"], band["hi"], color=col, alpha=0.18, linewidth=0)
            ax.plot(grid, band["smooth"], color=col, lw=1.5, marker=mark.get(label, "o"),
                    markersize=3, markevery=10, label=label)
            hi_lim = max(hi_lim, p_max, float(np.nanmax(band["hi"])))
            print(f"  {tgt:<9} {label:<13} n_boot_valid {band['n_boot_valid']}/{N_BOOT}  "
                  f"smooth at p=0.10: {np.interp(0.10, grid, band['smooth']):.3f}")

        ax.plot([0, 1], [0, 1], color=S.REF_COLOR, lw=S.REF_LW, ls="--", label="perfect")
        # Calibration plot wants matched x/y range (both axes are
        # probability), not fixed aspect=equal which distorts when data
        # spans only [0, 0.3-0.5] rather than the full [0, 1].
        lim = min(1.0, hi_lim * 1.1 + 0.02)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency (loess)")
        ax.set_title(tgt)
        S.epv_annotation(ax, tgt, cell="full_features", loc="upper left")

        # Predicted-probability rug at the bottom of the panel (pmcalplot
        # convention; shows where data density supports the smooth) - small
        # vertical ticks per architecture, anchored at y=0 with axes-fraction
        # height so the rug never dominates the data region. Each architecture
        # gets its own row so the per-arch p distribution is readable.
        rug_h = 0.035  # axes-fraction height of one rug strip
        for i, (label, _y, p) in enumerate(cached):
            col = S.arch_color(label)
            counts, edges = np.histogram(p, bins=40, range=(0, lim))
            if counts.max() > 0:
                normalised = counts / counts.max()
                centres = 0.5 * (edges[:-1] + edges[1:])
                base = i * rug_h * lim
                for c, h in zip(centres, normalised):
                    if h > 0:
                        ax.plot([c, c], [base, base + h * rug_h * lim],
                                color=col, lw=1.0, alpha=0.85)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("Flexible (loess) calibration - Park 2016 SHD, n=62\n"
                 "patient-day bootstrap 95% band; per-architecture predicted-probability rug at the bottom of each panel",
                 y=1.04, fontsize=10)
    print("saved", S.save(fig, HERE / "figures" / "fig_c4_calibration_flexible"))


if __name__ == "__main__":
    main()
