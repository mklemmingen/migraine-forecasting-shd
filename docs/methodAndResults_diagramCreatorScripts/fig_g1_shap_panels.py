"""Figure 2 (working-notes ID: G1) - SHAP beeswarm panels for both targets.

Body-facing figure for the Figure 2 slot. Both panels are drawn natively here
from the leaves' frozen insights/shap_matrix_*.npz rather than composited from
the per-target PNGs, because the blind reader audit turned up four defects that
a PIL paste of two independently-rendered panels cannot fix:

  * the two panels carried different x-scales (+0.10 vs +0.07) while stacked
    vertically, so the eye compared magnitudes that were not comparable;
  * the zero reference rule was drawn over the dense marker clusters instead of
    behind them, so it read as data;
  * neither panel carried a letter, making them awkward to cite from the body;
  * every row label was a raw code identifier with underscores.

Drawing both panels inside one Figure lets them share an axis, puts the rule
behind the points, and emits vector PDF. The shared
experiment/_explain/_plots.plot_beeswarm primitive is deliberately NOT modified:
it still backs the leaf-level insight pass and fig_h2, whose per-target PNGs
remain the diagnostic rendering.

Content: panel (a) headache headline cell, panel (b) migraine headline cell.
Points are one row-by-feature SHAP value, coloured by that feature's own value
(dark low, bright high), showing the direction
and magnitude of each feature's effect on next-day positive-class probability.

Usage: python fig_g1_shap_panels.py
"""
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
OUT = FIGS / "fig_g1_shap_panels"

TOP_N = 12

# S.TARGET["migraine"] is the vermillion used for MARKS. As text on white it is
# 3.87:1 and fails WCAG 1.4.3, so the bold panel word uses the darker twin the
# m-series established, at 4.70:1. Blue already passes.
TARGET_TEXT = {"headache": S.TARGET["headache"], "migraine": "#C25100"}

# The point ramp must be unique to this figure. Blue and orange already mean
# headache and migraine (and label this figure's own panels), and Okabe-Ito is
# otherwise spent on architectures: green is XGBoost, purple TabPFN, orange
# window-MLP, skyblue GRU, grey pooled. A truncated `magma` was chosen on
# measurements rather than by eye. Against the earlier purple/green it is better
# on all three criteria that matter here: worst-case contrast on white 3.03:1
# vs 1.89:1 (WCAG 1.4.11 asks 3:1 for meaningful graphics), separation of the
# two ends under the worst of protanopia/deuteranopia/tritanopia 68.6 dE vs
# 61.7, and closest approach to any colour already used in the paper 25.7 dE vs
# 15.3. The top is truncated at 0.68 because full magma ends in a pale salmon
# that drops to 2.02:1 on white. Ordering is carried mainly by LIGHTNESS, which
# survives every colour-vision deficiency, so the ramp still reads low-to-high
# when hue information is unavailable.
VAL_RAMP = ("magma", 0.10, 0.68)   # (colormap, low cut, high cut)

# Raw column names are code identifiers. The audit's reader put it plainly:
# "this is a figure prepared for the people who built the model". These are the
# same quantities in the words the Methods section uses for them.
LABEL = {
    "migraine_rate_last3":          "migraine days, last 3",
    "migraine_rate_last7":          "migraine days, last 7",
    "headache_free_streak":         "headache-free streak",
    "days_since_last_migraine":     "days since last migraine",
    "migraine_yesterday":           "migraine yesterday",
    "inappropriate_lighting":       "harsh lighting",
    "inappropriate_lighting_today": "harsh lighting, today",
    "menstruation_today":           "menstruation, today",
    "days_since_last_record":       "days since last diary entry",
    "consecutive_sedentary_days":   "consecutive sedentary days",
    "weather_instability_3day":     "weather instability, 3 day",
    "sleep_debt_3day":              "sleep debt, 3 day",
    "sleep_variability_7day":       "sleep variability, 7 day",
    "exercise_days_7day":           "exercise days, last 7",
    "moderate_exercise_min":        "moderate exercise, minutes",
    "consecutive_trigger_days":     "consecutive trigger days",
    "dow":                          "day of week",
}


def _pretty(name: str) -> str:
    return LABEL.get(name, name.replace("_", " "))


def _load():
    """Resolve the two headline leaves and their stashed SHAP matrices."""
    _sys.path.insert(0, str(HERE))
    import fig_h2_shap_beeswarm as h2

    data = h2._F.load_figdata(h2._F.latest_figdata(h2.ADDITION2))
    out = {}
    for h in data.get("headline_explain", []):
        npzs = sorted((Path(h["leaf_dir"]) / "insights").glob("shap_matrix_*.npz"))
        if not npzs:
            continue
        z = np.load(npzs[-1], allow_pickle=True)
        out[h["target"]] = (
            [str(f) for f in z["feature_names"]], z["shap"], z["values"],
        )
    return out


def _panel(ax, names, matrix, values, letter, target, xlim, cmap):
    """One per-row beeswarm. Returns the scatter for the shared colour bar."""
    mean_abs = np.abs(matrix).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:TOP_N][::-1]

    # The rule goes down FIRST and stays behind every marker. Drawn over the
    # clusters it read as data rather than as the no-effect reference.
    ax.axvline(0.0, color="#9a9a9a", lw=0.9, zorder=1)
    for row in range(len(order)):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color="#f4f4f4", zorder=0)

    rng = np.random.default_rng(0)
    sc = None
    for row, j in enumerate(order):
        sv = matrix[:, j]
        fv = values[:, j].astype(float)
        if np.isfinite(fv).any():
            lo, hi = np.nanpercentile(fv, [2, 98])
        else:
            lo, hi = 0.0, 1.0
        norm = np.clip((fv - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        jitter = rng.uniform(-0.17, 0.17, size=len(sv))
        sc = ax.scatter(sv, np.full(len(sv), row) + jitter, c=norm, cmap=cmap,
                        vmin=0.0, vmax=1.0, s=9, alpha=0.65,
                        edgecolors="none", zorder=3)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_pretty(names[j]) for j in order], fontsize=7.6)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(*xlim)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Panel letter plus the target in words, and the row count the reader could
    # not recover: the two panels rest on very different numbers of rows.
    ax.text(0.0, 1.045, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")
    ax.text(0.105, 1.045, target.upper(), transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left",
            color=TARGET_TEXT[target])
    ax.text(1.0, 1.045, f"{matrix.shape[0]} scored days", transform=ax.transAxes,
            fontsize=7.4, va="bottom", ha="right", color="#7a7a7a")
    return sc


def main() -> None:
    S.apply()
    loaded = _load()
    missing = [t for t in ("headache", "migraine") if t not in loaded]
    if missing:
        raise SystemExit(
            f"no stashed shap_matrix_*.npz for {', '.join(missing)}; re-run the "
            f"insight pass for that leaf"
        )

    # One shared x-scale across both panels. The panels are stacked and were
    # previously drawn at +0.10 and +0.07, which invited exactly the
    # cross-panel magnitude comparison the mismatch invalidated. The extreme
    # values are NOT clipped: a handful of large attributions is a real
    # property of these cells, not chart junk to be tidied away.
    lo_v, hi_v = 0.0, 0.0
    for names, m, _v in loaded.values():
        top = np.argsort(np.abs(m).mean(axis=0))[::-1][:TOP_N]
        lo_v = min(lo_v, float(m[:, top].min()))
        hi_v = max(hi_v, float(m[:, top].max()))
    pad = (hi_v - lo_v) * 0.04
    xlim = (lo_v - pad, hi_v + pad)

    _base = plt.get_cmap(VAL_RAMP[0])
    cmap = LinearSegmentedColormap.from_list(
        "shd_featvalue",
        [_base(x) for x in np.linspace(VAL_RAMP[1], VAL_RAMP[2], 32)])

    # article.tex's text block is 372pt = 5.15in wide, NOT the 183mm S.COL_DOUBLE
    # assumes. A 7.2in canvas placed at width=\textwidth is downscaled to 0.72,
    # which is what dropped this figure's row labels to about 5pt on the page and
    # drew the "cannot be printed as laid out" objection. Drawing at the real text
    # width means the point sizes below are the point sizes that reach the reader.
    fig, axes = plt.subplots(
        2, 1, figsize=(4.25, 6.0), sharex=True,
        gridspec_kw={"hspace": 0.26},
    )
    sc = None
    for ax, (letter, tgt) in zip(axes, (("a", "headache"), ("b", "migraine"))):
        names, m, v = loaded[tgt]
        sc = _panel(ax, names, m, v, letter, tgt, xlim, cmap)

    axes[-1].set_xlabel(
        "SHAP value: how much this feature moved that day's predicted "
        "probability\n(left = pushed the prediction down, right = pushed it up)")

    cbar = fig.colorbar(sc, ax=axes, fraction=0.022, pad=0.015)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["low", "high"])
    cbar.set_label("feature value\n(scaled within each row)", fontsize=7.4)
    cbar.ax.tick_params(labelsize=7.2)
    cbar.outline.set_visible(False)

    print("saved", S.save(fig, OUT))
    plt.close(fig)


if __name__ == "__main__":
    main()
