"""Addition 2 figure generators (shared).

The five cross-leaf comparison figures, factored out of compare.py so both the
report driver (experiment/2/compare.py) and the per-figure scripts in
docs/methodAndResults_diagramCreatorScripts/ render them from one source. Each
plotter is a pure function of figure-ready data: compare.py freezes that data to
experiment/2/figdata_<ts>.json (rotated into experiment/2/results/<date>/), and
the docs wrappers read it back, so a paper figure pins to one run rather than to
the live selection state.

Colours route through _style (canonical split hues + Okabe-Ito); the plotters
never read parquet or walk results - prevalence and parsed SHAP rankings arrive
already in the data.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import apply, save, OI, SPLIT, GREY, SOFT, FAINT, MUTED

# Split-type hue from the canonical SPLIT palette (chrono blue / stratified
# vermillion / patient green), plus the one-line honest/leaky/generalisation tag.
SPLIT_ORDER = ("chrono", "stratified", "patient")
SPLIT_FIG_LABELS = {
    "chrono":     "chronological (honest)",
    "stratified": "stratified (leaky)",
    "patient":    "patient (generalisation)",
}


def latest_figdata(addition2_dir):
    """Newest ``figdata_*.json`` in experiment/2/, or None when no run has
    frozen one yet. Filenames embed a sortable ``YYYYmmddTHHMMSS`` stamp."""
    files = sorted(Path(addition2_dir).glob("figdata_*.json"))
    return files[-1] if files else None


def load_figdata(path):
    """Parse a frozen Addition-2 figure-data JSON into its dict."""
    return json.loads(Path(path).read_text())


def _cell_label(target, fset):
    return f"{target}\n{fset.replace('_features', '')}"


def split_auroc_figure(headlines: list[dict], out_png) -> Path | None:
    """Grouped horizontal bar of the best (headline) hold-out AUROC per
    (target, feature_set) cell, one bar per split type with 95% CI whiskers.

    The chronological bar is the deployable forecast; stratified exposes the
    leakage inflation carried by history / rolling features; patient is
    generalisation to unseen patients. A bar whose 95% CI reaches chance is
    hatched (not significantly forecastable).
    """
    cells, by_cell = [], {}
    for s in headlines:
        if s.get("role") != "headline":
            continue
        key = (s["target"], s["feature_set"])
        if key not in by_cell:
            by_cell[key] = {}
            cells.append(key)
        by_cell[key][s["splittype"]] = s
    if not cells:
        return None
    cells.sort()
    apply()
    n, g = len(cells), len(SPLIT_ORDER)
    bh = 0.8 / g
    base = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * n * g / 2 + 1.4))
    any_subchance = False
    for gi, st in enumerate(SPLIT_ORDER):
        ys, vals, los, his, subchance = [], [], [], [], []
        for ci, key in enumerate(cells):
            s = by_cell[key].get(st)
            if s is None:
                continue
            ys.append(base[ci] + (g / 2 - gi - 0.5) * bh)
            vals.append(s["auroc_mean"])
            los.append(s["auroc_mean"] - s["auroc_lo"])
            his.append(s["auroc_hi"] - s["auroc_mean"])
            subchance.append(s["auroc_lo"] <= 0.5)
        bars = ax.barh(ys, vals, height=bh, color=SPLIT[st],
                       xerr=[los, his],
                       error_kw={"elinewidth": 0.8, "capsize": 2},
                       label=SPLIT_FIG_LABELS[st])
        for patch, sc in zip(bars.patches, subchance):
            if sc:
                patch.set_hatch("////")
                patch.set_edgecolor("white")
                any_subchance = True
    ax.axvline(0.5, color=SOFT, lw=0.9, ls=":", zorder=0)
    handles, _labels = ax.get_legend_handles_labels()
    if any_subchance:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor=FAINT, hatch="////",
                             edgecolor="white", label="CI reaches chance (ns)"))
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    ax.set_yticks(base)
    ax.set_yticklabels([_cell_label(t, fs) for t, fs in cells], fontsize=8)
    ax.set_xlim(0.45, max(0.95, max(s["auroc_hi"] for c in by_cell.values()
                                    for s in c.values()) + 0.03))
    ax.set_xlabel("best hold-out AUROC (95% CI); dotted line = chance (0.5); "
                  "hatched = CI reaches chance", fontsize=9)
    ax.set_title("Discrimination by split type, per cell (headline model)",
                 fontsize=10)
    save(fig, out_png)
    plt.close(fig)
    return out_png


def auprc_lift_figure(headlines: list[dict], out_png) -> Path | None:
    """Grouped horizontal bars of AUPRC lift over the no-skill baseline per
    cell, one bar per split, reference line at 1.0 (no skill).

    Lift = AUPRC / test-set prevalence puts both targets on a common scale
    (headache prevalence far exceeds migraine's). Each entry carries its own
    leaf's test prevalence in ``prevalence`` so the plotter reads no parquet.
    """
    cells, by_cell = [], {}
    for s in headlines:
        if s.get("role") != "headline" or s.get("auprc_mean") is None:
            continue
        prev = s.get("prevalence")
        if not prev:
            continue
        key = (s["target"], s["feature_set"])
        if key not in by_cell:
            by_cell[key] = {}
            cells.append(key)
        by_cell[key][s["splittype"]] = (s, prev)
    if not cells:
        return None
    cells.sort()
    apply()
    n, g = len(cells), len(SPLIT_ORDER)
    bh = 0.8 / g
    base = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * n * g / 2 + 1.4))
    xmax = 1.0
    for gi, st in enumerate(SPLIT_ORDER):
        ys, vals, los, his = [], [], [], []
        for ci, key in enumerate(cells):
            sp = by_cell[key].get(st)
            if sp is None:
                continue
            s, prev = sp
            lift = s["auprc_mean"] / prev
            ys.append(base[ci] + (g / 2 - gi - 0.5) * bh)
            vals.append(lift)
            los.append((s["auprc_mean"] - s["auprc_lo"]) / prev if s.get("auprc_lo") else 0.0)
            his.append((s["auprc_hi"] - s["auprc_mean"]) / prev if s.get("auprc_hi") else 0.0)
            xmax = max(xmax, lift + his[-1])
        ax.barh(ys, vals, height=bh, color=SPLIT[st],
                xerr=[los, his], error_kw={"elinewidth": 0.8, "capsize": 2},
                label=SPLIT_FIG_LABELS[st])
    ax.axvline(1.0, color=SOFT, lw=0.9, ls=":", zorder=0)
    ax.set_yticks(base)
    ax.set_yticklabels([_cell_label(t, fs) for t, fs in cells], fontsize=8)
    ax.set_xlim(0, xmax * 1.05)
    ax.set_xlabel("AUPRC lift over no-skill baseline (AUPRC / test prevalence; "
                  "dotted line = 1.0 = no skill)", fontsize=9)
    ax.set_title("Precision-recall skill by split type, per cell (headline)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out_png)
    plt.close(fig)
    return out_png


def calib_slope_figure(headlines: list[dict], out_png) -> Path | None:
    """Dot plot of the headline calibration slope per cell, one marker per
    split, against the perfect-calibration line at 1.0 with the degenerate
    zones (<= 0 inverted, > 5 mis-scaled) shaded.
    """
    cells, by_cell = [], {}
    for s in headlines:
        if s.get("role") != "headline" or s.get("calib_slope") is None:
            continue
        key = (s["target"], s["feature_set"])
        if key not in by_cell:
            by_cell[key] = {}
            cells.append(key)
        by_cell[key][s["splittype"]] = s["calib_slope"]
    if not cells:
        return None
    cells.sort()
    apply()
    n = len(cells)
    base = np.arange(n)[::-1]
    vals = [v for c in by_cell.values() for v in c.values()]
    xmax = max(2.2, max(vals) + 0.3)
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * n + 1.4))
    ax.axvspan(xmax * -0.02, 0.0, color=OI["vermillion"], alpha=0.10, zorder=0)
    ax.axvspan(5.0, xmax, color=OI["vermillion"], alpha=0.10, zorder=0)
    ax.axvline(1.0, color=SOFT, lw=1.0, ls="--", zorder=1,
               label="perfect calibration (1.0)")
    for gi, st in enumerate(SPLIT_ORDER):
        ys = [base[ci] + (1 - gi) * 0.18 for ci, k in enumerate(cells) if st in by_cell[k]]
        xs = [by_cell[k][st] for k in cells if st in by_cell[k]]
        ax.scatter(xs, ys, s=55, color=SPLIT[st], zorder=3,
                   edgecolor="white", label=SPLIT_FIG_LABELS[st])
    ax.set_yticks(base)
    ax.set_yticklabels([_cell_label(t, fs) for t, fs in cells], fontsize=8)
    ax.set_xlim(xmax * -0.02, xmax)
    ax.set_xlabel("calibration slope (1.0 = perfect; shaded zones excluded "
                  "from selection)", fontsize=9)
    ax.set_title("Calibration of the headline model, by split type", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", ncol=1, framealpha=0.95)
    save(fig, out_png)
    plt.close(fig)
    return out_png


def cross_arch_figure(h, r, sel, out_png, top_n: int = 8) -> Path | None:
    """Grouped horizontal bar comparing two architectures' relative feature
    attribution (headline vs runner-up) for one cross-family cell.

    Each model's attribution is normalised to its share of that model's total
    mean |SHAP| (%), because the two explainers' absolute magnitudes are not
    comparable; rank agreement is reported in the report's overlap table.
    """
    h_map, r_map = dict(h["ranking"]), dict(r["ranking"])
    h_total = sum(abs(v) for v in h_map.values()) or 1.0
    r_total = sum(abs(v) for v in r_map.values()) or 1.0
    h_share = {f: 100.0 * abs(v) / h_total for f, v in h_map.items()}
    r_share = {f: 100.0 * abs(v) / r_total for f, v in r_map.items()}
    feats: list[str] = []
    for f, _ in list(h["ranking"][:top_n]) + list(r["ranking"][:top_n]):
        if f not in feats:
            feats.append(f)
    feats.sort(key=lambda f: max(h_share.get(f, 0.0), r_share.get(f, 0.0)), reverse=True)
    feats = feats[:12]
    if not feats:
        return None
    apply()
    y = np.arange(len(feats))[::-1]
    bw = 0.4
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(feats) + 1.3))
    ax.barh(y + bw / 2, [h_share.get(f, 0.0) for f in feats], height=bw,
            color=OI["blue"], label=f"headline ({h['arch_family']})")
    ax.barh(y - bw / 2, [r_share.get(f, 0.0) for f in feats], height=bw,
            color=OI["orange"], label=f"runner-up ({r['arch_family']})")
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("relative attribution: share of each model's total mean |SHAP| (%)")
    ax.set_title(f"{sel['target']} / {sel['feature_set']} - {sel['splittype']}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, out_png)
    plt.close(fig)
    return out_png


def park_scatter_figure(shared, or_rank, shap_rank, rho, sel, out_png) -> Path | None:
    """Scatter of Park-2016 odds-ratio rank (x) against the model's mean
    |SHAP| rank (y) for the shared triggers, with the agreement diagonal.
    """
    if len(shared) < 3:
        return None
    apply()
    n = len(shared)
    xs = [or_rank[f] for f in shared]
    ys = [shap_rank[f] for f in shared]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([1, n], [1, n], color=MUTED, lw=1.0, ls="--", zorder=1,
            label="perfect agreement")
    ax.scatter(xs, ys, s=70, color=OI["blue"], zorder=3, edgecolor="white")
    for f, x, y in zip(shared, xs, ys):
        ax.annotate(f.replace("_today", ""), (x, y), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_xlabel("Park 2016 odds-ratio rank (1 = strongest trigger)")
    ax.set_ylabel("model mean |SHAP| rank (1 = most weighted)")
    ax.set_title(f"{sel['role']} {sel['family']} - Spearman rho = {rho:+.2f}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, out_png)
    plt.close(fig)
    return out_png
