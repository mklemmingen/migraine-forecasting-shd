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

from _style import apply, save, cc_by_footer, OI, ARCH, SPLIT, GREY, SOFT, FAINT, MUTED, INK, leaf_slug


def _arch_var(leaf_dir):
    """Extract the ARCH-VAR token (first segment) from the canonical 4-token
    slug. Returns "?" if leaf_dir is missing or unparseable."""
    if not leaf_dir:
        return "?"
    s = leaf_slug(leaf_dir)
    return s.split(" / ")[0] if s else "?"

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
        ys, vals, los, his, subchance, slugs = [], [], [], [], [], []
        for ci, key in enumerate(cells):
            s = by_cell[key].get(st)
            if s is None:
                continue
            ys.append(base[ci] + (g / 2 - gi - 0.5) * bh)
            vals.append(s["auroc_mean"])
            los.append(s["auroc_mean"] - s["auroc_lo"])
            his.append(s["auroc_hi"] - s["auroc_mean"])
            subchance.append(s["auroc_lo"] <= 0.5)
            slugs.append(_arch_var(s.get("leaf_dir")))
        bars = ax.barh(ys, vals, height=bh, color=SPLIT[st], alpha=0.85,
                       xerr=[los, his],
                       error_kw={"elinewidth": 0.8, "capsize": 2},
                       label=SPLIT_FIG_LABELS[st])
        # Per-bar slug annotation: each (target, feature_set, split) headline
        # may decode to a different ARCH-VAR (e.g. TabPFN-v2.6 vs v3-default)
        # depending on composite-rule selection. Surface the variant inline.
        for y_val, x_val, hi, slug in zip(ys, vals, his, slugs):
            ax.text(x_val + hi + 0.008, y_val, slug,
                    fontsize=6, va="center", color=INK)
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
    ax.set_xlabel("best hold-out AUROC (95% CI)", fontsize=9)
    ax.set_title("Discrimination by split type, per cell (headline model) · Park 2016 SHD (n=62)",
                 fontsize=10)
    cc_by_footer(fig)
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
    any_noskill = False
    for gi, st in enumerate(SPLIT_ORDER):
        ys, vals, los, his, noskill, slugs = [], [], [], [], [], []
        for ci, key in enumerate(cells):
            sp = by_cell[key].get(st)
            if sp is None:
                continue
            s, prev = sp
            lift = s["auprc_mean"] / prev
            lo = (s["auprc_mean"] - s["auprc_lo"]) / prev if s.get("auprc_lo") else 0.0
            hi = (s["auprc_hi"] - s["auprc_mean"]) / prev if s.get("auprc_hi") else 0.0
            ys.append(base[ci] + (g / 2 - gi - 0.5) * bh)
            vals.append(lift); los.append(lo); his.append(hi)
            noskill.append(lift - lo <= 1.0)   # 95% CI reaches the no-skill line
            slugs.append(_arch_var(s.get("leaf_dir")))
            xmax = max(xmax, lift + hi)
        bars = ax.barh(ys, vals, height=bh, color=SPLIT[st],
                       xerr=[los, his], error_kw={"elinewidth": 0.8, "capsize": 2},
                       label=SPLIT_FIG_LABELS[st])
        # Per-bar slug annotation: surface the ARCH-VAR of each cell's headline
        # so a reader can tell when TabPFN-v2.6 vs v3-default is the chosen leaf.
        for y_val, x_val, hi, slug in zip(ys, vals, his, slugs):
            ax.text(x_val + hi + xmax * 0.01, y_val, slug,
                    fontsize=6, va="center", color=INK)
        # Hatch bars whose CI reaches no-skill: not significantly above the base
        # rate (same honesty convention as the split-AUROC chance hatching).
        for patch, ns in zip(bars.patches, noskill):
            if ns:
                patch.set_hatch("////")
                patch.set_edgecolor("white")
                any_noskill = True
    ax.axvline(1.0, color=SOFT, lw=0.9, ls=":", zorder=0)
    ax.set_yticks(base)
    ax.set_yticklabels([_cell_label(t, fs) for t, fs in cells], fontsize=8)
    ax.set_xlim(0, xmax * 1.05)
    ax.set_xlabel("AUPRC lift over no-skill baseline (AUPRC / test prevalence)",
                  fontsize=9)
    ax.set_title("Precision-recall skill by split type, per cell (headline) · Park 2016 SHD (n=62)",
                 fontsize=10)
    handles, _labels = ax.get_legend_handles_labels()
    if any_noskill:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor=FAINT, hatch="////", edgecolor="white",
                             label="CI reaches no-skill (ns)"))
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    cc_by_footer(fig)
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
        by_cell[key][s["splittype"]] = s
    if not cells:
        return None
    cells.sort()
    apply()
    n = len(cells)
    base = np.arange(n)[::-1]
    vals = [s_data["calib_slope"] for c in by_cell.values() for s_data in c.values()]
    xmax = max(2.2, max(vals) + 0.3)
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * n + 1.4))
    ax.axvspan(xmax * -0.02, 0.0, color=OI["vermillion"], alpha=0.10, zorder=0)
    ax.axvspan(5.0, xmax, color=OI["vermillion"], alpha=0.10, zorder=0)
    ax.axvline(1.0, color=SOFT, lw=1.0, ls="--", zorder=1,
               label="perfect calibration (1.0)")
    for gi, st in enumerate(SPLIT_ORDER):
        ys = [base[ci] + (1 - gi) * 0.18 for ci, k in enumerate(cells) if st in by_cell[k]]
        xs = [by_cell[k][st]["calib_slope"] for k in cells if st in by_cell[k]]
        x_lo = [by_cell[k][st].get("calib_slope_lo") for k in cells if st in by_cell[k]]
        x_hi = [by_cell[k][st].get("calib_slope_hi") for k in cells if st in by_cell[k]]
        slugs = [_arch_var(by_cell[k][st].get("leaf_dir"))
                 for k in cells if st in by_cell[k]]
        # Per-marker CI whiskers from the per-leaf bootstrap (parsed from
        # holdout_Calibration Slope `mean [lo - hi]`).
        for x_val, y_val, lo, hi in zip(xs, ys, x_lo, x_hi):
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [y_val, y_val], color=SPLIT[st], lw=1.0,
                        alpha=0.4, zorder=2)
        ax.scatter(xs, ys, s=55, color=SPLIT[st], zorder=3,
                   edgecolor="white", label=SPLIT_FIG_LABELS[st])
        # Per-marker slug annotation: each (target, feature_set, split) headline
        # may decode to a different ARCH-VAR. Surface the variant inline.
        for x_val, y_val, slug in zip(xs, ys, slugs):
            ax.text(x_val + xmax * 0.012, y_val, slug,
                    fontsize=6, va="center", color=INK)
    ax.set_yticks(base)
    ax.set_yticklabels([_cell_label(t, fs) for t, fs in cells], fontsize=8)
    ax.set_xlim(xmax * -0.02, xmax)
    ax.set_xlabel("calibration slope (shaded zones = excluded from selection)",
                  fontsize=9)
    ax.set_title("Calibration of the headline model, by split type\n"
                 "Park 2016 SHD, n=62; whiskers = per-leaf bootstrap 95% CI on slope",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", ncol=1, frameon=False)
    fig.text(0.99, 0.005, "CC BY 4.0", ha="right", va="bottom",
             fontsize=6.5, color=GREY, alpha=0.7)
    cc_by_footer(fig)
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
    # Colour each bar by its architecture family (xgboost green / tabpfn purple),
    # not the headline/runner-up role, so the hue matches the model's canonical
    # colour everywhere else in the paper rather than a generic blue/orange.
    ax.barh(y + bw / 2, [h_share.get(f, 0.0) for f in feats], height=bw,
            color=ARCH.get(h["arch_family"], OI["blue"]),
            label=f"headline ({h['arch_family']})")
    ax.barh(y - bw / 2, [r_share.get(f, 0.0) for f in feats], height=bw,
            color=ARCH.get(r["arch_family"], OI["orange"]),
            label=f"runner-up ({r['arch_family']})")
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("relative attribution: share of each model's total mean |SHAP| (%)")
    # Title names the cell, headline architecture family, and runner-up family.
    # The figdata cross_arch block does not carry the per-variant leaf_dir so
    # we name the families directly (legend below adds the role).
    ax.set_title(
        f"{sel['target']} / {sel['feature_set']} / {sel['splittype']} · Park 2016 SHD (n=62)\n"
        f"headline: {h.get('arch_family', '?')}  |  "
        f"runner-up: {r.get('arch_family', '?')}",
        fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    # Integrity caveat: these are single-fit shares on a small, imbalanced dataset.
    ax.text(0.0, -0.16,
            "Single-fit attribution shares; small bar-length differences are not significant.",
            transform=ax.transAxes, ha="left", fontsize=7.5, style="italic", color=GREY)
    cc_by_footer(fig)
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
    # Points in this model's canonical family colour (the title names the family),
    # so the scatter is colour-consistent with the rest of the paper.
    ax.scatter(xs, ys, s=70, color=ARCH.get(sel.get("family"), OI["blue"]),
               zorder=3, edgecolor="white")
    for f, x, y in zip(shared, xs, ys):
        ax.annotate(f.replace("_today", ""), (x, y), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_xlabel("Park 2016 odds-ratio rank (1 = strongest trigger)")
    ax.set_ylabel("model mean |SHAP| rank (1 = most weighted)")
    # Report p, n, and Fisher-z 95% CI with rho: on ~6 shared triggers any
    # coefficient is highly uncertain, so a bare rho would read as far more
    # conclusive than it is.
    p = sel.get("p")
    # Fisher z-transform CI (Bonett-Wright 2000 standard form). At n=6 this is
    # asymptotic — surfaced anyway so the reader sees the uncertainty width.
    if n > 3 and -1 < rho < 1:
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(n - 3)
        rho_lo = float(np.tanh(z - 1.96 * se))
        rho_hi = float(np.tanh(z + 1.96 * se))
        ci_str = f", 95% CI [{rho_lo:+.2f}, {rho_hi:+.2f}]"
    else:
        ci_str = ""
    stat = (f"rho = {rho:+.2f}"
            + (f", p = {p:.2f}, n = {n}" if p is not None else f", n = {n}")
            + ci_str)
    # Title names the role + architecture family (figdata park block does not
    # carry the per-variant leaf_dir; the family identifies the model class).
    arch = sel.get("architecture") or sel.get("family") or "?"
    ax.set_title(f"{sel['role']} {arch} - Spearman {stat} · Park 2016 SHD (n=62)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    cc_by_footer(fig)
    save(fig, out_png)
    plt.close(fig)
    return out_png
