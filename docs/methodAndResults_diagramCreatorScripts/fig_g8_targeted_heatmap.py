"""Figure G8 — targeted-hypothesis-test AUROC heatmap, with the
main-text BH-FDR-significant pair from §3a overlaid.

Companion to ``fig_g7_significance_heatmap.py`` (which carries the
exhaustive Bonferroni overlay over the 487-test all-pairs family at
ratio 70_30). G8 carries the *targeted* main-text overlay: the
18-test BH-FDR family curated to address load-bearing paper claims
(cross-family headline-vs-runner-up, within-family-tie at the
headache full chrono cell, HP-ladder, AutoTabPFN-vs-XGB). Same
AUROC-landscape backdrop as G7 for visual continuity.

Why two figures rather than one with both overlays:
  - Pre-registered separate test families have separate multiplicity
    bounds (BH-FDR over m=18 here; Bonferroni FWER over m=487 in G7).
  - One overlay per figure makes the reader's job easier and
    forecloses the misread that the two families have the same
    correction.

The figure consumes the most recent
``experiment/_eval/paired_delong_*.json`` (written by
``experiment/_eval/run_paired_delong.py``).

Usage: python fig_g8_targeted_heatmap.py
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_FIG_DIR = _THIS.parent / "figures"
_FIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(_REPO / "experiment"))
import _style as S  # noqa: E402


CELLS = [
    ("headache", "full_features",       "chrono"),
    ("headache", "full_features",       "stratified"),
    ("headache", "full_features",       "patient"),
    ("headache", "spano_features",      "chrono"),
    ("headache", "spano_features",      "stratified"),
    ("headache", "spano_features",      "patient"),
    ("headache", "no_rolling_features", "chrono"),
    ("headache", "no_rolling_features", "stratified"),
    ("headache", "no_rolling_features", "patient"),
    ("migraine", "full_features",       "chrono"),
    ("migraine", "full_features",       "stratified"),
    ("migraine", "full_features",       "patient"),
    ("migraine", "spano_features",      "chrono"),
    ("migraine", "spano_features",      "stratified"),
    ("migraine", "spano_features",      "patient"),
    ("migraine", "no_rolling_features", "chrono"),
    ("migraine", "no_rolling_features", "stratified"),
    ("migraine", "no_rolling_features", "patient"),
    ("migraine", "park_features",       "chrono"),
    ("migraine", "park_features",       "stratified"),
    ("migraine", "park_features",       "patient"),
]

ARCHS = [
    "xgb_NonHP",
    "xgb_HP020", "xgb_HP050", "xgb_HP100", "xgb_HP200", "xgb_HP500",
    "tabpfn_2-5-real", "tabpfn_2-5-finetuned", "tabpfn_2-6",
    "tabpfn_3-default", "tabpfn_3-binary",
    "autotabpfn",
]


def _parse_mean(s):
    if not s:
        return None
    m = re.match(r"\s*([0-9.\-]+)", s)
    return float(m.group(1)) if m else None


def _csv_lookup(rows, target, feature_set, split, arch):
    for r in rows:
        if (r["target"] != target or r["feature_set"] != feature_set
                or r["splittype"] != split or r["datasplit"] != "70_30"):
            continue
        a = r["architecture"]
        v = r["version"].strip()
        hp = r["hyperparameter"].strip()
        hpv = r["hp_variant"].strip()
        hps = r["hp_strategy"].strip()
        if a == "stacked_2xgb_meta_lr":
            if arch == "xgb_NonHP" and hp == "":
                return _parse_mean(r["holdout_AUROC"])
            if arch.startswith("xgb_HP") and hp == "HyperparameterTuned" and hps == "single_AUROC":
                if hpv == arch.replace("xgb_", ""):
                    return _parse_mean(r["holdout_AUROC"])
        elif a == "tabpfn":
            if "auto" in v.lower():
                if arch == "autotabpfn":
                    return _parse_mean(r["holdout_AUROC"])
            else:
                short = v.replace("version_", "")
                if arch == f"tabpfn_{short}":
                    return _parse_mean(r["holdout_AUROC"])
    return None


def _latest_paired_json():
    candidates = sorted((_REPO / "experiment" / "_eval").glob("paired_delong_*.json"))
    return candidates[-1] if candidates else None


def main():
    S.apply()

    csv_path = sorted((_REPO / "experiment").glob("comparison_*.csv"))[-1]
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    M = np.full((len(CELLS), len(ARCHS)), np.nan, dtype=float)
    for i, (t, fs, sp) in enumerate(CELLS):
        for j, arch in enumerate(ARCHS):
            v = _csv_lookup(rows, t, fs, sp, arch)
            if v is not None:
                M[i, j] = v

    headlines = {}
    for i in range(len(CELLS)):
        row = M[i]
        if np.isfinite(row).any():
            headlines[i] = int(np.nanargmax(row))

    # Tested-by-G6 markers (small open circle) and significant-pair markers
    # (thick edge) from the 18-test BH-FDR family.
    tested_cells = set()
    sig_cells = set()
    paired_src = _latest_paired_json()
    if paired_src is not None:
        paired = json.loads(paired_src.read_text())
        for t in paired["tests"]:
            parts = t["cell"].split("/")
            if len(parts) != 3:
                continue
            tgt, feat, sp = parts
            feat_full = f"{feat}_features"
            try:
                row_idx = CELLS.index((tgt, feat_full, sp))
            except ValueError:
                continue
            for arch_lbl in (t["arch_a"], t["arch_b"]):
                norm = (arch_lbl.replace("stacked_2xgb_", "xgb_")
                                .replace("autotabpfn_v2-5-auto", "autotabpfn")
                                .replace("tabpfn_v", "tabpfn_"))
                if norm in ARCHS:
                    col = ARCHS.index(norm)
                    tested_cells.add((row_idx, col))
                    if t.get("sig_at_q05"):
                        sig_cells.add((row_idx, col))

    fig, ax = plt.subplots(figsize=S.figsize(cols="double", h=9.5))

    cmap = plt.get_cmap("RdBu_r")
    vmin, vmax = 0.35, 0.85
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    cmap.set_bad(S.FAINT)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            txt_color = "white" if (v > 0.72 or v < 0.45) else S.INK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.0, color=txt_color)

    # Headline marker (small black filled circle).
    for i, j in headlines.items():
        ax.plot(j + 0.34, i - 0.34, marker="o", markersize=4,
                markerfacecolor=S.INK, markeredgecolor=S.INK, linestyle="none")

    # Tested-by-G6 marker (small open circle bottom-left corner).
    for (i, j) in tested_cells - sig_cells:
        ax.plot(j - 0.34, i + 0.34, marker="o", markersize=4,
                markerfacecolor="white", markeredgecolor=S.INK,
                markeredgewidth=1.0, linestyle="none")

    # Significant pair overlay (thick black edge).
    for (i, j) in sig_cells:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor="none", edgecolor=S.INK, lw=1.8))

    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels(ARCHS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([f"{t}/{fs.replace('_features','')}/{sp}"
                        for (t, fs, sp) in CELLS], fontsize=8)
    ax.set_xlabel("Architecture variant")
    ax.set_title(
        "AUROC heatmap with main-text BH-FDR overlay (18-test targeted family)\n"
        "cell-headline • • • ;  tested-in-G6 ○ ○ ○ ;  thick edge = BH-FDR-significant pair"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("hold-out AUROC (point estimate)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    headache_max = sum(1 for c in CELLS if c[0] == "headache") - 0.5
    ax.axhline(headache_max, color=S.SOFT, lw=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_stem = _FIG_DIR / "fig_g8_targeted_heatmap"
    S.save(fig, out_stem)
    print(f"source CSV:    {csv_path.relative_to(_REPO)}")
    if paired_src is not None:
        print(f"source paired: {paired_src.relative_to(_REPO)}")
    print(f"tested cells: {len(tested_cells)} ;  sig cells: {len(sig_cells)}")


if __name__ == "__main__":
    main()
