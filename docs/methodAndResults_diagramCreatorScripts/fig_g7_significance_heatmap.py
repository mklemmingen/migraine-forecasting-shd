"""Figure G7 — supplementary AUROC landscape heatmap across the 70_30
sweep, with BH-FDR-significant paired-DeLong comparisons overlaid.

Rows: (target, feature_set, split) cells at ratio=70_30 (the headline
ratio). Columns: every architecture variant present in the sweep.
Cell colour: AUROC point estimate from the canonical sweep CSV
(`experiment/comparison_*.csv`). Cells with overlaid annotation
(thick black edge) carry the architectures involved in
BH-FDR-significant paired-DeLong tests from `fig_g6`'s 18-test
family. The headline architecture per cell is marked with a small
filled circle.

The figure consumes:
  - `experiment/comparison_*.csv` for the AUROC landscape
  - `experiment/_eval/paired_delong_*.json` for significance overlay

so the figure pins to the same canonical sweep and the same 18-test
family the main-text §3a table is built from.

Usage: python fig_g7_significance_heatmap.py
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


# Cells covered (excluding headache/park).
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

# Architecture column order; missing cells render grey.
ARCHS = [
    "xgb_NonHP",
    "xgb_HP020", "xgb_HP050", "xgb_HP100", "xgb_HP200", "xgb_HP500",
    "tabpfn_v2-5-real", "tabpfn_v2-5-finetuned", "tabpfn_v2-6",
    "tabpfn_v3-default", "tabpfn_v3-binary",
    "autotabpfn",
]


def _parse_mean(s: str):
    """Parse '0.793 [0.701 - 0.873]' → 0.793."""
    if not s:
        return None
    m = re.match(r"\s*([0-9.\-]+)", s)
    return float(m.group(1)) if m else None


def _csv_lookup(rows, target, feature_set, split, arch):
    """Return AUROC for a (cell, arch) tuple, or None if not in the sweep."""
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


def _latest_delong_json():
    """Prefer the exhaustive-grid JSON (Bonferroni FWER over the all-pairs
    family at ratio=70_30) when present; fall back to the 18-test
    paired_delong family if not. The exhaustive run carries
    architecture-vs-headline significance per cell, which is the natural
    overlay for the AUROC heatmap."""
    eval_dir = _REPO / "experiment" / "_eval"
    exhaustive = sorted(eval_dir.glob("exhaustive_delong_*.json"))
    if exhaustive:
        return exhaustive[-1], "exhaustive"
    paired = sorted(eval_dir.glob("paired_delong_*.json"))
    return (paired[-1], "paired") if paired else (None, None)


def main():
    S.apply()

    csv_path = sorted((_REPO / "experiment").glob("comparison_*.csv"))[-1]
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Build (rows × archs) AUROC matrix.
    M = np.full((len(CELLS), len(ARCHS)), np.nan, dtype=float)
    for i, (t, fs, sp) in enumerate(CELLS):
        for j, arch in enumerate(ARCHS):
            v = _csv_lookup(rows, t, fs, sp, arch)
            if v is not None:
                M[i, j] = v

    # Identify per-row headline (highest-AUROC architecture in the row).
    headlines = {}
    for i in range(len(CELLS)):
        row = M[i]
        if np.isfinite(row).any():
            headlines[i] = int(np.nanargmax(row))

    # Overlay significant pairs from either the exhaustive Bonferroni
    # family (preferred) or the targeted BH-FDR 18-test family (fallback).
    sig_cells = set()
    delong_src, delong_kind = _latest_delong_json()
    if delong_src is not None:
        delong = json.loads(delong_src.read_text())
        if delong_kind == "exhaustive":
            # Mark every architecture that participates in any Bonferroni-
            # significant pair within its cell.
            for cell_dict in delong["cells"]:
                parts = cell_dict["cell"].split("/")
                if len(parts) != 3:
                    continue
                tgt, feat, sp = parts
                feat_full = f"{feat}_features"
                try:
                    row_idx = CELLS.index((tgt, feat_full, sp))
                except ValueError:
                    continue
                for pair in cell_dict["pairs"]:
                    if not pair.get("sig_bonferroni"):
                        continue
                    for arch_lbl in (pair["arch_a"], pair["arch_b"]):
                        if arch_lbl in ARCHS:
                            sig_cells.add((row_idx, ARCHS.index(arch_lbl)))
        else:  # paired (18-test family)
            for t in delong["tests"]:
                if not t.get("sig_at_q05"):
                    continue
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
                                    .replace("autotabpfn_v2-5-auto", "autotabpfn"))
                    if norm in ARCHS:
                        sig_cells.add((row_idx, ARCHS.index(norm)))

    fig, ax = plt.subplots(figsize=S.figsize(cols="double", h=9.5))

    # Diverging colormap centred at chance (0.5): warm = high AUROC, cool = low.
    # _metric_palette uses RdBu warm=bad; for AUROC we want warm=good, so RdBu_r.
    cmap = plt.get_cmap("RdBu_r")
    vmin, vmax = 0.35, 0.85
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    cmap.set_bad(S.FAINT)

    # Cell-text: AUROC value, white-on-dark or black-on-light by intensity.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            txt_color = "white" if (v > 0.72 or v < 0.45) else S.INK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.0, color=txt_color)

    # Headline marker: small black filled circle in cell corner.
    for i, j in headlines.items():
        ax.plot(j + 0.34, i - 0.34, marker="o", markersize=4,
                markerfacecolor=S.INK, markeredgecolor=S.INK, linestyle="none")

    # BH-FDR-significant overlay: thick black edge.
    for (i, j) in sig_cells:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor="none", edgecolor=S.INK, lw=1.8))

    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels(ARCHS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([f"{t}/{fs.replace('_features','')}/{sp}"
                        for (t, fs, sp) in CELLS], fontsize=8)
    ax.set_xlabel("Architecture variant")
    overlay_label = ("exhaustive Bonferroni" if delong_kind == "exhaustive"
                     else "BH-FDR main-text §3a")
    ax.set_title(
        "Supplementary AUROC heatmap at ratio 70_30 across the sweep\n"
        f"headline per cell • • • ;   thick edges = {overlay_label}-significant pair"
    )

    # Side-row colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("hold-out AUROC (point estimate)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Lines between target groups for readability
    headache_max = sum(1 for c in CELLS if c[0] == "headache") - 0.5
    ax.axhline(headache_max, color=S.SOFT, lw=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_stem = _FIG_DIR / "fig_g7_significance_heatmap"
    S.save(fig, out_stem)
    print(f"source CSV:    {csv_path.relative_to(_REPO)}")
    if delong_src is not None:
        print(f"source DeLong: {delong_src.relative_to(_REPO)}")
    print(f"rows {len(CELLS)} × cols {len(ARCHS)} = {len(CELLS)*len(ARCHS)} potential cells")
    n_present = int(np.isfinite(M).sum())
    print(f"populated cells: {n_present}, BH-FDR-significant overlays: {len(sig_cells)}")


if __name__ == "__main__":
    main()
