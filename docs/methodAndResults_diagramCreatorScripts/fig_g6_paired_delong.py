"""Figure G6 - paired DeLong forest plot for cross-architecture
significance (Section 3a).

One row per paired test (cell + arch pair); ΔAUC point with 95%
bootstrap-CI whiskers from `delong_paired_test`. The vertical reference
line is ΔAUC = 0 (no difference). FDR-significant tests at q ≤ 0.05
are highlighted in the brand-emphasis colour with a filled marker; the
rest are de-emphasised grey with hollow markers. Sorted by raw DeLong
p-value (most significant on top), so a reader scanning the figure
top-down sees the strongest evidence first.

The figure consumes the most recent
`experiment/_eval/paired_delong_<ts>.json` (written by
`experiment/_eval/run_paired_delong.py`), so the figure pins to the
exact 18-test run that produced the §3a table.

Usage: python fig_g6_paired_delong.py
"""
# §11 compliance: paired DeLong forest with 95% DeLong CIs + q-values.
#   §11.1 PASS (DeLong CI whiskers + q-values)
#   §11.3 caption: cohort+n in rendered title
#   §11.6, §11.7 via renderer (migraine `full_features` rows present)
#   §11.10 no banned adjectives; §11.11 self-check this block
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_EVAL_DIR = _REPO / "experiment" / "_eval"
_FIG_DIR = _THIS.parent / "figures"
_FIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(_REPO / "experiment"))
import _style as S  # noqa: E402


def _latest_json() -> Path:
    candidates = sorted(_EVAL_DIR.glob("paired_delong_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no paired_delong_*.json under {_EVAL_DIR}; run "
            "experiment/_eval/run_paired_delong.py first"
        )
    return candidates[-1]


_ARCH_TOKEN = {
    "tabpfn_v2-5-finetuned": "TabPFN-v2.5f",
    "tabpfn_v2-5-auto": "TabPFN-v2.5a",
    "tabpfn_v2-5-real": "TabPFN-v2.5r",
    "tabpfn_v2-6": "TabPFN-v2.6",
    "tabpfn_v3-default": "TabPFN-v3d",
    "tabpfn_v3-binary": "TabPFN-v3b",
    "autotabpfn_v2-5-auto": "AutoTabPFN-v2.5a",
}


def _arch_to_var(arch_lbl: str) -> str:
    """Translate internal arch labels to the canonical ARCH-VAR slug token."""
    if arch_lbl in _ARCH_TOKEN:
        return _ARCH_TOKEN[arch_lbl]
    if arch_lbl.startswith("stacked_2xgb_"):
        return "XGB-" + arch_lbl[len("stacked_2xgb_"):]
    if arch_lbl == "blended_xgb_lr_spano2026":
        return "XGB-blended"
    if arch_lbl == "add4_window_mlp":
        return "Seq-windowMLP"
    return arch_lbl


def _normalize_cell(cell: str) -> str:
    return cell.replace("no_rolling", "no-roll").replace("full_features", "full")


def _row_label(r: dict) -> str:
    """Compact y-axis label using the canonical 4-token slug stem."""
    return (f"{_normalize_cell(r['cell'])}/{r['ratio'].replace('_', '-')}  "
            f"{_arch_to_var(r['arch_a'])} vs {_arch_to_var(r['arch_b'])}")


def main():
    S.apply()

    src = _latest_json()
    payload = json.loads(src.read_text())
    rows = payload["tests"]
    rows = sorted(rows, key=lambda r: r["p"])

    n = len(rows)
    deltas = np.array([r["delta"] for r in rows])
    ci_lo = np.array([r["ci_lo"] for r in rows])
    ci_hi = np.array([r["ci_hi"] for r in rows])
    sig = np.array([bool(r["sig_at_q05"]) for r in rows])
    labels = [_row_label(r) for r in rows]
    qvals = [r["q"] for r in rows]

    y_pos = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=S.figsize(cols="double", h=max(4.5, 0.34 * n + 1.2)))

    sig_color = S.OI["vermillion"]
    nonsig_color = S.MUTED

    for i in range(n):
        c = sig_color if sig[i] else nonsig_color
        ax.hlines(y_pos[i], ci_lo[i], ci_hi[i], color=c, lw=1.4, alpha=0.9)

    for i in range(n):
        c = sig_color if sig[i] else nonsig_color
        face = c if sig[i] else "white"
        ax.plot(deltas[i], y_pos[i], marker="o", markersize=6,
                markerfacecolor=face, markeredgecolor=c, markeredgewidth=1.4,
                linestyle="none")

    S.refline(ax, x=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    x_anno = max(ci_hi.max(), 0.05) + 0.012
    for i in range(n):
        c = sig_color if sig[i] else nonsig_color
        ax.text(x_anno, y_pos[i], f"q={qvals[i]:.3f}",
                va="center", ha="left", fontsize=7.5, color=c)

    ax.set_xlabel("ΔAUC (arch A − arch B), 95% DeLong CI")
    n_sig = int(sig.sum())
    ax.set_title(
        f"Paired DeLong + BH-FDR (q ≤ 0.05) across {n} cross-architecture tests; "
        f"{n_sig} significant after multiplicity correction"
    )

    x_lo = min(ci_lo.min(), -0.02) - 0.012
    x_hi = x_anno + 0.06
    ax.set_xlim(x_lo, x_hi)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Frameless legend, positioned below the x-axis label so it never
    # occludes the row-wise CI lines or q-value annotations (§6 rule 2).
    legend_elements = [
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor=sig_color, markeredgecolor=sig_color,
               markeredgewidth=1.4, markersize=6,
               label="q ≤ 0.05 (BH-FDR sig)"),
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="white", markeredgecolor=nonsig_color,
               markeredgewidth=1.4, markersize=6, label="q > 0.05"),
        Line2D([0], [0], color=nonsig_color, lw=1.4,
               label="95% DeLong CI"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=8)

    fig.tight_layout()

    out_stem = _FIG_DIR / "fig_g6_paired_delong"
    S.save(fig, out_stem)
    print(f"source: {src.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
