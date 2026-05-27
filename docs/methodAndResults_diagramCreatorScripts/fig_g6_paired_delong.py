"""Figure G6 — paired DeLong forest plot for cross-architecture
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
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def _row_label(r: dict) -> str:
    """Compact y-axis label: cell + arch-A vs arch-B."""
    return f"{r['cell']}/{r['ratio']}  {r['arch_a']} vs {r['arch_b']}"


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

    fig.tight_layout()

    out_stem = _FIG_DIR / "fig_g6_paired_delong"
    S.save(fig, out_stem)
    print(f"source: {src.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
