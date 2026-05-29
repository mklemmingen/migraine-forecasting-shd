"""Figure S1 (supplementary) - training wall-clock per architecture class.

Two-panel figure of header-precise per-leaf training wall-clock from
`experiment/wall_clock_*.csv`.

Top panel: strip plot, one row per architecture class (sorted by class
median ascending), one marker per leaf, jittered, log x-axis (seconds).
A short solid tick marks each class median; the `n=N` count of leaves per
class is annotated on each y-tick label. Matches the A4-base-rates strip
pattern in `figure_design_requirements.md` §7.

Bottom panel: Dolan-More performance profile [dolanmore2002profile,
p. 201] adapted to wall-clock. Cells are
(target, feature_set, split, ratio) tuples; for each cell and each
architecture class the per-class fastest leaf is taken; the per-cell
ratio r(cell, class) = t(cell, class) / min over classes; the profile
at tolerance tau is the fraction of cells with r <= tau. A curve in the
upper-left dominates: it is within a factor tau of the fastest on a
larger fraction of cells. Cells where a class has no leaves are recorded
as failed (r = infinity, contribute to denominator only); the caption
discloses this. Matches the fig_f2 within-family performance-profile
pattern.

Scope and exclusions disclosed in the caption: training-phase only
(no eval wrapper exists on disk per paper_rigor_checklist.md §6);
Addition 4 sequence baselines absent (no _running_output/ or results/
directories exist for them); AutoTabPFN-precise channel under-reports
because the AutoGluon-internal `total runtime` excludes ~2.3x wrapper
overhead per cross-validated leaf, so the header-precise channel is the
honest budget.

Usage: python fig_s1_wallclock_train.py
"""
# §11 compliance: training wall-clock per architecture class.
#   §11.1 CIs: distributions per class; no point-estimate CI applicable
#   §11.3 caption: cohort+n in rendered title; per-class N on y-tick labels
#   §11.6 CC BY footer via S.cc_by_footer
#   §11.7 N/A (this figure does not depend on the migraine full_features cell)
#   §11.10 no banned adjectives; §11.11 self-check this block
# Paper caption (for LaTeX):
#   Training-phase wall-clock per leaf across 465 wrapped training runs (236
#   Addition 0 XGBoost + 229 Addition 1 TabPFN family), extracted from the
#   "# Generated:" and "# Finished:" ISO-timestamp markers in each leaf's
#   _running_output/training_*.txt header and footer. Top: strip distribution
#   per architecture class, log x-axis, median tick per class. Bottom:
#   Dolan-More performance profile [dolanmore2002profile, p. 201] of class wall-clock
#   relative to fastest in cell (cells = target x feature_set x split x ratio);
#   curves rising fastest in the top-left dominate. Eval phase is excluded
#   (not wrapped on disk); Addition 4 sequence baselines are excluded (no
#   _running_output/ or results/ directories). AMD Radeon RX 7900 XT via
#   PyTorch ROCm.
from __future__ import annotations

import csv
import sys as _sys
from pathlib import Path as _Path

import matplotlib.pyplot as plt
import numpy as np

_EXP = _Path(__file__).resolve().parents[2] / "experiment"
_sys.path.insert(0, str(_EXP))
import _style as S  # noqa: E402

HERE = _Path(__file__).resolve().parent

# Architecture-class taxonomy. Each leaf is mapped to exactly one class by
# substring rules over its relative-to-experiment leaf_path. Ordered for
# stable y-axis ordering after median-sort.
CLASSES = [
    ("XGB NonHP",          "0/", "/NonHP"),
    ("XGB HP-tuned",       "0/", "/HyperparameterTuned"),
    ("TabPFN vanilla",     "1/", "tabpfn/version_2-6"),
    ("TabPFN vanilla-v3d", "1/", "tabpfn/version_3-default"),
    ("TabPFN vanilla-v3b", "1/", "tabpfn/version_3-binary"),
    ("TabPFN fine-tuned",  "1/", "tabpfn/version_2-5-finetuned"),
    ("Real-TabPFN",        "1/", "tabpfn/version_2-5-real"),
    ("AutoTabPFN",         "1/", "tabpfn/version_2-5-auto"),
]


def _classify(leaf_path: str) -> str | None:
    for name, add_prefix, marker in CLASSES:
        if leaf_path.startswith(add_prefix) and marker in leaf_path:
            return name
    return None


def _class_color(class_name: str) -> str:
    # Family-anchor colour from S.ARCH, then within-family brightness step
    # for the four TabPFN variants per design guide §1.0 (many-variant role).
    if class_name.startswith("XGB"):
        base = S.ARCH["xgboost"]
        # NonHP slightly lighter than HP-tuned to distinguish single-fit vs Optuna.
        if "NonHP" in class_name:
            return _lighten(base, 0.30)
        return base
    if class_name.startswith("TabPFN") or class_name.startswith("Real-TabPFN") or class_name.startswith("AutoTabPFN"):
        base = S.ARCH["tabpfn"]
        # Brightness step within the TabPFN family. AutoTabPFN darkest (heaviest
        # compute), vanilla lightest (instant in-context inference).
        order = {"TabPFN vanilla": 0.45, "TabPFN vanilla-v3d": 0.35,
                 "TabPFN vanilla-v3b": 0.25, "TabPFN fine-tuned": 0.10,
                 "Real-TabPFN": -0.05, "AutoTabPFN": -0.20}
        step = order.get(class_name, 0.0)
        return _lighten(base, step) if step > 0 else _darken(base, -step)
    return S.GREY


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(round(r * 255)))),
        max(0, min(255, int(round(g * 255)))),
        max(0, min(255, int(round(b * 255)))))


def _lighten(hex_color: str, amount: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex((r + (1 - r) * amount, g + (1 - g) * amount,
                        b + (1 - b) * amount))


def _darken(hex_color: str, amount: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex((r * (1 - amount), g * (1 - amount), b * (1 - amount)))


def _cell_key(leaf_path: str) -> tuple[str, str, str, str] | None:
    """Group leaves by (target, feature_set, split, ratio)."""
    parts = leaf_path.split("/")
    # Robust to addition prefix; locate target then feature_set then look for
    # ratio + split markers anywhere in the path.
    if len(parts) < 4:
        return None
    target = parts[1] if parts[1] in ("headache", "migraine") else None
    feature_set = parts[2] if len(parts) > 2 else None
    if target is None or feature_set is None:
        return None
    ratios = {"70_30", "70_15_15", "80_20"}
    splits = {"chrono", "stratified", "patient"}
    ratio = next((p for p in parts if p in ratios), None)
    split = next((p for p in parts if p in splits), None)
    if ratio is None or split is None:
        return None
    return (target, feature_set, split, ratio)


def _latest_csv() -> _Path:
    candidates = sorted(_EXP.glob("wall_clock_*.csv"))
    if not candidates:
        raise SystemExit("no wall_clock_*.csv in experiment/ - run extract_wallclock.py first")
    return candidates[-1]


def _load() -> list[dict]:
    rows = []
    with _latest_csv().open() as f:
        for row in csv.DictReader(f):
            if not row["header_duration_s"]:
                continue
            cls = _classify(row["leaf_path"])
            if cls is None:
                continue
            duration = float(row["header_duration_s"])
            cell = _cell_key(row["leaf_path"])
            rows.append({"class": cls, "duration_s": duration,
                         "cell": cell, "leaf_path": row["leaf_path"]})
    return rows


def _strip_panel(ax, rows: list[dict], rng) -> list[str]:
    """Draws the top strip panel and returns class order (top-to-bottom)."""
    by_class: dict[str, list[float]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r["duration_s"])
    # Sort classes by median ascending (fastest on top)
    order = sorted(by_class.keys(), key=lambda c: float(np.median(by_class[c])))
    DISPLAY_FLOOR = 0.5  # clamp zero-second NonHP leaves for log display
    for i, cls in enumerate(order):
        vals = np.maximum(np.asarray(by_class[cls], dtype=float), DISPLAY_FLOOR)
        y = i + (rng.random(len(vals)) - 0.5) * 0.55
        color = _class_color(cls)
        ax.scatter(vals, y, s=14, color=color, alpha=0.65,
                   edgecolor="white", lw=0.4)
        med = float(np.median(vals))
        ax.plot([med, med], [i - 0.30, i + 0.30], color=S.INK, lw=1.4)
        # Inline median annotation (small offset above the tick)
        ax.text(med, i + 0.36,
                _fmt_seconds(med),
                ha="center", va="bottom", fontsize=7, color=S.INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{cls}\nn={len(by_class[cls])}" for cls in order],
                       fontsize=8)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xscale("log")
    ax.set_xlim(DISPLAY_FLOOR, 5000)
    ax.set_xlabel("training wall-clock per leaf (seconds, log scale)")
    ax.axvline(1.0, color=S.REF_COLOR, lw=S.REF_LW, ls=":", alpha=0.6)
    ax.text(1.0, 0.98, "1 s", ha="left", va="top", fontsize=6.5, color=S.SOFT,
            transform=ax.get_xaxis_transform())
    ax.set_title("Training wall-clock per leaf, by architecture class",
                 fontsize=10)
    return order


def _fmt_seconds(v: float) -> str:
    if v < 60:
        return f"{v:.0f}s"
    if v < 3600:
        return f"{v/60:.1f}m"
    return f"{v/3600:.1f}h"


def _profile_panel(ax, rows: list[dict], order: list[str]) -> None:
    """Dolan-More performance profile of wall-clock per cell.

    Per cell (target x feature_set x split x ratio), find the per-class
    minimum duration. Compute r(cell, class) = class_min / cell_min. Cells
    without a leaf in class are recorded as missing (denominator only). The
    profile is the cumulative fraction of cells with r <= tau.
    """
    # Cell -> class -> min duration (across leaves of that class in that cell).
    # Apply a 0.5 s floor so 0-second NonHP leaves do not create a degenerate
    # cell_min that drives every other class's ratio to infinity. The clamp is
    # justified because second-resolution timestamps cannot distinguish below 1 s.
    FLOOR = 0.5
    cells: dict[tuple, dict[str, float]] = {}
    for r in rows:
        if r["cell"] is None:
            continue
        cells.setdefault(r["cell"], {})
        prev = cells[r["cell"]].get(r["class"], float("inf"))
        clamped = max(r["duration_s"], FLOOR)
        if clamped < prev:
            cells[r["cell"]][r["class"]] = clamped
    n_cells = len(cells)
    if n_cells == 0:
        return
    # tau grid: log-spaced from 1 to 10^4 (wide enough for our spread)
    tau = np.logspace(0, 4, 200)
    for cls in order:
        ratios = []
        for cell, class_min in cells.items():
            cell_min = min(class_min.values())
            if cls in class_min:
                ratios.append(class_min[cls] / cell_min)
            # cells where this class has no leaf contribute to denominator only
        ratios = np.asarray(ratios)
        if len(ratios) == 0:
            continue
        # CDF over tau: fraction of n_cells (NOT len(ratios)) with r <= tau,
        # so missing cells count as failed (r = inf) per Dolan-More convention.
        rho = np.array([float(np.sum(ratios <= t)) / n_cells for t in tau])
        ax.plot(tau, rho, color=_class_color(cls), lw=1.5, label=cls)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1e4)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"$\tau$ (wall-clock ratio to fastest in cell)")
    ax.set_ylabel(r"$\rho_s(\tau)$ (fraction of cells)")
    ax.set_title(
        f"Performance profile (Dolan-More) of training wall-clock, "
        f"{n_cells} cells",
        fontsize=10)
    ax.axhline(1.0, color=S.REF_COLOR, lw=S.REF_LW, ls=":", alpha=0.6)
    # Direct labelling on the right of each curve (per design guide §6:
    # prefer direct labelling over a legend when few lines). Spread overlapping
    # labels vertically by a minimum gap so co-saturating curves do not stack.
    label_ys: list[tuple[float, str, str]] = []
    for cls in order:
        ratios = []
        for cell, class_min in cells.items():
            if cls in class_min:
                ratios.append(class_min[cls] / min(class_min.values()))
        if not ratios:
            continue
        rho_at_max = float(np.sum(np.array(ratios) <= 1e4)) / n_cells
        label_ys.append((rho_at_max, cls, _class_color(cls)))
    label_ys.sort(key=lambda t: t[0])
    MIN_GAP = 0.055
    adjusted = []
    for y, cls, c in label_ys:
        target_y = y
        if adjusted and target_y < adjusted[-1][0] + MIN_GAP:
            target_y = adjusted[-1][0] + MIN_GAP
        adjusted.append((target_y, cls, c))
    for y, cls, c in adjusted:
        ax.text(1.15e4, min(y, 1.02), cls,
                fontsize=7, va="center", ha="left", color=c)


def main():
    S.apply()
    rng = np.random.default_rng(0)
    rows = _load()
    if not rows:
        raise SystemExit("no header-precise rows after classification")
    print(f"  source {_latest_csv().name} ({len(rows)} classified leaves)")
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=S.figsize("double", 7.2),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.42})
    order = _strip_panel(ax_top, rows, rng)
    _profile_panel(ax_bot, rows, order)
    S.panel_label(ax_top, "a")
    S.panel_label(ax_bot, "b")
    fig.subplots_adjust(right=0.78)  # reserve right margin for direct labels
    S.cc_by_footer(fig)
    out = HERE / "figures" / "fig_s1_wallclock_train"
    print("saved", S.save(fig, out))
    for cls in order:
        vals = [r["duration_s"] for r in rows if r["class"] == cls]
        if vals:
            vals.sort()
            print(f"  {cls:<22} n={len(vals):<4} median {_fmt_seconds(np.median(vals))} "
                  f"max {_fmt_seconds(vals[-1])} total {_fmt_seconds(sum(vals))}")


if __name__ == "__main__":
    main()
