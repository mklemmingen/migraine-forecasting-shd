"""Figure C3 - calibration reliability diagrams.

Observed vs predicted next-day probability by quantile bin for the headline
architectures (Additions 0/1/4, full_features, chronological 70/15/15), one panel
per target, with the calibration slope annotated per model. Reuses the Addition 5
prediction worker (per-addition subprocess; GPU only for TabPFN), so no model code
is re-implemented here. Points sagging below the diagonal at high predicted risk
are the over-confidence fingerprint (slope < 1) flagged in results_findings.md
Section 6.

Usage: python fig_c3_calibration.py
"""
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

WORKER = EXP / "5" / "_personal" / "_predict_worker.py"
N_BINS = 8


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


def _discover():
    leaves = {}
    for tgt in ("headache", "migraine"):
        ls = []
        for m in (EXP / "0" / tgt / "full_features").rglob("NonHP/model.joblib"):
            if "stacked_2xgb" in str(m) and "/70_15_15/chrono/" in str(m):
                ls.append(("XGBoost stack", m.parent)); break
        d1 = EXP / "1" / tgt / "full_features/tabpfn/version_3-default/70_15_15/chrono"
        if (d1 / "model.joblib").exists():
            ls.append(("TabPFN", d1))
        d4 = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
        if (d4 / "model.joblib").exists():
            ls.append(("window-MLP", d4))
        leaves[tgt] = ls
    return leaves


def main():
    S.apply()
    leaves = _discover()
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.2))
    for _ax, _lt in zip(axes.ravel(), "abcdefgh"):
        S.panel_label(_ax, _lt)
    mark = {"XGBoost stack": "o", "TabPFN": "s", "window-MLP": "^"}  # CVD/grayscale reinforcement
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
            sizes = 12 + 120 * ns / ns.max()
            col = S.arch_color(label)
            mk = mark.get(label, "o")
            ax.plot(xs, ys, "-", color=col, lw=1.2, marker=mk, markersize=4)
            ax.scatter(xs, ys, s=sizes, color=col, marker=mk,
                       label=f"{label} (slope {slope:.2f})")
            hi = max(hi, xs.max(), ys.max())
            print(f"  {tgt:<9} {label:<13} slope {slope:+.2f}  bins {len(xs)}")
        lim = min(1.0, hi * 1.1 + 0.02)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")   # honest 45-degree perfect line
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"{tgt} (full_features, chrono)")
        ax.legend(loc="upper left")
    fig.suptitle("Reliability diagrams", y=1.02)
    print("saved", S.save(fig, HERE / "figures" / "fig_c3_calibration"))


if __name__ == "__main__":
    main()
