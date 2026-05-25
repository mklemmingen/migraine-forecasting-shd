"""Figure D3 - decision-curve (net benefit) analysis.

Net benefit vs threshold probability for the headline architectures (Additions
0/1/4, full_features, chronological), one panel per target, against the treat-all
and treat-none default strategies (Vickers 2006). A model is clinically useful
over the threshold range where its curve sits above both defaults. Reuses the
Addition 5 prediction worker and the Addition 6 net-benefit primitive.

Usage: python fig_d3_decision_curve.py
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
sys.path.insert(0, str(EXP / "6" / "_value"))
import decision_curve as DC  # noqa: E402

WORKER = EXP / "5" / "_personal" / "_predict_worker.py"


def _env(addition):
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""; e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf):
    add = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"d3_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(WORKER), str(leaf), str(out)],
                           capture_output=True, text=True, timeout=1800, env=_env(add))
        if r.returncode != 0 or not out.exists():
            print("  skip", leaf.name); return None
        z = np.load(out)
        return z["y"].astype(float), z["p"].astype(float)
    finally:
        out.unlink(missing_ok=True)


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
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.3))
    for _ax, _lt in zip(axes.ravel(), "abcdefgh"):
        S.panel_label(_ax, _lt)
    mark = {"XGBoost stack": "o", "TabPFN": "s", "window-MLP": "^"}  # CVD/grayscale reinforcement
    for ax, tgt in zip(axes, ("headache", "migraine")):
        ref = None
        ymin = ymax = 0.0
        for label, leaf in leaves[tgt]:
            r = _predict(leaf)
            if r is None:
                continue
            dc = DC.decision_curve(*r)
            ref = dc
            ax.plot(dc["thresholds"], dc["model"], lw=1.6, color=S.arch_color(label),
                    marker=mark.get(label, "o"), markevery=8, markersize=4, label=label)
            ymin = min(ymin, float(dc["model"].min())); ymax = max(ymax, float(dc["model"].max()))
            print(f"  {tgt:<9} {label:<13} max net benefit {np.max(dc['model']):.3f}")
        if ref is not None:
            ax.plot(ref["thresholds"], ref["treat_all"], "--", color=S.GREY, lw=1, label="treat all")
            ax.plot(ref["thresholds"], ref["treat_none"], ":", color=S.REF_COLOR, lw=1, label="treat none")
            # scale to the model curves (the message); treat-all may clip below
            ax.set_ylim(ymin - 0.03, ymax + 0.03)
        ax.set_xlabel("threshold probability")
        ax.set_ylabel("net benefit")
        ax.set_title(f"{tgt} (full_features, chrono)")
        ax.legend(fontsize=7.5)
    fig.suptitle("Decision-curve analysis", y=1.02)
    print("saved", S.save(fig, HERE / "figures" / "fig_d3_decision_curve"))


if __name__ == "__main__":
    main()
