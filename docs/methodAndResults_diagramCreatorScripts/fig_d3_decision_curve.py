"""Figure D3 - decision-curve (net benefit) analysis.

Net benefit vs threshold probability overlaying the composite-best XGBoost
(Add-0) and TabPFN (Add-1) leaves resolved from the latest
experiment/2/figdata_*.json, plus a documented window-MLP sequence
representative (Add-4) on the same full_features/chrono cell. One panel per
target, against the treat-all and treat-none default strategies (Vickers 2006).
A model is clinically useful over the threshold range where its curve sits
above both defaults. Reuses the Addition 5 prediction worker and the
Addition 6 net-benefit primitive.

Usage: python fig_d3_decision_curve.py
"""
# §11 compliance:
#   §11.1 PASS: net-benefit patient-day bootstrap 95% CI band per architecture
#               (DC.decision_curve_ci, n_boot=500)
#   §11.3 caption: cohort+n in suptitle; §11.6 footer; §11.7 EPV migraine; §11.11 self-check
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

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
_F = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_F)

WORKER = EXP / "5" / "_personal" / "_predict_worker.py"


def _resolve_leaf(headlines, target, family):
    for role in ("headline", "runner_up"):
        for e in headlines:
            if (e.get("target") == target and e.get("feature_set") == "full_features"
                    and e.get("splittype") == "chrono" and e.get("role") == role
                    and e.get("family") == family and e.get("leaf_dir")):
                return Path(e["leaf_dir"])
    return None


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
            print(f"  skip {leaf.name} (returncode={r.returncode}, out_exists={out.exists()})")
            if r.stderr:
                for line in r.stderr.strip().splitlines()[-10:]:
                    print(f"    {line}")
            return None
        z = np.load(out)
        return z["y"].astype(float), z["p"].astype(float)
    finally:
        out.unlink(missing_ok=True)


def _discover(headlines):
    leaves = {}
    for tgt in ("headache", "migraine"):
        ls = []
        d0 = _resolve_leaf(headlines, tgt, "xgboost")
        if d0 is not None and (d0 / "model.joblib").exists():
            ls.append(("XGBoost stack", d0))
        d1 = _resolve_leaf(headlines, tgt, "tabpfn")
        if d1 is not None and (d1 / "model.joblib").exists():
            ls.append(("TabPFN", d1))
        d4 = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
        if (d4 / "model.joblib").exists():
            ls.append(("window-MLP", d4))
        leaves[tgt] = ls
    return leaves


def main():
    S.apply()
    figdata_path = _F.latest_figdata(EXP / "2")
    if figdata_path is None:
        raise SystemExit("no figdata_*.json in experiment/2/ - run experiment/2/compare.py first")
    headlines = _F.load_figdata(figdata_path).get("headlines", [])
    print(f"  source {figdata_path.name} ({len(headlines)} headline rows)")
    leaves = _discover(headlines)
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
            dc = DC.decision_curve_ci(*r)
            ref = dc
            col = S.arch_color(label)
            ax.fill_between(dc["thresholds"], dc["model_ci_low"], dc["model_ci_high"],
                            color=col, alpha=0.15, linewidth=0)
            ax.plot(dc["thresholds"], dc["model"], lw=1.5, color=col,
                    marker=mark.get(label, "o"), markevery=8, markersize=4,
                    label=f"{label}")
            ymin = min(ymin, float(np.nanmin(dc["model_ci_low"])))
            ymax = max(ymax, float(np.nanmax(dc["model_ci_high"])))
            print(f"  {tgt:<9} {label:<13} max net benefit {np.max(dc['model']):.3f}")
        if ref is not None:
            ax.plot(ref["thresholds"], ref["treat_all"], "--", color=S.REF_COLOR, lw=1, label="treat all")
            ax.plot(ref["thresholds"], ref["treat_none"], ":", color=S.REF_COLOR, lw=1, label="treat none")
            # scale to the model curves (the message); treat-all may clip below
            ax.set_ylim(ymin - 0.03, ymax + 0.03)
        ax.set_xlabel("threshold probability")
        ax.set_ylabel("net benefit")
        ax.set_title(tgt)
        ax.legend(fontsize=8, loc="upper right")
        S.epv_annotation(ax, tgt, cell="full_features", loc="lower right")
    fig.suptitle("Decision-curve analysis (Park 2016 SHD, n=62)\n"
                 "Net benefit per architecture with patient-day bootstrap 95% CI band",
                 y=1.04, fontsize=10)
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_d3_decision_curve"))


if __name__ == "__main__":
    main()
