"""Figure D4 - Brier skill versus per-patient climatology.

Brier skill score (1 - BS_model / BS_reference) of the headline architectures
against each patient's own TRAIN-set base rate, per target. Positive means the
model's probabilities beat the trivial per-patient prior; the migraine bars are
negative despite AUROC ~0.76 - high discrimination that does not translate into
probabilistic value (Murphy quality-vs-value; Addition 6 Section 9a). Reuses the
Addition 5 prediction worker and the Addition 6 skill primitive.

Usage: python fig_d4_brier_skill.py
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
sys.path.insert(0, str(EXP / "6" / "_value"))
import skill as SK  # noqa: E402
from _dataRead.read import load_raw, TARGET_COL  # noqa: E402

WORKER = EXP / "5" / "_personal" / "_predict_worker.py"


def _env(addition):
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""; e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf):
    add = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"d4_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(WORKER), str(leaf), str(out)],
                           capture_output=True, text=True, timeout=1800, env=_env(add))
        if r.returncode != 0 or not out.exists():
            print("  skip", leaf.name); return None
        z = np.load(out)
        return z["y"].astype(float), z["p"].astype(float), z["pid"].astype(str)
    finally:
        out.unlink(missing_ok=True)


def _climatology(target):
    tr = load_raw(str(REPO / "data" / "processed" / target / "70_15_15" / "chrono" / "diary_train.parquet"))
    rates = tr.groupby("patient_id")[TARGET_COL].mean()
    rates.index = rates.index.astype(str)
    return rates.to_dict(), float(tr[TARGET_COL].mean())


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
    targets = ("headache", "migraine")
    labels = ["XGBoost stack", "TabPFN", "window-MLP"]
    skills = {t: {} for t in targets}
    for tgt in targets:
        rates, cohort = _climatology(tgt)
        for label, leaf in leaves[tgt]:
            r = _predict(leaf)
            if r is None:
                continue
            y, p, pid = r
            ref = SK.per_patient_climatology(pid, rates, cohort)
            skills[tgt][label] = SK.brier_skill_score(y, p, ref)
            print(f"  {tgt:<9} {label:<13} Brier skill {skills[tgt][label]:+.3f}")
    fig, ax = plt.subplots(figsize=S.figsize("double", 4.2))
    x = np.arange(len(labels)); w = 0.38
    for i, tgt in enumerate(targets):
        vals = [skills[tgt].get(l, np.nan) for l in labels]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=S.TARGET[tgt], label=tgt)
        for b, v in zip(bars, vals):
            if v == v:   # skip NaN
                ax.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=7,
                            xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
    ax.axhline(0, color=S.REF_COLOR, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Brier skill vs per-patient climatology")
    ax.set_title("Probabilistic value over the patient base rate")
    ax.text(0.02, 0.04, "below 0 = worse than predicting the patient's own base rate",
            transform=ax.transAxes, fontsize=7.5, style="italic", color=S.GREY)
    ax.legend(title="target")
    print("saved", S.save(fig, HERE / "figures" / "fig_d4_brier_skill"))


if __name__ == "__main__":
    main()
