"""Figure B7 - sequence vs tabular discrimination (Addition 4).

Hold-out (val+test) AUROC of the sequence model (window-MLP over gap-aware
temporal windows) against the tabular architectures (XGBoost stack, TabPFN),
per target, full_features chronological. The sequence model does not beat the
tabular models on this cohort - day-to-day order adds little once history
features are present. Computed via the Addition 5 prediction worker so all
architectures are scored on the same horizon (the internal Addition 4 cells were
not emitted to the comparison table).

Note: only the window-MLP sequence variant was trained; the GRU and TCN variants
were scaffolded (`experiment/4/.../version_{gru,tcn}/`) but not fitted, so the
sequence family is represented here by the window-MLP.

Usage: python fig_b7_sequence_vs_tabular.py
"""
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import _figstyle as S
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiment"
WORKER = EXP / "5" / "_personal" / "_predict_worker.py"
ORDER = ["XGBoost stack", "TabPFN", "window-MLP (seq)"]
COL = {"XGBoost stack": "#1b9e77", "TabPFN": "#7570b3", "window-MLP (seq)": "#d95f02"}


def _env(addition):
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""; e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _auroc(leaf):
    add = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"b7_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(WORKER), str(leaf), str(out)],
                           capture_output=True, text=True, timeout=1800, env=_env(add))
        if r.returncode != 0 or not out.exists():
            print("  skip", leaf.name); return None
        z = np.load(out)
        return float(roc_auc_score(z["y"], z["p"]))
    finally:
        out.unlink(missing_ok=True)


def _leaves(tgt):
    out = {}
    for m in (EXP / "0" / tgt / "full_features").rglob("NonHP/model.joblib"):
        if "stacked_2xgb" in str(m) and "/70_15_15/chrono/" in str(m):
            out["XGBoost stack"] = m.parent; break
    out["TabPFN"] = EXP / "1" / tgt / "full_features/tabpfn/version_3-default/70_15_15/chrono"
    out["window-MLP (seq)"] = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
    return out


def main():
    S.apply()
    aurocs = {t: {} for t in ("headache", "migraine")}
    for tgt in aurocs:
        for label, leaf in _leaves(tgt).items():
            if not (leaf / "model.joblib").exists():
                continue
            a = _auroc(leaf)
            if a is not None:
                aurocs[tgt][label] = a
                print(f"  {tgt:<9} {label:<18} AUROC {a:.3f}")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(ORDER)); w = 0.38
    for i, tgt in enumerate(("headache", "migraine")):
        vals = [aurocs[tgt].get(l, np.nan) for l in ORDER]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=S.TARGET[tgt], label=tgt, alpha=0.85)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                        ha="center", fontsize=7.5)
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("AUROC (val+test horizon)")
    ax.set_title("Sequence vs tabular discrimination (full_features, chrono)")
    ax.legend(title="target")
    print("saved", S.save(fig, HERE / "figures" / "fig_b7_sequence_vs_tabular"))


if __name__ == "__main__":
    main()
