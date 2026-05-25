"""Figure B7 - sequence vs tabular discrimination (Addition 4).

Hold-out (val+test) AUROC of the three sequence architectures (window-MLP, GRU,
TCN over gap-aware temporal windows) against the tabular architectures (XGBoost
stack, TabPFN), per target, full_features chronological. No sequence variant beats
the tabular models on this cohort: explicit temporal modelling adds nothing once
the rolling/lag history features are present, consistent with Addition 3 finding
that day-to-day dependence decays within a few days. Computed via the Addition 5
prediction worker so every architecture is scored on the same horizon (the
internal Addition 4 cells were not emitted to the comparison table).

Usage: python fig_b7_sequence_vs_tabular.py
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiment"
WORKER = EXP / "5" / "_personal" / "_predict_worker.py"
ORDER = ["XGBoost stack", "TabPFN", "window-MLP", "GRU", "TCN"]
TABULAR = {"XGBoost stack", "TabPFN"}


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
            print("  skip", leaf.name, (r.stderr.strip().splitlines() or ["?"])[-1])
            return None
        z = np.load(out)
        return float(roc_auc_score(z["y"], z["p"]))
    finally:
        out.unlink(missing_ok=True)


def _leaves(tgt):
    seqbase = EXP / "4" / tgt / "full_features" / "sequence"
    out = {}
    for m in (EXP / "0" / tgt / "full_features").rglob("NonHP/model.joblib"):
        if "stacked_2xgb" in str(m) and "/70_15_15/chrono/" in str(m):
            out["XGBoost stack"] = m.parent; break
    out["TabPFN"] = EXP / "1" / tgt / "full_features/tabpfn/version_3-default/70_15_15/chrono"
    out["window-MLP"] = seqbase / "version_window-mlp/70_15_15/chrono"
    out["GRU"] = seqbase / "version_gru/70_15_15/chrono"
    out["TCN"] = seqbase / "version_tcn/70_15_15/chrono"
    return out


def main():
    S.apply()
    aurocs = {t: {} for t in ("headache", "migraine")}
    for tgt in aurocs:
        for label, leaf in _leaves(tgt).items():
            if not (leaf / "model.joblib").exists():
                print(f"  {tgt} {label}: no model.joblib"); continue
            a = _auroc(leaf)
            if a is not None:
                aurocs[tgt][label] = a
                print(f"  {tgt:<9} {label:<14} AUROC {a:.3f}")
    fig, ax = plt.subplots(figsize=S.figsize("double", 4.3))
    x = np.arange(len(ORDER)); w = 0.38
    for i, tgt in enumerate(("headache", "migraine")):
        vals = [aurocs[tgt].get(l, np.nan) for l in ORDER]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=S.TARGET[tgt], label=tgt)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                        ha="center", fontsize=7)
    S.refline(ax, y=0.5)
    ax.axvline(1.5, color=S.FAINT, lw=1, ls=":")        # tabular | sequence divider
    ax.text(0.5, 0.83, "tabular", ha="center", fontsize=8, color=S.GREY,
            transform=ax.get_xaxis_transform())
    ax.text(3.0, 0.83, "sequence", ha="center", fontsize=8, color=S.GREY,
            transform=ax.get_xaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("AUROC (val+test horizon)")
    ax.set_title("Sequence vs tabular discrimination (full_features, chrono)")
    ax.legend(title="target", loc="upper right")
    print("saved", S.save(fig, HERE / "figures" / "fig_b7_sequence_vs_tabular"))


if __name__ == "__main__":
    main()
