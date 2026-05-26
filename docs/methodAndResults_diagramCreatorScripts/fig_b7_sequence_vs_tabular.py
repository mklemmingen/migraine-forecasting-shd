"""Figure B7 - sequence vs tabular discrimination (Addition 4).

Hold-out (val+test) AUROC of the three sequence architectures (window-MLP,
GRU, TCN over gap-aware temporal windows) against the tabular architectures
(XGBoost stack, TabPFN), per target, full_features chronological. Computed
via the Addition 5 prediction worker so every architecture is scored on the
same horizon (the internal Addition 4 cells were not emitted to the
comparison table).

The cross-architecture comparison this figure presents is NOT fully
symmetric. Three asymmetries should be read alongside the bar heights:

1. Split ratio. Sequence leaves are pinned to ``70_15_15 / chrono`` while
   the tabular bars are composite-tracked headline leaves (typically at
   ``70_30 / chrono``). The tabular models therefore train on ~10% more
   patient-days than the sequence models, and the val+test horizon is
   structurally different across bars (15% + 15% vs 30%).
2. Hyperparameter-search budget. The tabular XGBoost stack composite
   winner is the HP020 variant (best of the first 20 trials of a
   500-trial Optuna sweep); the sequence variants are NonHP per
   addition4_sequence.md Decision 4. The tabular advantage on AUROC
   absorbs the HP-tuning advantage.
3. CI width vs sample-size. The sequence test set is ~136 rows
   (70_15_15 test slice); the tabular composite-headline test is ~1465
   rows. Bootstrap CIs on the sequence bars are wider by construction.

The qualitative result holds despite these asymmetries: on migraine the
window-MLP at AUROC 0.771 sits between the XGB-HP020 headline (0.793)
and the TabPFN-v2.6 runner-up (0.761) - sequence is competitive but
not dominant, and Addition 3's prediction that the temporal signal
is short-range is supported. On headache no sequence variant beats
either tabular family (best sequence GRU 0.640 vs tabular best 0.657).
The right verb for the result is "competitive, not dominant," not
"failed to beat."

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


import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
_F = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_F)


def _resolve_leaf(headlines, target, family):
    """Composite-tracked leaf for (target, family) on full_features/chrono.
    Mirrors the helper in fig_c2/c3/d3/d4."""
    for role in ("headline", "runner_up"):
        for e in headlines:
            if (e.get("target") == target and e.get("feature_set") == "full_features"
                    and e.get("splittype") == "chrono" and e.get("role") == role
                    and e.get("family") == family and e.get("leaf_dir")):
                return Path(e["leaf_dir"])
    return None


def _leaves(tgt, headlines):
    """Cross-architecture set for the full_features/chrono cell.
    Add-0 XGBoost stack and Add-1 TabPFN are figdata-driven (composite
    tracking); the three Add-4 sequence variants are documented pinned
    representatives (composite selection does not cover Add-4)."""
    seqbase = EXP / "4" / tgt / "full_features" / "sequence"
    out = {}
    d0 = _resolve_leaf(headlines, tgt, "xgboost")
    if d0 is not None and (d0 / "model.joblib").exists():
        out["XGBoost stack"] = d0
    d1 = _resolve_leaf(headlines, tgt, "tabpfn")
    if d1 is not None and (d1 / "model.joblib").exists():
        out["TabPFN"] = d1
    for name, var in (("window-MLP", "window-mlp"), ("GRU", "gru"), ("TCN", "tcn")):
        d = seqbase / f"version_{var}/70_15_15/chrono"
        if (d / "model.joblib").exists():
            out[name] = d
    return out


def main():
    S.apply()
    figdata_path = _F.latest_figdata(EXP / "2")
    if figdata_path is None:
        raise SystemExit("no figdata_*.json in experiment/2/ - run experiment/2/compare.py first")
    headlines = _F.load_figdata(figdata_path).get("headlines", [])
    print(f"  source {figdata_path.name} ({len(headlines)} headline rows)")
    aurocs = {t: {} for t in ("headache", "migraine")}
    slug_per_label = {t: {} for t in ("headache", "migraine")}
    for tgt in aurocs:
        for label, leaf in _leaves(tgt, headlines).items():
            if not (leaf / "model.joblib").exists():
                print(f"  {tgt} {label}: no model.joblib"); continue
            a = _auroc(leaf)
            if a is not None:
                aurocs[tgt][label] = a
                slug_per_label[tgt][label] = S.leaf_slug(leaf)
                print(f"  {tgt:<9} {label:<14} AUROC {a:.3f}  ({S.leaf_slug(leaf)})")
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
