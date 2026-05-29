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
and the TabPFN-v2.5f runner-up (0.761) - sequence is competitive but
not dominant, and Addition 3's prediction that the temporal signal
is short-range is supported. On headache no sequence variant beats
either tabular family (best sequence GRU 0.640 vs tabular best 0.657).
The right verb for the result is "competitive, not dominant," not
"failed to beat." Note that the TabPFN composite-tracked runner-up
is target-specific: v2.5-finetuned on migraine, v2.6 on headache - the
per-bar x-tick label discloses each variant explicitly.

Usage: python fig_b7_sequence_vs_tabular.py
"""
# §11 compliance: sequence vs tabular bars on full_features/chrono.
#   §11.1 CIs: per-bar patient-day bootstrap (n_boot=1000) 95% CI whiskers from the
#       worker's (y, p) test arrays; renders the wider sequence CIs the disclosure cites
#   §11.3 caption: cohort+n in title
#   §11.6 footer + §11.7 EPV (migraine bars) + §11.11 self-check
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


def _auroc_with_ci(leaf, n_boot: int = 1000, seed: int = 42):
    """Returns (auroc, ci_low, ci_high, n_rows) via patient-day bootstrap on (y, p),
    or None when the leaf has no recoverable predictions."""
    add = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"b7_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(WORKER), str(leaf), str(out)],
                           capture_output=True, text=True, timeout=1800, env=_env(add))
        if r.returncode != 0 or not out.exists():
            print("  skip", leaf.name, (r.stderr.strip().splitlines() or ["?"])[-1])
            return None
        z = np.load(out)
        y, p = z["y"], z["p"]
        point = float(roc_auc_score(y, p))
        rng = np.random.default_rng(seed)
        n = len(y)
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb, pb = y[idx], p[idx]
            if yb.min() == yb.max():
                continue
            boot.append(roc_auc_score(yb, pb))
        boot = np.asarray(boot, dtype=float)
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 10
                  else (float("nan"), float("nan")))
        return point, float(lo), float(hi), int(n)
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
    cis = {t: {} for t in ("headache", "migraine")}
    slug_per_label = {t: {} for t in ("headache", "migraine")}
    for tgt in aurocs:
        for label, leaf in _leaves(tgt, headlines).items():
            if not (leaf / "model.joblib").exists():
                print(f"  {tgt} {label}: no model.joblib"); continue
            res = _auroc_with_ci(leaf)
            if res is not None:
                a, lo, hi, n_rows = res
                aurocs[tgt][label] = a
                cis[tgt][label] = (lo, hi)
                slug_per_label[tgt][label] = S.leaf_slug(leaf)
                print(f"  {tgt:<9} {label:<14} AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]  "
                      f"n={n_rows}  ({S.leaf_slug(leaf)})")
    fig, ax = plt.subplots(figsize=S.figsize("double", 4.3))
    x = np.arange(len(ORDER)); w = 0.38
    for i, tgt in enumerate(("headache", "migraine")):
        vals = [aurocs[tgt].get(l, np.nan) for l in ORDER]
        lo_arr = [max(0.0, vals[j] - cis[tgt].get(ORDER[j], (vals[j], vals[j]))[0])
                  if vals[j] == vals[j] else 0.0 for j in range(len(ORDER))]
        hi_arr = [max(0.0, cis[tgt].get(ORDER[j], (vals[j], vals[j]))[1] - vals[j])
                  if vals[j] == vals[j] else 0.0 for j in range(len(ORDER))]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=S.TARGET[tgt], label=tgt,
                      yerr=[lo_arr, hi_arr], capsize=2.0, ecolor=S.SOFT,
                      error_kw={"elinewidth": 0.9})
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.2f}",
                        ha="center", fontsize=7)
    S.refline(ax, y=0.5)
    ax.axvline(1.5, color=S.FAINT, lw=1, ls=":")        # tabular | sequence divider
    ax.text(0.5, 0.83, "tabular", ha="center", fontsize=8, color=S.GREY,
            transform=ax.get_xaxis_transform())
    ax.text(3.0, 0.83, "sequence", ha="center", fontsize=8, color=S.GREY,
            transform=ax.get_xaxis_transform())
    # Slug-bearing tick labels: every reported AUROC must name the model+cell
    # that produced it. Where the headache and migraine bars use the same
    # ARCH-VAR (e.g. window-MLP, GRU, TCN), one slug suffices; where they
    # diverge (TabPFN-v2.6 on headache vs TabPFN-v2.5f on migraine via the
    # composite-tracked headline), both are disclosed.
    def _slug_arch(s):
        return s.split(" / ")[0] if s else "?"
    tick_labels = []
    for fam in ORDER:
        h_a = _slug_arch(slug_per_label["headache"].get(fam, ""))
        m_a = _slug_arch(slug_per_label["migraine"].get(fam, ""))
        if h_a == m_a and h_a != "?":
            tick_labels.append(f"{fam}\n{h_a}")
        elif h_a == "?" and m_a == "?":
            tick_labels.append(fam)
        else:
            tick_labels.append(f"{fam}\nh: {h_a}\nm: {m_a}")
    ax.set_xticks(x); ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.set_ylim(0.5, 0.88)
    ax.set_ylabel("AUROC (val+test horizon)")
    ax.set_title("Sequence vs tabular discrimination (Park 2016 SHD, n=62; full_features, chrono)")
    S.epv_annotation(ax, "migraine", cell="full_features", loc="upper left")
    ax.legend(title="target", loc="upper right")
    # Three structural asymmetries the reader must keep in mind alongside bar heights:
    # tabular = 70/30 + HP-tuned; sequence = 70/15/15 + NonHP. The visible XGB-HP020
    # vs Seq-windowMLP 0.79 vs 0.77 gap absorbs these.
    ax.text(0.5, -0.32,
            "tabular bars: 70/30 / composite-tracked / HP-tuned · "
            "sequence bars: 70/15/15 / NonHP / wider bootstrap CI",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color=S.SOFT, style="italic")
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_b7_sequence_vs_tabular"))


if __name__ == "__main__":
    main()
