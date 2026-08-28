"""Figure D4 - Brier skill versus per-patient climatology.

Brier skill score (1 - BS_model / BS_reference) of the composite-best
architectures (Add-0 XGBoost and Add-1 TabPFN from the latest
experiment/2/figdata_*.json, plus a documented window-MLP sequence
representative from Add-4) against each patient's own TRAIN-set base rate, per
target. The per-patient climatology is computed from the same train parquet
that backs the resolved headline cell (so e.g. migraine uses 70_30/chrono
when the composite migraine headline lives there). Positive means the model's
probabilities beat the trivial per-patient prior; the migraine bars are
negative despite AUROC ~0.76 - high discrimination that does not translate into
probabilistic value (Murphy quality-vs-value; Addition 6 Section 9a). Reuses
the Addition 5 prediction worker and the Addition 6 skill primitive.

Usage: python fig_d4_brier_skill.py
"""
# §11 compliance:
#   §11.1 PASS: per-architecture Brier-skill bootstrap 95% CI rendered as whiskers
#               (SK.brier_skill_ci, n_boot=1000, patient-day resampling unit)
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
                return e
    return None


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


def _climatology(target, datasplit, splittype):
    tr = load_raw(str(REPO / "data" / "processed" / target / datasplit / splittype / "diary_train.parquet"))
    rates = tr.groupby("patient_id")[TARGET_COL].mean()
    rates.index = rates.index.astype(str)
    return rates.to_dict(), float(tr[TARGET_COL].mean())


def _discover(headlines):
    leaves = {}
    cells = {}
    for tgt in ("headache", "migraine"):
        ls = []
        e0 = _resolve_leaf(headlines, tgt, "xgboost")
        if e0 is not None and (Path(e0["leaf_dir"]) / "model.joblib").exists():
            ls.append(("XGBoost stack", Path(e0["leaf_dir"])))
        e1 = _resolve_leaf(headlines, tgt, "tabpfn")
        if e1 is not None and (Path(e1["leaf_dir"]) / "model.joblib").exists():
            ls.append(("TabPFN", Path(e1["leaf_dir"])))
        d4 = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
        if (d4 / "model.joblib").exists():
            ls.append(("window-MLP", d4))
        leaves[tgt] = ls
        # Climatology must come from the headline cell's train parquet. Prefer
        # the xgboost entry's split; fall back to tabpfn if no xgboost row.
        e = e0 or e1
        cells[tgt] = (e["datasplit"], e["splittype"]) if e else ("70_30", "chrono")
    return leaves, cells


def main():
    S.apply()
    figdata_path = _F.latest_figdata(EXP / "2")
    if figdata_path is None:
        raise SystemExit("no figdata_*.json in experiment/2/ - run experiment/2/compare.py first")
    headlines = _F.load_figdata(figdata_path).get("headlines", [])
    print(f"  source {figdata_path.name} ({len(headlines)} headline rows)")
    leaves, cells = _discover(headlines)
    targets = ("headache", "migraine")
    labels = ["XGBoost stack", "TabPFN", "window-MLP"]
    skills = {t: {} for t in targets}
    skill_cis = {t: {} for t in targets}
    slugs: dict = {t: {} for t in targets}
    for tgt in targets:
        ds, sp = cells[tgt]
        print(f"  {tgt} climatology: {ds}/{sp}")
        rates, cohort = _climatology(tgt, ds, sp)
        for label, leaf in leaves[tgt]:
            r = _predict(leaf)
            if r is None:
                continue
            y, p, pid = r
            ref = SK.per_patient_climatology(pid, rates, cohort)
            ci = SK.brier_skill_ci(y, p, ref)
            skills[tgt][label] = ci["estimate"]
            skill_cis[tgt][label] = (ci["ci_low"], ci["ci_high"])
            slugs[tgt][label] = S.leaf_slug(leaf)
            print(f"  {tgt:<9} {label:<13} Brier skill {ci['estimate']:+.3f} "
                  f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({S.leaf_slug(leaf)})")
    fig, ax = plt.subplots(figsize=S.figsize("double", 4.4))
    x = np.arange(len(labels)); w = 0.38
    for i, tgt in enumerate(targets):
        vals = [skills[tgt].get(l, np.nan) for l in labels]
        lows = [skill_cis[tgt].get(l, (np.nan, np.nan))[0] for l in labels]
        highs = [skill_cis[tgt].get(l, (np.nan, np.nan))[1] for l in labels]
        yerr = np.array([
            [v - lo if (v == v and lo == lo) else 0 for v, lo in zip(vals, lows)],
            [hi - v if (v == v and hi == hi) else 0 for v, hi in zip(vals, highs)],
        ])
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=S.TARGET[tgt], alpha=0.85,
                       yerr=yerr, capsize=3, ecolor=S.SOFT, label=tgt)
        for b, v, lo, hi in zip(bars, vals, lows, highs):
            if v == v:
                # Centred above/below the bar top, the bracket landed straight on
                # the error whisker. Anchored to the bar's right edge and centred
                # on its own value it sits BESIDE the interval it reports, which
                # also makes clear which bar it belongs to.
                ax.annotate(f"{v:+.2f}\n[{lo:+.2f}, {hi:+.2f}]",
                            (b.get_x() + b.get_width(), v),
                            ha="left", va="center", fontsize=6.5,
                            xytext=(4, 0), textcoords="offset points")
    ax.axhline(0, color=S.REF_COLOR, lw=1)
    # Slug-bearing tick labels: the same family name can decode to different
    # leaves per target (e.g. TabPFN-v2.6 on headache vs TabPFN-v2.5f on
    # migraine via the composite-tracked headline). Disclose ARCH-VAR
    # alongside the family name so the reader can verify the cell.
    def _slug_arch(s):
        return s.split(" / ")[0] if s else "?"
    tick_labels = []
    for fam in labels:
        h_a = _slug_arch(slugs["headache"].get(fam, ""))
        m_a = _slug_arch(slugs["migraine"].get(fam, ""))
        if h_a == m_a and h_a != "?":
            tick_labels.append(f"{fam}\n{h_a}")
        elif h_a == "?" and m_a == "?":
            tick_labels.append(fam)
        else:
            tick_labels.append(f"{fam}\nh: {h_a}\nm: {m_a}")
    ax.set_xticks(x); ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.set_ylabel("Brier skill vs per-patient climatology")
    ax.set_title("Probabilistic value over the patient base rate\n"
                 "Park 2016 SHD, n=62; per-architecture bootstrap 95% CI (1000 iters, patient-day)")
    S.epv_annotation(ax, "migraine", cell="full_features", loc="lower right")
    ax.set_ylabel("Brier skill vs per-patient climatology\n(below 0 = worse than patient base rate)")
    ax.legend(title="target", loc="upper right")
    print("saved", S.save(fig, HERE / "figures" / "fig_d4_brier_skill"))


if __name__ == "__main__":
    main()
