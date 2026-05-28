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
    # Capture the resolved leaf per (target, family-label) so the x-axis
    # tick labels carry the slug rather than the generic family name.
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
            skills[tgt][label] = SK.brier_skill_score(y, p, ref)
            slugs[tgt][label] = S.leaf_slug(leaf)
            print(f"  {tgt:<9} {label:<13} Brier skill {skills[tgt][label]:+.3f}  ({S.leaf_slug(leaf)})")
    fig, ax = plt.subplots(figsize=S.figsize("double", 4.2))
    x = np.arange(len(labels)); w = 0.38
    for i, tgt in enumerate(targets):
        vals = [skills[tgt].get(l, np.nan) for l in labels]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=S.TARGET[tgt], alpha=0.85, label=tgt)
        for b, v in zip(bars, vals):
            if v == v:   # skip NaN
                ax.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=7,
                            xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
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
    ax.set_title("Probabilistic value over the patient base rate")
    ax.text(0.02, 0.04, "below 0 = worse than predicting the patient's own base rate",
            transform=ax.transAxes, fontsize=7.5, style="italic", color=S.GREY)
    ax.legend(title="target")
    print("saved", S.save(fig, HERE / "figures" / "fig_d4_brier_skill"))


if __name__ == "__main__":
    main()
