"""Driver for the clinical-forecast-value layer (Addition 6).

Post-hoc pass over the selected leaf models from Additions 0/1/4 (and, where
present, the Addition 5 regimes): regenerate their out-of-sample predictions
(reusing the Addition 5 subprocess worker, so the conflicting _model_architecture
packages and the TabPFN-GPU/CPU split stay isolated), then add the value layer:

  1. decision curves (net benefit vs threshold) overlaid across architectures per
     (target, feature_set) cell - the cross-addition value comparison;
  2. Brier skill vs each patient's TRAIN-set climatology (a leakage-free
     per-patient base rate);
  3. the operating point: net-benefit-optimal threshold, the MCC threshold for
     reference, and sensitivity at a tolerated false-alarm rate.

Adds no core model and makes no clinical-utility claim beyond the development
stage [vasey2022decideAI, p. 1]. The optional measured-weather ablation
(_value/weather_join.py) is left for when that external data is sourced.
Design and decisions: docs/addition6_clinical_value.md.
"""
import datetime as _dt
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent      # experiment/6/
EXP = HERE.parent                           # experiment/
REPO = EXP.parent
sys.path[0:0] = [str(EXP), str(HERE), str(HERE / "_value")]
import decision_curve as DC  # noqa: E402
import skill as SK  # noqa: E402
import operating_point as OP  # noqa: E402
from _dataRead.read import load_raw, TARGET_COL  # noqa: E402
from _eval.metrics_lib import find_operating_thresholds  # noqa: E402

WP_WORKER = EXP / "5" / "_personal" / "_predict_worker.py"
RATIOS = ("70_15_15", "70_30", "80_20")
SPLITS = ("chrono", "stratified", "patient")


def _dims(model_dir: Path) -> dict:
    parts = model_dir.relative_to(EXP).parts
    return {"addition": parts[0], "target": parts[1], "feature_set": parts[2],
            "architecture": parts[3] if len(parts) > 3 else "?",
            "ratio": next((p for p in parts if p in RATIOS), "?"),
            "split": next((p for p in parts if p in SPLITS), "?")}


def _env(addition: str) -> dict:
    env = os.environ.copy()
    if addition != "1":  # GPU only for TabPFN; CPU for XGBoost/sequence
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
    return env


def _predict(model_dir: Path):
    """(y, p, patient_id) on the leaf's val+test horizon via the shared worker."""
    d = _dims(model_dir)
    out = Path(tempfile.gettempdir()) / f"v6_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(WP_WORKER), str(model_dir), str(out)],
                           capture_output=True, text=True, timeout=1800, env=_env(d["addition"]))
        if r.returncode != 0 or not out.exists():
            raise RuntimeError((r.stderr.strip().splitlines() or ["worker failed"])[-1])
        z = np.load(out)
        return z["y"], z["p"], z["pid"], d
    finally:
        out.unlink(missing_ok=True)


def _train_climatology(target, ratio, split):
    tr = load_raw(str(REPO / "data" / "processed" / target / ratio / split / "diary_train.parquet"))
    rates = tr.groupby("patient_id")[TARGET_COL].mean()
    rates.index = rates.index.astype(str)
    return rates.to_dict(), float(tr[TARGET_COL].mean())


def value_for_leaf(model_dir: Path) -> tuple:
    y, p, pid, d = _predict(model_dir)
    rates, cohort = _train_climatology(d["target"], d["ratio"], d["split"])
    ref = SK.per_patient_climatology(pid, rates, cohort)
    dc = DC.decision_curve_ci(y, p)
    mcc_t, _ = find_operating_thresholds(pd.Series(y), p)
    om = OP.map_threshold(mcc_t, dc["thresholds"], dc["model"])
    thr, sens, fpr = OP.sensitivity_at_fpr(y, p, target_fpr=0.10)
    # threshold band where the model's net benefit beats treat-all and treat-none
    beats = (dc["model"] > np.maximum(dc["treat_all"], 0.0))
    band = dc["thresholds"][beats]
    skill_ci = SK.brier_skill_ci(y, p, ref)
    citl_ci = SK.citl_ci(y, p)
    row = {"addition": d["addition"], "target": d["target"],
           "feature_set": d["feature_set"], "architecture": d["architecture"],
           "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
           "brier_skill": skill_ci["estimate"],
           "brier_skill_ci_low": skill_ci["ci_low"],
           "brier_skill_ci_high": skill_ci["ci_high"],
           "citl": citl_ci["estimate"],
           "citl_ci_low": citl_ci["ci_low"],
           "citl_ci_high": citl_ci["ci_high"],
           "nb_optimal_threshold": om["net_benefit_optimal_threshold"],
           "nb_at_optimal": om["net_benefit_at_optimal"],
           "sens_at_fpr0.10": sens,
           "nb_positive_band": f"{band.min():.2f}-{band.max():.2f}" if band.size else "none"}
    return row, d, dc


def _plot_cell(cell, curves, out_png):
    plt.figure(figsize=(6, 4))
    any_ref = next(iter(curves.values()))
    t = any_ref["thresholds"]
    plt.plot(t, any_ref["treat_all"], "--", color="grey", lw=1, label="treat all")
    plt.plot(t, any_ref["treat_none"], ":", color="black", lw=1, label="treat none")
    for arch, dc in curves.items():
        plt.plot(dc["thresholds"], dc["model"], lw=1.5, label=arch)
    plt.ylim(bottom=min(-0.01, float(any_ref["treat_all"].min())))
    plt.xlabel("threshold probability"); plt.ylabel("net benefit")
    plt.title(f"Decision curve - {cell[0]} / {cell[1]}")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(out_png, dpi=130); plt.close()


def _load_figdata_headlines() -> list[dict]:
    """Headlines array from the latest experiment/2/figdata_*.json. The Addition
    6 leaves are resolved from this so they track the composite_sorted
    selection driving the headline table; mirrors the helper in
    run_personalization.py."""
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
    F = _ilu.module_from_spec(spec); spec.loader.exec_module(F)
    fp = F.latest_figdata(EXP / "2")
    if fp is None:
        return []
    return F.load_figdata(fp).get("headlines", [])


def _resolve_composite_leaf(headlines: list[dict], target: str, family: str) -> Path | None:
    """Composite-tracked leaf for (target, family) on the full_features/chrono
    headline cell. Prefer headline-role; runner-up is also valid when the
    cross-architecture headline lives in the other family."""
    for role in ("headline", "runner_up"):
        for e in headlines:
            if (e.get("target") == target and e.get("feature_set") == "full_features"
                    and e.get("splittype") == "chrono" and e.get("role") == role
                    and e.get("family") == family and e.get("leaf_dir")):
                return Path(e["leaf_dir"])
    return None


def default_leaves() -> list[Path]:
    """Cross-architecture set on the composite headline cell (full_features
    chronological): Add-0 XGBoost and Add-1 TabPFN resolved from
    experiment/2/figdata_*.json so they track composite_sorted; Add-4 sequence
    pinned to a documented window-MLP leaf as the cross-addition contrast."""
    headlines = _load_figdata_headlines()
    leaves: list[Path] = []
    for tgt in ("headache", "migraine"):
        d0 = _resolve_composite_leaf(headlines, tgt, "xgboost")
        if d0 is not None and (d0 / "model.joblib").exists():
            leaves.append(d0)
        d1 = _resolve_composite_leaf(headlines, tgt, "tabpfn")
        if d1 is not None and (d1 / "model.joblib").exists():
            leaves.append(d1)
        d4 = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
        if (d4 / "model.joblib").exists():
            leaves.append(d4)
    return leaves


def main(leaves=None):
    leaves = leaves or default_leaves()
    print(f"Clinical-forecast-value layer over {len(leaves)} leaf(s)")
    rows, curves = [], {}
    for leaf in leaves:
        try:
            row, d, dc = value_for_leaf(leaf)
            rows.append(row)
            curves.setdefault((d["target"], d["feature_set"]), {})[
                f"add{d['addition']} {d['architecture']}"] = dc
            print(f"  add{row['addition']} {row['target']:<8} {row['architecture']:<18} "
                  f"AUROC {row['auroc']:.3f} | Brier skill {row['brier_skill']:+.3f} "
                  f"[{row['brier_skill_ci_low']:+.3f}, {row['brier_skill_ci_high']:+.3f}] | "
                  f"CITL {row['citl']:.2f} [{row['citl_ci_low']:.2f}, {row['citl_ci_high']:.2f}] | "
                  f"NB+band {row['nb_positive_band']}")
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {leaf.relative_to(EXP)}: {type(e).__name__}: {e}")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for cell, cv in curves.items():
        _plot_cell(cell, cv, HERE / f"decision_curve_{cell[0]}_{cell[1]}_{ts}.png")
    if rows:
        out = HERE / f"value_summary_{ts}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved summary: {out} and decision-curve figures")


if __name__ == "__main__":
    main()
