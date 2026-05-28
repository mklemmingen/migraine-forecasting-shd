"""Two-fold leave-one-site-out external validation driver (Route A).

Geographic internal-external validation across the two SHD recruitment sites
[steyerberg2016validation, p. 245]: for each held-out site, every model is refit
on the other site and applied to the held-out site's unseen patients. Three
transport questions per fold (docs/external_validation_site.md Section 3):

  1. discrimination transport - pooled AUROC/AUPRC, in the standard bootstrap
     contract folded into comparison_*.html (datasplit=external,
     splittype=site_<held>);
  2. within-person transport - the precision-weighted within-person C-statistic
     on the held-out site (reusing _personal/within_person.py), testing whether
     the near-chance per-patient result replicates off-site;
  3. calibration transport - calibration-in-the-large and slope on the held-out
     site [huang2020calibration, p. 621]. Because the sites differ in base rate, a
     model trained on one must mis-set its mean predicted risk on the other, and
     that intercept drift is the distinctive transportability result.

The LR regimes run in-process; the architectures (Additions 0/1/4) refit through
_personal/_site_worker.py in a per-addition subprocess (the GPU is shown only to
Addition 1). Imbalance handling is unchanged (no reweighting; external threshold).
"""
import datetime as _dt
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent      # experiment/5/
EXP = HERE.parent
REPO = EXP.parent
sys.path[0:0] = [str(EXP), str(HERE / "_personal")]
import within_person as WP  # noqa: E402
import regimes as RG  # noqa: E402
import _site as SITE  # noqa: E402
from _dataRead.read import (  # noqa: E402
    load_raw, chronological_subsplit, NON_FEATURE_COLS, TARGET_COL)
from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
from _dataRead.filter_to_park_features import select_park_features  # noqa: E402
from _eval.metrics_lib import calibration_slope  # noqa: E402

SITE_WORKER = HERE / "_personal" / "_site_worker.py"
LOADERS = {"full_features": None,
           "no_rolling_features": select_non_rolling_features,
           "park_features": select_park_features}
ARCH_LABEL = {"0": "add0_stacked", "1": "add1_tabpfn", "4": "add4_window_mlp"}


def _env(addition: str) -> dict:
    env = os.environ.copy()
    if addition != "1":  # GPU only for TabPFN; CPU for XGBoost/sequence
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
    return env


def calibration_transport(y, p) -> dict:
    """Calibration-in-the-large (mean observed vs mean predicted) and slope on a
    held-out site. citl<0 and oe_ratio<1 both denote over-prediction - the
    expected sign when a higher-base-rate site's model is applied to a lower one."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    obs, pred = float(y.mean()), float(p.mean())
    return {"observed_rate": round(obs, 4), "mean_predicted": round(pred, 4),
            "oe_ratio": round(obs / pred, 3) if pred > 0 else float("nan"),
            "citl": round(obs - pred, 4), "cal_slope": round(calibration_slope(y, p), 3)}


def transport_bootstrap_ci(y, p, n_boot: int = 500, seed: int = 42) -> dict:
    """Patient-day bootstrap 95% CI on the external-site transport metrics.
    Returns dict of (lo, hi) tuples for ``auroc``, ``oe_ratio``, ``cal_slope``.

    Patient-day resampling per body §2.8; under-coverage caveat against
    patient-cluster bootstrap disclosed in body Methods."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    aucs, oes, slopes = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]; ps = p[idx]
        if 0 < ys.sum() < n:
            try:
                aucs.append(float(roc_auc_score(ys, ps)))
            except Exception:
                pass
            mp = float(ps.mean())
            if mp > 0:
                oes.append(float(ys.mean() / mp))
            try:
                slopes.append(float(calibration_slope(ys, ps)))
            except Exception:
                pass
    out = {}
    for name, arr in (("auroc", aucs), ("oe_ratio", oes), ("cal_slope", slopes)):
        if arr:
            a = np.array(arr)
            out[f"{name}_ci_low"] = round(float(np.percentile(a, 2.5)), 4)
            out[f"{name}_ci_high"] = round(float(np.percentile(a, 97.5)), 4)
        else:
            out[f"{name}_ci_low"] = float("nan")
            out[f"{name}_ci_high"] = float("nan")
    return out


def _arch_predict(leaf: Path, held: str, addition: str):
    out = Path(tempfile.gettempdir()) / f"siteA_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(SITE_WORKER), str(leaf), held, str(out)],
                           capture_output=True, text=True, timeout=2400, env=_env(addition))
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr.strip().splitlines() or ["worker failed"])[-1]
            print(f"    SKIP {ARCH_LABEL[addition]} site_{held}: {tail}")
            return None
        z = np.load(out, allow_pickle=True)
        return {k: z[k] for k in z.files}
    finally:
        out.unlink(missing_ok=True)


def _record(rows, model, target, feature_set, held, train_rate,
            y_val, p_val, y_test, p_test, pid, emit=True):
    """Append the transport-metrics row; emit the standard contract unless emit is
    False (the figure pass regenerates only the summary CSV, without re-writing the
    already-committed results_*.txt cells)."""
    if emit:
        out_dir = HERE / target / feature_set / model / "external" / f"site_{held}"
        RG.emit_holdout_results(str(out_dir), f"ROUTE A / {feature_set} / {model} / site_{held}",
                                pd.Series(np.asarray(y_val, dtype=float)), np.asarray(p_val, dtype=float),
                                pd.Series(np.asarray(y_test, dtype=float)), np.asarray(p_test, dtype=float))
    y = np.asarray(y_test, dtype=float); p = np.asarray(p_test, dtype=float)
    pid = np.asarray(pid).astype(str)
    within = WP.within_person_cstatistic(WP.per_patient_scores(y, p, pid, WP.MIN_POS))
    cal = calibration_transport(y, p)
    cis = transport_bootstrap_ci(y, p)
    within_ci_low = round(float(within["ci_low"]), 3) if within["ci_low"] == within["ci_low"] else float("nan")
    within_ci_high = round(float(within["ci_high"]), 3) if within["ci_high"] == within["ci_high"] else float("nan")
    row = {"target": target, "feature_set": feature_set, "model": model, "held_out": held,
           "n_test": int(len(y)), "train_rate": round(train_rate, 4),
           "auroc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else float("nan"),
           "auprc": round(float(average_precision_score(y, p)), 4),
           "within_cstat": round(float(within["estimate"]), 3) if within["estimate"] == within["estimate"] else float("nan"),
           "within_ci_low": within_ci_low, "within_ci_high": within_ci_high,
           "within_k": int(within["k_estimable"]), **cal, **cis}
    rows.append(row)
    print(f"  {model:<16} site_{held:<10} AUROC {row['auroc']:.3f} | within {row['within_cstat']} "
          f"(k={row['within_k']}) | O:E {cal['oe_ratio']} slope {cal['cal_slope']} "
          f"(pred {cal['mean_predicted']:.3f} vs obs {cal['observed_rate']:.3f})")
    return row


def _load_figdata_headlines() -> list[dict]:
    """Headlines array from the latest experiment/2/figdata_*.json. See the
    matching helper in run_personalization.py for rationale."""
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
    F = _ilu.module_from_spec(spec); spec.loader.exec_module(F)
    fp = F.latest_figdata(EXP / "2")
    if fp is None:
        return []
    return F.load_figdata(fp).get("headlines", [])


def _resolve_composite_leaf(headlines: list[dict], target: str, family: str) -> Path | None:
    """Composite-tracked leaf for (target, family) in the full_features/chrono
    headline cell. Mirrors the helper in run_personalization.py."""
    for role in ("headline", "runner_up"):
        for e in headlines:
            if (e.get("target") == target and e.get("feature_set") == "full_features"
                    and e.get("splittype") == "chrono" and e.get("role") == role
                    and e.get("family") == family and e.get("leaf_dir")):
                return Path(e["leaf_dir"])
    return None


def default_leaves() -> dict:
    """{target -> [Add 0/1/4 leaf model dirs]} on the composite headline cell.
    Add-0 and Add-1 are resolved from experiment/2/figdata_*.json so they track
    composite_sorted; Add-4 stays pinned to a documented window-MLP leaf as the
    cross-addition contrast (composite selection covers Add-0 and Add-1 only)."""
    headlines = _load_figdata_headlines()
    out: dict = {}
    for tgt in ("headache", "migraine"):
        ls: list[Path] = []
        d0 = _resolve_composite_leaf(headlines, tgt, "xgboost")
        if d0 is not None and (d0 / "model.joblib").exists():
            ls.append(d0)
        d1 = _resolve_composite_leaf(headlines, tgt, "tabpfn")
        if d1 is not None and (d1 / "model.joblib").exists():
            ls.append(d1)
        d4 = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
        if (d4 / "model.joblib").exists():
            ls.append(d4)
        out[tgt] = ls
    return out


def run_cell(target: str, feature_set: str, leaves: list, emit: bool = True) -> list:
    df = SITE.attach_site(load_raw(
        str(REPO / "data" / "processed" / target / "diary_cv5_timeseries.parquet"),
        loader=LOADERS.get(feature_set))).dropna(subset=["site"])
    fc = [c for c in df.columns if c not in NON_FEATURE_COLS and c != "site"]
    rows = []
    print(f"\n=== {target} / {feature_set} (sites: "
          + ", ".join(f"{s}={int((df.site == s).sum())}r/"
                      f"{df.loc[df.site == s, TARGET_COL].mean():.1%}" for s in SITE.SITES) + ") ===")
    for held in SITE.SITES:
        train = df[df["site"] != held]; test = df[df["site"] == held]
        if train.empty or test.empty:
            continue
        tr, ca = chronological_subsplit(train)
        train_rate = float(train[TARGET_COL].mean())
        # Under patient-disjoint site validation the held-out site's patients are
        # entirely unseen, so per_patient and partial_pool have nothing learned
        # for them and collapse exactly to the pooled global model; only pooled is
        # run (its row stands in for all three, and that collapse is itself the
        # cold-start result - personalisation cannot transfer to a new patient).
        p_val, p_test = RG.pooled(tr, ca, test, fc)
        _record(rows, "pooled_lr", target, feature_set, held, train_rate,
                ca[TARGET_COL], p_val, test[TARGET_COL], p_test, test["patient_id"], emit=emit)
        for leaf in leaves:
            add = leaf.relative_to(EXP).parts[0]
            z = _arch_predict(leaf, held, add)
            if z is None:
                continue
            _record(rows, ARCH_LABEL[add], target, feature_set, held, train_rate,
                    z["y_val"], z["p_val"], z["y_test"], z["p_test"], z["pid_test"], emit=emit)
    return rows


def main(feature_set: str = "full_features", emit: bool = True):
    leaves = default_leaves()
    rows = []
    for target in ("headache", "migraine"):
        rows += run_cell(target, feature_set, leaves.get(target, []), emit=emit)
    if rows:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = HERE / f"external_site_summary_{ts}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved summary: {out}")
        if emit:
            print("Site-holdout cells emit the standard contract -> comparison_*.html "
                  "(datasplit=external, splittype=site_<held>)")


if __name__ == "__main__":
    main(emit="--no-emit" not in sys.argv)
