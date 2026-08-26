"""Figure C2 - within-person discrimination (the load-bearing finding).

Per-patient hold-out AUROC for the TabPFN leaf in each target's headline cell
(full_features, chronological), pooled across the 5 expanding-window CV folds so
every patient with >=5 positive days is estimable. Each marker is one patient
with its Hanley-McNeil 95% CI; reference lines are chance (0.5), the pooled
AUROC with its bootstrap 95% CI band, and the precision-weighted within-person
C-statistic (DerSimonian-Laird random effects) with its CI band. The pooled
AUROC sits far above a per-patient centre near chance: pooled discrimination
is mostly between-patient base-rate separation, not within-person day-to-day
ranking.

Usage: python fig_c2_within_person.py   (refits per CV fold; minutes, GPU for TabPFN)
"""
# §11 compliance (figure_design_requirements.md, reviewer-derived 2026-05-28):
#   §11.1 CIs on headline metric:        Hanley-McNeil per-patient whiskers; bootstrap
#                                        CI band on pooled AUROC ref line; DL CI band on
#                                        within-person C-stat ref line
#   §11.2 estimability denominators:     k=len(est) annotated in xlabel + on panel
#   §11.3 self-contained caption:        S.caption_block() composed for suptitle
#   §11.7 EPV-5.5 annotation:            S.epv_annotation() on migraine panel
#   §11.10 no "substantial"/"large":     verified in captions
#   §11.11 self-check:                   this block
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
sys.path.insert(0, str(EXP / "5" / "_personal"))
import within_person as WP  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# Reuse the figdata loader the g/h scripts use so leaf-path resolution stays
# in lockstep with the composite_sorted selection in experiment/2/select.py.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
_F = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_F)

CV_WORKER = EXP / "5" / "_personal" / "_cv_oof_worker.py"
MIN_POS = 5


def _rebase_leaf(leaf_dir: str) -> Path:
    """Rebase a figdata ``leaf_dir`` onto this checkout.

    figdata stores absolute paths captured at run time, so they break whenever
    the repository is moved or cloned elsewhere. Everything from the
    ``experiment/`` component onwards is stable, so re-root that suffix on the
    current EXP parent and fall back to the literal path if the shape is
    unexpected.
    """
    p = Path(leaf_dir)
    parts = p.parts
    if "experiment" in parts:
        return EXP.parent.joinpath(*parts[parts.index("experiment"):])
    return p


def _resolve_tabpfn_leaf(headlines, target):
    """Return the figdata-tracked TabPFN leaf for the headline cell
    (full_features/chrono) of ``target``: prefer the headline if family=tabpfn,
    otherwise the runner-up. Returns None when neither row is tabpfn (e.g. if
    figdata pre-dates the leaf_dir field)."""
    for role in ("headline", "runner_up"):
        for e in headlines:
            if (e.get("target") == target and e.get("feature_set") == "full_features"
                    and e.get("splittype") == "chrono" and e.get("role") == role
                    and e.get("family") == "tabpfn" and e.get("leaf_dir")):
                return _rebase_leaf(e["leaf_dir"])
    return None


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


# The 3600 s default was sized for GPU runs. TabPFN's cost grows steeply with
# context length on CPU, where a single 5-fold leaf can exceed an hour, so the
# refit would time out and lose all of its work. Override with CV_TIMEOUT_S.
CV_TIMEOUT_S = int(os.environ.get("CV_TIMEOUT_S", 6 * 3600))

CV_CACHE = HERE / "figures" / "_cv_oof_cache"


def _cache_path(leaf: Path) -> Path:
    """Stable cache filename for a leaf's out-of-fold predictions."""
    return CV_CACHE / (leaf.relative_to(EXP).as_posix().replace("/", "__") + ".npz")


def _cv_predict(leaf: Path):
    """Return (y, p, patient_id) out-of-fold predictions for ``leaf``.

    The CV refit costs minutes per leaf on CPU, so the arrays are cached under
    ``figures/_cv_oof_cache/``. Earlier revisions deleted the worker's npz in a
    finally block, which meant every consumer paid the full refit again and the
    per-patient values existed nowhere on disk. Delete the cache file to force a
    recompute.
    """
    add = leaf.relative_to(EXP).parts[0]
    cached = _cache_path(leaf)
    if cached.exists():
        z = np.load(cached)
        print(f"  cache hit {cached.name}")
        return z["y"].astype(float), z["p"].astype(float), z["pid"].astype(str)
    out = Path(tempfile.gettempdir()) / f"c2_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(CV_WORKER), str(leaf), str(out)],
                           capture_output=True, text=True, timeout=CV_TIMEOUT_S, env=_env(add))
        if r.returncode != 0 or not out.exists():
            print(f"  FAIL {leaf.name} (returncode={r.returncode}, out_exists={out.exists()})")
            if r.stderr:
                print("  --- worker stderr ---")
                for line in r.stderr.strip().splitlines():
                    print(f"    {line}")
                print("  --- end stderr ---")
            return None
        z = np.load(out)
        CV_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cached, y=z["y"], p=z["p"], pid=z["pid"])
        print(f"  cached {cached.name}")
        return z["y"].astype(float), z["p"].astype(float), z["pid"].astype(str)
    finally:
        out.unlink(missing_ok=True)


def _bootstrap_pooled_auroc_ci(y, p, n_boot=1000, seed=42):
    """Patient-day bootstrap CI on pooled AUROC. Matches the Methods
    section's patient-day resampling discipline (the under-coverage caveat
    against patient-cluster bootstrap is disclosed there)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        if 0 < ys.sum() < n:
            aucs.append(roc_auc_score(ys, p[idx]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _panel(ax, tgt, y, p, pid):
    scores = WP.per_patient_scores(y, p, pid, MIN_POS)
    est = scores[scores["estimable"]].sort_values("auroc").reset_index(drop=True)
    within = WP.within_person_cstatistic(scores)
    pooled = float(roc_auc_score(y, p))
    pooled_lo, pooled_hi = _bootstrap_pooled_auroc_ci(y, p)
    x = np.arange(len(est))
    err = np.array([1.96 * np.sqrt(WP._hanley_mcneil_var(
        r.auroc, int(r.n_pos), int(r.n - r.n_pos))) for r in est.itertuples()])
    ax.errorbar(x, est["auroc"], yerr=err, fmt="o", ms=3, lw=0.5,
                color=S.TARGET[tgt], alpha=0.6, ecolor=S.FAINT, capsize=0)
    ax.axhspan(within["ci_low"], within["ci_high"], color=S.GREY,
               alpha=S.CI_ALPHA, zorder=0)
    ax.axhspan(pooled_lo, pooled_hi, color=S.ARCH["TabPFN"],
               alpha=0.12, zorder=0)
    S.refline(ax, y=0.5, label="chance (0.5)")
    ax.axhline(within["estimate"], color=S.SOFT, lw=1.5,
               label=f"within-person C {within['estimate']:.2f} "
                     f"[{within['ci_low']:.2f}-{within['ci_high']:.2f}]")
    ax.axhline(pooled, color=S.ARCH["TabPFN"], lw=1.6,
               label=f"pooled AUROC {pooled:.2f} "
                     f"[{pooled_lo:.2f}-{pooled_hi:.2f}]")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"patient, sorted by AUROC ({len(est)} of 63 estimable)")
    ax.set_ylabel("per-patient AUROC")
    ax.legend(loc="upper left")
    S.epv_annotation(ax, tgt, cell="full_features", loc="lower right")
    print(f"  {tgt:<9} pooled {pooled:.3f} [{pooled_lo:.3f}-{pooled_hi:.3f}] "
          f"| within {within['estimate']:.3f} "
          f"[{within['ci_low']:.3f}-{within['ci_high']:.3f}] | k={len(est)}")


def main():
    S.apply()
    figdata_path = _F.latest_figdata(EXP / "2")
    if figdata_path is None:
        raise SystemExit("no figdata_*.json in experiment/2/ - run experiment/2/compare.py first")
    headlines = _F.load_figdata(figdata_path).get("headlines", [])
    print(f"  source {figdata_path.name} ({len(headlines)} headline rows)")
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.6))
    for _ax, _lt in zip(axes.ravel(), "abcdefgh"):
        S.panel_label(_ax, _lt)
    for ax, tgt in zip(axes, ("headache", "migraine")):
        leaf = _resolve_tabpfn_leaf(headlines, tgt)
        if leaf is None:
            print(f"  no tabpfn entry for {tgt} in figdata; skipping panel "
                  "(re-run experiment/2/compare.py to refresh leaf_dir)")
            continue
        print(f"  {tgt} TabPFN leaf: {leaf.relative_to(EXP)}")
        r = _cv_predict(leaf)
        if r is not None:
            _panel(ax, tgt, *r)
    print("saved", S.save(fig, HERE / "figures" / "fig_c2_within_person"))


if __name__ == "__main__":
    main()
