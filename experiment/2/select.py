"""Headline / runner-up leaf selection for the Addition-2 insight pass.

Builds the (headline, runner-up) leaf list for the seven scientifically
defined ``(target, feature_set)`` cells (docs Section 4). The selection is
computed from the pure parsing layer (``_eval/_parsing`` + the holdout
contract + ``parse_mean_ci``), not the aggregator's script-local
``_collect_entries`` which carries module-level side effects.

Rules (docs Section 4):
  - Rule A (headline): top row per ``(target, feature_set)`` cell by mean
    test AUROC.
  - Rule B (runner-up): walking down the AUROC-ranked list, the first row
    whose 95% CI overlaps the headline's CI but whose architecture *family*
    differs. The runner-up answers "what almost won, and for the same
    reasons?".

Coverage is the seven cells with a scientific definition: the four feature
sets (full, spano, no_rolling, park) times the two targets, minus the
headache/park cell (Park's stepwise regression is migraine-specific, docs
Section 4). The result is at most fourteen leaf directories; cells lacking
a CI-overlapping cross-family runner-up contribute only the headline.
"""
from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR / "_eval"))

from _eval._parsing import (  # noqa: E402
    find_results_dirs, parse_path, find_latest_files, parse_file,
)
from _eval.sharedMetricPrinter import getContract  # noqa: E402
from _eval._metric_palette import parse_mean_ci  # noqa: E402

# The seven scientifically defined cells; headache/park_features is excluded
# (Park's stepwise trigger model is migraine-specific, docs Section 4).
TARGETS = ("headache", "migraine")
FEATURE_SETS = ("full_features", "spano_features", "no_rolling_features", "park_features")
EXCLUDED_CELLS = frozenset({("headache", "park_features")})


def arch_family(addition: str, architecture: str, version: str | None) -> str:
    """Map a leaf's path dimensions to its explainability dispatch family.

    Addition-0 architectures are calibrated XGBoost stacking/blending
    pipelines (``xgboost``); Addition-1 leaves are single-fit TabPFN
    (``tabpfn``) except the post-hoc-ensemble variant (``autotabpfn``),
    which has no single forward pass.
    """
    if addition == "0":
        return "xgboost"
    if version == "version_2-5-auto":
        return "autotabpfn"
    return "tabpfn"


def collect_holdout_rows() -> list[dict]:
    """Parse every leaf's latest hold-out result file into a flat row list.

    Each row carries the path dimensions, the architecture family, the
    leaf directory, and the parsed ``(mean, lo, hi)`` test AUROC. Rows with
    no parseable AUROC are dropped (they cannot be ranked or CI-compared).
    """
    contract = getContract()
    rows: list[dict] = []
    for rd in find_results_dirs(EXPERIMENT_DIR):
        dims = parse_path(rd, EXPERIMENT_DIR)
        if dims["addition"] not in ("0", "1"):
            continue
        holdout_file, _ = find_latest_files(rd)
        parsed = parse_file(holdout_file, contract)
        if not parsed:
            continue
        auroc = parse_mean_ci(parsed.get("AUROC"))
        if auroc is None or auroc[1] is None or auroc[2] is None:
            continue
        auprc = parse_mean_ci(parsed.get("AUPRC"))
        calib = parse_mean_ci(parsed.get("Calibration Slope"))
        rows.append({
            "addition": dims["addition"],
            "target": dims["target"],
            "feature_set": dims["feature_set"],
            "architecture": dims["architecture"],
            "version": dims["version"],
            "datasplit": dims["datasplit"],
            "splittype": dims["splittype"],
            "hp_strategy": dims["hp_strategy"],
            "hp_variant": dims.get("hp_variant"),
            "family": arch_family(dims["addition"], dims["architecture"], dims["version"]),
            "leaf_dir": rd.parent,
            "auroc_mean": auroc[0],
            "auroc_lo": auroc[1],
            "auroc_hi": auroc[2],
            "auprc_mean": auprc[0] if auprc else None,
            "auprc_lo": auprc[1] if auprc else None,
            "auprc_hi": auprc[2] if auprc else None,
            "calib_slope": calib[0] if calib else None,
            "calib_slope_lo": calib[1] if calib else None,
            "calib_slope_hi": calib[2] if calib else None,
        })
    return rows


def ci_overlap(a: dict, b: dict) -> bool:
    """True when the two rows' 95% AUROC CIs overlap.

    Two intervals ``[lo_a, hi_a]`` and ``[lo_b, hi_b]`` overlap iff
    ``lo_a <= hi_b`` and ``lo_b <= hi_a``. This is the inverse of the
    aggregator's strict CI-*separation* predicate; Rule B needs overlap, so
    the test is written directly rather than negating the separation rule
    (which only ever returns a single strict winner).
    """
    return a["auroc_lo"] <= b["auroc_hi"] and b["auroc_lo"] <= a["auroc_hi"]


SPLIT_TYPES = ("chrono", "stratified", "patient")


CALIB_MIN = 0.0   # slope <= 0 is Platt-inverted: calibrated probs anti-correlate
CALIB_MAX = 5.0   # slope this far above 1 means wildly mis-scaled probabilities
AUROC_TOL = 0.02  # AUROC differences below this are noise-level (treated as tied)
AUPRC_TOL = 0.02  # AUPRC differences below this are noise-level (treated as tied)


def _calibration_degenerate(row: dict) -> bool:
    """A leaf whose calibration slope is missing, non-positive (inverted) or
    wildly off has unreliable calibrated probabilities; SHAP on
    ``calibrated_proba`` would explain noise, so such leaves are de-prioritised.
    """
    s = row.get("calib_slope")
    return s is None or s <= CALIB_MIN or s > CALIB_MAX


def _calib_distance(row: dict) -> float:
    """Distance of the calibration slope from the ideal 1.0 (large if absent)."""
    s = row.get("calib_slope")
    return abs(s - 1.0) if s is not None else 1e9


def composite_sorted(group: list[dict]) -> list[dict]:
    """Sort a candidate group best-first for choosing the leaf to explain.

    Discrimination is primary, but only at a meaningful resolution: AUROC and
    AUPRC are bucketed at a noise-level tolerance, so leaves that differ by
    less than ``AUROC_TOL`` / ``AUPRC_TOL`` are treated as tied rather than
    letting a 0.001 edge decide. Calibration is a reliability factor, not an
    equal vote: degenerate calibration (inverted or wildly mis-scaled slope)
    sorts last so SHAP is never run on a model whose calibrated probabilities
    are meaningless, and among discrimination-tied leaves the one with
    calibration slope closest to 1 wins. The exact AUROC is the final
    tiebreak. This avoids both failure modes - calibration cannot override a
    real AUROC gap, and a trivial AUROC gap cannot override calibration.
    """
    def key(row):
        au = row.get("auroc_mean") or 0.0
        ap = row.get("auprc_mean") or 0.0
        return (
            _calibration_degenerate(row),          # False (0) before True
            -round(au / AUROC_TOL),                 # AUROC bucket (desc)
            -round(ap / AUPRC_TOL),                 # AUPRC bucket (desc)
            _calib_distance(row),                   # within tie: best calibration
            -au,                                    # exact AUROC tiebreak
        )
    return sorted(group, key=key)


def select_for_cell_split(rows, target, feature_set, split_type) -> list[dict]:
    """Headline + cross-family runner-up for one (target, feature_set,
    split_type), ranked by the multi-metric composite. The runner-up is the
    best-composite candidate of a different architecture family whose AUROC
    CI overlaps the headline's; absent that, the cell-split contributes the
    headline only.
    """
    cell = [r for r in rows if r["target"] == target
            and r["feature_set"] == feature_set and r["splittype"] == split_type]
    if not cell:
        return []
    ranked = composite_sorted(cell)
    headline = ranked[0]
    selection = [dict(headline, role="headline")]
    for cand in ranked[1:]:
        if cand["family"] != headline["family"] and ci_overlap(headline, cand):
            selection.append(dict(cand, role="runner_up"))
            break
    return selection


def select_by_split() -> list[dict]:
    """Headline + runner-up per (target, feature_set, split_type), so the
    cross-leaf comparison can be categorised by split type (chronological =
    honest, stratified/patient = the leakage and generalisation contrasts).
    """
    rows = collect_holdout_rows()
    out: list[dict] = []
    for target in TARGETS:
        for feature_set in FEATURE_SETS:
            if (target, feature_set) in EXCLUDED_CELLS:
                continue
            for split_type in SPLIT_TYPES:
                out.extend(select_for_cell_split(rows, target, feature_set, split_type))
    return out


def select_insight_leaves() -> list[dict]:
    """Full selection across all cells and split types (flattened)."""
    return select_by_split()


def _describe(sel: dict) -> str:
    ver = f"/{sel['version']}" if sel["version"] else ""
    auprc = f"{sel['auprc_mean']:.3f}" if sel.get("auprc_mean") is not None else "n/a"
    calib = f"{sel['calib_slope']:+.2f}" if sel.get("calib_slope") is not None else "n/a"
    return (f"{sel['role']:<10} {sel['target']}/{sel['feature_set']:<20} "
            f"{sel['splittype']:<11} AUROC={sel['auroc_mean']:.3f} AUPRC={auprc} "
            f"calib={calib} family={sel['family']:<10} {sel['architecture']}{ver} "
            f"{sel['datasplit']}")


if __name__ == "__main__":
    selections = select_by_split()
    n_cells = (len(TARGETS) * len(FEATURE_SETS) - len(EXCLUDED_CELLS)) * len(SPLIT_TYPES)
    print(f"Selected {len(selections)} insight leaves across up to {n_cells} "
          f"(cell x split) groups:\n")
    for s in selections:
        print("  " + _describe(s))
        print(f"             -> {s['leaf_dir'].relative_to(EXPERIMENT_DIR)}")
