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
        rows.append({
            "addition": dims["addition"],
            "target": dims["target"],
            "feature_set": dims["feature_set"],
            "architecture": dims["architecture"],
            "version": dims["version"],
            "datasplit": dims["datasplit"],
            "splittype": dims["splittype"],
            "hp_strategy": dims["hp_strategy"],
            "family": arch_family(dims["addition"], dims["architecture"], dims["version"]),
            "leaf_dir": rd.parent,
            "auroc_mean": auroc[0],
            "auroc_lo": auroc[1],
            "auroc_hi": auroc[2],
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


def select_for_cell(rows: list[dict], target: str, feature_set: str) -> list[dict]:
    """Apply Rule A then Rule B for one ``(target, feature_set)`` cell.

    Returns ``[headline]`` or ``[headline, runner_up]``. The runner-up is
    the first CI-overlapping row whose architecture family differs from the
    headline's; if none exists the cell contributes the headline only.
    """
    cell = [r for r in rows if r["target"] == target and r["feature_set"] == feature_set]
    if not cell:
        return []
    ranked = sorted(cell, key=lambda r: r["auroc_mean"], reverse=True)
    headline = ranked[0]
    selection = [dict(headline, role="headline")]
    for cand in ranked[1:]:
        if cand["family"] != headline["family"] and ci_overlap(headline, cand):
            selection.append(dict(cand, role="runner_up"))
            break
    return selection


def select_insight_leaves() -> list[dict]:
    """Compute the full headline + runner-up selection across the 7 cells."""
    rows = collect_holdout_rows()
    out: list[dict] = []
    for target in TARGETS:
        for feature_set in FEATURE_SETS:
            if (target, feature_set) in EXCLUDED_CELLS:
                continue
            out.extend(select_for_cell(rows, target, feature_set))
    return out


def _describe(sel: dict) -> str:
    ver = f"/{sel['version']}" if sel["version"] else ""
    return (f"{sel['role']:<10} {sel['target']}/{sel['feature_set']:<20} "
            f"AUROC={sel['auroc_mean']:.3f} [{sel['auroc_lo']:.3f}-{sel['auroc_hi']:.3f}] "
            f"family={sel['family']:<10} {sel['architecture']}{ver} "
            f"{sel['datasplit']}/{sel['splittype']}")


if __name__ == "__main__":
    selections = select_insight_leaves()
    print(f"Selected {len(selections)} insight leaves "
          f"across {len(TARGETS) * len(FEATURE_SETS) - len(EXCLUDED_CELLS)} cells:\n")
    for s in selections:
        print("  " + _describe(s))
        print(f"             -> {s['leaf_dir'].relative_to(EXPERIMENT_DIR)}")
