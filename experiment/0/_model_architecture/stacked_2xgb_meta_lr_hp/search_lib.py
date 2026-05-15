"""
Hyperparameter-search helpers for the stacked_2xgb_meta_lr_hp variant.

Pure logic: trajectory I/O, Pareto-frontier extraction, knee picking, and
the two search runners (RandomSampler + NSGA-II). The objective callable
that fits an XGBoost stacker and returns validation metrics lives in
``model.py`` so this file has no ML dependencies of its own.

Trajectory file format (``trajectory.jsonl``)
---------------------------------------------
One JSON object per line. Stable schema:

    {
      "trial_id":   int,                              # 0-indexed
      "params":     {n_estimators, learning_rate,
                     subsample, colsample_bytree,
                     min_child_weight},
      "metrics":    {auroc, auprc, brier, ece10,
                     calibration_slope, slope_dist_to_1, ...},
      "objectives": list[float]                       # empty for single-obj
                                                      # 2-tuple for NSGA-II
    }

Search space
------------
Five shared dimensions across the two base learners; structural max_depth
is held fixed (shallow=3, deep=6) so the stacked ensemble retains its
diverse-depth design intent. Search ranges follow XGBoost-on-small-tabular
conventions: n_estimators 50-500 keeps the per-tree budget reachable on a
~3.9k-row dataset; max_depth bands and min_child_weight regularise the
shallow/deep pair; subsample and colsample_bytree open a row/column
bagging dimension that defaults trivially to 1.0 in the NonHP baseline.

Selection rules
---------------
- single_AUROC tier (HP020, HP050, HP100, HP200, HP500): best validation
  AUROC among the first N trials of the deterministic RandomSampler run.
  Same seed across N values means the first 20 trials of a 500-trial run
  are bit-identical to a stand-alone 20-trial run, giving the strict
  HP020 subset HP050 subset HP100 subset HP200 subset HP500 nesting.
- pareto_<x>_<y> operating points (auroc_max | knee | slope_closest):
  selected from the non-dominated subset of the NSGA-II trajectory. The
  knee is picked by the maximum perpendicular distance to the chord
  spanning the two extreme points of the frontier, after normalising
  both axes to [0, 1] [Satopaa et al. 2011, IEEE ICDCSW].
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Search space and structural parameters
# ---------------------------------------------------------------------------

HP_SEARCH_SPACE = {
    "n_estimators":     {"low": 50,   "high": 500, "kind": "int"},
    "learning_rate":    {"low": 0.01, "high": 0.3, "kind": "float", "log": True},
    "subsample":        {"low": 0.6,  "high": 1.0, "kind": "float"},
    "colsample_bytree": {"low": 0.6,  "high": 1.0, "kind": "float"},
    "min_child_weight": {"low": 1,    "high": 10,  "kind": "int"},
}

STRUCTURAL_PARAMS = {
    # Fixed to preserve the shallow/deep diversity that motivates stacking.
    "max_depth_shallow": 3,
    "max_depth_deep":    6,
}

# Variant identifier conventions. Names sort lexicographically into the
# same order a human would read them, which matters for the aggregator's
# row sort and for the comparison table's vertical layout.
SINGLE_OBJ_TIERS  = ("HP020", "HP050", "HP100", "HP200", "HP500")
PARETO_VARIANTS   = ("auroc_max", "knee", "slope_closest")
PARETO_VARIANTS_AUPRC = ("auprc_max", "knee", "slope_closest")
TIER_TO_TRIAL_CAP = {"HP020": 20, "HP050": 50, "HP100": 100,
                     "HP200": 200, "HP500": 500}


# ---------------------------------------------------------------------------
# Optuna -> params dict
# ---------------------------------------------------------------------------

def suggest_params(trial) -> dict:
    """Map one Optuna trial to a concrete hyperparameter dict.

    Used identically by RandomSampler and NSGA-II runs. The trial's
    sampler decides the values; this function only declares the search
    space and the call shape.
    """
    out = {}
    for name, cfg in HP_SEARCH_SPACE.items():
        if cfg["kind"] == "int":
            out[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
        else:
            out[name] = trial.suggest_float(
                name, cfg["low"], cfg["high"], log=cfg.get("log", False)
            )
    return out


# ---------------------------------------------------------------------------
# Trajectory file I/O
# ---------------------------------------------------------------------------

def write_trajectory_record(path: Path, record: dict) -> None:
    """Append one JSON object as a line to the trajectory file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_trajectory(path: Path) -> list[dict]:
    """Read the trajectory JSON-lines file back as a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Pareto-frontier extraction
# ---------------------------------------------------------------------------

def _axis_order(a: float, b: float, higher_is_better: bool) -> tuple[bool, bool]:
    """For an axis where ``higher_is_better`` sets the direction, return
    ``(a_is_at_least_as_good_as_b, a_is_strictly_better_than_b)``.
    """
    ge = (a >= b) if higher_is_better else (a <= b)
    gt = (a >  b) if higher_is_better else (a <  b)
    return ge, gt


def pareto_frontier(
    trajectory: list[dict],
    x_key: str,
    y_key: str,
    x_higher: bool = True,
    y_higher: bool = False,
) -> list[dict]:
    """Return the non-dominated subset of trials in (x_key, y_key) space.

    Default direction: x is "higher is better" (e.g. AUROC), y is "lower
    is better" (e.g. slope_dist_to_1). Trial i is dominated when some
    other trial j is at least as good as i on both axes and strictly
    better on at least one.
    """
    pts = []
    for t in trajectory:
        metrics = t.get("metrics", {})
        x = metrics.get(x_key)
        y = metrics.get(y_key)
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if math.isnan(x) or math.isnan(y):
            continue
        pts.append((float(x), float(y), t))

    frontier = []
    for i, (xi, yi, ti) in enumerate(pts):
        dominated = False
        for j, (xj, yj, _) in enumerate(pts):
            if i == j:
                continue
            x_ge, x_gt = _axis_order(xj, xi, x_higher)
            y_ge, y_gt = _axis_order(yj, yi, y_higher)
            if x_ge and y_ge and (x_gt or y_gt):
                dominated = True
                break
        if not dominated:
            frontier.append(ti)
    return frontier


# ---------------------------------------------------------------------------
# Knee selection on a Pareto frontier
# ---------------------------------------------------------------------------

def pick_knee(
    frontier: list[dict],
    x_key: str,
    y_key: str,
) -> Optional[dict]:
    """Pick the knee of a Pareto frontier by max perpendicular distance to
    the chord through the two extreme points, after normalising both axes
    to [0, 1].

    Reference: Satopaa et al. 2011 [IEEE ICDCSW] Kneedle algorithm. The
    full Kneedle adds a spline-smoothing pass for noisy curves; at 500
    NSGA-II trials our frontier is already smooth enough that the
    smoothing step would resolve below its own window size, so we use
    the geometric core only.
    """
    if not frontier:
        return None
    if len(frontier) < 3:
        return frontier[len(frontier) // 2]

    xs = [t["metrics"][x_key] for t in frontier]
    ys = [t["metrics"][y_key] for t in frontier]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_span = (x_hi - x_lo) or 1.0
    y_span = (y_hi - y_lo) or 1.0

    sf = sorted(frontier, key=lambda t: t["metrics"][x_key])
    p0 = ((sf[0]["metrics"][x_key]  - x_lo) / x_span,
          (sf[0]["metrics"][y_key]  - y_lo) / y_span)
    p1 = ((sf[-1]["metrics"][x_key] - x_lo) / x_span,
          (sf[-1]["metrics"][y_key] - y_lo) / y_span)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    chord_len = math.sqrt(dx * dx + dy * dy) or 1.0

    best_d = -1.0
    best_t = sf[0]
    for t in sf:
        xi = (t["metrics"][x_key] - x_lo) / x_span
        yi = (t["metrics"][y_key] - y_lo) / y_span
        # 2D cross product magnitude / chord length = perpendicular distance.
        d = abs(dx * (p0[1] - yi) - (p0[0] - xi) * dy) / chord_len
        if d > best_d:
            best_d, best_t = d, t
    return best_t


# ---------------------------------------------------------------------------
# Variant -> trajectory record selection
# ---------------------------------------------------------------------------

def select_single_obj_tier(trajectory: list[dict], tier: str,
                           score_key: str = "auroc") -> Optional[dict]:
    """For an HP020 / HP050 / ... tier, return the best-``score_key`` trial
    among trials with trial_id < tier_cap. Deterministic given the seed.
    """
    cap = TIER_TO_TRIAL_CAP[tier]
    cands = [t for t in trajectory
             if t["trial_id"] < cap and t["metrics"].get(score_key) is not None]
    if not cands:
        return None
    return max(cands, key=lambda t: t["metrics"][score_key])


def select_pareto_point(
    trajectory: list[dict],
    point_kind: str,
    x_key: str,
    y_key: str,
    x_higher: bool = True,
    y_higher: bool = False,
) -> Optional[dict]:
    """Pick one of (auroc_max | auprc_max | knee | slope_closest) from the
    Pareto frontier of (x_key, y_key)."""
    frontier = pareto_frontier(trajectory, x_key, y_key, x_higher, y_higher)
    if not frontier:
        return None
    if point_kind in ("auroc_max", "auprc_max"):
        return max(frontier, key=lambda t: t["metrics"][x_key])
    if point_kind == "knee":
        return pick_knee(frontier, x_key, y_key)
    if point_kind == "slope_closest":
        return min(frontier, key=lambda t: t["metrics"][y_key])
    raise ValueError(f"unknown pareto point_kind: {point_kind!r}")


# ---------------------------------------------------------------------------
# Search runners
# ---------------------------------------------------------------------------

def run_random_search(
    objective_dict_fn: Callable[[dict], dict],
    n_trials: int,
    trajectory_path: Path,
    seed: int = 42,
) -> None:
    """Run an Optuna RandomSampler study for ``n_trials`` trials.

    ``objective_dict_fn`` takes a params dict and returns a metrics dict;
    each trial appends one trajectory record. Optuna's internal objective
    value is set to the AUROC for compatibility with introspection tools,
    but downstream variant selection consults the trajectory file rather
    than the study object.
    """
    import optuna
    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction="maximize")

    def _objective(trial):
        params  = suggest_params(trial)
        metrics = objective_dict_fn(params)
        write_trajectory_record(trajectory_path, {
            "trial_id":   trial.number,
            "params":     params,
            "metrics":    metrics,
            "objectives": [],
        })
        return float(metrics.get("auroc", 0.0))

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)


def run_nsga2_search(
    objective_dict_fn: Callable[[dict], dict],
    n_trials: int,
    trajectory_path: Path,
    objectives: list[tuple[str, str]],
    population_size: int = 50,
    seed: int = 42,
) -> None:
    """NSGA-II study with two objectives.

    ``objectives`` is a list of (metric_key, direction) where direction
    is 'maximize' or 'minimize'. With Optuna's default population_size=50
    and 500 trials, NSGA-II completes 10 generations: enough for the
    frontier to converge on smooth 2D problems at this search-space
    dimensionality.
    """
    import optuna
    sampler = optuna.samplers.NSGAIISampler(
        seed=seed, population_size=population_size)
    directions = [d for _, d in objectives]
    study = optuna.create_study(sampler=sampler, directions=directions)

    def _objective(trial):
        params  = suggest_params(trial)
        metrics = objective_dict_fn(params)
        obj_vals = tuple(float(metrics.get(k, 0.0)) for k, _ in objectives)
        write_trajectory_record(trajectory_path, {
            "trial_id":   trial.number,
            "params":     params,
            "metrics":    metrics,
            "objectives": list(obj_vals),
        })
        return obj_vals

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)