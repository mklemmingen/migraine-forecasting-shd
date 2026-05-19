"""Results-directory discovery and per-file parsing.

Walks ``experiment/`` looking for leaf ``results/`` folders, parses each
leaf's latest holdout and CV result text files, and post-processes the
parsed dictionaries into the shape the aggregator expects.

Path layout (mirrors the directory tree):
    experiment/<addition>/<target>/<feature_set>/<architecture>/
      [<version>]/<datasplit>/<splittype>/
      [<HyperparameterTuned>/<hp_strategy>/<hp_variant>]/results/

The ``_``-prefixed directories (``_eval``, ``_templates``, etc.) are
skipped so this discovery does not pick up template scaffolds or
runner-output directories.
"""
from pathlib import Path

RED   = "\033[91m"
RESET = "\033[0m"


def find_results_dirs(experiment_dir):
    """Return all leaf results/ directories under ``experiment_dir``
    that are not inside ``_``-prefixed folders.

    Skips ``experiment/results/`` itself, which is the aggregator's own
    archive of prior HTML/PNG outputs (not where leaves write their
    txt files).
    """
    found = []
    for p in experiment_dir.rglob("results"):
        if not p.is_dir():
            continue
        rel = p.relative_to(experiment_dir)
        if rel == Path("results"):
            continue   # top-level output folder, not a leaf
        if not any(part.startswith("_") for part in rel.parts):
            found.append(p)
    return sorted(found)


def parse_path(results_dir, experiment_dir):
    """Extract experiment dimensions from a results/ directory path."""
    rel   = results_dir.relative_to(experiment_dir)
    parts = list(rel.parts[:-1])   # drop trailing "results"

    addition     = parts[0] if len(parts) > 0 else "?"
    target       = parts[1] if len(parts) > 1 else "?"
    feature_set  = parts[2] if len(parts) > 2 else "?"
    architecture = parts[3] if len(parts) > 3 else "?"

    idx     = 4
    version = None
    if idx < len(parts) and parts[idx].startswith("version_"):
        version = parts[idx]
        idx    += 1

    datasplit = parts[idx] if idx < len(parts) else "?"
    idx      += 1
    splittype = parts[idx] if idx < len(parts) else None

    idx      += 1
    hyperparameter = (
        parts[idx]
        if idx < len(parts) and parts[idx] == "HyperparameterTuned"
        else None
    )
    # When the tuning state is HyperparameterTuned the next two segments
    # carry the search strategy and the variant identifier (e.g.,
    # "single_AUROC/HP020" or "pareto_AUROC_slope/knee"). NonHP cells
    # have parts[idx] == "NonHP" and neither hp_strategy nor hp_variant
    # is set.
    hp_strategy = None
    hp_variant  = None
    if hyperparameter is not None:
        idx += 1
        hp_strategy = parts[idx] if idx < len(parts) else None
        idx += 1
        hp_variant  = parts[idx] if idx < len(parts) else None

    return {
        "addition":     addition,
        "target":       target,
        "feature_set":  feature_set,
        "architecture": architecture,
        "version":      version,
        "datasplit":    datasplit,
        "splittype":    splittype,
        "hyperparameter": hyperparameter,
        "hp_strategy":  hp_strategy,
        "hp_variant":   hp_variant,
    }


def find_latest_files(results_dir):
    """Return (latest_holdout_path, latest_cv_path); either may be None."""
    files   = sorted(results_dir.glob("*.txt"), key=lambda f: f.name)
    holdout = [f for f in files if not f.stem.startswith("results_cv")]
    cv      = [f for f in files if f.stem.startswith("results_cv")]
    return (holdout[-1] if holdout else None, cv[-1] if cv else None)


def format_ts(filepath):
    """Parse 'YYYYMMDD_HHMMSS' from a filename and return a readable
    'YYYY-MM-DD HH:MM:SS' string. Returns "" when no timestamp present.
    """
    if filepath is None:
        return ""
    parts = filepath.stem.split("_")
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            d = p
            t = parts[i + 1] if i + 1 < len(parts) else "000000"
            return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"
    return ""


def parse_file(filepath, contract):
    """Match each non-structural ContractLine prefix against file lines.

    Returns {key: value_str} with None for any unmatched key. Returns
    None when ``filepath`` itself is None (no result file for this leaf).
    """
    if filepath is None:
        return None
    lines  = filepath.read_text().splitlines()
    result = {}
    for cl in contract:
        if cl.structural:
            continue
        for line in lines:
            if line.startswith(cl.prefix):
                result[cl.key] = line[len(cl.prefix):].strip()
                break
        else:
            result[cl.key] = None
    return result


def cv_mean_std(raw):
    """Extract 'mean +/- std' from a CV metric line's value string.

    Raw format from sharedMetricPrinter:
        '  0.581  ...  | 0.584    0.051  '
    """
    if raw is None:
        return None
    if " | " in raw:
        right = raw.rsplit(" | ", 1)[1].strip()
        parts = right.split()
        if len(parts) >= 2:
            return f"{parts[0]} ± {parts[1]}"
    return raw


def postprocess_cv(cv_raw):
    """Convert raw CV parsed dict to {metric: 'mean +/- std'} format."""
    if cv_raw is None:
        return None
    non_metric = {"threshold_mcc", "threshold_sens"}
    return {
        k: (cv_mean_std(v) if k not in non_metric else v)
        for k, v in cv_raw.items()
    }


def warn_na(entry):
    """Print a coloured warning per missing metric so the operator
    sees which leaves are incomplete as the aggregator runs.
    """
    path_str = " / ".join(str(v) for v in entry["path"].values() if v)
    for label, metrics in [("hold-out", entry["holdout"]), ("CV", entry["cv"])]:
        if metrics is None:
            continue
        for key, val in metrics.items():
            if val is None:
                print(f"{RED}WARNING: N/A - '{key}' missing in {label}: {path_str}{RESET}")
