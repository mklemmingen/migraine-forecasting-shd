"""
run script that aggregates any and all results/ *LATEST* entries into one interactive table in html.

Uses the predefined structure of the folders as described in the Readme of:
`experiment/<NrAddition>/<headache/migraine>/<feature_set>/<architecture>/<*modelVersion>/<*dataSplit>/<*SplitType>/<*HyperparameterTuned>

to name the current approach in the tabular views.

Though to the multidimensional state of the experiments, namely (as of 26-05-08):
feature set
architecture
<version where applicable>
split in fractions
split in type

the tabular views are sorted by these 2 dimensions, allowing selections of the lower dimension to move to the
next applied tabular view, and so on, until the metrics as defined in the Readme at the lowest level are shown.

The complexity of the resulting html is scaled dynamically based on results/ folders found with valid last result and paths mapped.

Expects the prints in the results txt to be based as defined by sharedMetricPrinter.
-> this code checks the contract template by calling getContract() on it, to ensure any change in format can be
centralized in sharedMetricPrinter.py

The results are metadated by time in the last view.
the overall html has a human readable iso timestamp and a uuid to not allow accidental overwrites by sudden reruns.

fyi: the aggregate result runner will fill out N/A under values not found that were defined by contract,
but not in the results/ folders latest result txt.  - and will give any user running it a red text
in their terminal where supported.
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).parent / "_eval"))
from sharedMetricPrinter import (
    getContract, getContract_cv, METRICS_HOLDOUT, METRICS_CV
)

EXPERIMENT_DIR = Path(__file__).parent
RED   = "\033[91m"
RESET = "\033[0m"

# Output layout: the latest aggregator run lives directly under experiment/
# (next to this script) so the current HTML/PNG set is one click away. Any
# prior outputs matching OUTPUT_PATTERNS get archived into experiment/results/
# at the start of each run, keeping the root uncluttered.
# HTML files reference PNGs by basename, so HTML+PNG must stay co-located.
LATEST_DIR  = EXPERIMENT_DIR
ARCHIVE_DIR = EXPERIMENT_DIR / "results"
OUTPUT_PATTERNS = (
    "results_*.html",
    "comparison_*.html",
    "venn_counts_*.png",
    "venn_names_*.png",
    "tree_*.png",
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_results_dirs():
    """Return all leaf results/ directories that are not inside _-prefixed folders.

    Skips the top-level experiment/results/ directory itself - that's the
    aggregator's own archive of prior HTML/PNG outputs, not where leaves write
    their txt files.
    """
    found = []
    for p in EXPERIMENT_DIR.rglob("results"):
        if not p.is_dir():
            continue
        rel = p.relative_to(EXPERIMENT_DIR)
        if rel == Path("results"):
            continue  # top-level output folder - not a leaf
        if not any(part.startswith("_") for part in rel.parts):
            found.append(p)
    return sorted(found)


def parse_path(results_dir):
    """Extract experiment dimensions from a results/ directory path."""
    rel   = results_dir.relative_to(EXPERIMENT_DIR)
    parts = list(rel.parts[:-1])  # drop trailing "results"

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
    hyperparameter = parts[idx] if idx < len(parts) and parts[idx] == "HyperparameterTuned" else None
    # When the tuning state is HyperparameterTuned, the next two segments
    # carry the search strategy and the variant identifier (e.g.,
    # "single_AUROC/HP020" or "pareto_AUROC_slope/knee"). NonHP cells have
    # parts[idx] == "NonHP" and neither hp_strategy nor hp_variant is set.
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


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def find_latest_files(results_dir):
    """Return (latest_holdout_path, latest_cv_path); either may be None."""
    files   = sorted(results_dir.glob("*.txt"), key=lambda f: f.name)
    holdout = [f for f in files if not f.stem.startswith("results_cv")]
    cv      = [f for f in files if f.stem.startswith("results_cv")]
    return (holdout[-1] if holdout else None, cv[-1] if cv else None)


def format_ts(filepath):
    """Parse 'YYYYMMDD_HHMMSS' from filename and return a readable string."""
    if filepath is None:
        return ""
    parts = filepath.stem.split("_")
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            d = p
            t = parts[i + 1] if i + 1 < len(parts) else "000000"
            return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"
    return ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_file(filepath, contract):
    """
    Match each non-structural ContractLine prefix against file lines.
    Returns {key: value_str} with None for any unmatched key.
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
    """
    Extract 'mean ± std' from a CV metric line's value string.
    Raw format: '  0.581  ...  | 0.584    0.051  '
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
    """Convert raw CV parsed dict to {metric: 'mean ± std'} format."""
    if cv_raw is None:
        return None
    non_metric = {"threshold_mcc", "threshold_sens"}
    return {
        k: (cv_mean_std(v) if k not in non_metric else v)
        for k, v in cv_raw.items()
    }


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def warn_na(entry):
    path_str = " / ".join(str(v) for v in entry["path"].values() if v)
    for label, metrics in [("hold-out", entry["holdout"]), ("CV", entry["cv"])]:
        if metrics is None:
            continue
        for key, val in metrics.items():
            if val is None:
                print(f"{RED}WARNING: N/A - '{key}' missing in {label}: {path_str}{RESET}")


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_JS = r"""
const DATA = $data;
const METRICS_HOLDOUT = $metrics_holdout;
const METRICS_CV = $metrics_cv;

const DIMS = ["addition", "target", "feature_set", "arch_ver"];
const DIM_LABELS = {
  addition: "Addition", target: "Target",
  feature_set: "Feature Set", arch_ver: "Architecture"
};

let navPath = [];

function archVer(e) {
  return e.path.version
    ? e.path.architecture + " (" + e.path.version + ")"
    : e.path.architecture;
}
function splitKey(e) {
  return e.path.splittype
    ? e.path.datasplit + " / " + e.path.splittype
    : e.path.datasplit;
}
function getVal(e, dim) {
  return dim === "arch_ver" ? archVer(e) : e.path[dim];
}
function filterData() {
  return DATA.filter(e => navPath.every(s => getVal(e, s.dim) === s.val));
}
function uniqueVals(data, dim) {
  return [...new Set(data.map(e => getVal(e, dim)))].sort();
}

function renderBreadcrumb() {
  const bc = document.getElementById("breadcrumb");
  if (!navPath.length) { bc.innerHTML = ""; return; }
  const parts = ['<a href="#" onclick="navTo(-1);return false">All</a>'];
  navPath.forEach(function(s, i) {
    parts.push('<span class="sep">›</span>');
    if (i < navPath.length - 1)
      parts.push('<a href="#" onclick="navTo(' + i + ');return false">' + esc(s.val) + '</a>');
    else
      parts.push('<strong>' + esc(s.val) + '</strong>');
  });
  bc.innerHTML = parts.join(" ");
}

function navTo(idx) { navPath = idx < 0 ? [] : navPath.slice(0, idx + 1); render(); }
function navigate(dim, val) { navPath.push({dim: dim, val: val}); render(); }

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function renderNavTable(data, dim) {
  const vals = uniqueVals(data, dim);
  let h = '<table><tr><th>' + DIM_LABELS[dim] + '</th><th>Experiments</th></tr>';
  vals.forEach(function(val) {
    const count = data.filter(function(e) { return getVal(e, dim) === val; }).length;
    h += '<tr class="nav-row"><td class="nav-label">' + esc(val) + '</td><td>' + count + '</td></tr>';
  });
  h += '</table>';
  return { html: h, vals: vals };
}

function renderValue(v) {
  if (v === null || v === undefined)
    return '<span class="na">N/A</span>';
  return esc(v);
}

function renderMetricsTable(metrics, names) {
  let h = '<table><tr><th>Metric</th><th>Value</th></tr>';
  names.forEach(function(m) {
    const v = metrics ? metrics[m] : null;
    h += '<tr><td>' + esc(m) + '</td><td>' + renderValue(v) + '</td></tr>';
  });
  return h + '</table>';
}

function renderLeaf(data) {
  const splits = [...new Set(data.map(e => splitKey(e)))].sort();
  let h = '';
  splits.forEach(function(sk) {
    const entry = data.find(function(e) { return splitKey(e) === sk; });
    const hot = entry.holdout_ts ? ' <span class="ts">(' + esc(entry.holdout_ts) + ')</span>' : '';
    const cvt = entry.cv_ts     ? ' <span class="ts">(' + esc(entry.cv_ts)      + ')</span>' : '';
    let thresh = '';
    if (entry.threshold_mcc || entry.threshold_sens) {
      thresh = '<div class="thresh">MCC threshold: ' + esc(entry.threshold_mcc || 'N/A') +
               ' &nbsp;··· sens≥0.5 threshold: ' + esc(entry.threshold_sens || 'N/A') + '</div>';
    }
    h += '<div class="result-block">';
    h += '<h3>' + esc(sk) + '</h3>';
    h += '<div class="result-pair">';
    h += '<div class="result-section"><h4>Hold-out Test' + hot + '</h4>';
    h += thresh;
    h += renderMetricsTable(entry.holdout, METRICS_HOLDOUT);
    h += '</div>';
    h += '<div class="result-section"><h4>Cross-Validation (5-Fold)' + cvt + '</h4>';
    h += renderMetricsTable(entry.cv, METRICS_CV);
    h += '</div>';
    h += '</div></div>';
  });
  return h || '<p style="color:#888">No result files found for this path.</p>';
}

function render() {
  renderBreadcrumb();
  const filtered = filterData();
  const level    = navPath.length;
  const content  = document.getElementById("content");
  if (level >= DIMS.length) {
    content.innerHTML = renderLeaf(filtered);
  } else {
    const result = renderNavTable(filtered, DIMS[level]);
    content.innerHTML = result.html;
    document.querySelectorAll('.nav-row').forEach(function(row, i) {
      row.addEventListener('click', function() { navigate(DIMS[level], result.vals[i]); });
    });
  }
}

render();
"""


def build_html(all_entries, iso_timestamp, uid):
    data_json            = json.dumps(all_entries, ensure_ascii=False, indent=2)
    metrics_holdout_json = json.dumps(METRICS_HOLDOUT)
    metrics_cv_json      = json.dumps(METRICS_CV)

    js = Template(_JS).safe_substitute(
        data=data_json,
        metrics_holdout=metrics_holdout_json,
        metrics_cv=metrics_cv_json,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Results - {iso_timestamp}</title>
<style>
  body {{
    font-family: 'Courier New', monospace;
    max-width: 1200px; margin: 24px auto; padding: 0 20px;
    background: #fafafa; color: #222;
  }}
  h1 {{ font-size: 1.25em; margin-bottom: 4px; }}
  .meta {{ color: #777; font-size: 0.8em; margin-bottom: 20px; }}
  .breadcrumb {{ margin-bottom: 14px; font-size: 0.9em; }}
  .breadcrumb .sep {{ color: #bbb; margin: 0 4px; }}
  .breadcrumb a {{ color: #1a73e8; text-decoration: none; }}
  .breadcrumb a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: left; font-size: 0.87em; }}
  th {{ background: #f0f4f8; font-weight: bold; }}
  tr.nav-row {{ cursor: pointer; }}
  tr.nav-row:hover {{ background: #e8f0fe; }}
  td.nav-label::after {{ content: " \203a"; color: #aaa; }}
  .na {{ color: #c00; font-style: italic; }}
  .result-block {{
    margin-bottom: 22px; border: 1px solid #e0e0e0;
    border-radius: 4px; padding: 14px 16px; background: #fff;
  }}
  .result-block h3 {{ font-size: 0.95em; margin: 0 0 12px; color: #333; }}
  .result-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .result-section h4 {{ font-size: 0.82em; margin: 0 0 5px; color: #555; }}
  .thresh {{ font-size: 0.77em; color: #888; margin-bottom: 5px; }}
  .ts {{ font-size: 0.78em; color: #999; font-weight: normal; }}
  @media (max-width: 700px) {{ .result-pair {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Experiment Results</h1>
<div class="meta">Generated: {iso_timestamp} &nbsp;&middot;&nbsp; ID: {uid}</div>
<div class="breadcrumb" id="breadcrumb"></div>
<div id="content"></div>
<script>
{js}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

# (key-in-holdout-dict, display-label, higher-is-better, ref-min, ref-max)
# All 10 metrics are shown in cells. The CV-source MCC (renamed to "MCC (Cal-Optimal)"
# under cross-validation) is auto-substituted via the lookup_key branch in build_comparison_html.
COMPARISON_METRICS = [
    ("AUROC",               "AUROC",       True,  0.5,  1.0),
    ("AUPRC",               "AUPRC",       True,  0.0,  1.0),
    ("Brier Score",         "Brier ↓",     False, 0.0,  0.25),  # ~baseline at 5% prevalence
    ("ECE10",               "ECE10 ↓",     False, 0.0,  0.10),  # well-calibrated < 0.05
    # Calibration slope is the logistic-recalibration coefficient on the
    # held-out set; 1.0 = perfect, <1.0 = over-confident (overfit), >1.0
    # = under-confident. Required by TRIPOD+AI alongside ECE/Brier.
    # vmin/vmax framed around 1.0: [0.0, 2.0]; the closer to 1.0 the
    # better, but the global color scale is symmetric only via this
    # 0..2 range. Older results files without this key render as N/A
    # until those leaves are re-evaluated.
    ("Calibration Slope",   "CalSlope",    True,  0.0,  2.0),
    ("MCC (Optimal)",       "MCC",         True,  0.0,  0.5),   # practical upper bound ~0.5
    ("Sensitivity (>=0.5)", "Sens≥0.5",    True,  0.0,  1.0),
    # Accuracy: kept for completeness but base-rate-dominated on
    # imbalanced targets. On the migraine target (5% positive rate) a
    # constant-negative predictor reaches ~95% accuracy. Use MCC / AUPRC
    # / Brier as the primary headline; accuracy is for completeness only.
    ("Accuracy",            "Acc (base!)", True,  0.5,  1.0),
    ("Precision",           "Prec",        True,  0.0,  1.0),
    ("Recall",              "Recall",      True,  0.0,  1.0),
    ("F1",                  "F1",          True,  0.0,  1.0),
]


def compute_color(val_str, higher_better, val_min, val_max):
    """Map a metric string (e.g. '0.519 [...]') to an HSL background colour."""
    if val_str is None:
        return "#e4e4e4"
    try:
        val = float(str(val_str).split()[0])
    except (ValueError, IndexError):
        return "#ffffff"
    t = (val - val_min) / (val_max - val_min) if val_max != val_min else 0.5
    t = max(0.0, min(1.0, t))
    if not higher_better:
        t = 1.0 - t
    return f"hsl({round(t * 120)}, 60%, 91%)"


def parse_mean_ci(val_str):
    """Parse the per-leaf bootstrap output ``"mean [lo - hi]"``.

    Returns ``(mean, lo, hi)`` as floats, or ``None`` if the string is
    missing or unparseable. The evaluator templates write the CI as
    1000-iteration bootstrap 2.5/97.5 percentiles (see ``_scaffold_leaves.py``
    ``run_bootstrap_evaluation``), so the bracketed range is a 95% CI.
    For CV-source values the same format is used; for the singleton
    cells (e.g. accuracy that is not bootstrapped) the bracket may be
    absent and only the mean is returned.
    """
    if val_str is None:
        return None
    s = str(val_str)
    try:
        mean = float(s.split()[0])
    except (ValueError, IndexError):
        return None
    lo = hi = None
    lb = s.find("[")
    rb = s.find("]")
    if lb != -1 and rb != -1 and rb > lb:
        inner = s[lb + 1:rb]
        for sep in (" - ", " – ", " — ", ", ", " to "):
            if sep in inner:
                parts = inner.split(sep)
                if len(parts) == 2:
                    try:
                        lo = float(parts[0].strip())
                        hi = float(parts[1].strip())
                    except ValueError:
                        lo = hi = None
                    break
    return mean, lo, hi


# ---------------------------------------------------------------------------
# Feature-set Venn diagrams (PNGs generated alongside the HTML)
# ---------------------------------------------------------------------------

# Columns produced by data/pipeline/engineer.py via aggregation, lag, rolling
# windows, state-change detection, or cross-feature interactions. Everything
# else in the parquet is treated as an original SHD column (or a 1:1 rename
# of one - e.g. `stress` → `stress_today`).
#
# Why hard-coded: these are determined by inspection of engineer.py, not
# derivable from the parquet alone. Update both together.
ENGINEERED_FEATURES = {
    # Migraine history (rolling / lag / state-derived from migraine_today)
    'migraine_yesterday', 'migraine_rate_last3', 'migraine_rate_last7',
    'headache_free_streak', 'days_since_last_migraine',
    # Stress derivatives
    'stress_drop_today', 'consecutive_stress_days',
    # Sleep derivatives
    'any_sleep_issue_today', 'sleep_debt_3day', 'sleep_disruption_today',
    'sleep_variability_7day', 'recent_weekend_sleep_issues',
    # Weather derivatives
    'consecutive_weather_changes', 'weather_instability_3day',
    'weather_change_yesterday', 'weather_headache_interaction',
    # Cross-trigger combination
    'consecutive_trigger_days',
    # Exercise derivatives
    'exercise_today', 'consecutive_exercise_days',
    'consecutive_sedentary_days', 'exercise_days_7day',
    # Recording-gap features (engineered from date diff)
    'days_since_last_record', 'recording_gap_flag',
    # Calendar derivative
    'dow',
    # Target column: derived (sign-flipped from headache_free, or merged from
    # disability sheet in migraine mode) - not present verbatim in raw input.
    'migraine_today',
}

# Plot palette - kept centralised so both venns stay visually consistent.
VENN_COLORS = {
    'full':       '#2563eb',  # blue
    'spano':      '#dc2626',  # red
    'no_rolling': '#059669',  # green
    'park':       '#a16207',  # amber - Park (2016) stepwise-selected subset
}
CATEGORY_COLORS = {
    'engineered': '#7c2d92',  # dark purple - derived features
    'original':   '#14532d',  # dark green  - raw / 1:1 rename
}


def compute_feature_sets():
    """Load a representative parquet and compute the four feature sets.

    Returns dict {full, spano, no_rolling, park} of column-name sets, or
    None if no parquet is found (e.g. data pipeline hasn't been run yet).
    The Park set includes one derived feature (``hormonal_changes_today``)
    that is NOT a column in the engineered parquet itself - the Park
    filter computes it from menstruation_today OR ovulation_today to
    match Park et al.'s single "hormonal changes" trigger.
    """
    NON_FEATURE = {"entry_id", "patient_id", "date", "migraine_target", "cv_fold", "migraine_today"}
    candidates = [
        EXPERIMENT_DIR.parent / "data" / "processed" / "headache" / "70_15_15" / "chrono" / "diary_train.parquet",
        EXPERIMENT_DIR.parent / "data" / "processed" / "migraine"  / "70_15_15" / "chrono" / "diary_train.parquet",
    ]
    parquet = next((p for p in candidates if p.exists()), None)
    if parquet is None:
        return None

    # The filter modules use absolute imports from `_dataRead.*`; put the
    # *parent* (experiment/) on sys.path so those imports resolve.
    sys.path.insert(0, str(EXPERIMENT_DIR))
    import pandas as pd
    from _dataRead.filter_to_spano_features import select_spano_features
    from _dataRead.filter_to_no_rolling_features import select_non_rolling_features
    from _dataRead.filter_to_park_features import select_park_features

    full       = set(pd.read_parquet(parquet).columns)        - NON_FEATURE
    spano      = set(select_spano_features(str(parquet)).columns)      - NON_FEATURE
    no_rolling = set(select_non_rolling_features(str(parquet)).columns) - NON_FEATURE
    park       = set(select_park_features(str(parquet)).columns)        - NON_FEATURE
    return {"full": full, "spano": spano, "no_rolling": no_rolling, "park": park}


def _split_by_category(features):
    """Return (engineered_count, original_count) for a feature set."""
    eng = sum(1 for f in features if f in ENGINEERED_FEATURES)
    return eng, len(features) - eng


def generate_count_venn_png(feature_sets, out_path):
    """Render a 3-set Venn showing region counts, split by engineered/original."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3
    from matplotlib_venn.layout.venn3 import cost_based

    full       = feature_sets["full"]
    spano      = feature_sets["spano"]
    no_rolling = feature_sets["no_rolling"]

    fig, ax = plt.subplots(figsize=(13, 10))
    v = venn3(
        [full, spano, no_rolling],
        set_labels=(
            f'full ({len(full)})',
            f'spano ({len(spano)})',
            f'no_rolling ({len(no_rolling)})',
        ),
        set_colors=(VENN_COLORS['full'], VENN_COLORS['spano'], VENN_COLORS['no_rolling']),
        alpha=0.32,
        ax=ax,
        # spano ⊂ full ⊃ no_rolling is a near-subset configuration; the default
        # pairwise solver can't satisfy the implied triangle inequality and emits
        # "Bad circle positioning". The cost-based optimizer minimises log-area
        # error across all 7 regions and handles this case cleanly.
        layout_algorithm=cost_based.LayoutAlgorithm(),
    )

    regions = {
        '100': full - spano - no_rolling,
        '010': spano - full - no_rolling,
        '001': no_rolling - full - spano,
        '110': (full & spano) - no_rolling,
        '101': (full & no_rolling) - spano,
        '011': (spano & no_rolling) - full,
        '111': full & spano & no_rolling,
    }
    for rid, feats in regions.items():
        lbl = v.get_label_by_id(rid)
        if lbl is None:
            continue
        if not feats:
            lbl.set_text('')
            continue
        eng, orig = _split_by_category(feats)
        lbl.set_text(f"{len(feats)}\n({eng} eng + {orig} orig)")
        lbl.set_fontsize(10)
        lbl.set_fontweight('bold')

    # Pin set labels to known-good positions and colour them.
    # Auto-placement breaks for nested subsets (spano ⊂ full ⊃ no_rolling),
    # producing the "Bad circle positioning" warning and sometimes hiding labels.
    set_label_positions = {
        'full':       (-0.85,  0.65),
        'spano':       (0.85,  0.65),
        'no_rolling':  (0.00, -0.85),
    }
    for sid, color_key in zip(('A', 'B', 'C'), ('full', 'spano', 'no_rolling')):
        s_lbl = v.get_label_by_id(sid)
        if s_lbl is None:
            continue
        s_lbl.set_position(set_label_positions[color_key])
        s_lbl.set_horizontalalignment('center')
        s_lbl.set_fontsize(13)
        s_lbl.set_fontweight('bold')
        s_lbl.set_color(VENN_COLORS[color_key])

    ax.set_title("Feature-set inclusion - region counts (engineered + original SHD columns)",
                 fontsize=12, pad=14)

    # Legend explaining 'eng' / 'orig'
    ax.text(0.02, 0.02,
            "eng = engineered (rolling / lag / interaction / state-derived)\n"
            "orig = original SHD column or 1:1 rename",
            transform=ax.transAxes, fontsize=8.5, color='#444',
            verticalalignment='bottom',
            bbox=dict(facecolor='white', edgecolor='#bbb', boxstyle='round,pad=0.4'))

    # Park (2016) stepwise-selected subset is shown as a sidebar rather than a
    # 4th circle. It is a near-strict subset of `full` (5 of its 6 features are
    # columns of `full`), plus one derived feature `hormonal_changes_today`
    # (= menstruation_today OR ovulation_today) that is constructed inside the
    # filter to match Park et al.'s single "hormonal changes" trigger
    # representation. Forcing a 4-set Venn would visually flatten this almost-
    # subset relationship; the sidebar is more informative.
    park = feature_sets.get("park", set())
    if park:
        park_in_full = sorted(park & full)
        park_only    = sorted(park - full)
        park_lines = [
            f"Park (2016) stepwise-selected: {len(park)} features",
            f"  {len(park_in_full)} ⊂ full:  " + ", ".join(park_in_full),
        ]
        if park_only:
            park_lines.append(
                f"  {len(park_only)} derived (not in full): " + ", ".join(park_only)
            )
        ax.text(
            0.98, 0.02, "\n".join(park_lines),
            transform=ax.transAxes, fontsize=8.5, color=VENN_COLORS['park'],
            verticalalignment='bottom', horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(facecolor='white', edgecolor=VENN_COLORS['park'],
                      boxstyle='round,pad=0.5', linewidth=1.2),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def generate_names_venn_png(feature_sets, out_path):
    """Render a 3-set Venn whose regions list every feature name, colour-coded
    by engineered vs original. The default count labels are hidden; we place
    one ``ax.text`` per feature stacked vertically around each region centroid
    so individual names can carry their own colour and weight."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3
    from matplotlib_venn.layout.venn3 import cost_based

    full       = feature_sets["full"]
    spano      = feature_sets["spano"]
    no_rolling = feature_sets["no_rolling"]

    fig, ax = plt.subplots(figsize=(18, 13))
    v = venn3(
        [full, spano, no_rolling],
        set_labels=(
            f'full  ({len(full)} features)',
            f'spano  ({len(spano)} features)',
            f'no_rolling  ({len(no_rolling)} features)',
        ),
        set_colors=(VENN_COLORS['full'], VENN_COLORS['spano'], VENN_COLORS['no_rolling']),
        alpha=0.16,
        ax=ax,
        # See generate_count_venn_png - same near-subset configuration.
        layout_algorithm=cost_based.LayoutAlgorithm(),
    )

    regions = {
        '100': full - spano - no_rolling,
        '010': spano - full - no_rolling,
        '001': no_rolling - full - spano,
        '110': (full & spano) - no_rolling,
        '101': (full & no_rolling) - spano,
        '011': (spano & no_rolling) - full,
        '111': full & spano & no_rolling,
    }

    # Hide all default count labels; we'll replace them with stacked names.
    for rid in regions:
        lbl = v.get_label_by_id(rid)
        if lbl is not None:
            lbl.set_text('')

    LINE_H = 0.020  # vertical spacing per name in axes coords (figure units)
    for rid, feats in regions.items():
        if not feats:
            continue
        anchor = v.get_label_by_id(rid)
        if anchor is None:
            continue
        cx, cy = anchor.get_position()
        sorted_feats = sorted(feats)
        n = len(sorted_feats)
        start_y = cy + (n - 1) / 2.0 * LINE_H
        for i, f in enumerate(sorted_feats):
            y = start_y - i * LINE_H
            is_eng = f in ENGINEERED_FEATURES
            ax.text(
                cx, y, f,
                color=CATEGORY_COLORS['engineered'] if is_eng else CATEGORY_COLORS['original'],
                fontsize=7.2,
                ha='center', va='center',
                fontweight='bold' if is_eng else 'normal',
                fontfamily='monospace',
            )

    # Set labels: pinned to fixed positions, bigger, coloured to match their circle.
    # See generate_count_venn_png for the rationale behind manual positioning.
    set_label_positions = {
        'full':       (-0.85,  0.70),
        'spano':       (0.85,  0.70),
        'no_rolling':  (0.00, -0.85),
    }
    for sid, color_key in zip(('A', 'B', 'C'), ('full', 'spano', 'no_rolling')):
        s_lbl = v.get_label_by_id(sid)
        if s_lbl is None:
            continue
        s_lbl.set_position(set_label_positions[color_key])
        s_lbl.set_horizontalalignment('center')
        s_lbl.set_fontsize(14)
        s_lbl.set_fontweight('bold')
        s_lbl.set_color(VENN_COLORS[color_key])

    ax.set_title(
        "Feature-set inclusion - all feature names, colour-coded by origin",
        fontsize=13, pad=14,
    )

    # Legend (axes-relative, top-left corner)
    legend_text = (
        "● original SHD column or 1:1 rename"
    )
    ax.text(0.01, 0.985, legend_text,
            transform=ax.transAxes, fontsize=10,
            color=CATEGORY_COLORS['original'], fontweight='normal',
            verticalalignment='top', fontfamily='monospace')
    ax.text(0.01, 0.955,
            "● engineered (rolling / lag / interaction / state-derived)",
            transform=ax.transAxes, fontsize=10,
            color=CATEGORY_COLORS['engineered'], fontweight='bold',
            verticalalignment='top', fontfamily='monospace')

    # Park (2016) sidebar - same rationale as in generate_count_venn_png.
    park = feature_sets.get("park", set())
    if park:
        park_in_full = sorted(park & full)
        park_only    = sorted(park - full)
        park_lines = [
            f"Park (2016) [1, Tab. 4]: stepwise-selected subset, {len(park)} features",
        ]
        for f in park_in_full:
            park_lines.append(f"  ⊂ full   {f}")
        for f in park_only:
            park_lines.append(f"  derived  {f}")
        ax.text(
            0.99, 0.01, "\n".join(park_lines),
            transform=ax.transAxes, fontsize=8.5, color=VENN_COLORS['park'],
            verticalalignment='bottom', horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(facecolor='white', edgecolor=VENN_COLORS['park'],
                      boxstyle='round,pad=0.5', linewidth=1.2),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Leaf tree visualisation (matplotlib horizontal-phylogeny rendering)
# ---------------------------------------------------------------------------

# Path-segment levels in the order they appear under experiment/.
# Each tuple is (path_dim_key_in_parse_path, full-word column header).
TREE_LEVELS = (
    ('addition',       'Addition'),
    ('target',         'Target'),
    ('feature_set',    'Feature set'),
    ('architecture',   'Architecture'),
    ('version',        'Model version'),
    ('datasplit',      'Data split ratio'),
    ('splittype',      'Split strategy'),
    ('hyperparameter', 'Hyperparameter tuning'),
)

# One distinct hue per level - matches each column's colour in the diagram.
TREE_LEVEL_COLORS = [
    '#1a3550',  # addition           - deep navy
    '#2563eb',  # target             - blue
    '#0891b2',  # feature_set        - teal
    '#059669',  # architecture       - green
    '#65a30d',  # version            - olive
    '#ca8a04',  # datasplit          - amber
    '#dc2626',  # splittype          - red
    '#7c2d92',  # hyperparameter     - purple
]


def generate_tree_png(all_entries, out_path):
    """Render the discovered-leaves hierarchy as a matplotlib tree diagram.

    Layout: horizontal "phylogeny" - root on the left, leaves stack down on
    the right. Each level is its own column with a colour-coded header.
    Labels show full directory names (no truncation, no abbreviation).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class Node:
        __slots__ = ('label', 'level', 'children', 'x', 'y')

        def __init__(self, label, level):
            self.label = label
            self.level = level
            self.children = {}
            self.x = 0.0
            self.y = 0.0

    # --- build tree ---
    root = Node('experiment/', -1)
    for entry in all_entries:
        p = entry['path']
        node = root
        for lvl, (key, _hdr) in enumerate(TREE_LEVELS):
            val = p.get(key)
            if val is None:
                continue
            child_key = (lvl, val)
            if child_key not in node.children:
                node.children[child_key] = Node(str(val), lvl)
            node = node.children[child_key]

    # --- assign y by leaf-order DFS, parents = mean of children ---
    leaf_count = [0]

    def assign_y(node):
        if not node.children:
            node.y = float(leaf_count[0])
            leaf_count[0] += 1
            return
        for k in sorted(node.children):
            assign_y(node.children[k])
        ys = [node.children[k].y for k in node.children]
        node.y = (min(ys) + max(ys)) / 2.0

    assign_y(root)
    n_leaves = leaf_count[0]
    if n_leaves == 0:
        return

    # x = level index (root sits at -1)
    def assign_x(node):
        node.x = float(node.level)
        for child in node.children.values():
            assign_x(child)
    assign_x(root)

    # --- canvas ---
    n_levels = len(TREE_LEVELS)
    fig_w = max(20, n_levels * 2.6)
    fig_h = max(7, n_leaves * 0.34 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- draw edges (drawn first so nodes sit on top) ---
    def draw_edges(node):
        for child in node.children.values():
            ax.plot([node.x, node.x], [node.y, child.y],
                    color='#bbb', linewidth=0.7, zorder=1)
            ax.plot([node.x, child.x], [child.y, child.y],
                    color='#bbb', linewidth=0.7, zorder=1)
            draw_edges(child)
    draw_edges(root)

    # --- draw nodes + labels ---
    def draw_nodes(node):
        if node.level >= 0:
            color = TREE_LEVEL_COLORS[min(node.level, len(TREE_LEVEL_COLORS) - 1)]
            ax.scatter([node.x], [node.y], s=70, color=color, zorder=3,
                       edgecolors='white', linewidths=1.0)
            ax.text(node.x + 0.06, node.y, node.label,
                    ha='left', va='center', fontsize=8.0,
                    color=color, fontweight='bold',
                    fontfamily='monospace', zorder=4)
        else:
            # Root sits at x=-1 with a black dot.
            ax.scatter([node.x], [node.y], s=90, color='#222', zorder=3,
                       edgecolors='white', linewidths=1.0)
            ax.text(node.x + 0.06, node.y, node.label,
                    ha='left', va='center', fontsize=10,
                    color='#222', fontweight='bold',
                    fontfamily='monospace', zorder=4)
        for child in node.children.values():
            draw_nodes(child)
    draw_nodes(root)

    # --- column headers (full-word, colour-matched) ---
    header_y = -1.4
    for lvl, (_key, header) in enumerate(TREE_LEVELS):
        ax.text(lvl, header_y, header,
                ha='center', va='center', fontsize=10.5,
                color=TREE_LEVEL_COLORS[lvl], fontweight='bold',
                fontfamily='monospace')
        ax.axvline(lvl, ymin=0, ymax=1,
                   color=TREE_LEVEL_COLORS[lvl], linewidth=0.4,
                   alpha=0.10, zorder=0)

    ax.set_xlim(-1.4, n_levels + 0.5)
    ax.set_ylim(n_leaves + 0.6, header_y - 1.0)  # y inverted (top → bottom)
    ax.axis('off')
    ax.set_title(
        f"Discovered experiment leaves ({len(all_entries)} total) - "
        "experiment/<addition>/<target>/<feature_set>/<architecture>/"
        "[<version>]/<datasplit>/<splittype>/[<hyperparameter>]",
        fontsize=11, pad=18,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison HTML helpers
# ---------------------------------------------------------------------------

def _strict_ci_winner(parsed, higher_better):
    """Strict CI-separation rule for picking a single statistically clear
    winner from a list of ``(key, mean, lo, hi)`` tuples.

    For higher-is-better metrics the candidate is the entry with the
    highest lower bound; for lower-is-better metrics it is the entry with
    the lowest upper bound. The candidate wins only when its 95% CI is
    disjoint from every other entry's CI. Returns ``{winner_key}`` on a
    strict win, or an empty set when CIs overlap or when fewer than two
    entries carry a parseable CI.
    """
    if len(parsed) < 2:
        return set()
    if higher_better:
        winner        = max(parsed, key=lambda t: t[2])    # highest lo
        others_max_hi = max(t[3] for t in parsed if t[0] != winner[0])
        separated     = winner[2] > others_max_hi
    else:
        winner        = min(parsed, key=lambda t: t[3])    # lowest hi
        others_min_lo = min(t[2] for t in parsed if t[0] != winner[0])
        separated     = winner[3] < others_min_lo
    return {winner[0]} if separated else set()


_METRIC_NOTES = {
    "Accuracy": (
        "Base-rate-dominated on imbalanced targets. On migraine "
        "(~5% positive rate) a constant-negative predictor reaches "
        "~95% accuracy, so this column should not be read as a "
        "headline. MCC, AUPRC and Brier are the primary metrics."
    ),
    "Calibration Slope": (
        "Logistic-recalibration slope. 1.0 = perfect calibration; "
        "&lt;1.0 = predictions are too extreme (the classical "
        "overfitting fingerprint); &gt;1.0 = predictions are too "
        "conservative. Required by TRIPOD+AI alongside ECE/Brier."
    ),
    "AUPRC": (
        "Comparing AUPRC <i>across</i> the headache (~25% positive) "
        "and migraine (~5% positive) columns is misleading because "
        "the baseline differs; compare within a target only."
    ),
}


def _render_metric_legend_rows(metrics):
    """Build the metric-legend table rows: one row per metric, each
    showing the higher/lower-is-better direction, the (vmin, vmax) range,
    a three-step worst-mid-best color swatch, and an optional reading
    note from _METRIC_NOTES.
    """
    out = ""
    for m_key, label, higher, vmin, vmax in metrics:
        direction = "↑ higher = better" if higher else "↓ lower = better"
        worst_bg  = compute_color(str(vmin if higher else vmax), higher, vmin, vmax)
        best_bg   = compute_color(str(vmax if higher else vmin), higher, vmin, vmax)
        mid_bg    = compute_color(str((vmin + vmax) / 2), higher, vmin, vmax)
        note      = _METRIC_NOTES.get(m_key, "")
        note_html = f'<div class="metric-note">{note}</div>' if note else ""
        out += (
            f"<tr><td><b>{label}</b></td><td>{direction}</td>"
            f"<td>{vmin} … {vmax}</td>"
            f'<td>'
            f'<span class="sw" style="background:{worst_bg}">worst</span> → '
            f'<span class="sw" style="background:{mid_bg}">mid</span> → '
            f'<span class="sw" style="background:{best_bg}">best</span>'
            f"{note_html}"
            f'</td></tr>'
        )
    return out


def _render_fs_glossary_rows(row_keys, fs_labels, fs_gloss):
    """Glossary rows for the feature-set short labels actually present
    on the architecture axis. ``row_keys`` carries the architecture axis
    (addition, architecture, version, feature_set); feature_set is at
    position 3.
    """
    fs_present = sorted({fs_labels.get(rk[3], rk[3]) for rk in row_keys})
    return "".join(
        f'<tr><td><b>[{s}]</b></td><td>{fs_gloss.get(s, "(no description)")}</td></tr>'
        for s in fs_present
    )


def _render_blended_caveat(row_keys):
    """Methodological caveat for the blended Spano replication. Empty
    when that architecture is absent. Architecture is row_keys[i][1].
    """
    has_blended = any(rk[1] == "blended_xgb_lr_spano2026" for rk in row_keys)
    if not has_blended:
        return ""
    return """
    <div class="caveat">
      <h3>Why <code>blended_xgb_lr_spano2026</code> appears only at the canonical 70_15_15 / chrono cell</h3>
      <p>The architecture's held-out calibration set drives <b>four sequential optimisation steps</b>: per-base isotonic + Platt calibrators, alpha grid search, final-calibrator selection, and downstream operating-threshold selection. Because the isotonic calibrators effectively memorise the calibration set, threshold-derived metrics (MCC, Sensitivity ≥ 0.5, F1) on test are unreliable; AUROC and AUPRC remain trustworthy because they are rank-based and calibration-invariant.</p>
      <p>The architecture is preserved as a faithful replication of the prior bachelor-thesis baseline (Spano 2026, single operating point). Fanning it out across ratios and split types would add cells whose threshold metrics could not be cleanly compared. The methodologically clean comparator is <code>stacked_2xgb_meta_lr</code>, which uses two-parameter Platt calibration only and is fanned out to the full grid.</p>
    </div>
    """


def _render_data_cell(rk, ck, entry, row_best, col_best_for_col, metrics):
    """Render the full ``<td>`` for one (row, column) cell.

    Returns the empty-cell placeholder when ``entry`` is None. Otherwise
    emits one ``.mrow`` block per metric, each carrying its hue-coded
    background plus optional left-edge / upper-edge strict-CI-separation
    marker. ``row_best`` is the per-metric winner set for this row,
    ``col_best_for_col`` is the per-metric winner set for this column;
    the two markers are independent (a cell can carry neither, one, or
    both). A small CV / H+CV badge in the corner records the source of
    the numeric values.
    """
    if entry is None:
        return '<td class="empty">-</td>'

    src       = entry.get("holdout") or {}
    is_cv_src = not src
    if is_cv_src:
        src = entry.get("cv") or {}
    has_cv = bool(entry.get("cv"))

    metric_rows = []
    for m_key, label, higher, vmin, vmax in metrics:
        cv_key = "MCC (Cal-Optimal)" if m_key == "MCC (Optimal)" else m_key
        val_str = src.get(cv_key if is_cv_src else m_key)
        bg      = compute_color(val_str, higher, vmin, vmax)
        parsed  = parse_mean_ci(val_str)
        if parsed is None:
            mean_disp = "N/A"
            ci_disp   = ""
            na_cls    = ' class="mv na"'
        else:
            mean, lo, hi = parsed
            mean_disp = f"{mean:.3f}"
            ci_disp   = (f'<span class="ci">[{lo:.3f}-{hi:.3f}]</span>'
                         if lo is not None and hi is not None else "")
            na_cls    = ' class="mv"'

        # Left edge: cell wins its row on this metric (strict CI separation).
        # Upper edge: cell wins its column on this metric (same rule, transposed).
        classes = ["mrow"]
        if ck in row_best.get(m_key, set()):
            classes.append("mrow-best")
        if rk in col_best_for_col.get(m_key, set()):
            classes.append("mrow-best-col")
        mrow_cls = " ".join(classes)
        metric_rows.append(
            f'<div class="{mrow_cls}" style="background:{bg}">'
            f'<span class="ml">{label}</span>'
            f'<span{na_cls}>{mean_disp}{ci_disp}</span>'
            f'</div>'
        )

    if is_cv_src:
        src_tag = '<div class="src-tag">CV</div>'
    elif has_cv:
        src_tag = '<div class="src-tag">H+CV</div>'
    else:
        src_tag = ""
    return f'<td class="dcell">{src_tag}{"".join(metric_rows)}</td>'


def _render_column_headers(col_keys):
    """Build the two-row column header for the comparison table.

    Row 1 groups consecutive columns that share the same ``target`` under
    a single ``<th>``; row 2 carries the per-column ratio / split label.
    The corner ``<th>`` spans the two row-header columns (addition chip
    plus the arch / ver / fs text block) and both header rows.
    """
    # target_groups: list of [target_name, span] pairs
    target_groups: list[list] = []
    last_target = None
    for target, _ds, _st in col_keys:
        if last_target is None or target != last_target:
            target_groups.append([target, 1])
            last_target = target
        else:
            target_groups[-1][1] += 1

    group_row_cells = [
        '<th class="corner" rowspan="2" colspan="2">'
        '<div class="corner-axis">rows ↓ Architecture</div>'
        '<div class="corner-axis">cols → Data Package</div></th>'
    ]
    for target, span in target_groups:
        group_row_cells.append(
            f'<th class="add-hdr" colspan="{span}">{target}</th>'
        )

    detail_row_cells = []
    for _target, ds, st in col_keys:
        split_disp = st if st else ""
        detail_row_cells.append(
            f'<th class="col-hdr">'
            f'<span class="c-arch">{ds}</span>'
            + (f'<br><span class="c-ver">{split_disp}</span>' if split_disp else "")
            + f'</th>'
        )

    return (
        f'<tr>{"".join(group_row_cells)}</tr>'
        f'<tr>{"".join(detail_row_cells)}</tr>'
    )


def _compute_best_sets(entries_with_src, metrics):
    """For each metric in ``metrics``, return the singleton-or-empty set
    of strictly CI-separated winner keys among ``entries_with_src``.

    ``entries_with_src`` is a list of ``(key, src, is_cv_source)`` tuples
    sharing one axis (e.g. all the cells in one row, or all the cells in
    one column). Returns ``{m_key: set_of_winner_keys}``. Cells whose
    source dict carries no parseable CI for the metric are silently
    skipped, so a CI-separation result is never claimed on point
    estimates. The MCC name shifts from "MCC (Optimal)" to
    "MCC (Cal-Optimal)" when the row's source is CV.
    """
    result: dict[str, set[tuple]] = {}
    for m_key, _label, higher, _vmin, _vmax in metrics:
        cv_key = "MCC (Cal-Optimal)" if m_key == "MCC (Optimal)" else m_key
        parsed = []
        for key, src, is_cv in entries_with_src:
            v_tuple = parse_mean_ci(src.get(cv_key if is_cv else m_key))
            if v_tuple is None:
                continue
            mean, lo, hi = v_tuple
            if lo is None or hi is None:
                continue
            parsed.append((key, mean, lo, hi))
        result[m_key] = _strict_ci_winner(parsed, higher)
    return result


def build_comparison_html(all_entries, iso_timestamp, uid,
                          venn_count_filename=None,
                          venn_names_filename=None,
                          tree_filename=None):
    """
    One big colour-coded flat table.
    Rows  = architectures  (addition / architecture / version / feature_set).
    Cols  = data packages  (target / datasplit / splittype).
    Cells = all 10 metrics, each with its own hue-coded background.
    Source: hold-out test results (fall back to CV mean if hold-out absent).

    The architecture axis sits on rows because that axis grows with each
    new model variant added to the study (TabPFN versions, AutoTabPFN,
    Fine-tuned, future additions), while the data-package axis stays
    near-constant (a fixed grid of target x ratio x split with one
    target-specific extension for the Park feature set). Putting the
    growing axis on rows keeps the table comfortable to read vertically
    as new variants land.

    Layout (top → bottom): table → colour legend → feature-set glossary →
    count Venn → names Venn → leaf tree (matplotlib PNG) →
    blended caveat (when applicable).
    """

    # Feature-set short labels - explicit map so adding new sets requires a
    # one-line edit. Previous heuristic `"full" if "full" in fs else "spano"`
    # silently mislabelled `no_rolling_features` as `[spano]`, hiding it
    # behind the actual spano column in the rendered table.
    FS_LABELS = {
        "full_features": "full",
        "no_rolling_features": "no_rolling",
        "spano_features": "spano",
        "park_features": "park",
    }
    FS_GLOSS = {
        "full": "all engineered features (today's triggers + rolling/lag/interaction)",
        "no_rolling": "today's triggers only - no temporal aggregation, lag, or streaks",
        "spano": "Spano (2026) feature subset - matches the prior-work replication",
        "park": "Park (2016) [Tab. 4] stepwise-selected triggers: stress, hormonal_changes, noise, alcohol, overeating, travel (migraine target only)",
    }

    # Architecture-axis sort: (addition, architecture, version, feature_set).
    # - addition first so all rows from the same experiment number cluster.
    # - architecture alphabetical inside each addition.
    # - version is the natural sub-sort within an architecture (tabpfn 2-6 vs 2-7).
    # - feature_set as the next tiebreaker so [full] / [no_rolling] / [spano]
    #   variants of the same architecture stay adjacent.
    # - hp_label distinguishes HP variants from their NonHP siblings inside
    #   the same (addition, architecture, version, feature_set) group; empty
    #   string for NonHP cells means they sort first within each group.
    def arch_key(e):
        p = e["path"]
        hp_label = ""
        if p.get("hp_strategy") and p.get("hp_variant"):
            hp_label = f"{p['hp_strategy']}/{p['hp_variant']}"
        return (p["addition"], p["architecture"], p["version"] or "", p["feature_set"], hp_label)

    # Data-package-axis sort: (target, datasplit, splittype).
    # target first so all headache columns sit together, then migraine,
    # which keeps the headache-vs-migraine visual divider intact.
    def data_key(e):
        p = e["path"]
        return (p["target"], p["datasplit"], p["splittype"] or "")

    row_keys = sorted(set(arch_key(e) for e in all_entries))
    col_keys = sorted(set(data_key(e) for e in all_entries))
    lookup   = {(arch_key(e), data_key(e)): e for e in all_entries}

    header_html = _render_column_headers(col_keys)

    # ---- column-best precomputation -----------------------------------
    # For each (col_key, m_key) pair, find the row whose CI is strictly
    # disjoint from every other row's CI in that column. Same strict
    # separation rule used for row-best, just transposed. A cell can
    # win in its row, in its column, in both, or neither - the two
    # markers are independent.
    best_col_set_per_metric: dict[tuple, dict[str, set[tuple]]] = {}
    for ck in col_keys:
        # Gather every row's source dict for this column, then defer the
        # per-metric strict-CI comparison to _compute_best_sets.
        col_entries: list[tuple[tuple, dict, bool]] = []
        for rk in row_keys:
            entry = lookup.get((rk, ck))
            if entry is None:
                continue
            src       = entry.get("holdout") or {}
            is_cv_src = not src
            if is_cv_src:
                src = entry.get("cv") or {}
            if src:
                col_entries.append((rk, src, is_cv_src))
        best_col_set_per_metric[ck] = _compute_best_sets(col_entries, COMPARISON_METRICS)

    # ---- data rows ----
    # One row per architecture-axis key. The row header is split into two
    # cells: a narrow colored chip carrying the addition number, and the
    # arch/ver/fs text block.
    #
    # Two independent best-cell markers fire per metric per cell:
    #   - Left edge (.mrow-best): cell's CI is strictly disjoint from
    #     every other cell's CI in the same row for this metric. Reads
    #     as "for this architecture, this data-package gives the best
    #     result we can statistically distinguish from the others".
    #   - Upper edge (.mrow-best-col): same rule applied down a column.
    #     Reads as "for this data-package on this metric, this
    #     architecture wins versus all others in a statistically
    #     distinguishable way".
    # The two markers are independent; a cell can have neither, one,
    # or both.
    rows_html = ""
    for rk in row_keys:
        addition, arch, ver, fs, hp_label = rk
        arch_disp = arch.replace("_", " ")
        ver_disp  = ver.replace("version_", "") if ver else ""
        fs_short  = FS_LABELS.get(fs, fs)

        # First pass: collect numeric values per metric across this row's
        # data cells so we can identify the row's best column per metric.
        # CV-source cells fall back to CV-named keys for MCC.
        row_entries: list[tuple[tuple, dict, bool]] = []
        for ck in col_keys:
            entry = lookup.get((rk, ck))
            if entry is None:
                row_entries.append((ck, {}, False))
                continue
            src       = entry.get("holdout") or {}
            is_cv_src = not src
            if is_cv_src:
                src = entry.get("cv") or {}
            row_entries.append((ck, src, is_cv_src))

        # Strict CI-separation best-set for this row, per metric. At this
        # dataset's sample size most rows have overlapping CIs and
        # correctly show no marker ("no clear winner at this sample
        # size"); the rare strict winners stand out as lighthouse-contrast
        # highlights.
        best_set_per_metric = _compute_best_sets(
            [(ck, src, is_cv) for ck, src, is_cv in row_entries if src],
            COMPARISON_METRICS,
        )

        parts = [
            f'<div class="rh-line"><span class="rh-key">arch:</span> {arch_disp}</div>',
        ]
        if ver_disp:
            parts.append(f'<div class="rh-line"><span class="rh-key">ver:</span> v{ver_disp}</div>')
        parts.append(f'<div class="rh-line"><span class="rh-key">fs:</span> [{fs_short}]</div>')
        if hp_label:
            # HP variant rows get a "HP:" line below the fs label, e.g.
            # "HP: single_AUROC/HP020" or "HP: pareto_AUROC_slope/knee".
            parts.append(f'<div class="rh-line"><span class="rh-key">HP:</span> {hp_label}</div>')

        # Two row-header cells: addition chip + arch/ver/fs text block.
        add_chip_cell = (
            f'<td class="add-chip add-{addition}">'
            f'<span class="add-chip-text">Addition&nbsp;{addition}</span>'
            f'</td>'
        )
        cells = add_chip_cell + f'<td class="row-hdr">{"".join(parts)}</td>'

        for ck, _src, _is_cv in row_entries:
            entry = lookup.get((rk, ck))
            cells += _render_data_cell(
                rk, ck, entry,
                row_best=best_set_per_metric,
                col_best_for_col=best_col_set_per_metric.get(ck, {}),
                metrics=COMPARISON_METRICS,
            )

        rows_html += f"<tr>{cells}</tr>\n"

    legend_rows      = _render_metric_legend_rows(COMPARISON_METRICS)
    fs_glossary_rows = _render_fs_glossary_rows(row_keys, FS_LABELS, FS_GLOSS)
    caveat_html      = _render_blended_caveat(row_keys)

    # Venn - region counts (full / spano / no_rolling), engineered + original split
    venn_count_html = (f"""
    <div class="venn">
      <h3>Feature-set inclusion (region counts)</h3>
      <p>Computed from the canonical <code>diary_train.parquet</code> by running both filters
         (<code>remove_non_spano_features</code>, <code>remove_rolling_features</code>) at aggregation time.
         <b>full</b> = all engineered columns; <b>spano</b> = Spano-faithful subset; <b>no_rolling</b> = drops rolling/lag/interaction features.
         Each region label shows <i>total (engineered + original SHD columns)</i>.</p>
      <img src="{venn_count_filename}" alt="Feature-set Venn - region counts (full / spano / no_rolling)">
    </div>
    """ if venn_count_filename else "")

    # Venn - every feature name, colour-coded (engineered vs original / 1:1 rename)
    venn_names_html = (f"""
    <div class="venn">
      <h3>Feature-set inclusion (every feature by name)</h3>
      <p>Same three sets as above, but each region lists the actual feature names.
         <span style="color:{CATEGORY_COLORS['original']}">●</span> <b>green</b> = original SHD column or 1:1 rename.
         <span style="color:{CATEGORY_COLORS['engineered']};font-weight:bold">●</span> <b>purple bold</b> = engineered (rolling, lag, interaction, or state-derived).</p>
      <img src="{venn_names_filename}" alt="Feature-set Venn - feature names colour-coded by origin">
    </div>
    """ if venn_names_filename else "")

    # Matplotlib tree diagram of all leaves discovered in this aggregation run.
    tree_html = (f"""
    <div class="tree">
      <h3>Discovered leaves ({len(all_entries)} total)</h3>
      <p>One row per discovered <code>results/</code> directory. Path levels mirror the directory layout:
         <code>experiment/&lt;addition&gt;/&lt;target&gt;/&lt;feature_set&gt;/&lt;architecture&gt;/[&lt;version&gt;]/&lt;datasplit&gt;/&lt;splittype&gt;/[&lt;hyperparameter&gt;]</code>.
         Each column has its own colour; node labels use the directory name verbatim.</p>
      <img src="{tree_filename}" alt="Tree diagram of all discovered experiment leaves">
    </div>
    """ if tree_filename else "")

    return f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>Comparison Table - {iso_timestamp}</title>
    <style>
      body {{
        font-family: 'Courier New', monospace;
        margin: 20px; background: #f8f8f8; color: #222; font-size: 0.83em;
      }}
      h1   {{ font-size: 1.2em; margin-bottom: 3px; }}
      .meta {{ color: #777; font-size: 0.82em; margin-bottom: 18px; }}
      .wrap {{ overflow-x: auto; }}
    
      table {{ border-collapse: collapse; }}
      th, td {{ border: 1px solid #c8c8c8; padding: 0; vertical-align: top; }}
    
      .corner {{
        background: #c8d3df; padding: 8px 12px;
        text-align: left; min-width: 160px; vertical-align: middle;
        font-size: 0.78em; position: sticky; left: 0; z-index: 2;
      }}
      .corner-axis {{ display: block; line-height: 1.5; color: #1a3550; }}
      .add-hdr {{
        background: #1a3550; color: #fff; text-align: center;
        font-weight: bold; font-size: 0.88em; letter-spacing: 0.04em;
        padding: 4px 10px; border-bottom: 2px solid #6f8aa6;
        position: sticky; top: 0; z-index: 1;
      }}
      .col-hdr {{
        background: #dde3ea; text-align: center;
        padding: 6px 10px; min-width: 120px;
        position: sticky; top: 28px; z-index: 1;
      }}
      .c-arch {{ font-weight: bold; font-size: 0.92em; display: block; }}
      .c-ver  {{ color: #444; font-size: 0.78em; display: block; }}
      .c-fs   {{
        color: #5a3300; background: #fff3d4; font-size: 0.74em; display: inline-block;
        padding: 0 6px; margin-top: 4px; border-radius: 3px; font-weight: bold;
      }}
    
      .row-hdr {{
        background: #dde3ea; padding: 6px 10px; font-weight: bold;
        font-size: 0.83em; min-width: 140px; vertical-align: middle;
        text-align: left; position: sticky; left: 28px; z-index: 1;
      }}
      .rh-target {{ font-size: 1.05em; font-weight: bold; color: #1a3550; }}
      .rh-line {{ font-weight: normal; color: #444; font-size: 0.92em; margin-top: 1px; }}
      .rh-key {{ color: #888; font-weight: normal; }}

      /* Addition chip: leftmost narrow column on each data row. The chip's
         background colour encodes the experiment-addition number so a
         reader can group rows visually before reading the arch label.
         Width is kept narrow because the chip is a categorical marker,
         not a label - the text rotates vertically inside the chip. */
      .add-chip {{
        width: 28px; min-width: 28px; padding: 6px 0;
        vertical-align: middle; text-align: center;
        position: sticky; left: 0; z-index: 1;
        background: #888; color: #fff;
      }}
      .add-chip-text {{
        writing-mode: vertical-rl; transform: rotate(180deg);
        font-size: 0.72em; font-weight: bold; letter-spacing: 0.06em;
        white-space: nowrap;
      }}
      /* Per-addition colours. Extend this list as new additions land. */
      .add-chip.add-0 {{ background: #2f5b8a; }}  /* deep blue */
      .add-chip.add-1 {{ background: #c47218; }}  /* amber/copper */
      .add-chip.add-2 {{ background: #5a4585; }}  /* violet */
      .add-chip.add-3 {{ background: #2f7a5b; }}  /* forest green */
      .add-chip.add-4 {{ background: #8a2f5b; }}  /* magenta */

      /* Best-cell markers. Both rules use strict CI separation: the
         winning cell's 95% bootstrap CI does not overlap any other
         cell's CI in the same row (row-best) or same column (col-best)
         for that metric. The two markers are independent; a cell can
         carry one, both, or neither. The visual is intentionally a
         thin coloured edge (no outline, no shadow) so the eye finds
         the lighthouse-contrast cells without crowding their content.

         - Left edge, dark green: row-best
           "for this architecture, this data-package is statistically
            distinguishable from the rest of the row."

         - Upper edge, deep violet: column-best
           "for this data-package, this architecture is statistically
            distinguishable from the rest of the column."
      */
      .mrow.mrow-best {{
        border-left: 3px solid #0d3a0d;
        padding-left: 3px;
      }}
      .mrow.mrow-best-col {{
        border-top: 3px solid #5a3a8a;
        padding-top: 0px;
      }}

      /* Marker-key swatch in the best-cell legend block. The swatch is
         a small inline box that wears the same border-edge style as the
         in-table marker, so the legend visually matches the cells. */
      .mk-sample {{
        display: inline-block; width: 22px; height: 14px;
        background: #f3f3f3; border: 1px solid #c8c8c8;
        vertical-align: middle; margin-right: 4px;
      }}
      .mk-sample.mk-row  {{ border-left: 3px solid #0d3a0d; }}
      .mk-sample.mk-col  {{ border-top: 3px solid #5a3a8a; }}
      .mk-sample.mk-both {{
        border-left: 3px solid #0d3a0d;
        border-top: 3px solid #5a3a8a;
      }}
    
      /* Axis-info panel above the table */
      .axis-info {{
        margin-bottom: 14px; padding: 10px 14px; background: #fff;
        border: 1px solid #c8d3df; border-radius: 4px; font-size: 0.85em;
      }}
      .axis-info h2 {{ margin: 0 0 8px; font-size: 0.95em; color: #1a3550; }}
      .axis-info dl {{ margin: 0; }}
      .axis-info dt {{ font-weight: bold; color: #1a3550; margin-top: 6px; }}
      .axis-info dt:first-of-type {{ margin-top: 0; }}
      .axis-info dd {{ margin: 2px 0 0 18px; color: #444; }}
    
      .dcell {{ padding: 0; min-width: 120px; background: #fff; }}
      .src-tag {{
        font-size: 0.65em; color: #888; text-align: right;
        padding: 1px 4px; background: #f0f0f0; border-bottom: 1px solid #ddd;
      }}
      .mrow {{
        display: flex; justify-content: space-between;
        padding: 2px 6px; border-bottom: 1px solid rgba(0,0,0,0.07);
        font-size: 0.79em;
      }}
      .mrow:last-child {{ border-bottom: none; }}
      .ml {{ color: #555; }}
      .mv {{ font-weight: bold; display: inline-flex; align-items: baseline; gap: 4px; }}
      .mv.na {{ color: #b00; font-style: italic; font-weight: normal; }}
      .ci {{
        font-weight: normal; font-size: 0.85em; color: #555;
        letter-spacing: -0.02em;
      }}
      /* Per-metric note shown under each legend swatch row. Plain bordered
         box - no left-bar accent, no orange chrome - so it does not visually
         compete with the metric row it belongs to. */
      .metric-note {{
        font-size: 0.85em; color: #333; background: #fafafa;
        border: 1px solid #d4d4d4; border-radius: 2px;
        margin-top: 6px; padding: 5px 8px; max-width: 380px;
      }}

      .empty {{
        background: #f2f2f2; text-align: center; color: #ccc;
        font-size: 1.3em; padding: 18px; min-width: 120px;
      }}

      /* Colour legend. Cell borders are intentionally visible (#aaa) so each
         row/column reads as its own boxed unit; without explicit lines the
         metric/direction/range/swatch columns blur into one another. */
      .legend {{
        margin-top: 22px; max-width: 640px;
        border: 1px solid #aaa; border-radius: 3px;
        padding: 10px 14px; background: #fff;
      }}
      .legend h3 {{ font-size: 0.88em; margin: 0 0 7px; }}
      .legend table {{ border-collapse: collapse; width: 100%; }}
      .legend th, .legend td {{
        border: 1px solid #b8b8b8; padding: 5px 8px;
        font-size: 0.78em; vertical-align: middle;
      }}
      .legend th {{
        background: #ececec; color: #1a3550; text-align: left;
        font-weight: 600; letter-spacing: 0.02em;
      }}
      .legend tbody tr:nth-child(odd) td {{ background: #fbfbfb; }}
      .sw {{
        display: inline-block; padding: 1px 5px;
        border-radius: 2px; border: 1px solid rgba(0,0,0,0.12);
        font-size: 0.9em;
      }}

      /* Methodological caveat block. Plain bordered box matching the legend
         and metric-note style, no left-bar accent. */
      .caveat {{
        margin-top: 22px; max-width: 760px;
        border: 1px solid #b8b8b8; border-radius: 3px;
        padding: 10px 14px; background: #fafafa; color: #222;
      }}
      .caveat h3 {{ font-size: 0.88em; margin: 0 0 7px; color: #222; }}
      .caveat p {{ font-size: 0.82em; margin: 4px 0; color: #222; line-height: 1.5; }}
      .caveat code {{
        font-family: 'Courier New', monospace; font-size: 0.95em;
        background: #fff; padding: 0 4px; border-radius: 2px;
      }}

      /* Feature-set Venn diagrams (count + names variants share the same chrome) */
      .venn {{
        margin-top: 22px; max-width: 1100px;
        border: 1px solid #ddd; border-radius: 3px;
        padding: 10px 14px; background: #fff;
      }}
      .venn h3 {{ font-size: 0.88em; margin: 0 0 7px; }}
      .venn p {{ font-size: 0.78em; margin: 4px 0 8px; color: #555; line-height: 1.4; }}
      .venn img {{ max-width: 100%; height: auto; display: block; }}

      /* Tree visualisation of the leaf hierarchy (matplotlib PNG) */
      .tree {{
        margin-top: 22px; max-width: 1400px;
        border: 1px solid #ddd; border-radius: 3px;
        padding: 10px 14px; background: #fff;
      }}
      .tree h3 {{ font-size: 0.88em; margin: 0 0 7px; }}
      .tree p {{ font-size: 0.78em; margin: 4px 0 8px; color: #555; line-height: 1.4; }}
      .tree img {{ max-width: 100%; height: auto; display: block; }}
    </style>
    </head>
    <body>
    <h1>Architecture Comparison Table</h1>
    <div class="meta">
      Generated: {iso_timestamp} &nbsp;·&nbsp; ID: {uid}
    </div>
    
    <div class="axis-info">
      <h2>How to read this table</h2>
      <dl>
        <dt>Rows (Y-axis) - Data Package</dt>
        <dd>Each row is one (target × split-ratio × split-strategy) combination.
            <b>target</b> = headache or migraine.
            <b>ratio</b> = train/val/test split sizes (e.g. 70_15_15 or 70_30 / 80_20 with chronological cal sub-split).
            <b>split</b> = how rows are assigned: <i>chrono</i> (date-percentile cuts), <i>stratified</i> (class-balanced random shuffle), <i>patient</i> (whole-patient holdout).</dd>
        <dt>Columns (X-axis) - Architecture</dt>
        <dd>Each column is one (model × version × feature-set) combination. Top line: model name. Middle line (if present): version. Bottom yellow tag <b>[…]</b>: feature-set abbreviation - see glossary below.</dd>
        <dt>Cells</dt>
        <dd>All 10 metrics on the locked test set. Each metric has its own colour scale (see <i>Colour legend</i>). The MCC row substitutes "Cal-Optimal" for "Optimal" automatically when the source is CV. A cell tagged <b>H+CV</b> has both hold-out and 5-fold CV results (cell shows hold-out). A cell tagged <b>CV</b> only has CV results - used as fallback when hold-out is missing. Empty (-) means no result file for that combination.</dd>
      </dl>
    </div>
    
    <div class="wrap">
    <table>
      <thead>{header_html}</thead>
      <tbody>
    {rows_html}  </tbody>
    </table>
    </div>

    <div class="legend">
      <h3>Best-cell markers (strict CI separation)</h3>
      <table>
        <tr><th>Marker</th><th>Meaning</th></tr>
        <tr>
          <td><span class="mk-sample mk-row">&nbsp;&nbsp;&nbsp;</span> left edge, dark green</td>
          <td><b>Row-best</b>: for this architecture, this data-package's 95% CI on this metric does not overlap any other data-package's CI in the row. Statistically distinguishable best across data packages.</td>
        </tr>
        <tr>
          <td><span class="mk-sample mk-col">&nbsp;&nbsp;&nbsp;</span> upper edge, deep violet</td>
          <td><b>Column-best</b>: for this data-package, this architecture's 95% CI on this metric does not overlap any other architecture's CI in the column. Statistically distinguishable best across architectures.</td>
        </tr>
        <tr>
          <td><span class="mk-sample mk-both">&nbsp;&nbsp;&nbsp;</span> both edges</td>
          <td>Cell wins both directions independently. Rare on small data: not only is this architecture the clear winner on this data-package, but this data-package is also the clear winner for this architecture.</td>
        </tr>
        <tr>
          <td>no edge</td>
          <td>No statistically distinguishable winner in this row/column for this metric at our sample size. The default reading; absence of marker is itself an honest scientific signal.</td>
        </tr>
      </table>
    </div>

    <div class="legend">
      <h3>Colour legend (per-metric scale)</h3>
      <table>
        <tr><th>Metric</th><th>Direction</th><th>Reference range</th><th>Scale</th></tr>
        {legend_rows}
      </table>
    </div>

    <div class="legend">
      <h3>Feature-set glossary</h3>
      <table>
        <tr><th>Tag</th><th>Meaning</th></tr>
        {fs_glossary_rows}
      </table>
    </div>
    {venn_count_html}
    {venn_names_html}
    {tree_html}
    {caveat_html}
    </body>
    </html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _archive_dest_for(src_path, dest_root):
    """Return ``<dest_root>/<YYYY-MM-DD>/<basename>`` for an aggregator
    output file, where the date is taken from the file's own mtime."""
    from datetime import datetime
    day_str = datetime.fromtimestamp(src_path.stat().st_mtime).strftime("%Y-%m-%d")
    day_dir = dest_root / day_str
    day_dir.mkdir(exist_ok=True)
    return day_dir / src_path.name


def _migrate_flat_archive(archive_dir):
    """One-time bring-up: if any aggregator outputs are still sitting
    directly under ``experiment/results/`` (from an older flat-archive
    version of this function), file them into their YYYY-MM-DD subfolder
    so the directory is uniformly date-bucketed going forward."""
    migrated = 0
    for pattern in OUTPUT_PATTERNS:
        for src in archive_dir.glob(pattern):
            if not src.is_file():
                continue
            dest = _archive_dest_for(src, archive_dir)
            if dest == src:
                continue
            if dest.exists():
                dest.unlink()
            src.rename(dest)
            migrated += 1
    if migrated:
        print(f"Migrated {migrated} flat-archived file{'s' if migrated != 1 else ''} "
              f"into {archive_dir}/YYYY-MM-DD/ subfolders")


def archive_previous_outputs():
    """Move any prior aggregator outputs from LATEST_DIR into ARCHIVE_DIR.

    Each archived file is filed under a dated subfolder
    ``experiment/results/YYYY-MM-DD/`` (ISO-8601 calendar date) inferred
    from the file's own mtime. This keeps the archive browsable by
    day for a project that produces many aggregator runs - dropping
    ~7 outputs at a time into a single flat directory rapidly becomes
    unreadable; per-day folders preserve the run-grouping a reader
    actually cares about.

    The flat scan over LATEST_DIR remains non-recursive so leaf .txt
    files inside experiment/<.../>results/ are untouched.
    """
    ARCHIVE_DIR.mkdir(exist_ok=True)
    _migrate_flat_archive(ARCHIVE_DIR)
    moved = 0
    per_day_counts: dict[str, int] = {}
    for pattern in OUTPUT_PATTERNS:
        for src in LATEST_DIR.glob(pattern):
            if not src.is_file():
                continue
            dest = _archive_dest_for(src, ARCHIVE_DIR)
            if dest.exists():
                # Re-run with identical timestamp; keep the new version.
                dest.unlink()
            day_str = dest.parent.name
            src.rename(dest)
            moved += 1
            per_day_counts[day_str] = per_day_counts.get(day_str, 0) + 1
    if moved:
        per_day_summary = ", ".join(
            f"{day}={n}" for day, n in sorted(per_day_counts.items())
        )
        print(f"Archived {moved} prior output file{'s' if moved != 1 else ''} "
              f"into {ARCHIVE_DIR}/YYYY-MM-DD/ ({per_day_summary})")


def main():
    contract    = getContract()
    contract_cv = getContract_cv()

    results_dirs = find_results_dirs()
    if not results_dirs:
        print("No results/ directories found.")
        return

    print(f"Found {len(results_dirs)} results director{'y' if len(results_dirs) == 1 else 'ies'}.")

    all_entries = []
    for rd in results_dirs:
        path_dims             = parse_path(rd)
        holdout_file, cv_file = find_latest_files(rd)

        holdout_raw = parse_file(holdout_file, contract)
        cv_raw      = parse_file(cv_file, contract_cv)
        cv_clean    = postprocess_cv(cv_raw)

        # Separate thresholds from metric values in the hold-out dict
        threshold_mcc  = (holdout_raw or {}).get("threshold_mcc")
        threshold_sens = (holdout_raw or {}).get("threshold_sens")
        holdout_metrics = (
            {k: v for k, v in holdout_raw.items() if k not in ("threshold_mcc", "threshold_sens")}
            if holdout_raw else None
        )

        entry = {
            "path":           path_dims,
            "holdout":        holdout_metrics,
            "cv":             cv_clean,
            "threshold_mcc":  threshold_mcc,
            "threshold_sens": threshold_sens,
            "holdout_ts":     format_ts(holdout_file),
            "cv_ts":          format_ts(cv_file),
        }
        all_entries.append(entry)
        warn_na(entry)

    iso_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    short_uid     = str(uuid.uuid4())[:8]
    ts_flat       = iso_timestamp.replace(":", "").replace("-", "")

    archive_previous_outputs()

    html = build_html(all_entries, iso_timestamp, short_uid)
    output = LATEST_DIR / f"results_{ts_flat}_{short_uid}.html"
    output.write_text(html, encoding="utf-8")
    print(f"Saved: {output}")

    # Venn PNGs - count variant + per-name variant. Skipped silently if no
    # parquet is available yet (e.g. data pipeline hasn't been run).
    venn_count_filename = None
    venn_names_filename = None
    feature_sets = compute_feature_sets()
    if feature_sets is not None:
        venn_count_path = LATEST_DIR / f"venn_counts_{ts_flat}_{short_uid}.png"
        try:
            generate_count_venn_png(feature_sets, venn_count_path)
            venn_count_filename = venn_count_path.name
            print(f"Saved: {venn_count_path}")
        except Exception as e:
            print(f"{RED}Count-Venn render failed: {e}{RESET}")

        venn_names_path = LATEST_DIR / f"venn_names_{ts_flat}_{short_uid}.png"
        try:
            generate_names_venn_png(feature_sets, venn_names_path)
            venn_names_filename = venn_names_path.name
            print(f"Saved: {venn_names_path}")
        except Exception as e:
            print(f"{RED}Names-Venn render failed: {e}{RESET}")

    # Tree diagram (matplotlib horizontal phylogeny over all leaves).
    tree_filename = None
    tree_path = LATEST_DIR / f"tree_{ts_flat}_{short_uid}.png"
    try:
        generate_tree_png(all_entries, tree_path)
        tree_filename = tree_path.name
        print(f"Saved: {tree_path}")
    except Exception as e:
        print(f"{RED}Tree render failed: {e}{RESET}")

    html_cmp = build_comparison_html(
        all_entries, iso_timestamp, short_uid,
        venn_count_filename=venn_count_filename,
        venn_names_filename=venn_names_filename,
        tree_filename=tree_filename,
    )
    output_cmp = LATEST_DIR / f"comparison_{ts_flat}_{short_uid}.html"
    output_cmp.write_text(html_cmp, encoding="utf-8")
    print(f"Saved: {output_cmp}")


if __name__ == "__main__":
    main()
