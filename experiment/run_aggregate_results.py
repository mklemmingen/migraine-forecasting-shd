"""
run script that aggregates any and all results/ *LATEST* entries into one interactive table in html.

Uses the predefined structure of the folders as described in the Readme of:
`experiment/<NrAddition>/<feature_set>/<architecture>/<*dataSplit>/<*SplitType>`

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


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_results_dirs():
    """Return all leaf results/ directories that are not inside _-prefixed folders.

    Skips the top-level experiment/results/ directory itself — that's where THIS
    aggregator writes its own HTML output, not where leaves write their txt files.
    """
    found = []
    for p in EXPERIMENT_DIR.rglob("results"):
        if not p.is_dir():
            continue
        rel = p.relative_to(EXPERIMENT_DIR)
        if rel == Path("results"):
            continue  # top-level output folder — not a leaf
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

    return {
        "addition":     addition,
        "target":       target,
        "feature_set":  feature_set,
        "architecture": architecture,
        "version":      version,
        "datasplit":    datasplit,
        "splittype":    splittype,
        "hyperparameter": hyperparameter,
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
                print(f"{RED}WARNING: N/A — '{key}' missing in {label}: {path_str}{RESET}")


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
<title>Experiment Results — {iso_timestamp}</title>
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
COMPARISON_METRICS = [
    ("AUROC",               "AUROC",    True,  0.5,  1.0),
    ("AUPRC",               "AUPRC",    True,  0.0,  1.0),
    ("MCC (Optimal)",       "MCC",      True,  0.0,  0.5),   # practical upper bound ~0.5
    ("Sensitivity (>=0.5)", "Sens≥0.5", True,  0.0,  1.0),
    ("Brier Score",         "Brier ↓",  False, 0.0,  0.25),  # lower is better
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


def build_comparison_html(all_entries, iso_timestamp, uid):
    """
    One big colour-coded flat table.
    Rows  = data packages  (target / datasplit / splittype).
    Cols  = architectures  (feature_set / architecture / version).
    Cells = top-5 metrics, each with its own hue-coded background.
    Source: hold-out test results (fall back to CV mean if hold-out absent).
    """

    # Feature-set short labels — explicit map so adding new sets requires a
    # one-line edit. Previous heuristic `"full" if "full" in fs else "spano"`
    # silently mislabelled `no_rolling_features` as `[spano]`, hiding it
    # behind the actual spano column in the rendered table.
    FS_LABELS = {
        "full_features": "full",
        "no_rolling_features": "no_rolling",
        "spano_features": "spano",
    }
    FS_GLOSS = {
        "full": "all engineered features (today's triggers + rolling/lag/interaction)",
        "no_rolling": "today's triggers only — no temporal aggregation, lag, or streaks",
        "spano": "Spano (2026) feature subset — matches the prior-work replication",
    }

    # Column sort order: (addition, architecture, version, feature_set).
    # — addition first so all columns from the same experiment number cluster.
    # — architecture alphabetical inside each addition.
    # — version is the natural sub-sort within an architecture (tabpfn 2-6 vs 2-7).
    # — feature_set as the final tiebreaker so [full] / [no_rolling] / [spano]
    #   variants of the same architecture stay adjacent.
    def col_key(e):
        p = e["path"]
        return (p["addition"], p["architecture"], p["version"] or "", p["feature_set"])

    def row_key(e):
        p = e["path"]
        return (p["target"], p["datasplit"], p["splittype"] or "")

    col_keys = sorted(set(col_key(e) for e in all_entries))
    row_keys = sorted(set(row_key(e) for e in all_entries))
    lookup   = {(row_key(e), col_key(e)): e for e in all_entries}

    # ---- column headers ----
    # Two-row header: row 1 is the addition group spanning its columns,
    # row 2 has the per-column architecture / version / feature-set labels.
    addition_groups: list[tuple[str, int]] = []
    last_addition = None
    for addition, _arch, _ver, _fs in col_keys:
        if last_addition is None or addition != last_addition:
            addition_groups.append([addition, 1])
            last_addition = addition
        else:
            addition_groups[-1][1] += 1

    group_row_cells = [
        '<th class="corner" rowspan="2">'
        '<div class="corner-axis">rows ↓ Data Package</div>'
        '<div class="corner-axis">cols → Architecture</div></th>'
    ]
    for addition, span in addition_groups:
        group_row_cells.append(
            f'<th class="add-hdr" colspan="{span}">Addition {addition}</th>'
        )

    detail_row_cells = []
    for addition, arch, ver, fs in col_keys:
        arch_disp = arch.replace("_", " ")
        ver_disp  = ver.replace("version_", "") if ver else ""
        fs_short  = FS_LABELS.get(fs, fs)
        detail_row_cells.append(
            f'<th class="col-hdr">'
            f'<span class="c-arch">{arch_disp}</span>'
            + (f'<br><span class="c-ver">v{ver_disp}</span>' if ver_disp else "")
            + f'<br><span class="c-fs">[{fs_short}]</span>'
            f'</th>'
        )

    header_html = (
        f'<tr>{"".join(group_row_cells)}</tr>'
        f'<tr>{"".join(detail_row_cells)}</tr>'
    )

    # ---- data rows ----
    rows_html = ""
    for rk in row_keys:
        target, ds, st = rk
        # Label each row component so a reader knows which axis is which.
        parts = []
        parts.append(f'<div class="rh-target">{target}</div>')
        parts.append(f'<div class="rh-line"><span class="rh-key">ratio:</span> {ds}</div>')
        if st:
            parts.append(f'<div class="rh-line"><span class="rh-key">split:</span> {st}</div>')
        cells = f'<td class="row-hdr">{"".join(parts)}</td>'

        for ck in col_keys:
            entry = lookup.get((rk, ck))
            if entry is None:
                cells += '<td class="empty">—</td>'
                continue

            src       = entry.get("holdout") or {}
            has_cv    = bool(entry.get("cv"))
            is_cv_src = not src  # fall back to CV if no hold-out
            if is_cv_src:
                src = entry.get("cv") or {}

            metric_rows = []
            for m_key, label, higher, vmin, vmax in COMPARISON_METRICS:
                # CV keys differ: "MCC (Optimal)" → "MCC (Cal-Optimal)"
                lookup_key = m_key
                if is_cv_src and m_key == "MCC (Optimal)":
                    lookup_key = "MCC (Cal-Optimal)"
                val_str = src.get(lookup_key)
                bg      = compute_color(val_str, higher, vmin, vmax)
                display = val_str.split()[0] if val_str else "N/A"
                na_cls  = ' class="na"' if val_str is None else ""
                metric_rows.append(
                    f'<div class="mrow" style="background:{bg}">'
                    f'<span class="ml">{label}</span>'
                    f'<span class="mv"{na_cls}>{display}</span>'
                    f'</div>'
                )

            src_tag = ""
            if is_cv_src:
                src_tag = '<div class="src-tag">CV</div>'
            elif has_cv:
                src_tag = '<div class="src-tag">H+CV</div>'

            cells += f'<td class="dcell">{src_tag}{"".join(metric_rows)}</td>'

        rows_html += f"<tr>{cells}</tr>\n"

    # ---- legend ----
    legend_rows = ""
    for _, label, higher, vmin, vmax in COMPARISON_METRICS:
        direction = "↑ higher = better" if higher else "↓ lower = better"
        worst_bg  = compute_color(str(vmin if higher else vmax), higher, vmin, vmax)
        best_bg   = compute_color(str(vmax if higher else vmin), higher, vmin, vmax)
        mid_val   = (vmin + vmax) / 2
        mid_bg    = compute_color(str(mid_val), higher, vmin, vmax)
        legend_rows += (
            f"<tr><td><b>{label}</b></td><td>{direction}</td>"
            f"<td>{vmin} … {vmax}</td>"
            f'<td>'
            f'<span class="sw" style="background:{worst_bg}">worst</span> → '
            f'<span class="sw" style="background:{mid_bg}">mid</span> → '
            f'<span class="sw" style="background:{best_bg}">best</span>'
            f'</td></tr>'
        )

    # Glossary rows for feature-set abbreviations actually present in the table.
    # col_keys is now (addition, architecture, version, feature_set) — feature_set is element [3].
    fs_present_short = sorted({FS_LABELS.get(ck[3], ck[3]) for ck in col_keys})
    fs_glossary_rows = "".join(
        f'<tr><td><b>[{s}]</b></td><td>{FS_GLOSS.get(s, "(no description)")}</td></tr>'
        for s in fs_present_short
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Comparison Table — {iso_timestamp}</title>
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
    text-align: left; position: sticky; left: 0; z-index: 1;
  }}
  .rh-target {{ font-size: 1.05em; font-weight: bold; color: #1a3550; }}
  .rh-line {{ font-weight: normal; color: #444; font-size: 0.92em; margin-top: 1px; }}
  .rh-key {{ color: #888; font-weight: normal; }}

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
  .mv {{ font-weight: bold; }}
  .mv.na {{ color: #b00; font-style: italic; font-weight: normal; }}

  .empty {{
    background: #f2f2f2; text-align: center; color: #ccc;
    font-size: 1.3em; padding: 18px; min-width: 120px;
  }}

  /* Legend */
  .legend {{
    margin-top: 22px; max-width: 600px;
    border: 1px solid #ddd; border-radius: 3px;
    padding: 10px 14px; background: #fff;
  }}
  .legend h3 {{ font-size: 0.88em; margin: 0 0 7px; }}
  .legend table {{ border-collapse: collapse; width: 100%; }}
  .legend td {{ border: 1px solid #eee; padding: 3px 7px; font-size: 0.76em; vertical-align: middle; }}
  .sw {{
    display: inline-block; padding: 1px 5px;
    border-radius: 2px; border: 1px solid rgba(0,0,0,0.12);
    font-size: 0.9em;
  }}
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
    <dt>Rows (Y-axis) — Data Package</dt>
    <dd>Each row is one (target × split-ratio × split-strategy) combination.
        <b>target</b> = headache or migraine.
        <b>ratio</b> = train/val/test split sizes (e.g. 70_15_15 or 70_30 / 80_20 with chronological cal sub-split).
        <b>split</b> = how rows are assigned: <i>chrono</i> (date-percentile cuts), <i>stratified</i> (class-balanced random shuffle), <i>patient</i> (whole-patient holdout).</dd>
    <dt>Columns (X-axis) — Architecture</dt>
    <dd>Each column is one (model × version × feature-set) combination. Top line: model name. Middle line (if present): version. Bottom yellow tag <b>[…]</b>: feature-set abbreviation — see glossary below.</dd>
    <dt>Cells</dt>
    <dd>Top-5 metrics on the locked test set. Each metric has its own colour scale (see <i>Colour legend</i>). A cell tagged <b>H+CV</b> has both hold-out and 5-fold CV results (cell shows hold-out). A cell tagged <b>CV</b> only has CV results — used as fallback when hold-out is missing. Empty (—) means no result file for that combination.</dd>
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
  <h3>Feature-set glossary</h3>
  <table>
    <tr><th>Tag</th><th>Meaning</th></tr>
    {fs_glossary_rows}
  </table>
</div>

<div class="legend">
  <h3>Colour legend (per-metric scale)</h3>
  <table>
    <tr><th>Metric</th><th>Direction</th><th>Reference range</th><th>Scale</th></tr>
    {legend_rows}
  </table>
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    output_dir = EXPERIMENT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    html = build_html(all_entries, iso_timestamp, short_uid)
    output = output_dir / f"results_{ts_flat}_{short_uid}.html"
    output.write_text(html, encoding="utf-8")
    print(f"Saved: {output}")

    html_cmp = build_comparison_html(all_entries, iso_timestamp, short_uid)
    output_cmp = output_dir / f"comparison_{ts_flat}_{short_uid}.html"
    output_cmp.write_text(html_cmp, encoding="utf-8")
    print(f"Saved: {output_cmp}")


if __name__ == "__main__":
    main()
