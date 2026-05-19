"""Interactive tree-explorer HTML.

Generates the legacy ``results_<ts>_<uid>.html`` output: a JS-driven
breadcrumb explorer that lets a reader drill down through addition ->
target -> feature_set -> architecture -> per-cell holdout + CV results.
Complements the flat comparison table by providing an alternative
hierarchical view of the same data.
"""
import json
from string import Template


# All client-side state lives in DATA and METRICS_*; navigation walks
# through the four DIMS via breadcrumb clicks.
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


def build_html(all_entries, iso_timestamp, uid, metrics_holdout, metrics_cv):
    """Render the interactive tree-explorer HTML as a string."""
    data_json            = json.dumps(all_entries, ensure_ascii=False, indent=2)
    metrics_holdout_json = json.dumps(metrics_holdout)
    metrics_cv_json      = json.dumps(metrics_cv)

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
