"""Aggregate every leaf ``results/*.txt`` into one comparison table.

Walks ``experiment/`` to discover per-leaf result files, parses the
latest hold-out and 5-fold CV output of each leaf, and writes three
outputs, each stamped ``<ts>_<uid>``:

- ``comparison_<ts>.html`` - self-contained colour-coded heatmap (RdBu
  palette dispatched by metric kind) with strict CI-separation row-best /
  column-best markers and the feature-set glossary.
- ``results_<ts>.html`` - interactive tree explorer of the same leaves.
- ``comparison_<ts>.csv`` - the flat metric table, one row per leaf. This
  is the durable data interface for the publication figures: the
  per-figure scripts in ``docs/methodAndResults_diagramCreatorScripts/``
  read it back (via ``_parsing.entries_from_csv``) and render the
  critical-difference, performance-profile, slopegraph, Venn, and
  leaf-tree figures into ``docs/.../figures/`` from a pinned snapshot, so
  figure generation is decoupled from this aggregation pass.

Each result file's column contract is owned by ``sharedMetricPrinter``;
this script trusts that contract and re-renders the same metric set in
the heatmap. The directory layout it walks mirrors:
    experiment/<addition>/<target>/<feature_set>/<architecture>/
      [<version>]/<datasplit>/<splittype>/
      [<HyperparameterTuned>/<hp_strategy>/<hp_variant>]/results/
"""
import csv
import sys
import uuid
from datetime import datetime
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
LATEST_DIR     = EXPERIMENT_DIR
ARCHIVE_DIR    = EXPERIMENT_DIR / "results"

RED   = "\033[91m"
RESET = "\033[0m"

# Sub-modules live in _eval/ alongside the metric contract; flat-style
# imports are used throughout this package so the discovery-time
# sys.path entry is enough to resolve every helper.
sys.path.insert(0, str(EXPERIMENT_DIR / "_eval"))
from sharedMetricPrinter import (   # noqa: E402
    getContract, getContract_cv, METRICS_HOLDOUT, METRICS_CV,
)
from _parsing import (   # noqa: E402
    find_results_dirs, parse_path, find_latest_files,
    format_ts, parse_file, postprocess_cv, warn_na, PATH_DIMS,
)
from _archival import archive_previous_outputs   # noqa: E402
from _html_to_pdf import html_to_pdf   # noqa: E402
from _interactive_html import build_html   # noqa: E402
from _comparison_html import build_comparison_html   # noqa: E402


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def _collect_entries():
    """Discover every leaf results/ directory and parse its latest
    hold-out and 5-fold CV result file into one list of entries.

    Each entry carries the parsed path dimensions, the hold-out metric
    dict (or None), the CV metric dict (or None), the operating
    thresholds extracted from the hold-out file, and the timestamps of
    both source files for display in the interactive tree explorer.
    """
    contract    = getContract()
    contract_cv = getContract_cv()
    results_dirs = find_results_dirs(EXPERIMENT_DIR)
    if not results_dirs:
        return []
    print(f"Found {len(results_dirs)} results "
          f"director{'y' if len(results_dirs) == 1 else 'ies'}.")

    all_entries = []
    for rd in results_dirs:
        path_dims             = parse_path(rd, EXPERIMENT_DIR)
        holdout_file, cv_file = find_latest_files(rd)

        holdout_raw = parse_file(holdout_file, contract)
        cv_raw      = parse_file(cv_file, contract_cv)
        cv_clean    = postprocess_cv(cv_raw)

        # Separate thresholds from metric values in the hold-out dict.
        threshold_mcc  = (holdout_raw or {}).get("threshold_mcc")
        threshold_sens = (holdout_raw or {}).get("threshold_sens")
        holdout_metrics = (
            {k: v for k, v in holdout_raw.items()
             if k not in ("threshold_mcc", "threshold_sens")}
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
    return all_entries


def _write_csv(all_entries, out_path):
    """Flatten every leaf entry into one row and write to ``out_path``.

    This CSV is the durable interface the per-figure scripts in
    ``docs/methodAndResults_diagramCreatorScripts/`` consume (via
    ``_parsing.entries_from_csv``): one row per discovered results/ leaf,
    every path dimension and metric in its own column, so a figure can be
    re-rendered from a pinned snapshot without re-walking ``experiment/``.

    Hold-out cells carry the original 'mean [lo-hi]' string; CV cells
    carry 'mean +/- std' as produced by ``postprocess_cv``. Missing
    cells are emitted as empty strings so the CSV round-trips through
    pandas/Excel without dtype surprises.
    """
    path_cols    = list(PATH_DIMS)
    holdout_cols = [f"holdout_{m}" for m in METRICS_HOLDOUT]
    cv_cols      = [f"cv_{m}"      for m in METRICS_CV]
    header = (
        path_cols
        + ["threshold_mcc", "threshold_sens", "holdout_ts", "cv_ts"]
        + holdout_cols + cv_cols
    )

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for entry in all_entries:
            row = [entry["path"].get(c, "") or "" for c in path_cols]
            row += [
                entry.get("threshold_mcc") or "",
                entry.get("threshold_sens") or "",
                entry.get("holdout_ts") or "",
                entry.get("cv_ts") or "",
            ]
            holdout = entry.get("holdout") or {}
            cv      = entry.get("cv") or {}
            row += [holdout.get(m) or "" for m in METRICS_HOLDOUT]
            row += [cv.get(m)      or "" for m in METRICS_CV]
            writer.writerow(row)


def main():
    all_entries = _collect_entries()
    if not all_entries:
        print("No results/ directories found.")
        return

    iso_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    short_uid     = str(uuid.uuid4())[:8]
    ts_flat       = iso_timestamp.replace(":", "").replace("-", "")

    archive_previous_outputs(LATEST_DIR, ARCHIVE_DIR)

    # Interactive tree explorer.
    html = build_html(all_entries, iso_timestamp, short_uid,
                      METRICS_HOLDOUT, METRICS_CV)
    output = LATEST_DIR / f"results_{ts_flat}_{short_uid}.html"
    output.write_text(html, encoding="utf-8")
    print(f"Saved: {output}")
    html_to_pdf(output)

    html_cmp = build_comparison_html(all_entries, iso_timestamp, short_uid)
    output_cmp = LATEST_DIR / f"comparison_{ts_flat}_{short_uid}.html"
    output_cmp.write_text(html_cmp, encoding="utf-8")
    print(f"Saved: {output_cmp}")
    html_to_pdf(output_cmp)

    output_csv = LATEST_DIR / f"comparison_{ts_flat}_{short_uid}.csv"
    _write_csv(all_entries, output_csv)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
