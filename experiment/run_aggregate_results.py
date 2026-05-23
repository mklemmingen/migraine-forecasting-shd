"""Aggregate every leaf ``results/*.txt`` into one comparison table.

Walks ``experiment/`` to discover per-leaf result files, parses the
latest hold-out and 5-fold CV output of each leaf, and produces a
self-contained ``comparison_<ts>_<uid>.html`` with:

- Colour-coded heatmap (RdBu palette dispatched by metric kind).
- Strict CI-separation row-best / column-best markers.
- Cross-architecture benchmark figures (critical-difference diagram,
  Dolan-More performance profile, rank slopegraph) for AUROC per target.
- Feature-set Venn diagrams (region counts + every feature name).
- Tree diagram of the discovered leaf hierarchy.
- A legacy ``results_<ts>_<uid>.html`` interactive tree explorer is
  also written alongside the comparison HTML.

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
    format_ts, parse_file, postprocess_cv, warn_na,
)
from _archival import archive_previous_outputs   # noqa: E402
from _html_to_pdf import html_to_pdf   # noqa: E402
from _interactive_html import build_html   # noqa: E402
from _comparison_html import build_comparison_html   # noqa: E402
from _venn_diagrams import (   # noqa: E402
    compute_feature_sets, generate_count_venn_png, generate_names_venn_png,
)
from _tree_diagram import generate_tree_png   # noqa: E402
from _benchmark_visuals import render_benchmark_triplet   # noqa: E402


# ---------------------------------------------------------------------------
# Benchmark supplementary figures
# ---------------------------------------------------------------------------

def _build_benchmark_section(all_entries, out_dir, ts_flat, short_uid):
    """Generate critical-difference, performance-profile, and rank
    slopegraph PNGs for the headline AUROC metric per target, and
    return the HTML embed block. Returns empty string when generation
    fails or no qualifying cells are available.

    Grid restricted to ``full_features`` so the Friedman matched-design
    assumption holds: every cell carries the same set of competing
    architectures and HP variants.
    """
    triplets_per_target: dict[str, dict[str, str]] = {}
    for target in ("headache", "migraine"):
        try:
            files = render_benchmark_triplet(
                all_entries, "AUROC", target, out_dir, ts_flat, short_uid,
                metric_label="AUROC", higher_is_better=True,
                restrict_feature_sets={"full_features"},
            )
            for _tag, fname in files.items():
                print(f"Saved: {out_dir / fname}")
            triplets_per_target[target] = files
        except Exception as e:
            print(f"{RED}Benchmark viz failed for {target}: {e}{RESET}")
            triplets_per_target[target] = {}

    if not any(triplets_per_target.values()):
        return ""

    sections = []
    for target, files in triplets_per_target.items():
        if not files:
            continue
        imgs = []
        if "cd" in files:
            imgs.append(
                f'<div class="bench-fig"><h4>Critical Difference (Friedman + Nemenyi)</h4>'
                f'<img src="{files["cd"]}" alt="CD diagram - {target} AUROC"></div>'
            )
        if "perfprofile" in files:
            imgs.append(
                f'<div class="bench-fig"><h4>Performance profile (Dolan-More)</h4>'
                f'<img src="{files["perfprofile"]}" alt="Performance profile - {target} AUROC"></div>'
            )
        if "slopegraph" in files:
            imgs.append(
                f'<div class="bench-fig"><h4>Rank slopegraph across (ratio, split)</h4>'
                f'<img src="{files["slopegraph"]}" alt="Slopegraph - {target} AUROC"></div>'
            )
        if imgs:
            sections.append(
                f'<div class="bench-target"><h3>{target.capitalize()} - '
                f'AUROC across full_features cells</h3>{"".join(imgs)}</div>'
            )
    if not sections:
        return ""

    return f"""
    <div class="benchmark">
      <h3>Cross-architecture benchmark visualisations (AUROC)</h3>
      <p>Supplementary statistical and rank views to complement the colour-coded heatmap above:</p>
      <ul>
        <li><b>Critical Difference (CD)</b> [Demsar 2006, JMLR] - Friedman omnibus + Nemenyi post-hoc on per-cell ranks. Architectures connected by a heavy horizontal bar are not significantly different at alpha=0.05. The "CD = X" bar at the top is the minimum mean-rank gap required for significance.</li>
        <li><b>Performance profile</b> [Dolan & More 2002, Math. Prog.] - cumulative fraction of cells where each architecture is within factor tau of the best architecture in that cell. Curves that rise quickly (top-left) dominate; flatter curves are unstable.</li>
        <li><b>Rank slopegraph</b> [Tufte 2001] - per-architecture rank trajectory across (ratio, split) cells. Stable architectures are flat; unstable architectures crisscross.</li>
      </ul>
      <p>Grid restricted to <code>full_features</code> so every cell carries the same set of competing architectures (Friedman matched-design requirement).</p>
      {"".join(sections)}
    </div>
    """


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


def _generate_figures(all_entries, ts_flat, short_uid):
    """Generate the supplementary PNGs (two Venns, tree, benchmark
    triplet) and return their basenames keyed for HTML embedding.
    """
    figs = {
        "venn_count": None, "venn_names": None,
        "tree": None, "benchmark_html": "",
    }

    feature_sets = compute_feature_sets(EXPERIMENT_DIR)
    if feature_sets is not None:
        venn_count_path = LATEST_DIR / f"venn_counts_{ts_flat}_{short_uid}.png"
        try:
            generate_count_venn_png(feature_sets, venn_count_path)
            figs["venn_count"] = venn_count_path.name
            print(f"Saved: {venn_count_path}")
        except Exception as e:
            print(f"{RED}Count-Venn render failed: {e}{RESET}")

        venn_names_path = LATEST_DIR / f"venn_names_{ts_flat}_{short_uid}.png"
        try:
            generate_names_venn_png(feature_sets, venn_names_path)
            figs["venn_names"] = venn_names_path.name
            print(f"Saved: {venn_names_path}")
        except Exception as e:
            print(f"{RED}Names-Venn render failed: {e}{RESET}")

    tree_path = LATEST_DIR / f"tree_{ts_flat}_{short_uid}.png"
    try:
        generate_tree_png(all_entries, tree_path)
        figs["tree"] = tree_path.name
        print(f"Saved: {tree_path}")
    except Exception as e:
        print(f"{RED}Tree render failed: {e}{RESET}")

    figs["benchmark_html"] = _build_benchmark_section(
        all_entries, LATEST_DIR, ts_flat, short_uid,
    )
    return figs


def _write_csv(all_entries, out_path):
    """Flatten every leaf entry into one row and write to ``out_path``.

    One row per discovered results/ leaf. Path dimensions, both
    thresholds, both source timestamps, and every metric from
    ``METRICS_HOLDOUT`` and ``METRICS_CV`` get their own column.
    Hold-out cells carry the original 'mean [lo-hi]' string; CV cells
    carry 'mean +/- std' as produced by ``postprocess_cv``. Missing
    cells are emitted as empty strings so the CSV round-trips through
    pandas/Excel without dtype surprises.
    """
    path_cols = [
        "addition", "target", "feature_set", "architecture", "version",
        "datasplit", "splittype", "hyperparameter", "hp_strategy", "hp_variant",
    ]
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

    # Legacy interactive tree explorer.
    html = build_html(all_entries, iso_timestamp, short_uid,
                      METRICS_HOLDOUT, METRICS_CV)
    output = LATEST_DIR / f"results_{ts_flat}_{short_uid}.html"
    output.write_text(html, encoding="utf-8")
    print(f"Saved: {output}")
    html_to_pdf(output)

    figs = _generate_figures(all_entries, ts_flat, short_uid)

    html_cmp = build_comparison_html(
        all_entries, iso_timestamp, short_uid,
        venn_count_filename=figs["venn_count"],
        venn_names_filename=figs["venn_names"],
        tree_filename=figs["tree"],
        benchmark_html=figs["benchmark_html"],
    )
    output_cmp = LATEST_DIR / f"comparison_{ts_flat}_{short_uid}.html"
    output_cmp.write_text(html_cmp, encoding="utf-8")
    print(f"Saved: {output_cmp}")
    html_to_pdf(output_cmp)

    output_csv = LATEST_DIR / f"comparison_{ts_flat}_{short_uid}.csv"
    _write_csv(all_entries, output_csv)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
