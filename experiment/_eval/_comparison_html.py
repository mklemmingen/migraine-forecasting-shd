"""Architecture comparison HTML.

Renders the flat colour-coded comparison table that is the primary
output of ``run_aggregate_results.py``:

- Rows are architectures (addition / architecture / version / feature_set,
  with HP variants as sibling rows under their NonHP parent).
- Columns are data packages (target / datasplit / splittype).
- Each cell carries all eleven metrics with their own per-metric
  colour scale (RdBu palette dispatched by ``MetricSpec.kind``).

Strict CI-separation markers on the left edge (row-best) and top edge
(column-best) flag cells whose 95% bootstrap CI is disjoint from every
other cell on the same axis - "no marker" is the default, which is
itself an honest signal at this dataset's sample size.

The table is followed by:
- Best-cell-marker legend
- Per-metric colour legend
- Feature-set glossary
- Optional Venn / tree / benchmark figure embeds
- Methodological caveat for the blended Spano replicator
"""
from _metric_palette import (
    COMPARISON_METRICS,
    compute_color,
    parse_mean_ci,
)
from _venn_diagrams import CATEGORY_COLORS


# Feature-set short labels and the glossary that explains them. Explicit
# map (rather than substring matching) so adding new sets requires a
# one-line edit and "no_rolling" is never silently mislabelled as
# "spano".
_FS_LABELS = {
    "full_features":        "full",
    "no_rolling_features":  "no_rolling",
    "spano_features":       "spano",
    "park_features":        "park",
}
_FS_GLOSS = {
    "full":       "all engineered features (today's triggers + rolling/lag/interaction)",
    "no_rolling": "today's triggers only - no temporal aggregation, lag, or streaks",
    "spano":      "Spano (2026) feature subset - matches the prior-work replication",
    "park":       "Park (2016) [Tab. 4] stepwise-selected triggers: stress, hormonal_changes, "
                  "noise, alcohol, overeating, travel (migraine target only)",
}


_METRIC_NOTES = {
    "Accuracy": (
        "Base-rate-dominated on imbalanced targets. On migraine (~5% "
        "positive rate) a constant-negative predictor reaches ~95% "
        "accuracy, so this column should not be read as a headline. "
        "MCC, AUPRC and Brier are the primary metrics."
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


# ---------------------------------------------------------------------------
# CI-separation helpers (per-row and per-column "clear winner" markers)
# ---------------------------------------------------------------------------

def _strict_ci_winner(parsed, higher_better):
    """Strict CI-separation rule for picking a single statistically
    clear winner from a list of ``(key, mean, lo, hi)`` tuples.

    For higher-is-better metrics the candidate is the entry with the
    highest lower bound; for lower-is-better metrics it is the entry
    with the lowest upper bound. The candidate wins only when its 95%
    CI is disjoint from every other entry's CI. Returns
    ``{winner_key}`` on a strict win, or an empty set when CIs overlap
    or when fewer than two entries carry a parseable CI.
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


def _compute_best_sets(entries_with_src, metrics):
    """For each metric in ``metrics``, return the singleton-or-empty
    set of strictly CI-separated winner keys among
    ``entries_with_src``.

    ``entries_with_src`` is a list of ``(key, src, is_cv_source)``
    tuples sharing one axis (e.g. all the cells in one row, or all the
    cells in one column). Returns ``{m_key: set_of_winner_keys}``.
    Cells whose source dict carries no parseable CI for the metric are
    silently skipped, so a CI-separation result is never claimed on
    point estimates. The MCC name shifts from "MCC (Optimal)" to
    "MCC (Cal-Optimal)" when the source is CV.
    """
    result: dict[str, set[tuple]] = {}
    for spec in metrics:
        # Target metrics (CalSlope) have no "highest" or "lowest"
        # winner; the CI-winner concept would need a CI-distance-to-
        # optimum test that the current _strict_ci_winner does not
        # implement. Skip rather than mark spurious winners.
        if spec.kind == "target":
            result[spec.key] = set()
            continue
        higher = (spec.kind != "min")
        lookup_key = spec.cv_alias if spec.cv_alias else spec.key
        parsed = []
        for key, src, is_cv in entries_with_src:
            v_tuple = parse_mean_ci(
                src.get(lookup_key if is_cv and spec.cv_alias else spec.key)
            )
            if v_tuple is None:
                continue
            mean, lo, hi = v_tuple
            if lo is None or hi is None:
                continue
            parsed.append((key, mean, lo, hi))
        result[spec.key] = _strict_ci_winner(parsed, higher)
    return result


# ---------------------------------------------------------------------------
# HTML fragment helpers
# ---------------------------------------------------------------------------

def _legend_swatches_for(spec):
    """Build the (direction, range, swatch-html) tuple for one metric's
    legend row. Each kind emits a three-stop swatch that matches the
    coloring rule actually applied to data cells.
    """
    def sw(label, bg):
        return f'<span class="sw" style="background:{bg}">{label}</span>'

    if spec.kind == "max":
        return (
            "↑ higher = better",
            f"{spec.bad} … {spec.good}",
            (sw("worst", compute_color(str(spec.bad), spec)) + " → "
             + sw("mid", compute_color(str((spec.bad + spec.good) / 2), spec)) + " → "
             + sw("best", compute_color(str(spec.good), spec))),
        )
    if spec.kind == "min":
        return (
            "↓ lower = better",
            f"{spec.good} … {spec.bad}",
            (sw("best", compute_color(str(spec.good), spec)) + " → "
             + sw("mid", compute_color(str((spec.bad + spec.good) / 2), spec)) + " → "
             + sw("worst", compute_color(str(spec.bad), spec))),
        )
    if spec.kind == "diverge":
        return (
            f"pivot = {spec.pivot} (random); negative = anti-predictive",
            f"{spec.bad} … {spec.good}",
            (sw("anti", compute_color(str(spec.bad), spec)) + " → "
             + sw(f"{spec.pivot}", compute_color(str(spec.pivot), spec)) + " → "
             + sw("good", compute_color(str(spec.good), spec))),
        )
    if spec.kind == "target":
        opt = spec.pivot
        far = opt + spec.bad
        return (
            f"target = {opt} (deviation in either direction is worse)",
            f"|val - {opt}| up to {spec.bad}",
            (sw(f"|{far}|", compute_color(str(far), spec)) + " → "
             + sw(f"{opt}", compute_color(str(opt), spec)) + " → "
             + sw(f"|{opt - spec.bad}|", compute_color(str(opt - spec.bad), spec))),
        )
    return ("(unknown kind)", "", "")


def _render_metric_legend_rows(metrics):
    """Build the metric-legend table rows: one row per metric showing
    the higher/lower-is-better direction, the reference range, a
    three-step worst-mid-best colour swatch, and an optional reading
    note from ``_METRIC_NOTES``.
    """
    out = ""
    for spec in metrics:
        direction, range_str, swatches = _legend_swatches_for(spec)
        note      = _METRIC_NOTES.get(spec.key, "")
        note_html = f'<div class="metric-note">{note}</div>' if note else ""
        out += (
            f"<tr><td><b>{spec.label}</b></td><td>{direction}</td>"
            f"<td>{range_str}</td>"
            f'<td>{swatches}{note_html}</td></tr>'
        )
    return out


def _render_fs_glossary_rows(row_keys, fs_labels, fs_gloss):
    """Glossary rows for the feature-set short labels actually present
    on the architecture axis. ``row_keys`` carries the architecture
    axis (addition, architecture, version, feature_set); feature_set
    is at position 3.
    """
    fs_present = sorted({fs_labels.get(rk[3], rk[3]) for rk in row_keys})
    return "".join(
        f'<tr><td><b>[{s}]</b></td><td>{fs_gloss.get(s, "(no description)")}</td></tr>'
        for s in fs_present
    )


def _render_blended_caveat(row_keys):
    """Methodological caveat for the blended Spano replication.
    Returns the empty string when that architecture is absent.
    Architecture is at ``row_keys[i][1]``.
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

    Returns the empty-cell placeholder when ``entry`` is None.
    Otherwise emits one ``.mrow`` block per metric, each carrying its
    hue-coded background plus optional left-edge / upper-edge
    strict-CI-separation marker. ``row_best`` is the per-metric winner
    set for this row, ``col_best_for_col`` is the per-metric winner
    set for this column; the two markers are independent (a cell can
    carry neither, one, or both). A small CV / H+CV badge in the
    corner records the source of the numeric values.
    """
    if entry is None:
        return '<td class="empty">-</td>'

    src       = entry.get("holdout") or {}
    is_cv_src = not src
    if is_cv_src:
        src = entry.get("cv") or {}
    has_cv = bool(entry.get("cv"))

    metric_rows = []
    for spec in metrics:
        lookup_key = spec.cv_alias if (is_cv_src and spec.cv_alias) else spec.key
        val_str = src.get(lookup_key)
        bg      = compute_color(val_str, spec)
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
        if ck in row_best.get(spec.key, set()):
            classes.append("mrow-best")
        if rk in col_best_for_col.get(spec.key, set()):
            classes.append("mrow-best-col")
        mrow_cls = " ".join(classes)
        metric_rows.append(
            f'<div class="{mrow_cls}" style="background:{bg}">'
            f'<span class="ml">{spec.label}</span>'
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

    Row 1 groups consecutive columns that share the same ``target``
    under a single ``<th>``; row 2 carries the per-column ratio /
    split label. The corner ``<th>`` spans the two row-header columns
    (addition chip plus the arch / ver / fs text block) and both
    header rows.
    """
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


# ---------------------------------------------------------------------------
# Embed helpers (Venn / tree PNGs reference their file by basename)
# ---------------------------------------------------------------------------

def _venn_count_html(filename):
    if not filename:
        return ""
    return f"""
    <div class="venn">
      <h3>Feature-set inclusion (region counts)</h3>
      <p>Computed from the canonical <code>diary_train.parquet</code> by running both filters
         (<code>remove_non_spano_features</code>, <code>remove_rolling_features</code>) at aggregation time.
         <b>full</b> = all engineered columns; <b>spano</b> = Spano-faithful subset; <b>no_rolling</b> = drops rolling/lag/interaction features.
         Each region label shows <i>total (engineered + original SHD columns)</i>.</p>
      <img src="{filename}" alt="Feature-set Venn - region counts (full / spano / no_rolling)">
    </div>
    """


def _venn_names_html(filename):
    if not filename:
        return ""
    return f"""
    <div class="venn">
      <h3>Feature-set inclusion (every feature by name)</h3>
      <p>Same three sets as above, but each region lists the actual feature names.
         <span style="color:{CATEGORY_COLORS['original']}">●</span> <b>green</b> = original SHD column or 1:1 rename.
         <span style="color:{CATEGORY_COLORS['engineered']};font-weight:bold">●</span> <b>purple bold</b> = engineered (rolling, lag, interaction, or state-derived).</p>
      <img src="{filename}" alt="Feature-set Venn - feature names colour-coded by origin">
    </div>
    """


def _tree_html_section(filename, n_leaves):
    if not filename:
        return ""
    return f"""
    <div class="tree">
      <h3>Discovered leaves ({n_leaves} total)</h3>
      <p>One row per discovered <code>results/</code> directory. Path levels mirror the directory layout:
         <code>experiment/&lt;addition&gt;/&lt;target&gt;/&lt;feature_set&gt;/&lt;architecture&gt;/[&lt;version&gt;]/&lt;datasplit&gt;/&lt;splittype&gt;/[&lt;hyperparameter&gt;]</code>.
         Each column has its own colour; node labels use the directory name verbatim.</p>
      <img src="{filename}" alt="Tree diagram of all discovered experiment leaves">
    </div>
    """


# ---------------------------------------------------------------------------
# CSS block (kept inline so each comparison HTML is a single self-
# contained file portable to any machine without an asset-server).
# ---------------------------------------------------------------------------

_COMPARISON_CSS = """
      body {
        font-family: 'Courier New', monospace;
        margin: 20px; background: #f8f8f8; color: #222; font-size: 0.83em;
      }
      h1   { font-size: 1.2em; margin-bottom: 3px; }
      .meta { color: #777; font-size: 0.82em; margin-bottom: 18px; }
      .wrap { overflow-x: auto; }

      table { border-collapse: collapse; }
      th, td { border: 1px solid #c8c8c8; padding: 0; vertical-align: top; }

      .corner {
        background: #c8d3df; padding: 8px 12px;
        text-align: left; min-width: 160px; vertical-align: middle;
        font-size: 0.78em; position: sticky; left: 0; z-index: 2;
      }
      .corner-axis { display: block; line-height: 1.5; color: #1a3550; }
      .add-hdr {
        background: #1a3550; color: #fff; text-align: center;
        font-weight: bold; font-size: 0.88em; letter-spacing: 0.04em;
        padding: 4px 10px; border-bottom: 2px solid #6f8aa6;
        position: sticky; top: 0; z-index: 1;
      }
      .col-hdr {
        background: #dde3ea; text-align: center;
        padding: 6px 10px; min-width: 120px;
        position: sticky; top: 28px; z-index: 1;
      }
      .c-arch { font-weight: bold; font-size: 0.92em; display: block; }
      .c-ver  { color: #444; font-size: 0.78em; display: block; }
      .c-fs   {
        color: #5a3300; background: #fff3d4; font-size: 0.74em; display: inline-block;
        padding: 0 6px; margin-top: 4px; border-radius: 3px; font-weight: bold;
      }

      .row-hdr {
        background: #dde3ea; padding: 6px 10px; font-weight: bold;
        font-size: 0.83em; min-width: 140px; vertical-align: middle;
        text-align: left; position: sticky; left: 28px; z-index: 1;
      }
      .rh-target { font-size: 1.05em; font-weight: bold; color: #1a3550; }
      .rh-line { font-weight: normal; color: #444; font-size: 0.92em; margin-top: 1px; }
      .rh-key { color: #888; font-weight: normal; }

      /* Addition chip: leftmost narrow column on each data row. The chip's
         background colour encodes the experiment-addition number so a
         reader can group rows visually before reading the arch label.
         Width is kept narrow because the chip is a categorical marker,
         not a label - the text rotates vertically inside the chip. */
      .add-chip {
        width: 28px; min-width: 28px; padding: 6px 0;
        vertical-align: middle; text-align: center;
        position: sticky; left: 0; z-index: 1;
        background: #888; color: #fff;
      }
      .add-chip-text {
        writing-mode: vertical-rl; transform: rotate(180deg);
        font-size: 0.72em; font-weight: bold; letter-spacing: 0.06em;
        white-space: nowrap;
      }
      .add-chip.add-0 { background: #2f5b8a; }  /* deep blue */
      .add-chip.add-1 { background: #c47218; }  /* amber/copper */
      .add-chip.add-2 { background: #5a4585; }  /* violet */
      .add-chip.add-3 { background: #2f7a5b; }  /* forest green */
      .add-chip.add-4 { background: #8a2f5b; }  /* magenta */

      /* Best-cell markers (strict CI separation).
         Left edge dark green = row-best; top edge deep violet = column-best.
         Independent markers: a cell can carry neither, one, or both. */
      .mrow.mrow-best     { border-left: 3px solid #0d3a0d; padding-left: 3px; }
      .mrow.mrow-best-col { border-top:  3px solid #5a3a8a; padding-top:  0px; }

      .mk-sample {
        display: inline-block; width: 22px; height: 14px;
        background: #f3f3f3; border: 1px solid #c8c8c8;
        vertical-align: middle; margin-right: 4px;
      }
      .mk-sample.mk-row  { border-left: 3px solid #0d3a0d; }
      .mk-sample.mk-col  { border-top:  3px solid #5a3a8a; }
      .mk-sample.mk-both {
        border-left: 3px solid #0d3a0d;
        border-top:  3px solid #5a3a8a;
      }

      .axis-info {
        margin-bottom: 14px; padding: 10px 14px; background: #fff;
        border: 1px solid #c8d3df; border-radius: 4px; font-size: 0.85em;
      }
      .axis-info h2 { margin: 0 0 8px; font-size: 0.95em; color: #1a3550; }
      .axis-info dl { margin: 0; }
      .axis-info dt { font-weight: bold; color: #1a3550; margin-top: 6px; }
      .axis-info dt:first-of-type { margin-top: 0; }
      .axis-info dd { margin: 2px 0 0 18px; color: #444; }

      .dcell { padding: 0; min-width: 120px; background: #fff; }
      .src-tag {
        font-size: 0.65em; color: #888; text-align: right;
        padding: 1px 4px; background: #f0f0f0; border-bottom: 1px solid #ddd;
      }
      .mrow {
        display: flex; justify-content: space-between;
        padding: 2px 6px; border-bottom: 1px solid rgba(0,0,0,0.07);
        font-size: 0.79em;
      }
      .mrow:last-child { border-bottom: none; }
      .ml { color: #555; }
      .mv { font-weight: bold; display: inline-flex; align-items: baseline; gap: 4px; }
      .mv.na { color: #b00; font-style: italic; font-weight: normal; }
      .ci { font-weight: normal; font-size: 0.85em; color: #555; letter-spacing: -0.02em; }
      .metric-note {
        font-size: 0.85em; color: #333; background: #fafafa;
        border: 1px solid #d4d4d4; border-radius: 2px;
        margin-top: 6px; padding: 5px 8px; max-width: 380px;
      }

      .empty {
        background: #f2f2f2; text-align: center; color: #ccc;
        font-size: 1.3em; padding: 18px; min-width: 120px;
      }

      .legend {
        margin-top: 22px; max-width: 640px;
        border: 1px solid #aaa; border-radius: 3px;
        padding: 10px 14px; background: #fff;
      }
      .legend h3 { font-size: 0.88em; margin: 0 0 7px; }
      .legend table { border-collapse: collapse; width: 100%; }
      .legend th, .legend td {
        border: 1px solid #b8b8b8; padding: 5px 8px;
        font-size: 0.78em; vertical-align: middle;
      }
      .legend th {
        background: #ececec; color: #1a3550; text-align: left;
        font-weight: 600; letter-spacing: 0.02em;
      }
      .legend tbody tr:nth-child(odd) td { background: #fbfbfb; }
      .sw {
        display: inline-block; padding: 1px 5px;
        border-radius: 2px; border: 1px solid rgba(0,0,0,0.12);
        font-size: 0.9em;
      }

      .caveat {
        margin-top: 22px; max-width: 760px;
        border: 1px solid #b8b8b8; border-radius: 3px;
        padding: 10px 14px; background: #fafafa; color: #222;
      }
      .caveat h3 { font-size: 0.88em; margin: 0 0 7px; color: #222; }
      .caveat p { font-size: 0.82em; margin: 4px 0; color: #222; line-height: 1.5; }
      .caveat code {
        font-family: 'Courier New', monospace; font-size: 0.95em;
        background: #fff; padding: 0 4px; border-radius: 2px;
      }

      .venn {
        margin-top: 22px; max-width: 1100px;
        border: 1px solid #ddd; border-radius: 3px;
        padding: 10px 14px; background: #fff;
      }
      .venn h3 { font-size: 0.88em; margin: 0 0 7px; }
      .venn p { font-size: 0.78em; margin: 4px 0 8px; color: #555; line-height: 1.4; }
      .venn img { max-width: 100%; height: auto; display: block; }

      .tree {
        margin-top: 22px; max-width: 1400px;
        border: 1px solid #ddd; border-radius: 3px;
        padding: 10px 14px; background: #fff;
      }
      .tree h3 { font-size: 0.88em; margin: 0 0 7px; }
      .tree p { font-size: 0.78em; margin: 4px 0 8px; color: #555; line-height: 1.4; }
      .tree img { max-width: 100%; height: auto; display: block; }

      .benchmark {
        margin-top: 22px; max-width: 1400px;
        border: 1px solid #ddd; border-radius: 3px;
        padding: 10px 14px; background: #fff;
      }
      .benchmark h3 { font-size: 0.88em; margin: 0 0 7px; }
      .benchmark h4 { font-size: 0.82em; margin: 12px 0 5px; color: #444; }
      .benchmark p  { font-size: 0.78em; margin: 4px 0 8px; color: #555; line-height: 1.4; }
      .benchmark ul { font-size: 0.78em; color: #555; line-height: 1.5; }
      .bench-target { margin: 18px 0; padding: 8px 0; border-top: 1px solid #eee; }
      .bench-fig { margin: 6px 0 18px; }
      .bench-fig img { max-width: 100%; height: auto; display: block; }
"""


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------

def build_comparison_html(all_entries, iso_timestamp, uid,
                          venn_count_filename=None,
                          venn_names_filename=None,
                          tree_filename=None,
                          benchmark_html=""):
    """Render the architecture comparison HTML as a string.

    Rows are architectures, columns are data packages, cells carry the
    eleven metrics with per-metric colour scales. The architecture axis
    is on rows because it grows as new model variants land (TabPFN
    versions, AutoTabPFN, Fine-tuned, future additions), while the
    data-package axis stays near-constant (target x ratio x split with
    one target-specific extension for the Park feature set). Putting
    the growing axis on rows keeps the table comfortable to read
    vertically as new variants land.

    Layout (top -> bottom): heatmap table -> best-cell-marker legend
    -> colour legend -> feature-set glossary -> Venn / tree / benchmark
    embeds -> blended Spano caveat (when applicable).
    """
    # Architecture-axis sort: (addition, architecture, version,
    # feature_set, hp_label). HP variants sort after NonHP siblings
    # because hp_label is empty for NonHP cells.
    def arch_key(e):
        p = e["path"]
        hp_label = ""
        if p.get("hp_strategy") and p.get("hp_variant"):
            hp_label = f"{p['hp_strategy']}/{p['hp_variant']}"
        return (
            p["addition"], p["architecture"], p["version"] or "",
            p["feature_set"], hp_label,
        )

    # Data-package-axis sort: (target, datasplit, splittype). Target
    # first so all headache columns sit together, then migraine.
    def data_key(e):
        p = e["path"]
        return (p["target"], p["datasplit"], p["splittype"] or "")

    row_keys = sorted(set(arch_key(e) for e in all_entries))
    col_keys = sorted(set(data_key(e) for e in all_entries))
    lookup   = {(arch_key(e), data_key(e)): e for e in all_entries}

    header_html = _render_column_headers(col_keys)

    # Per-column strict-CI winner sets, computed once before the row
    # loop so each cell can ask "am I the column winner on metric X?"
    # without re-walking the column.
    best_col_set_per_metric: dict[tuple, dict[str, set[tuple]]] = {}
    for ck in col_keys:
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

    rows_html = ""
    for rk in row_keys:
        addition, arch, ver, fs, hp_label = rk
        arch_disp = arch.replace("_", " ")
        ver_disp  = ver.replace("version_", "") if ver else ""
        fs_short  = _FS_LABELS.get(fs, fs)

        # Per-row source dicts for the row-best CI-separation comparison.
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

        best_set_per_metric = _compute_best_sets(
            [(ck, src, is_cv) for ck, src, is_cv in row_entries if src],
            COMPARISON_METRICS,
        )

        # Two row-header cells: the coloured addition chip plus the
        # arch / ver / fs text block.
        parts = [
            f'<div class="rh-line"><span class="rh-key">arch:</span> {arch_disp}</div>',
        ]
        if ver_disp:
            parts.append(f'<div class="rh-line"><span class="rh-key">ver:</span> v{ver_disp}</div>')
        parts.append(f'<div class="rh-line"><span class="rh-key">fs:</span> [{fs_short}]</div>')
        if hp_label:
            parts.append(f'<div class="rh-line"><span class="rh-key">HP:</span> {hp_label}</div>')

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
    fs_glossary_rows = _render_fs_glossary_rows(row_keys, _FS_LABELS, _FS_GLOSS)
    caveat_html      = _render_blended_caveat(row_keys)

    venn_count_html = _venn_count_html(venn_count_filename)
    venn_names_html = _venn_names_html(venn_names_filename)
    tree_html       = _tree_html_section(tree_filename, len(all_entries))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Comparison Table - {iso_timestamp}</title>
<style>{_COMPARISON_CSS}</style>
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
    <dd>All 11 metrics on the locked test set. Each metric has its own colour scale (see <i>Colour legend</i>). The MCC row substitutes "Cal-Optimal" for "Optimal" automatically when the source is CV. A cell tagged <b>H+CV</b> has both hold-out and 5-fold CV results (cell shows hold-out). A cell tagged <b>CV</b> only has CV results - used as fallback when hold-out is missing. Empty (-) means no result file for that combination.</dd>
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
{benchmark_html}
{caveat_html}
</body>
</html>"""
