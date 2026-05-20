"""Feature-set Venn diagrams for the comparison report.

Two PNG variants:

- :func:`generate_count_venn_png`: 3-set Venn with region counts split
  into engineered vs original SHD columns.
- :func:`generate_names_venn_png`: same regions but every feature name
  is rendered in-place, colour-coded by origin.

The Park (2016) [Tab. 4] stepwise-selected subset is shown as a sidebar
rather than a 4th circle because it is a near-strict subset of ``full``
(5 of its 6 features are columns of ``full``) plus one derived feature
``hormonal_changes_today`` (= menstruation_today OR ovulation_today)
that is constructed inside the filter to match Park et al.'s single
"hormonal changes" trigger. Forcing a 4-set Venn would flatten this
almost-subset relationship visually; the sidebar is more informative.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn.layout.venn3 import cost_based

from _figstyle import apply_journal_style, save_journal_figure


# Columns produced by data/pipeline/engineer.py via aggregation, lag,
# rolling windows, state-change detection, or cross-feature interactions.
# Everything else in the parquet is treated as an original SHD column
# (or a 1:1 rename, e.g. ``stress`` -> ``stress_today``).
#
# These are determined by inspection of engineer.py, not derivable from
# the parquet alone. Update both together.
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
    # Target column: derived (sign-flipped from headache_free, or merged
    # from disability sheet in migraine mode) - not present verbatim in
    # raw input.
    'migraine_today',
}

# Plot palette - kept centralised so both Venn variants stay visually
# consistent and the HTML legend colours match the figure.
VENN_COLORS = {
    'full':       '#2563eb',   # blue
    'spano':      '#dc2626',   # red
    'no_rolling': '#059669',   # green
    'park':       '#a16207',   # amber - Park (2016) stepwise-selected
}
# Feature-origin colours form a deliberately separate system from the
# set hues above: a neutral grey for raw columns and one warm accent for
# engineered ones. Neither shares a hue with the set circles (blue / red
# / green / amber) nor with their pairwise blends (purple, cyan, olive),
# so a reader never confuses "which set" with "which origin".
CATEGORY_COLORS = {
    'engineered': '#ea580c',   # orange  - derived / engineered features
    'original':   '#475569',   # slate grey - raw SHD column or 1:1 rename
}


def compute_feature_sets(experiment_dir: Path):
    """Load a representative parquet and compute the four feature sets.

    Returns ``{full, spano, no_rolling, park}`` of column-name sets, or
    None if no parquet is found (e.g. the data pipeline has not been
    run yet). The Park set includes one derived feature
    (``hormonal_changes_today``) that is NOT a column in the engineered
    parquet itself - the Park filter computes it from
    ``menstruation_today OR ovulation_today`` to match Park et al.'s
    single "hormonal changes" trigger.
    """
    NON_FEATURE = {
        "entry_id", "patient_id", "date",
        "migraine_target", "cv_fold", "migraine_today",
    }
    candidates = [
        experiment_dir.parent / "data" / "processed" / "headache" / "70_15_15" / "chrono" / "diary_train.parquet",
        experiment_dir.parent / "data" / "processed" / "migraine"  / "70_15_15" / "chrono" / "diary_train.parquet",
    ]
    parquet = next((p for p in candidates if p.exists()), None)
    if parquet is None:
        return None

    # The filter modules use absolute imports from ``_dataRead.*``; put
    # the *parent* (experiment/) on sys.path so those imports resolve.
    sys.path.insert(0, str(experiment_dir))
    import pandas as pd
    from _dataRead.filter_to_spano_features import select_spano_features
    from _dataRead.filter_to_no_rolling_features import select_non_rolling_features
    from _dataRead.filter_to_park_features import select_park_features

    full       = set(pd.read_parquet(parquet).columns)                  - NON_FEATURE
    spano      = set(select_spano_features(str(parquet)).columns)       - NON_FEATURE
    no_rolling = set(select_non_rolling_features(str(parquet)).columns) - NON_FEATURE
    park       = set(select_park_features(str(parquet)).columns)        - NON_FEATURE
    return {"full": full, "spano": spano, "no_rolling": no_rolling, "park": park}


def _split_by_category(features):
    """Return (engineered_count, original_count) for a feature set."""
    eng = sum(1 for f in features if f in ENGINEERED_FEATURES)
    return eng, len(features) - eng


def _build_venn_regions(full, spano, no_rolling):
    """The seven non-empty regions of a three-set Venn, keyed by the
    matplotlib_venn region ID convention (``'100'`` = only-A, etc.).
    """
    return {
        '100': full - spano - no_rolling,
        '010': spano - full - no_rolling,
        '001': no_rolling - full - spano,
        '110': (full & spano) - no_rolling,
        '101': (full & no_rolling) - spano,
        '011': (spano & no_rolling) - full,
        '111': full & spano & no_rolling,
    }


# Auto-placement breaks for nested subsets (spano subset of full,
# no_rolling subset of full); the default solver emits "Bad circle
# positioning" and sometimes hides labels. Pin to fixed positions.
_SET_LABEL_POSITIONS = {
    'full':       (-0.85,  0.65),
    'spano':       (0.85,  0.65),
    'no_rolling':  (0.00, -0.85),
}
_SET_LABEL_POSITIONS_NAMES_VARIANT = {
    'full':       (-0.85,  0.70),
    'spano':       (0.85,  0.70),
    'no_rolling':  (0.00, -0.55),
}


def _pin_set_labels(venn, positions, fontsize, color_lookup):
    """Move the three set labels to known-good positions, recolour, and
    bold them so they match the circle they describe.
    """
    for sid, color_key in zip(('A', 'B', 'C'), ('full', 'spano', 'no_rolling')):
        s_lbl = venn.get_label_by_id(sid)
        if s_lbl is None:
            continue
        s_lbl.set_position(positions[color_key])
        s_lbl.set_horizontalalignment('center')
        s_lbl.set_fontsize(fontsize)
        s_lbl.set_fontweight('bold')
        s_lbl.set_color(color_lookup[color_key])


def _outline_circles(ax, subsets, layout):
    """Draw each set's circle as a coloured outline matching its label.

    The translucent fills blend in the overlaps (and no_rolling, a near
    subset of full, reads as cyan rather than its own green); a solid
    set-coloured ring on every circle keeps each set's hue tied to its
    boundary so circle, ring and label always agree.
    """
    circles = venn3_circles(subsets, ax=ax, layout_algorithm=layout,
                            linewidth=2.2)
    colours = (VENN_COLORS['full'], VENN_COLORS['spano'],
               VENN_COLORS['no_rolling'])
    for circle, colour in zip(circles, colours):
        if circle is not None:
            circle.set_edgecolor(colour)
            circle.set_alpha(0.95)
    return circles


def _park_sidebar(ax, park, full, lines_factory):
    """Render the Park (2016) subset as a sidebar callout in the
    lower-right of the axes. ``lines_factory(park_in_full, park_only)``
    returns the formatted text lines to display.
    """
    if not park:
        return
    park_in_full = sorted(park & full)
    park_only    = sorted(park - full)
    lines = lines_factory(park_in_full, park_only)
    ax.text(
        0.98, 0.02, "\n".join(lines),
        transform=ax.transAxes, fontsize=8.5,
        color=VENN_COLORS['park'],
        verticalalignment='bottom', horizontalalignment='right',
        fontfamily='monospace',
        bbox=dict(
            facecolor='white', edgecolor=VENN_COLORS['park'],
            boxstyle='round,pad=0.5', linewidth=1.2,
        ),
    )


def generate_count_venn_png(feature_sets, out_path):
    """Render a 3-set Venn showing region counts, split by engineered
    vs original SHD columns.
    """
    apply_journal_style()
    full       = feature_sets["full"]
    spano      = feature_sets["spano"]
    no_rolling = feature_sets["no_rolling"]

    apply_journal_style()
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    layout = cost_based.LayoutAlgorithm()
    v = venn3(
        [full, spano, no_rolling],
        set_labels=(
            f'full ({len(full)})',
            f'spano ({len(spano)})',
            f'no_rolling ({len(no_rolling)})',
        ),
        set_colors=(
            VENN_COLORS['full'], VENN_COLORS['spano'], VENN_COLORS['no_rolling'],
        ),
        alpha=0.32,
        ax=ax,
        # spano subset full superset no_rolling is a near-subset
        # configuration; the default pairwise solver cannot satisfy the
        # implied triangle inequality and emits "Bad circle positioning".
        # The cost-based optimizer minimises log-area error across all 7
        # regions and handles this case cleanly.
        layout_algorithm=layout,
    )
    # Coloured circle outlines anchor each set's hue to its boundary, so
    # the green no_rolling ring matches its green label even where the
    # fill blends to cyan inside the full circle it is nearly a subset of.
    _outline_circles(ax, [full, spano, no_rolling], layout)

    regions = _build_venn_regions(full, spano, no_rolling)
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

    _pin_set_labels(v, _SET_LABEL_POSITIONS, 13, VENN_COLORS)

    ax.set_title(
        "Feature-set inclusion - region counts (engineered + original SHD columns)",
        fontsize=12, pad=14,
    )

    # Legend explaining 'eng' / 'orig'. Anchored top-left so it never
    # collides with the wider Park sidebar pinned to the bottom-right.
    ax.text(
        0.01, 0.99,
        "eng = engineered (rolling / lag / interaction / state-derived)\n"
        "orig = original SHD column or 1:1 rename",
        transform=ax.transAxes, fontsize=8.5, color='#444',
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='#bbb', boxstyle='round,pad=0.4'),
    )

    def _count_lines(park_in_full, park_only):
        lines = [
            f"Park (2016) stepwise-selected: {len(park_in_full) + len(park_only)} features",
            f"  {len(park_in_full)} subset full:  " + ", ".join(park_in_full),
        ]
        if park_only:
            lines.append(
                f"  {len(park_only)} derived (not in full): " + ", ".join(park_only)
            )
        return lines
    _park_sidebar(ax, feature_sets.get("park", set()), full, _count_lines)

    plt.tight_layout()
    save_journal_figure(fig, out_path)
    plt.close(fig)


# Region display titles and the order they appear in the name panel
# (most-populated / most-relevant regions first).
_REGION_TITLES = {
    '111': 'full ∩ spano ∩ no_rolling',
    '100': 'full only',
    '110': 'full ∩ spano (not no_rolling)',
    '101': 'full ∩ no_rolling (not spano)',
    '010': 'spano only',
    '001': 'no_rolling only',
    '011': 'spano ∩ no_rolling (not full)',
}
_REGION_ORDER = ('111', '100', '110', '101', '010', '001', '011')


def _render_name_panel(ax, regions, park=None):
    """Lay out each non-empty region's feature names as a column-flowed
    list of individually colour-coded, non-overlapping text entries.

    Region blocks flow top-to-bottom down a column; when a column fills,
    the next block continues in the next column. Every name is its own
    ``ax.text`` so engineered (purple bold) and original (green) entries
    are distinguishable and never overlap.
    """
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    n_cols = 2
    col_w = 1.0 / n_cols
    line_h = 0.023
    top = 0.99
    max_rows = int((top - 0.02) / line_h)

    state = {'col': 0, 'row': 0}

    def place(text, *, indent, color, weight, size):
        # Move to next column if the current one is full.
        if state['row'] >= max_rows:
            state['row'] = 0
            state['col'] += 1
        if state['col'] >= n_cols:
            return  # out of space (should not happen at this feature count)
        x = state['col'] * col_w + indent
        y = top - state['row'] * line_h
        ax.text(x, y, text, color=color, fontweight=weight, fontsize=size,
                ha='left', va='top', fontfamily='monospace',
                transform=ax.transAxes)
        state['row'] += 1

    def blank():
        state['row'] += 1

    blocks = [(rid, regions.get(rid)) for rid in _REGION_ORDER if regions.get(rid)]
    if park:
        blocks.append(('park', park))

    for rid, feats in blocks:
        title = (_REGION_TITLES.get(rid)
                 if rid != 'park' else 'Park (2016) [Tab. 4] subset')
        # Keep a block together: if it would split awkwardly near the
        # bottom, push it to the next column first.
        needed = 1 + len(feats)
        if state['row'] + needed > max_rows and state['row'] > 0:
            state['row'] = 0
            state['col'] += 1
        place(f"{title}  ({len(feats)})", indent=0.005,
              color='#111', weight='bold', size=9.5)
        for f in sorted(feats):
            is_eng = f in ENGINEERED_FEATURES
            place(
                f, indent=0.03,
                color=(CATEGORY_COLORS['engineered'] if is_eng
                       else CATEGORY_COLORS['original']),
                weight='bold' if is_eng else 'normal', size=8.0,
            )
        blank()


def generate_names_venn_png(feature_sets, out_path):
    """Two-panel feature-name figure: a 3-set Venn (with region counts)
    on the left for set-structure context, and a column-flowed,
    colour-coded list of every feature name per region on the right.

    The earlier single-panel version stacked all names at the region
    centroids, which overlapped badly at this feature count; splitting
    the names into a dedicated panel makes every name individually
    legible and non-overlapping.
    """
    apply_journal_style()
    full       = feature_sets["full"]
    spano      = feature_sets["spano"]
    no_rolling = feature_sets["no_rolling"]
    regions    = _build_venn_regions(full, spano, no_rolling)

    apply_journal_style()
    fig, (ax_venn, ax_list) = plt.subplots(
        1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.0, 1.05]},
    )

    # --- left: Venn with region counts (structure context) ---
    layout = cost_based.LayoutAlgorithm()
    v = venn3(
        [full, spano, no_rolling],
        set_labels=(
            f'full  ({len(full)})',
            f'spano  ({len(spano)})',
            f'no_rolling  ({len(no_rolling)})',
        ),
        set_colors=(
            VENN_COLORS['full'], VENN_COLORS['spano'], VENN_COLORS['no_rolling'],
        ),
        alpha=0.28,
        ax=ax_venn,
        layout_algorithm=layout,
    )
    _outline_circles(ax_venn, [full, spano, no_rolling], layout)
    for rid, feats in regions.items():
        lbl = v.get_label_by_id(rid)
        if lbl is None:
            continue
        lbl.set_text(str(len(feats)) if feats else '')
        lbl.set_fontsize(13)
        lbl.set_fontweight('bold')
    _pin_set_labels(v, _SET_LABEL_POSITIONS_NAMES_VARIANT, 14, VENN_COLORS)
    ax_venn.set_title("Feature-set structure (region counts)",
                      fontsize=12, pad=10)

    # --- right: colour-coded name lists per region ---
    _render_name_panel(ax_list, regions, park=feature_sets.get("park"))
    ax_list.set_title("All feature names by region (colour = origin)",
                      fontsize=12, pad=10)

    # Origin colour key for the name panel, placed underneath it (not in
    # the top margin where it crowded the title). Centred under the right
    # panel, the two halves meeting at the split point.
    fig.text(0.70, 0.035, "● original SHD column or 1:1 rename",
             ha='right', fontsize=9.5, color=CATEGORY_COLORS['original'],
             fontfamily='monospace')
    fig.text(0.71, 0.035, "● engineered: rolling / lag / interaction / state-derived",
             ha='left', fontsize=9.5, color=CATEGORY_COLORS['engineered'],
             fontweight='bold', fontfamily='monospace')

    fig.suptitle("Feature-set inclusion - every feature name, colour-coded by origin",
                 fontsize=14, y=0.94)
    plt.tight_layout(rect=[0, 0.06, 1, 0.91])
    save_journal_figure(fig, out_path)
    plt.close(fig)
