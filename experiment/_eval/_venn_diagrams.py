"""Feature-set inclusion diagrams for the comparison report.

The three engineered sets nest: ``spano`` and ``no_rolling`` are strict
subsets of ``full`` (each is ``full`` with columns removed), so a
three-circle Venn would be the wrong chart type - it always draws three
mutually-overlapping circles and would show ``spano`` / ``no_rolling``
bulging outside ``full`` into regions that are empty. These figures use a
**nested Euler** layout instead: ``full`` is the outer container and the
two subsets are drawn wholly inside it, overlapping each other. The layout
is schematic (circle areas are not to scale); the exact sizes live in the
region-count labels.

Two PNG variants:

- :func:`generate_count_venn_png`: region counts split into engineered vs
  original SHD columns.
- :func:`generate_names_venn_png`: the same structure with every feature
  name listed in a side panel, colour-coded by origin.

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
from matplotlib.patches import Circle

from _style import apply, save, panel_label, FEATURE_SET, OI, GREY, INK, SOFT, FAINT


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
# Canonical feature-set hues (single source = _style.FEATURE_SET), so the Venn
# matches every other figure: full=blue, spano=vermillion, no_rolling=green,
# park=orange (Okabe-Ito; the vermillion+green pair is CVD-safe, unlike red+green).
VENN_COLORS = dict(FEATURE_SET)
# Feature-origin colours form a deliberately separate system from the
# set hues above: a neutral grey for raw columns and one warm accent for
# engineered ones. Neither shares a hue with the set circles (blue / red
# / green / amber) nor with their pairwise blends (purple, cyan, olive),
# so a reader never confuses "which set" with "which origin".
CATEGORY_COLORS = {
    'engineered': OI["purple"],  # purple - distinct from the set hues (incl. park orange)
    'original':   GREY,          # neutral grey - raw SHD column or 1:1 rename
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
    """The seven regions of three sets (full=A, spano=B, no_rolling=C),
    keyed by a binary membership ID: ``'100'`` = in-A-only, ``'110'`` =
    in-A-and-B-not-C, ``'111'`` = in all three, etc. With these nested sets
    only '100', '110', '101', '111' are non-empty.
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


# Schematic nested-Euler geometry. spano and no_rolling are strict subsets of
# full (verified: the spano-only, no_rolling-only, and spano-no_rolling-outside-
# full regions are all empty), so both are drawn as circles wholly inside the
# full circle and overlapping each other. Circle AREAS are schematic, not to
# scale - the region labels carry the exact counts; the geometry carries only
# the (correct) containment + mutual-overlap topology. Positions are tuned so
# each region label lands unambiguously inside its own region.
_EULER = {
    'full':       (0.00, 0.00, 1.00),   # (cx, cy, r) - outer container, outline only
    'spano':      (-0.27, 0.05, 0.60),  # inner subset, left
    'no_rolling': (0.27, 0.05, 0.56),   # inner subset, right
}
_EULER_REGION_XY = {
    '100': (0.00, 0.78),    # full only (top annulus)
    '110': (-0.52, 0.05),   # spano, not no_rolling (left lobe)
    '101': (0.52, 0.05),    # no_rolling, not spano (right lobe)
    '111': (0.00, 0.05),    # all three (centre lens)
}
_EULER_SET_LABEL = {
    'full':       (0.00, 1.12, 'center', 'bottom'),
    'spano':      (-1.08, 0.05, 'right', 'center'),
    'no_rolling': (1.08, 0.05, 'left', 'center'),
}


def _draw_nested_euler(ax, regions, set_sizes, region_label_fn):
    """Draw the schematic nested-Euler diagram (full outer; spano and
    no_rolling inner, overlapping) and label the four non-empty regions.

    ``region_label_fn(region_id, features)`` returns the text for each region
    (count only, or count plus engineered/original split); an empty string
    suppresses the label. ``full`` is drawn as an outline only so its
    exclusive annulus stays clean white, while the two subsets are filled so
    their hues - and their overlap - read directly.
    """
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.25, 1.30)
    ax.set_aspect('equal')
    ax.axis('off')

    fcx, fcy, fr = _EULER['full']
    ax.add_patch(Circle((fcx, fcy), fr, facecolor='none',
                        edgecolor=VENN_COLORS['full'], linewidth=2.4, zorder=2))
    for key in ('spano', 'no_rolling'):
        cx, cy, r = _EULER[key]
        ax.add_patch(Circle((cx, cy), r, facecolor=VENN_COLORS[key], alpha=0.30,
                            edgecolor=VENN_COLORS[key], linewidth=2.2, zorder=1))

    for rid, (x, y) in _EULER_REGION_XY.items():
        txt = region_label_fn(rid, regions.get(rid) or set())
        if txt:
            ax.text(x, y, txt, ha='center', va='center', fontsize=10,
                    fontweight='bold', color=INK, zorder=4)

    for key, (x, y, ha, va) in _EULER_SET_LABEL.items():
        ax.text(x, y, f"{key} ({set_sizes[key]})", ha=ha, va=va,
                fontsize=13, fontweight='bold', color=VENN_COLORS[key], zorder=4)


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
        color=INK,
        verticalalignment='bottom', horizontalalignment='right',
        multialignment='right',
        fontfamily='monospace',
        bbox=dict(
            facecolor='white', edgecolor=SOFT,
            boxstyle='round,pad=0.5', linewidth=1.2,
        ),
    )


def generate_count_venn_png(feature_sets, out_path):
    """Render the nested-Euler diagram with region counts split by
    engineered vs original SHD columns.
    """
    apply()
    full       = feature_sets["full"]
    spano      = feature_sets["spano"]
    no_rolling = feature_sets["no_rolling"]
    regions    = _build_venn_regions(full, spano, no_rolling)
    sizes      = {'full': len(full), 'spano': len(spano), 'no_rolling': len(no_rolling)}

    fig, ax = plt.subplots(figsize=(9.5, 9.0))

    def _count_label(_rid, feats):
        if not feats:
            return ''
        eng, orig = _split_by_category(feats)
        return f"{len(feats)}\n({eng} eng + {orig} orig)"
    _draw_nested_euler(ax, regions, sizes, _count_label)

    ax.set_title(
        "Feature-set inclusion - nested Euler "
        "(spano, no_rolling subset full; schematic, areas not to scale)",
        fontsize=12, pad=14,
    )

    # Legend explaining 'eng' / 'orig'. Anchored top-left so it never
    # collides with the wider Park sidebar pinned to the bottom-right.
    ax.text(
        0.01, 0.99,
        "eng = engineered (rolling / lag / interaction / state-derived)\n"
        "orig = original SHD column or 1:1 rename",
        transform=ax.transAxes, fontsize=8.5, color=SOFT,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor=FAINT, boxstyle='round,pad=0.4'),
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
    save(fig, out_path)
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
              color=INK, weight='bold', size=9.5)
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
    """Two-panel feature-name figure: the nested-Euler diagram (with region
    counts) on the left for set-structure context, and a column-flowed,
    colour-coded list of every feature name per region on the right.

    The names live in a dedicated panel rather than at the region centroids
    because at this feature count centroid-stacked names overlap badly; the
    side panel keeps every name individually legible.
    """
    apply()
    full       = feature_sets["full"]
    spano      = feature_sets["spano"]
    no_rolling = feature_sets["no_rolling"]
    regions    = _build_venn_regions(full, spano, no_rolling)
    sizes      = {'full': len(full), 'spano': len(spano), 'no_rolling': len(no_rolling)}

    fig, (ax_venn, ax_list) = plt.subplots(
        1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.0, 1.05]},
    )

    # --- left: nested Euler with region counts (structure context) ---
    _draw_nested_euler(ax_venn, regions, sizes,
                       lambda _rid, feats: str(len(feats)) if feats else '')
    ax_venn.set_title("Feature-set structure - nested Euler (schematic)",
                      fontsize=12, pad=10)
    panel_label(ax_venn, "a")

    # --- right: colour-coded name lists per region ---
    _render_name_panel(ax_list, regions, park=feature_sets.get("park"))
    ax_list.set_title("All feature names by region (colour = origin)",
                      fontsize=12, pad=10)
    panel_label(ax_list, "b")

    # Origin colour key for the name panel, placed underneath it (not in
    # the top margin where it crowded the title). Centred under the right
    # panel, the two halves meeting at the split point.
    fig.text(0.70, 0.16, "● original SHD column or 1:1 rename",
             ha='right', fontsize=9.5, color=CATEGORY_COLORS['original'],
             fontfamily='monospace')
    fig.text(0.71, 0.16, "● engineered: rolling / lag / interaction / state-derived",
             ha='left', fontsize=9.5, color=CATEGORY_COLORS['engineered'],
             fontweight='bold', fontfamily='monospace')

    fig.suptitle("Feature-set inclusion - every feature name, colour-coded by origin",
                 fontsize=14, y=0.94)
    plt.tight_layout(rect=[0, 0.06, 1, 0.91])
    save(fig, out_path)
    plt.close(fig)
