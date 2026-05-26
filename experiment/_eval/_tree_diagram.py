"""Leaf-hierarchy visualisation for the comparison report.

Renders the discovered ``results/`` directories as an **icicle plot**
(Shneiderman's partition layout [1]): one horizontal stripe per path
level, with each node drawn as a rectangle whose width is proportional
to its descendant-leaf count and whose horizontal position aligns with
its parent's column. Children sit directly under their parent so the
hierarchy reads top-to-bottom along the column.

Why icicle over the prior horizontal phylogeny:
  - Andrews & Sanguinetti 2019 [2] (IEEE InfoVis user study): icicle
    plots outperform treemaps, sunburst, and node-link diagrams for
    hierarchies with many leaves on both navigation speed and
    hierarchy-comprehension tasks.
  - Munzner 2014 [3]: position-along-axis is the most accurate visual
    channel; icicle preserves position-coding at every level (treemap
    mixes position with area, sunburst uses angular position).
  - Z-pattern reading: icicle aligns with how readers naturally scan
    (left-to-right, top-to-bottom).
  - Fixed height: an n-level icicle stays approximately ``n * row_h``
    tall regardless of the leaf count, where the horizontal-phylogeny
    layout grew linearly with leaves and quickly became unreadable.

References
----------
[1] B. Shneiderman, "Tree visualization with tree-maps: 2D
    space-filling approach," ACM TOG, vol. 11, no. 1, pp. 92-99, 1992.
[2] K. Andrews and B. Sanguinetti, "Interactive Visualisation of
    Hierarchical Quantitative Data: An Evaluation," in IEEE Conf. on
    Information Visualisation, 2019.
[3] T. Munzner, *Visualization Analysis and Design*, CRC Press, 2014.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import apply, save, INK, FAINT, TARGET, ARCH, SPLIT, FEATURE_SET


# Path-segment levels in the order they appear under experiment/.
# Each tuple is (parse_path key, full-word column header). The full ten
# levels are listed; ``hp_strategy`` and ``hp_variant`` only fire for
# HyperparameterTuned cells (NonHP cells stop at the splittype level
# and skip the three HP-related rows).
TREE_LEVELS = (
    ('addition',       'Addition'),
    ('target',         'Target'),
    ('feature_set',    'Feature set'),
    ('architecture',   'Architecture'),
    ('version',        'Model version'),
    ('datasplit',      'Data split ratio'),
    ('splittype',      'Split strategy'),
    ('hyperparameter', 'HP-tuned?'),
    ('hp_strategy',    'HP search strategy'),
    ('hp_variant',     'HP operating point'),
)

# Cell colour carries ENTITY IDENTITY, not depth. Depth is already encoded by row
# position (top -> bottom), so a depth colour ramp would be redundant; spending the
# colour channel on the canonical entity hues instead tells the reader which target /
# feature set / model / split a cell is, consistent with every other figure. Levels
# without a canonical entity (addition index, model version, split ratio, the three
# HP rows) stay a neutral grey so the branded rows stand out.
_LEVEL_PALETTE = {
    1: TARGET,        # headache / migraine
    2: FEATURE_SET,   # full / spano / no_rolling / park (dirs carry a _features suffix)
    3: ARCH,          # model family hue
    6: SPLIT,         # chrono / stratified / patient / site
}
_STRUCTURAL_FILL = FAINT


def _node_fill(level, label):
    """Canonical entity colour for a node's level, else the neutral fill."""
    palette = _LEVEL_PALETTE.get(level)
    if palette is None:
        return _STRUCTURAL_FILL
    key = label.replace('_features', '') if level == 2 else label
    return palette.get(key, _STRUCTURAL_FILL)


def _label_ink(hex_fill):
    """White on a dark fill, INK on a light one, by relative luminance, so cell
    labels stay legible across the dark-to-light span of the Okabe-Ito palette.
    """
    h = hex_fill.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if luminance > 0.6 else 'white'


class _Node:
    __slots__ = ('label', 'level', 'children',
                 'leaf_count', 'x0', 'x1')

    def __init__(self, label, level):
        self.label = label
        self.level = level
        self.children = {}
        self.leaf_count = 0
        self.x0 = 0.0
        self.x1 = 0.0


def _build_tree(all_entries):
    """Construct the leaf-hierarchy tree. Children are stored in an
    insertion-order dict; the layout pass sorts them by key so the
    rendered icicle is deterministic across runs.
    """
    root = _Node('experiment/', -1)
    for entry in all_entries:
        p = entry['path']
        node = root
        for lvl, (key, _hdr) in enumerate(TREE_LEVELS):
            val = p.get(key)
            if val is None:
                continue
            child_key = (lvl, val)
            if child_key not in node.children:
                node.children[child_key] = _Node(str(val), lvl)
            node = node.children[child_key]
    return root


def _compute_leaf_counts(node):
    """Post-order pass: each node's ``leaf_count`` = sum of children's
    leaf counts (1 if leaf). Returns the count for convenience.
    """
    if not node.children:
        node.leaf_count = 1
        return 1
    total = 0
    for child in node.children.values():
        total += _compute_leaf_counts(child)
    node.leaf_count = total
    return total


def _assign_x_extents(node, x0, x1):
    """Pre-order pass: divide the parent's [x0, x1] interval among the
    children proportionally to their leaf counts (sorted by node key).
    Children sit horizontally adjacent under the parent's footprint,
    which preserves the parent-child column alignment that lets a
    reader trace a path down the icicle.
    """
    node.x0 = x0
    node.x1 = x1
    if not node.children:
        return
    width = x1 - x0
    total = node.leaf_count or 1
    cursor = x0
    for k in sorted(node.children):
        child = node.children[k]
        child_w = width * (child.leaf_count / total)
        _assign_x_extents(child, cursor, cursor + child_w)
        cursor += child_w


def _label_for_width(label, width_units, min_per_char=0.012):
    """Truncate ``label`` to fit a rectangle of width ``width_units``
    (axes coordinates, where the icicle spans [0, 1]). Returns the
    empty string when the rectangle is too narrow for any label.

    ``min_per_char`` is calibrated to the chosen fontsize (8.5pt at a
    ~14-inch-wide canvas). At narrower widths the label is truncated
    with an ellipsis; below ~3 characters the label is dropped entirely
    so the cell stays clean rather than carrying a meaningless stub.
    """
    if width_units <= 0:
        return ""
    max_chars = max(1, int(width_units / min_per_char))
    if max_chars < 3:
        return ""
    if len(label) <= max_chars:
        return label
    if max_chars < 5:
        return label[:max_chars]
    return label[:max_chars - 1] + "…"


# Compact display names for verbose path segments, so more cells show a
# full (un-truncated) label. Keys are the on-disk directory names.
_SHORT_LABELS = {
    'stacked_2xgb_meta_lr': 'stacked_2xgb',
    'blended_xgb_lr_spano2026': 'blended_xgb_lr',
    'full_features': 'full',
    'no_rolling_features': 'no_rolling',
    'spano_features': 'spano',
    'park_features': 'park',
    'HyperparameterTuned': 'HP-tuned',
}


def _short_node_label(label):
    """Map a verbose path-segment to its compact display form, and trim
    the redundant ``version_`` prefix so version cells read 'v2-6' etc.
    """
    if label in _SHORT_LABELS:
        return _SHORT_LABELS[label]
    if label.startswith('version_'):
        return label.replace('version_', 'v')
    return label


def _draw_node_rects(ax, node, row_h, row_pad):
    """Recursively render each node as a coloured rectangle within its
    row stripe. ``row_h`` is the per-level stripe height in data units;
    ``row_pad`` is the vertical gap between stripes for visual
    separation.
    """
    if node.level >= 0:
        color = _node_fill(node.level, node.label)
        y_top = node.level * row_h
        y_bot = y_top + row_h - row_pad
        width = node.x1 - node.x0
        # The rectangle itself.
        ax.add_patch(Rectangle(
            (node.x0, y_top), width, row_h - row_pad,
            facecolor=color, edgecolor='white', linewidth=0.6,
        ))
        # Centred label, ink chosen for legibility on this cell's fill;
        # compacted then truncated to fit the cell width.
        label = _label_for_width(_short_node_label(node.label), width)
        if label:
            ax.text(
                (node.x0 + node.x1) / 2.0, (y_top + y_bot) / 2.0,
                label, ha='center', va='center',
                color=_label_ink(color), fontsize=8.5, fontweight='bold',
                fontfamily='monospace',
            )
    for child in node.children.values():
        _draw_node_rects(ax, child, row_h, row_pad)


def generate_tree_png(all_entries, out_path):
    """Render the discovered-leaves hierarchy as an icicle plot.

    Layout: each path level is a horizontal stripe; nodes within a
    stripe are coloured rectangles whose width is proportional to the
    number of descendant leaves and whose horizontal position aligns
    with their parent's column. Labels render in white at the centre
    of each rectangle and truncate with an ellipsis when the cell is
    too narrow.

    The plot height stays approximately constant regardless of the
    leaf count: ``n_levels * row_h`` plus a thin header band for the
    column titles. The width is wide enough that the average leaf at
    the bottom row gets enough pixels to be discernible (cell counts
    vs widths are summarised in the title).
    """
    apply()
    root = _build_tree(all_entries)
    if not root.children:
        return
    _compute_leaf_counts(root)
    _assign_x_extents(root, x0=0.0, x1=1.0)

    n_levels   = len(TREE_LEVELS)
    n_leaves   = root.leaf_count
    row_h      = 1.0           # data-unit height per level stripe
    row_pad    = 0.10          # vertical gap inside each stripe
    header_h   = 0.6           # band above row 0 for level-name labels
    total_h_in = max(3.5, n_levels * 0.55 + 0.8)
    fig_w_in   = 14.0          # matches the comparison HTML's max width
                               # so the icicle fits without horizontal scroll
    fig, ax = plt.subplots(figsize=(fig_w_in, total_h_in))

    _draw_node_rects(ax, root, row_h=row_h, row_pad=row_pad)

    # Stripe labels on the left margin (one per row).
    for lvl, (_key, header) in enumerate(TREE_LEVELS):
        y_mid = lvl * row_h + (row_h - row_pad) / 2.0
        ax.text(
            -0.005, y_mid, header,
            ha='right', va='center', fontsize=9,
            fontweight='bold', fontfamily='monospace',
            color=INK,   # monotone ramp -> dark labels for legibility (not per-level hue)
        )

    ax.set_xlim(-0.12, 1.005)
    ax.set_ylim(n_levels * row_h, -header_h)   # y inverted (top -> bottom)
    ax.axis('off')
    ax.set_title(
        f"Discovered experiment leaves ({n_leaves} total) - icicle plot "
        "[Shneiderman 1992, Andrews 2019]: row = path level, cell width "
        "is proportional to descendant-leaf count, children align under parent",
        fontsize=10, pad=10,
    )

    plt.tight_layout()
    save(fig, out_path)
    plt.close(fig)
