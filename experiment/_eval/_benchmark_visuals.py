"""Cross-architecture, cross-cell benchmark visualisations.

Generates supplementary figures alongside the standard heatmap output
of run_aggregate_results.py:

1. Critical Difference (CD) diagram - Demsar 2006 [JMLR] [1].
   Friedman omnibus + Nemenyi post-hoc on per-cell ranks. Architectures
   joined by a horizontal bar at the top of the diagram are not
   significantly different at alpha=0.05.

2. Dolan-More performance profile - Dolan and More 2002 [Math. Prog.] [2].
   Cumulative fraction of cells where each architecture achieves a metric
   within factor tau of the best architecture in that cell. Top-left
   architectures are robust across cells; bottom-right architectures
   occasionally win big but lose often.

3. Rank slopegraph - Tufte 2001.
   Per-architecture rank line across (ratio, split) conditions, one line
   per architecture. Reveals cells where an architecture's rank
   collapses; stable architectures are flat lines, unstable ones
   crisscross.

All three consume one (cell x architecture) matrix per metric. Rows
with any missing architecture value are dropped to preserve Friedman's
matched-design assumption.

References
----------
[1] J. Demsar, "Statistical Comparisons of Classifiers over Multiple
    Data Sets," J. Machine Learning Research, vol. 7, pp. 1-30, 2006.
[2] E. D. Dolan and J. J. More, "Benchmarking Optimization Software
    with Performance Profiles," Math. Prog., vol. 91, pp. 201-213, 2002.
    doi: 10.1007/s101070100263.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare

from _figstyle import apply_journal_style, save_journal_figure


# Nemenyi q-alpha critical values for alpha=0.05, from Demsar 2006
# Table 5. Indexed by k (number of compared classifiers). The full
# Nemenyi post-hoc critical difference is CD = q_alpha * sqrt(k(k+1)/6N)
# where N is the number of data sets (cells here).
_NEMENYI_Q_ALPHA_05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
    20: 3.544,
}


def _parse_mean(val_str):
    """Extract the leading float from a 'mean [lo - hi]' string."""
    if val_str is None:
        return None
    try:
        return float(str(val_str).split()[0])
    except (ValueError, IndexError):
        return None


def build_metric_matrix(all_entries, metric_key, target,
                        restrict_feature_sets=None,
                        cv_alias=None):
    """Build a (cell x architecture) matrix for one metric and target.

    Cells are (feature_set, ratio, split). Architectures are
    (architecture, version, hp_strategy, hp_variant) so HP variants,
    NonHP, AND model versions (e.g. tabpfn v2-6 vs v3-default) appear
    as separate competitors. Including ``version`` in the key is
    critical: TabPFN has six evaluated versions sharing one
    ``architecture`` name, and without the version field every TabPFN
    cell would collapse onto one row with whichever version happened
    to be processed last winning - a silent data-quality bug that
    makes Friedman/Nemenyi output meaningless.

    Source preference is holdout > cv; when reading from CV,
    ``cv_alias`` is consulted (e.g. MCC is keyed "MCC (Cal-Optimal)"
    in CV result files).

    Returns (matrix, cell_labels, arch_labels). ``matrix`` has NaN for
    missing values.
    """
    rows = {}
    arch_set = set()
    for e in all_entries:
        p = e["path"]
        if p["target"] != target:
            continue
        if restrict_feature_sets and p["feature_set"] not in restrict_feature_sets:
            continue
        holdout = e.get("holdout") or {}
        cv      = e.get("cv") or {}
        if holdout:
            val_str = holdout.get(metric_key)
        else:
            val_str = cv.get(cv_alias) if cv_alias else cv.get(metric_key)
        val = _parse_mean(val_str)
        if val is None:
            continue
        cell_key = (p["feature_set"], p["datasplit"], p["splittype"] or "-")
        arch_key = (
            p["architecture"],
            p.get("version") or "-",
            p["hp_strategy"] or "NonHP",
            p["hp_variant"] or "-",
        )
        rows.setdefault(cell_key, {})[arch_key] = val
        arch_set.add(arch_key)

    cell_labels = sorted(rows.keys())
    arch_labels = sorted(arch_set)
    matrix = np.full((len(cell_labels), len(arch_labels)), np.nan)
    for i, ck in enumerate(cell_labels):
        for j, ak in enumerate(arch_labels):
            v = rows.get(ck, {}).get(ak)
            if v is not None:
                matrix[i, j] = v
    return matrix, cell_labels, arch_labels


def _rank_with_average_ties(row):
    """Return ranks of ``row`` with average-tie handling (1 = smallest)."""
    order = np.argsort(row, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(row):
        j = i + 1
        while j < len(row) and row[order[j]] == row[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for idx in order[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def friedman_nemenyi(matrix, higher_is_better=True,
                     min_coverage_frac=0.7):
    """Friedman omnibus + Nemenyi critical difference.

    The Friedman test requires a matched design (every classifier
    evaluated on every dataset), so any cell with even one NaN is
    dropped. When the dataset has sparse-coverage architectures (e.g.
    AutoTabPFN intentionally scaffolded to a subset of cells), naively
    dropping rows can collapse the cell count to almost nothing. This
    function therefore first drops architecture columns whose per-cell
    coverage is below ``min_coverage_frac``, then drops residual
    incomplete rows.

    Returns ``(mean_ranks, cd, p_value, n_cells_used, n_arch_used,
    kept_arch_indices, dropped_arch_indices)`` or ``None`` when there
    is not enough complete data after filtering.
    """
    n_total_cells = matrix.shape[0]
    if n_total_cells == 0:
        return None
    coverage = (~np.isnan(matrix)).sum(axis=0) / float(n_total_cells)
    kept = np.where(coverage >= min_coverage_frac)[0]
    dropped = np.where(coverage < min_coverage_frac)[0]
    if kept.size < 2:
        return None
    sub = matrix[:, kept]

    valid = ~np.isnan(sub).any(axis=1)
    m = sub[valid]
    n, k = m.shape
    if n < 2 or k < 2:
        return None
    if k not in _NEMENYI_Q_ALPHA_05:
        return None
    sign = -1.0 if higher_is_better else 1.0
    ranks = np.apply_along_axis(_rank_with_average_ties, 1, sign * m)
    mean_ranks = ranks.mean(axis=0)
    try:
        _, p_value = friedmanchisquare(*[m[:, j] for j in range(k)])
    except ValueError:
        return None
    q_alpha = _NEMENYI_Q_ALPHA_05[k]
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))
    return mean_ranks, cd, p_value, n, k, kept.tolist(), dropped.tolist()


def _short_arch_label(arch_tuple):
    """Compress an (architecture, version, hp_strategy, hp_variant)
    tuple to a one-line label. NonHP cells without a version render as
    just the architecture name. TabPFN cells fold the version into the
    label so each sub-version (v2-6, v3-default, etc.) reads as its own
    competitor.
    """
    arch, version, strategy, variant = arch_tuple
    arch_short = (
        "stacked2xgb" if arch == "stacked_2xgb_meta_lr"
        else "blended"  if arch == "blended_xgb_lr_spano2026"
        else arch
    )
    if version and version != "-":
        # tabpfn 'version_2-6' -> 'tabpfn:v2-6'
        v_clean = version.replace("version_", "v")
        arch_short = f"{arch_short}:{v_clean}"
    if strategy == "NonHP":
        return arch_short
    return f"{arch_short}:{strategy}/{variant}"


# Family-aware palette: each architecture-family gets a distinct hue
# anchor, and within-family HP-budget tiers walk a saturation/lightness
# gradient on that anchor. NonHP baselines and other architectures (e.g.
# tabpfn) get bold solid lines; HP perturbations get thinner dashed or
# dotted lines so the eye can immediately tell "what's a baseline" vs
# "what's a tuning variant". This addresses the core readability problem
# of matplotlib's tab10/tab20 cycles when k >= 10: similar shades
# scattered across unrelated competitors.
_FAMILY_ANCHORS = {
    "stacked_NonHP":             "#1f1f1f",   # near-black: canonical XGBoost stack
    "stacked_single_AUROC":      "#d62728",   # red gradient -> 5 budget tiers
    "stacked_pareto_AUROC":      "#2ca02c",   # green gradient -> 3 frontier picks
    "stacked_pareto_AUPRC":      "#9467bd",   # purple gradient -> 3 frontier picks
    "tabpfn_v2_5_family":        "#17becf",   # teal: v2-5 sub-family (real, finetuned, auto)
    "tabpfn_v2_6":               "#1f77b4",   # blue: v2-6 standalone
    "tabpfn_v3_family":          "#0b3d77",   # deep navy: v3 sub-family (default, binary)
    "blended":                   "#ff7f0e",   # orange: Spano replicator
    "other":                     "#7f7f7f",   # grey fallback
}

_FAMILY_LINESTYLE = {
    "stacked_NonHP":             ("-",  2.4),
    "stacked_single_AUROC":      ("--", 1.3),
    "stacked_pareto_AUROC":      (":",  1.6),
    "stacked_pareto_AUPRC":      ("-.", 1.6),
    "tabpfn_v2_5_family":        ("-",  2.0),
    "tabpfn_v2_6":               ("-",  2.4),
    "tabpfn_v3_family":          ("-",  2.0),
    "blended":                   ("-",  2.4),
    "other":                     ("-",  1.4),
}


def _arch_family(arch_tuple):
    """Classify an (architecture, version, hp_strategy, hp_variant)
    tuple into a plotting family. Returns ``(family_key, gradient_position)``;
    gradient_position is in [0, 1] for within-family ordering so the
    palette steps consistently from "canonical" to "most-perturbed".

    TabPFN is sub-classified by version major: v2-5 family groups the
    Real, Fine-tuned, and AutoTabPFN variants; v2-6 is the lone
    Addition-1 baseline; v3 family groups the v3-default and v3-binary
    checkpoints. Each sub-family keeps its own hue so the eye can
    follow "all the v3 cells" or "all the v2-5 cells" without reading
    labels.
    """
    arch, version, strategy, variant = arch_tuple
    if arch == "stacked_2xgb_meta_lr":
        if strategy == "NonHP":
            return ("stacked_NonHP", 0.5)
        if strategy == "single_AUROC":
            order = {"HP020": 0.0, "HP050": 0.25, "HP100": 0.5,
                     "HP200": 0.75, "HP500": 1.0}
            return ("stacked_single_AUROC", order.get(variant, 0.5))
        if strategy == "pareto_AUROC_slope":
            order = {"auroc_max": 0.0, "knee": 0.5, "slope_closest": 1.0}
            return ("stacked_pareto_AUROC", order.get(variant, 0.5))
        if strategy == "pareto_AUPRC_slope":
            order = {"auprc_max": 0.0, "knee": 0.5, "slope_closest": 1.0}
            return ("stacked_pareto_AUPRC", order.get(variant, 0.5))
        return ("other", 0.5)
    if arch == "blended_xgb_lr_spano2026":
        return ("blended", 0.5)
    if arch.startswith("tabpfn") or arch == "tabpfn":
        v = version or ""
        # v2-5 family: real, finetuned, auto. Position ordering keeps
        # related variants visually adjacent in legends.
        v25_order = {
            "version_2-5-real":      0.0,
            "version_2-5-finetuned": 0.5,
            "version_2-5-auto":      1.0,
        }
        if v in v25_order:
            return ("tabpfn_v2_5_family", v25_order[v])
        if v == "version_2-6":
            return ("tabpfn_v2_6", 0.5)
        v3_order = {
            "version_3-default": 0.0,
            "version_3-binary":  1.0,
        }
        if v in v3_order:
            return ("tabpfn_v3_family", v3_order[v])
        return ("tabpfn_v2_6", 0.5)   # unknown version falls back to baseline hue
    return ("other", 0.5)


def _hex_to_rgb(hx):
    h = hx.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _adjust_brightness(hex_color, position):
    """Walk a saturation gradient on ``hex_color``: position=0 is the
    darkest/most-saturated end, position=1 is the lightest end. Used
    for within-family HP-budget tier shading.
    """
    r, g, b = _hex_to_rgb(hex_color)
    # Blend with white by amount (0.0 to 0.55) - lower budget = darker.
    blend = 0.15 + 0.40 * position
    r = round(r + (255 - r) * blend)
    g = round(g + (255 - g) * blend)
    b = round(b + (255 - b) * blend)
    return (r / 255.0, g / 255.0, b / 255.0)


def _family_style_for(arch_tuple):
    """Return (color_rgb, linestyle, linewidth) for one architecture."""
    family, position = _arch_family(arch_tuple)
    anchor = _FAMILY_ANCHORS.get(family, _FAMILY_ANCHORS["other"])
    color = _adjust_brightness(anchor, position)
    linestyle, linewidth = _FAMILY_LINESTYLE.get(
        family, _FAMILY_LINESTYLE["other"])
    return color, linestyle, linewidth


# Distinct marker per within-family member so the within-family colour
# gradient is reinforced by shape (the 5 single_AUROC tiers, the 5
# tabpfn versions, etc. each get their own dot type). Indexed by the
# gradient position bucket.
_WITHIN_FAMILY_MARKERS = ("o", "s", "^", "D", "v", "P")


def _family_marker(arch_tuple):
    """Return a marker shape that varies with the within-family
    gradient position, so members of one family are distinguishable by
    dot shape as well as colour lightness.
    """
    _family, position = _arch_family(arch_tuple)
    idx = int(round(position * (len(_WITHIN_FAMILY_MARKERS) - 1)))
    return _WITHIN_FAMILY_MARKERS[min(idx, len(_WITHIN_FAMILY_MARKERS) - 1)]


def render_cd_diagram(mean_ranks, cd, arch_labels, out_path,
                      title="", p_value=None, n_cells=None,
                      arch_tuples=None, dropped_archs=None):
    """Render a Demsar-style CD diagram.

    Architectures hang from a horizontal mean-rank axis (rank 1, best,
    on the right). The field is split down the middle: better-ranked
    architectures label out to the right and worse-ranked ones to the
    left, each stacked on its own row so connectors never share a text
    line. Architectures joined by a heavy horizontal bar on the axis are
    not significantly different at alpha=0.05 (their mean ranks differ
    by less than CD).

    The figure is built to read in black-and-white print: connectors are
    thin and translucent and sit behind near-black label text, while a
    small filled marker at each connector elbow carries the family colour
    as a secondary cue only.
    """
    apply_journal_style()
    k = len(mean_ranks)
    order = np.argsort(mean_ranks)
    sorted_ranks = mean_ranks[order]
    sorted_labels = [arch_labels[i] for i in order]
    sorted_tuples = ([arch_tuples[i] for i in order]
                     if arch_tuples is not None else [None] * k)

    # Cliques: maximal contiguous groups whose rank span is <= CD.
    cliques = []
    i = 0
    while i < k:
        j = k - 1
        while j > i and (sorted_ranks[j] - sorted_ranks[i]) > cd:
            j -= 1
        if j > i:
            cliques.append((i, j))
        i += 1
    deduped = []
    for c in cliques:
        if not any(c != d and c[0] >= d[0] and c[1] <= d[1] for d in cliques):
            deduped.append(c)
    cliques = deduped

    # Split the ranked field: the better half (lower mean rank) labels on
    # the right, the worse half on the left. Each side gets a contiguous
    # block of rows so a connector's horizontal run ends on an empty line.
    n_right = (k + 1) // 2          # better-ranked architectures
    right_idx = list(range(n_right))             # rows 0..n_right-1 (best first)
    left_idx  = list(range(k - 1, n_right - 1, -1))   # worst first
    rows_per_side = max(n_right, len(left_idx))

    # Vertical pitch between label rows, in axis units. A comfortable
    # fixed pitch is what keeps many-competitor diagrams (k up to ~18)
    # readable; the figure height tracks it so the pitch is constant on
    # the page regardless of k.
    pitch = 1.0
    top_y = 0.0                     # rank axis sits at y = 0
    # Significance-clique bars stack just below the axis; the label rows
    # begin immediately under that band (plus the optional all-tie note),
    # so there is no fixed vertical dead band between the rank line and
    # the labels however many clique levels there are.
    clique_step = 0.18
    clique_base = top_y + 0.13
    single_full_clique = (len(cliques) == 1 and cliques[0] == (0, k - 1))
    clique_band_top = clique_base + clique_step * max(len(cliques), 1)
    first_row_y = clique_band_top + (0.70 if single_full_clique else 0.40)
    bottom_y = first_row_y + (rows_per_side - 1) * pitch

    fig_h = 2.6 + 0.40 * rows_per_side
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    # Wide outer margins so the side labels, which grow outward from the
    # elbow toward the plot edge, have room; bbox_inches="tight" crops the
    # surplus on save.
    pad = max(3.0, 0.30 * k)
    ax.set_xlim(k + pad, 1 - pad)   # invert so rank 1 (best) on the right
    axis_top = top_y - 1.2          # room above the line for CD bar + title
    ax.set_ylim(bottom_y + 0.4, axis_top)
    ax.axis("off")

    text_dark = "#111111"
    # The rank axis is drawn by hand so the scale numbers sit directly on
    # the axis line. matplotlib's top-spine ticks would render at the axes
    # top, far above a data-positioned spine, leaving the scale stranded
    # from the line it labels (the canonical CD-diagram layout keeps them
    # together).
    ax.plot([1, k], [top_y, top_y], "-", color=text_dark, lw=1.8,
            solid_capstyle="round", zorder=3)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [top_y, top_y - 0.13], "-",
                color=text_dark, lw=1.2, zorder=3)
        ax.text(tick, top_y - 0.28, str(tick), ha="center", va="bottom",
                fontsize=9, color=text_dark)
    ax.text((1 + k) / 2.0, axis_top, "Mean rank (rightmost = best)",
            ha="center", va="bottom", fontsize=10.5, color=text_dark)

    # x where each side's horizontal connector run terminates, just inside
    # the plot edge. Best-ranked architectures sit toward rank 1 on the
    # right of the inverted axis and label out to the right edge;
    # worst-ranked architectures label out to the left edge. Sending each
    # side to its own edge keeps the two label columns from colliding.
    x_elbow_right = 1.0             # right edge (near rank 1)
    x_elbow_left  = float(k)        # left edge (near worst rank)
    text_gap = 0.55

    def _draw_side(row_indices, side):
        for row, r in enumerate(row_indices):
            rank = sorted_ranks[r]
            label = sorted_labels[r]
            y = first_row_y + row * pitch
            if sorted_tuples[r] is not None:
                color, _, _ = _family_style_for(sorted_tuples[r])
            else:
                color = (0.4, 0.4, 0.4)
            x_elbow = x_elbow_right if side == "right" else x_elbow_left
            # Connector: a solid, full-weight polyline that starts on the
            # rank axis, drops to the label row, then runs out to the
            # elbow. Rounded joins keep the corner clean; the dark label
            # text (higher zorder) still reads on top where they meet.
            ax.plot([rank, rank, x_elbow], [top_y, y, y],
                    "-", lw=1.7, color=color, alpha=1.0, zorder=2,
                    solid_capstyle="round", solid_joinstyle="round")
            # Endpoint dot where the connector meets its label.
            ax.plot([x_elbow], [y], marker="o", ms=4.5, color=color,
                    zorder=3)
            # Text sits beyond the elbow, growing outward toward the plot
            # edge so it never overlaps the connector lines or the marker.
            # On the inverted axis the right column grows toward smaller x
            # and the left column toward larger x.
            if side == "right":
                ax.text(x_elbow - text_gap, y, label, va="center",
                        ha="left", fontsize=9, color=text_dark, zorder=4)
            else:
                ax.text(x_elbow + text_gap, y, label, va="center",
                        ha="right", fontsize=9, color=text_dark, zorder=4)

    _draw_side(right_idx, "right")
    _draw_side(left_idx, "left")

    # Significance cliques: heavy bars stacked just below the axis line,
    # connecting the rank positions of architectures that are NOT
    # significantly different. Each level is offset so overlapping cliques
    # stay legible. (clique_step / clique_base set with the geometry above.)
    for level, (lo, hi) in enumerate(cliques):
        y = clique_base + clique_step * level
        ax.plot([sorted_ranks[lo] - 0.06, sorted_ranks[hi] + 0.06],
                [y, y], "-", color=text_dark, lw=5.5,
                solid_capstyle="round", zorder=5)
    if len(cliques) == 1 and cliques[0] == (0, k - 1):
        ax.text(
            (sorted_ranks[0] + sorted_ranks[-1]) / 2.0,
            clique_base + clique_step * len(cliques) + 0.34,
            "no pair significantly different at alpha=0.05",
            ha="center", va="top", fontsize=9, style="italic",
            color="#444444", zorder=4,
        )

    # CD scale bar just above the rank numbers, anchored at rank 1, so the
    # critical-difference span reads against the same scale without a gap.
    cd_y = top_y - 0.72
    ax.plot([1, 1 + cd], [cd_y, cd_y], "-", color=text_dark, lw=2.4,
            solid_capstyle="butt", zorder=4)
    ax.plot([1, 1], [cd_y - 0.09, cd_y + 0.09], "-", color=text_dark,
            lw=1.4, zorder=4)
    ax.plot([1 + cd, 1 + cd], [cd_y - 0.09, cd_y + 0.09], "-",
            color=text_dark, lw=1.4, zorder=4)
    ax.text(1 + cd / 2.0, cd_y - 0.16, f"CD = {cd:.2f}", ha="center",
            va="bottom", fontsize=9.5, color=text_dark, zorder=4)

    if title:
        sub = []
        if p_value is not None:
            sub.append(f"Friedman p = {p_value:.3g}")
        if n_cells is not None:
            sub.append(f"n_cells = {n_cells}")
        suffix = (" (" + ", ".join(sub) + ")") if sub else ""
        ax.set_title(title + suffix, fontsize=11, pad=14)

    # How to read the diagram (Demsar 2006): make the ranking method
    # explicit so the figure is self-contained for a reader unfamiliar
    # with CD diagrams.
    n_txt = str(n_cells) if n_cells is not None else "n"
    fig.text(
        0.5, 0.045,
        f"Each method is ranked 1 (best) to {k} within every cell, then "
        f"averaged over the {n_txt} cells to give its mean rank (dot). "
        "Methods joined by a bar differ by less than the",
        ha="center", va="bottom", fontsize=8.5, color="#444444",
    )
    fig.text(
        0.5, 0.018,
        "critical difference CD (Nemenyi post-hoc, alpha=0.05) and are "
        "therefore not significantly different.",
        ha="center", va="bottom", fontsize=8.5, color="#444444",
    )

    if dropped_archs:
        # Honest disclosure: which architectures were excluded from the
        # statistical test because their per-cell coverage was below
        # the matched-design threshold (e.g. AutoTabPFN, which is
        # intentionally scaffolded to a subset of cells).
        fig.text(
            0.5, 0.075,
            "Excluded from Friedman (incomplete cell coverage): "
            + ", ".join(dropped_archs),
            ha="center", va="bottom", fontsize=8.0,
            style="italic", color="#777777",
        )

    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.14)
    save_journal_figure(fig, out_path)
    plt.close(fig)


def render_performance_profile(matrix, arch_labels, out_path,
                               title="", higher_is_better=True,
                               tau_max=2.0, arch_tuples=None,
                               dropped_archs=None):
    """Dolan-More performance profile.

    The performance ratio r_p,s is metric_p,best / metric_p,s for
    higher-is-better metrics (always >= 1.0). The profile rho_s(tau) is
    the fraction of cells p where r_p,s <= tau. Architectures whose
    curve reaches y=1 at small tau dominate; architectures whose curve
    rises slowly only solve a small share of cells near-optimally.

    Each curve carries three independent channels so that up to ~15
    architectures stay separable: family hue, family line style, and a
    per-member marker shape. The marker is the channel that survives
    grayscale, where the within-family lightness gradient collapses.
    Markers are placed sparsely (``markevery``) and at a per-curve phase
    offset so coincident staircases do not stamp markers on top of one
    another. The x-axis is clipped to the 90th percentile of finite
    performance ratios, because the staircases saturate to rho=1 well
    before tau_max and the empty tail wastes plot area and squeezes the
    informative low-tau region.
    """
    valid = ~np.isnan(matrix).any(axis=1)
    m = matrix[valid]
    if m.size == 0:
        return
    n_cells, n_arch = m.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        if higher_is_better:
            best = m.max(axis=1, keepdims=True)
            ratios = np.where(m > 0, best / m, np.inf)
        else:
            best = m.min(axis=1, keepdims=True)
            ratios = np.where(best > 0, m / best, np.inf)
    ratios = np.where(np.isfinite(ratios), ratios, np.inf)

    # Adaptive upper tau: fit to the 90th percentile of finite ratios so
    # the informative low-tau region fills the axis. Clamp into a sane
    # band and never exceed the caller's tau_max so the saturated tail is
    # trimmed rather than padded.
    finite = ratios[np.isfinite(ratios)]
    if finite.size:
        x_max = float(np.percentile(finite, 90))
    else:
        x_max = tau_max
    x_max = max(1.05, min(x_max, tau_max))

    apply_journal_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    taus = np.linspace(1.0, x_max, 400)
    # Markevery base spacing and a per-curve phase so that overlapping
    # staircases reveal their separate markers instead of stamping them
    # on the same x positions. The phase fans the n_arch curves evenly
    # across one marker period so coincident plateaus interleave.
    base_every = max(28, taus.size // 11)
    phase_step = max(1, base_every // max(1, n_arch))
    for j in range(n_arch):
        rho = np.array([(ratios[:, j] <= t).sum() / n_cells for t in taus])
        if arch_tuples is not None:
            color, linestyle, linewidth = _family_style_for(arch_tuples[j])
            marker = _family_marker(arch_tuples[j])
        else:
            cmap = plt.get_cmap("tab20" if n_arch > 10 else "tab10")
            color, linestyle, linewidth = cmap(j % cmap.N), "-", 1.5
            marker = "o"
        phase = (j * phase_step) % base_every
        ax.plot(taus, rho, lw=linewidth, ls=linestyle, color=color,
                marker=marker, markevery=(phase, base_every), markersize=5.5,
                markeredgecolor="black", markeredgewidth=0.4,
                label=str(arch_labels[j]))
    ax.set_xlabel("tau (tolerance factor)")
    ax.set_ylabel("Fraction of cells with metric within tau x best")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1.0, x_max)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(f"{title} (n_cells = {n_cells})", fontsize=10)
    if dropped_archs:
        ax.text(
            0.5, -0.13,
            "Excluded (incomplete cell coverage): " + ", ".join(dropped_archs),
            transform=ax.transAxes, ha="center", fontsize=8.0,
            style="italic", color="#777",
        )
    # Legend outside the axes so it does not overlap the staircase
    # curves; family-grouped order makes baselines visually adjacent. The
    # swatch reproduces colour, line style, AND marker so a reader can
    # match each entry to its curve in grayscale.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=7.5, framealpha=0.9, ncol=1,
              handlelength=3.0, borderaxespad=0.2)
    fig.tight_layout(pad=0.6)
    save_journal_figure(fig, out_path)
    plt.close(fig)


def render_rank_slopegraph(matrix, cell_labels, arch_labels, out_path,
                           title="", higher_is_better=True,
                           arch_tuples=None, dropped_archs=None):
    """Rank-bumps slopegraph across cells.

    Each polyline is one architecture's rank trajectory across cells.
    Stable architectures look flat; unstable architectures swap ranks.

    Architecture names are placed as direct end-of-line labels in a
    dedicated right-hand margin outside the plotting area, so they never
    overlap the polylines or gridlines. Each label is anchored by a thin
    leader line that runs from the architecture's final-cell point to a
    family-coloured swatch beside the dark, near-black label text. The
    swatch carries the family colour and the per-line marker, keeping
    every line identifiable in colour and in grayscale, while the label
    text stays legible regardless of how pale a dimmed family colour is.
    """
    valid = ~np.isnan(matrix).any(axis=1)
    m = matrix[valid]
    cells_used = [cell_labels[i] for i, v in enumerate(valid) if v]
    if m.size == 0:
        return
    n_cells, n_arch = m.shape

    sign = -1.0 if higher_is_better else 1.0
    ranks = np.apply_along_axis(_rank_with_average_ties, 1, sign * m)

    # Per-line marker cycle so that, in grayscale, lines sharing a family
    # hue and line style remain separable by glyph shape.
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h",
                    "p", "d", "H", "8", "4", "+", "x"]

    xs = np.arange(n_cells)
    label_text = [str(a) for a in arch_labels]

    # Geometry of the right-hand label margin. The x range is widened past
    # the last cell so labels live outside the data area; the leader is
    # drawn in three segments: line endpoint -> a short horizontal stub at
    # the endpoint rank -> the de-collided label anchor. The text column
    # width tracks the longest label so the reserved margin stays tight.
    longest = max((len(t) for t in label_text), default=1)
    cell_step = 1.55                              # inches between cells
    text_w = (longest * 0.052 + 0.30) / cell_step  # label width in x-units
    x_last = float(n_cells - 1)
    x_stub = x_last + 0.22                        # short stub off each endpoint
    x_swatch = x_last + 0.55                      # family-colour swatch column
    x_text = x_last + 0.62                        # label text begins here
    x_right = x_text + text_w                     # outer x-limit, hugs text

    fig_w = 2.4 + cell_step * (n_cells + text_w)
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(fig_w, 1.4 + 0.29 * n_arch))

    line_style = []
    for j in range(n_arch):
        if arch_tuples is not None:
            color, linestyle, linewidth = _family_style_for(arch_tuples[j])
            marker_size = 5 + 2 * (linewidth - 1.3)   # baselines get bigger markers
        else:
            cmap = plt.get_cmap("tab20" if n_arch > 10 else "tab10")
            color, linestyle, linewidth = cmap(j % cmap.N), "-", 1.4
            marker_size = 5
        marker = marker_cycle[j % len(marker_cycle)]
        line_style.append((color, linestyle, linewidth, marker, marker_size))
        ax.plot(xs, ranks[:, j], marker=marker, lw=linewidth, ls=linestyle,
                ms=marker_size, color=color)

    ax.invert_yaxis()
    ax.set_yticks(range(1, n_arch + 1))
    ax.set_ylabel("Rank (1 = best)")
    ax.set_xticks(xs)
    # Drop the feature_set prefix from the cell tick labels when every
    # cell shares it (these benchmark plots are restricted to one
    # feature set, named in the title) - removes redundant repetition.
    one_fs = len({c[0] for c in cells_used}) == 1
    tick_labels = ["/".join(c[1:] if one_fs else c) for c in cells_used]
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_xlim(-0.35, x_right)

    # De-collide the end labels. Targets are each line's final-cell rank;
    # labels are nudged apart so adjacent (tied or near-tied) entries keep
    # a minimum vertical gap, then re-centred to stay near their targets.
    final_rank = ranks[-1, :]
    order = np.argsort(final_rank, kind="stable")   # top (rank 1) first
    min_gap = (n_arch + 0.6) / float(max(n_arch, 1))   # ~1 rank-unit gap
    placed = np.empty(n_arch, dtype=float)
    prev = -np.inf
    for j in order:
        y = max(final_rank[j], prev + min_gap)
        placed[j] = y
        prev = y
    # If spreading pushed the stack past the bottom, slide it back up so it
    # stays inside the rank band rather than overrunning the axis.
    overshoot = placed[order[-1]] - (n_arch + 0.2)
    if overshoot > 0:
        shift = np.zeros(n_arch)
        prev = np.inf
        for j in order[::-1]:
            y = min(placed[j], prev - min_gap)
            shift[j] = y
            prev = y
        placed = shift

    for j in range(n_arch):
        color, linestyle, linewidth, marker, marker_size = line_style[j]
        y_end = final_rank[j]
        y_lab = placed[j]
        # Leader: endpoint -> horizontal stub -> diagonal to the swatch row.
        ax.plot([x_last, x_stub, x_swatch], [y_end, y_end, y_lab],
                ls="-", lw=0.7, color="#888888", clip_on=False, zorder=1)
        # Family-colour swatch carrying the per-line marker.
        ax.plot([x_swatch], [y_lab], marker=marker, ms=marker_size,
                color=color, mec="#333333", mew=0.5,
                clip_on=False, zorder=3)
        # Dark, near-black label text for legibility regardless of hue.
        ax.text(x_text, y_lab, label_text[j], va="center", ha="left",
                fontsize=7.8, color="#111111", clip_on=False, zorder=4)

    if title:
        ax.set_title(f"{title} (n_cells = {n_cells})", fontsize=10)
    if dropped_archs:
        # Below the rotated x-tick labels so it never overlaps them.
        ax.text(
            0.0, -0.42,
            "Excluded (incomplete cell coverage): " + ", ".join(dropped_archs),
            transform=ax.transAxes, ha="left", fontsize=8.0,
            style="italic", color="#777",
        )
    fig.tight_layout(pad=0.6)
    save_journal_figure(fig, out_path)
    plt.close(fig)


def render_benchmark_triplet(all_entries, metric_key, target, out_dir,
                             ts_flat, short_uid,
                             metric_label=None,
                             higher_is_better=True,
                             restrict_feature_sets=None,
                             cv_alias=None,
                             label_fn=None):
    """Generate CD + performance-profile + slopegraph for one
    (metric, target) pair and return the produced filenames as a dict.

    Returns empty dict (and prints a note) when the matrix has fewer
    than 2 complete cells - the three plots all require matched
    multi-cell data.
    """
    matrix, cell_labels, arch_labels = build_metric_matrix(
        all_entries, metric_key, target,
        restrict_feature_sets=restrict_feature_sets,
        cv_alias=cv_alias,
    )
    label_fn = label_fn or _short_arch_label
    arch_text = [label_fn(a) for a in arch_labels]

    out = {}
    mlabel = metric_label or metric_key
    base = f"{target}_{metric_label or metric_key}".replace(" ", "_")

    # Drop sparse-coverage architectures once at the top so all three
    # views (CD, performance profile, slopegraph) see the same
    # matched-design subset. Without this filter the perf-profile and
    # slopegraph would compute their stats on the few cells where
    # *every* listed architecture has a value, which collapses to ~2
    # cells when even one architecture (e.g. AutoTabPFN, scaffolded
    # to a subset of cells) has incomplete coverage.
    n_total_cells = matrix.shape[0]
    if n_total_cells == 0:
        return out
    coverage = (~np.isnan(matrix)).sum(axis=0) / float(n_total_cells)
    kept_idx    = [j for j, c in enumerate(coverage) if c >= 0.7]
    dropped_idx = [j for j, c in enumerate(coverage) if c < 0.7]
    if len(kept_idx) < 2:
        return out

    matrix_kept   = matrix[:, kept_idx]
    arch_kept     = [arch_labels[j] for j in kept_idx]
    text_kept     = [arch_text[j]   for j in kept_idx]
    dropped_text  = [arch_text[j]   for j in dropped_idx]

    # CD diagram (skips quietly when k is outside the Nemenyi table or
    # there are fewer than 2 complete cells).
    nem = friedman_nemenyi(matrix_kept, higher_is_better=higher_is_better)
    if nem is not None:
        mean_ranks, cd, p_value, n_cells, _, _kept2, _dropped2 = nem
        cd_path = out_dir / f"cd_{base}_{ts_flat}_{short_uid}.png"
        render_cd_diagram(
            mean_ranks, cd, text_kept, cd_path,
            title=f"Critical Difference - {target} {mlabel}",
            p_value=p_value, n_cells=n_cells,
            arch_tuples=arch_kept,
            dropped_archs=dropped_text,
        )
        out["cd"] = cd_path.name

    # Performance profile (uses the same coverage-filtered architecture set).
    pp_path = out_dir / f"perfprofile_{base}_{ts_flat}_{short_uid}.png"
    render_performance_profile(
        matrix_kept, text_kept, pp_path,
        title=f"Performance profile (Dolan-More) - {target} {mlabel}",
        higher_is_better=higher_is_better,
        arch_tuples=arch_kept,
        dropped_archs=dropped_text,
    )
    out["perfprofile"] = pp_path.name

    # Slopegraph (same subset).
    sg_path = out_dir / f"slopegraph_{base}_{ts_flat}_{short_uid}.png"
    render_rank_slopegraph(
        matrix_kept, cell_labels, text_kept, sg_path,
        title=f"Rank slopegraph - {target} {mlabel}",
        higher_is_better=higher_is_better,
        arch_tuples=arch_kept,
        dropped_archs=dropped_text,
    )
    out["slopegraph"] = sg_path.name

    return out
