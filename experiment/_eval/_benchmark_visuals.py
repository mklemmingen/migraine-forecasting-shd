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

    Architectures are arranged along a horizontal axis by mean rank
    (lower = better). Architectures joined by a heavy horizontal bar
    are not significantly different at alpha=0.05 (their mean ranks
    differ by less than CD).
    """
    k = len(mean_ranks)
    order = np.argsort(mean_ranks)
    sorted_ranks = mean_ranks[order]
    sorted_labels = [arch_labels[i] for i in order]

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

    apply_journal_style()
    fig, ax = plt.subplots(figsize=(11, 1.5 + 0.32 * k))
    pad = 0.6
    ax.set_xlim(k + pad, 1 - pad)   # invert so rank 1 (best) on the right
    ax.set_ylim(-1.0, k + 1.5)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.5))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(1, k + 1))
    ax.set_xlabel("Mean rank (rightmost = best)")

    # Connect each architecture's rank position to a side label, with
    # family-coloured connectors so the eye can see at a glance which
    # branch of the architecture tree a rank belongs to.
    sorted_tuples = ([arch_tuples[i] for i in order]
                     if arch_tuples is not None else [None] * k)
    for r in range(k):
        rank = sorted_ranks[r]
        label = sorted_labels[r]
        if sorted_tuples[r] is not None:
            color, _, _ = _family_style_for(sorted_tuples[r])
        else:
            color = (0, 0, 0)
        if r < k / 2:
            x_label = k + 0.4
            ax.plot([rank, rank, x_label], [-0.5, r + 1, r + 1],
                    "-", lw=0.9, color=color)
            ax.text(x_label + 0.05, r + 1, label, va="center",
                    ha="left", fontsize=8.5, color=color)
        else:
            x_label = 0.6
            ax.plot([rank, rank, x_label], [-0.5, r + 1, r + 1],
                    "-", lw=0.9, color=color)
            ax.text(x_label - 0.05, r + 1, label, va="center",
                    ha="right", fontsize=8.5, color=color)

    # CD scale bar at top.
    ax.plot([1, 1 + cd], [-0.95, -0.95], "k-", lw=2.0, solid_capstyle="butt")
    ax.plot([1, 1], [-0.85, -1.05], "k-", lw=1.0)
    ax.plot([1 + cd, 1 + cd], [-0.85, -1.05], "k-", lw=1.0)
    ax.text(1 + cd / 2.0, -1.15, f"CD = {cd:.2f}", ha="center", fontsize=9)

    # Clique bars below the spine (architectures NOT significantly different).
    # When exactly one clique spans the entire field, the diagram is
    # making the negative-result statement "no pair is significantly
    # different at alpha=0.05" - annotate it explicitly so a reader
    # who is unfamiliar with CD diagrams does not miss the message.
    for level, (lo, hi) in enumerate(cliques):
        y = 0.0 - 0.08 * (level + 1)
        ax.plot([sorted_ranks[lo] - 0.02, sorted_ranks[hi] + 0.02],
                [y, y], "k-", lw=3.5, solid_capstyle="butt")
    if len(cliques) == 1 and cliques[0] == (0, k - 1):
        ax.text(
            (sorted_ranks[0] + sorted_ranks[-1]) / 2.0,
            -0.55, "(no pair significantly different at alpha=0.05)",
            ha="center", fontsize=8.5, style="italic", color="#555",
        )

    if title:
        sub = []
        if p_value is not None:
            sub.append(f"Friedman p = {p_value:.3g}")
        if n_cells is not None:
            sub.append(f"n_cells = {n_cells}")
        suffix = (" (" + ", ".join(sub) + ")") if sub else ""
        ax.set_title(title + suffix, fontsize=10)

    if dropped_archs:
        # Honest disclosure: which architectures were excluded from the
        # statistical test because their per-cell coverage was below
        # the matched-design threshold (e.g. AutoTabPFN, which is
        # intentionally scaffolded to a subset of cells).
        ax.text(
            0.5, -0.10,
            "Excluded from Friedman (incomplete cell coverage): "
            + ", ".join(dropped_archs),
            transform=ax.transAxes, ha="center", fontsize=8.0,
            style="italic", color="#777",
        )

    plt.tight_layout()
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

    # Fit the x-range to the informative region: most curves plateau
    # well before tau_max=2.0, so a fixed axis wastes the right half on
    # flat lines. Clip just past the 90th percentile of finite ratios
    # (a single anti-predictive cell can push the max ratio toward 2.0;
    # the percentile keeps that lone outlier from stretching the axis).
    finite = ratios[np.isfinite(ratios)]
    if finite.size:
        x_max = min(tau_max, max(1.15, float(np.percentile(finite, 90)) * 1.05))
    else:
        x_max = tau_max

    apply_journal_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    taus = np.linspace(1.0, x_max, 400)
    for j in range(n_arch):
        rho = np.array([(ratios[:, j] <= t).sum() / n_cells for t in taus])
        if arch_tuples is not None:
            color, linestyle, linewidth = _family_style_for(arch_tuples[j])
        else:
            cmap = plt.get_cmap("tab20" if n_arch > 10 else "tab10")
            color, linestyle, linewidth = cmap(j % cmap.N), "-", 1.5
        # Dim the thin HP-variant lines so the bold baselines (tabpfn
        # versions, stacked NonHP) read as the foreground.
        alpha = 1.0 if linewidth >= 1.6 else 0.65
        ax.plot(taus, rho, lw=linewidth, ls=linestyle, color=color,
                alpha=alpha, label=str(arch_labels[j]))
    ax.set_xlabel("tau (tolerance factor)")
    ax.set_ylabel("Fraction of cells with metric within tau x best")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1.0, x_max)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(f"{title} (n_cells = {n_cells})", fontsize=10)
    if dropped_archs:
        ax.text(
            0.5, -0.20,
            "Excluded (incomplete cell coverage): " + ", ".join(dropped_archs),
            transform=ax.transAxes, ha="center", fontsize=8.0,
            style="italic", color="#777",
        )
    # Legend outside the axes so it does not overlap the staircase
    # curves; family-grouped order makes baselines visually adjacent.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7.5, framealpha=0.9, ncol=1)
    plt.tight_layout()
    save_journal_figure(fig, out_path)
    plt.close(fig)


def render_rank_slopegraph(matrix, cell_labels, arch_labels, out_path,
                           title="", higher_is_better=True,
                           arch_tuples=None, dropped_archs=None):
    """Rank-bumps slopegraph across cells.

    Each polyline is one architecture's rank trajectory across cells.
    Stable architectures look flat; unstable architectures swap ranks.
    """
    valid = ~np.isnan(matrix).any(axis=1)
    m = matrix[valid]
    cells_used = [cell_labels[i] for i, v in enumerate(valid) if v]
    if m.size == 0:
        return
    n_cells, n_arch = m.shape

    sign = -1.0 if higher_is_better else 1.0
    ranks = np.apply_along_axis(_rank_with_average_ties, 1, sign * m)

    # Wider figure: extra right margin holds the direct end-labels that
    # replace the cramped legend, and taller rows keep the n_arch lines
    # vertically separable.
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(5.0 + n_cells * 1.8, 2.0 + 0.42 * n_arch))
    xs = np.arange(n_cells)

    styles = []  # (color, label) per architecture, for the end-labels
    for j in range(n_arch):
        if arch_tuples is not None:
            color, linestyle, linewidth = _family_style_for(arch_tuples[j])
            marker = _family_marker(arch_tuples[j])
            marker_size = 5 + 2 * (linewidth - 1.3)   # baselines: bigger markers
        else:
            cmap = plt.get_cmap("tab20" if n_arch > 10 else "tab10")
            color, linestyle, linewidth, marker = cmap(j % cmap.N), "-", 1.4, "o"
            marker_size = 5
        # Dim thin HP-variant lines so the bold baselines (tabpfn
        # versions, stacked NonHP) stay in the visual foreground.
        alpha = 1.0 if linewidth >= 1.6 else 0.6
        zo = 3 if linewidth >= 1.6 else 2
        ax.plot(xs, ranks[:, j], marker=marker, lw=linewidth, ls=linestyle,
                ms=marker_size, color=color, alpha=alpha, zorder=zo)
        styles.append((color, str(arch_labels[j])))

    # Direct end-of-line labels at the right edge, de-collided so labels
    # at tied/adjacent final ranks do not overlap (Tufte slopegraph
    # convention - lets the eye trace a line to its name without a legend).
    final_rank = ranks[:, -1]
    order = np.argsort(final_rank)            # top (best) first
    min_gap = 0.85                            # in rank units
    placed_y = []
    label_x = (n_cells - 1) + 0.08
    for j in order:
        y = final_rank[j]
        if placed_y and y - placed_y[-1] < min_gap:
            y = placed_y[-1] + min_gap
        placed_y.append(y)
        color, lbl = styles[j]
        ax.plot([n_cells - 1, label_x - 0.02], [final_rank[j], y],
                color=color, lw=0.5, alpha=0.5, zorder=1)
        ax.text(label_x, y, lbl, color=color, fontsize=7.5,
                ha="left", va="center", fontfamily="monospace")

    ax.invert_yaxis()
    ax.set_yticks(range(1, n_arch + 1))
    ax.set_ylabel("Rank (1 = best)")
    ax.set_xticks(xs)
    # Drop the feature_set prefix from the cell tick labels when every
    # cell shares it (these benchmark plots are restricted to one
    # feature set, named in the title) - removes redundant repetition.
    one_fs = len({c[0] for c in cells_used}) == 1
    tick_labels = ["/".join(c[1:] if one_fs else c) for c in cells_used]
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.3, label_x + 0.05)
    ax.margins(x=0)
    ax.grid(alpha=0.3, axis="y")
    if title:
        ax.set_title(f"{title} (n_cells = {n_cells})", fontsize=10)
    if dropped_archs:
        ax.text(
            0.5, -0.16,
            "Excluded (incomplete cell coverage): " + ", ".join(dropped_archs),
            transform=ax.transAxes, ha="center", fontsize=8.0,
            style="italic", color="#777",
        )
    plt.tight_layout()
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
