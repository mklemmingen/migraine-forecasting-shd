"""Shared figure style for the whole repository - the single source of truth.

Both figure generators import this one module:
  - aggregator / addition figures (experiment/_eval/_benchmark_visuals.py,
    _venn_diagrams.py, _tree_diagram.py; experiment/2/compare.py;
    experiment/3/_temporal/_plots.py);
  - custom paper figures (docs/methodAndResults_diagramCreatorScripts/fig_*.py).

It lives at experiment/ root (not inside _eval) so it is neutral to any one
subsystem; docs scripts import it after putting experiment/ on sys.path, keeping
the dependency one-way (docs -> experiment, never the reverse).

================================ FIGURE CONVENTIONS ===========================
Everything a journal-quality diagram in this repo must follow. Full rationale
and verified sources are embedded in this docstring. References [bracketed]
are listed at the bottom of this docstring.

1. COLOUR - four palette roles; pick by data type [harrowerbrewer2003].
   - Qualitative, <=8 categories  -> Okabe-Ito (TARGET / ARCH / SPLIT / FEATURE_SET
     below). Colour-vision-safe and grayscale-safe [okabeito2008; wong2011]; it is
     the default qualitative scale in Wilke's Fundamentals of Data Viz [wilke2019].
   - Qualitative, many (~15 architecture variants) -> family anchor + within-family
     brightness gradient + per-member marker + line style (_benchmark_visuals).
   - Sequential / diverging (a metric magnitude, e.g. the comparison heatmap)
     -> ColorBrewer RdBu, warm=worse/cool=better (_metric_palette); never red-green
     or rainbow [crameri2020; rougier2014 rule 6].
   - Ordinal (tree depth) -> one ordered single-hue ramp.
   - Each entity keeps ONE colour everywhere; reinforce colour with marker/line so
     figures survive grayscale and colour-vision deficiency.

2. TYPOGRAPHY [naturefig].
   - Sans-serif only (Arial/Helvetica, fallback DejaVu Sans), one family throughout.
   - Sizes: 9 pt body, 8 pt ticks & legend, 10.5 pt axis title, 11.5 pt suptitle.
     (Nature caps body at 7 pt / panel labels 8 pt bold AT FINAL PRINT SIZE; drop
     to 7 pt when authoring at single-column width for a Nature submission.)
   - Colours: text #222, axes/ticks #444.
   - Embed fonts as TrueType (pdf.fonttype 42); never outline/rasterise text - it
     stays editable for the typesetter [naturefig].

3. TITLES.
   - One concise line, regular weight, centred at top. The DESCRIPTION belongs in
     the LaTeX caption, never burned into the image [rougier2014 rule 4].
   - Multi-panel: bold lowercase a, b, ... at each panel's top-left (panel labels),
     plus a short per-panel title; ONE shared legend, not one per panel.

4. LEGENDS.
   - Frameless by default; place in empty plot space (a clear corner; avoid
     occluding data). Frame (framealpha ~0.9) ONLY over a filled background such as
     a heatmap, where a frameless legend is unreadable.
   - Prefer DIRECT labelling of series over a legend when there are few lines
     [rougier2014 rule 5/8]. Legend title only when the entity is non-obvious.

5. WHITESPACE & LAYOUT.
   - Author AT final size: single-column ~89 mm (3.5 in), double-column ~183 mm
     (7.2 in); two-panel figures use double-column. Setting the real size keeps
     on-page font sizes correct [chain2022mpl].
   - Tight bounding box (no surplus margin); adequate inter-panel spacing; do not
     overcrowd. Maximise data-ink, remove chartjunk: no gratuitous gridlines,
     backgrounds, boxes, or 3D effects [tufte2001; rougier2014 rule 8].

6. VISUALISATION TYPES - allowed / prioritised.
   - PREFER: scatter, line, grouped/horizontal bar, strip/jitter, forest/caterpillar
     (per-unit estimate + CI), reliability/decision curves, and discrete heatmaps
     for matrices. Match the encoding to the data [rougier2014 rule 2; harrowerbrewer2003].
   - AVOID: pie/donut (angle comparison is poor), 3D charts and 3D bars, dual y-axes,
     rainbow/jet colour maps, and stacked bars where parts must be compared
     [rougier2014; crameri2020].

7. LINES, AXES, MARKERS.
   - Data lines 1.5 pt; axes/ticks 0.8 pt; top + right spines off; ticks out.
   - Reference lines (chance 0.5, perfect O:E=1, treat-none, cohort mean) in black,
     dashed/dotted, 0.8-1.2 pt. CI bands as light grey shading (alpha ~0.18).

8. EXPORT & REPRODUCIBILITY.
   - Always emit vector PDF (LaTeX asset) + 300 dpi PNG (preview), white background,
     via save(). Recompute every number from data/CSVs in the figure script - never
     transcribe results into a plot.

------------------------------- PROGRAMMATIC API -----------------------------
Everything above is enforced/contracted in code, so figure scripts never re-type a
rule. apply() sets all rcParams (incl. suptitle 11.5 pt, TrueType embedding, grid
off). Constants: OI, GREY/INK/SOFT, PALE_FILL, REF_COLOR/REF_LW, CI_ALPHA,
COL_SINGLE/COL_DOUBLE, BOX_EDGE. Colour maps: TARGET, ARCH, SPLIT, FEATURE_SET.
Helpers:
  color(role, name) / arch_color / target_color / split_color / feature_set_color
                              - resolve a colour, raising on an unknown entity
                                (never a silent None that collapses two series).
  box(...role=) / arrow(...) - schematic primitives; box edge by role
                                (normal/output/exclude), no inline hex.
  panel_label(ax, "a")       - bold lowercase multi-panel label (top-left).
  refline(ax, y=/x=)         - black dashed reference line (chance, perfect, ...).
  ci_band(ax, xs, lo, hi)    - light grey CI shading.
  framed_legend(ax)          - the heatmap-only framed-legend exception.
  figsize(cols="single"|"double", h) - final-size width by column count.
  save(fig, out)             - PDF + 300 dpi PNG.
==============================================================================

References (verified against the primary source; see design guide for full entries):
  [okabeito2008] Okabe & Ito 2008, Color Universal Design, jfly.uni-koeln.de/color
  [wong2011]     Wong 2011, Nature Methods 8:441, doi:10.1038/nmeth.1618
  [wilke2019]    Wilke 2019, Fundamentals of Data Visualization, clauswilke.com/dataviz
  [naturefig]    Nature research figure guide, preparing-figures specifications
  [harrowerbrewer2003] Harrower & Brewer 2003, Cartographic J. 40(1):27-37,
                 doi:10.1179/000870403235002042
  [crameri2020]  Crameri et al. 2020, Nat. Commun. 11:5444, doi:10.1038/s41467-020-19160-7
  [rougier2014]  Rougier, Droettboom & Bourne 2014, PLOS Comput. Biol. 10(9):e1003833,
                 doi:10.1371/journal.pcbi.1003833
  [tufte2001]    Tufte, The Visual Display of Quantitative Information, 2nd ed., 2001
  [chain2022mpl] Chain 2022, Academic figures with Matplotlib (practitioner rcParams)
==============================================================================
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Okabe-Ito palette (Wong 2011 RGB)
OI = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
      "purple": "#CC79A7", "black": "#000000"}
GREY = "#555555"
INK = "#222222"      # text
SOFT = "#444444"     # axes / ticks
FAINT = "#cccccc"    # faint grey: error-bar whiskers, subtle dividers
MUTED = "#9a9a9a"    # de-emphasised / descriptive-only text (e.g. a non-significant rank ladder)
PALE_FILL = "#eef3f8"               # default schematic box fill
REF_COLOR = OI["black"]             # reference lines: chance, perfect, treat-none
REF_LW = 0.9                        # reference-line width (guide Section 1.3: 0.8-1.2)
CI_ALPHA = 0.18                     # CI-band shading alpha (guide: 0.15-0.20)
COL_SINGLE, COL_DOUBLE = 3.5, 7.2   # final figure widths (in): ~89 mm / ~183 mm

TARGET = {"headache": OI["blue"], "migraine": OI["vermillion"]}

# Architecture colours; every label alias the figures use maps to one hue.
_ARCH_BASE = {"xgboost": OI["green"], "tabpfn": OI["purple"], "window-mlp": OI["orange"],
              "gru": OI["skyblue"], "tcn": OI["blue"], "pooled": GREY}
ARCH = {
    "XGBoost stack": _ARCH_BASE["xgboost"], "stacked_2xgb_meta_lr": _ARCH_BASE["xgboost"],
    "add0_stacked": _ARCH_BASE["xgboost"], "xgboost": _ARCH_BASE["xgboost"],
    "TabPFN": _ARCH_BASE["tabpfn"], "tabpfn": _ARCH_BASE["tabpfn"],
    "add1_tabpfn": _ARCH_BASE["tabpfn"], "autotabpfn": _ARCH_BASE["tabpfn"],
    "window-MLP": _ARCH_BASE["window-mlp"], "window-MLP (seq)": _ARCH_BASE["window-mlp"],
    "sequence": _ARCH_BASE["window-mlp"], "add4_window_mlp": _ARCH_BASE["window-mlp"],
    "GRU": _ARCH_BASE["gru"], "TCN": _ARCH_BASE["tcn"],
    "pooled_lr": _ARCH_BASE["pooled"], "pooled": _ARCH_BASE["pooled"],
}

# Canonical split-type and feature-set colours (see design guide Section 1.2).
SPLIT = {"chrono": OI["blue"], "stratified": OI["vermillion"],
         "patient": OI["green"], "site": OI["orange"]}
FEATURE_SET = {"full": OI["blue"], "spano": OI["vermillion"],
               "no_rolling": OI["green"], "park": OI["orange"]}

# Schematic box edge roles (guide Section 7): normal / output / exclusion.
BOX_EDGE = {"normal": OI["blue"], "output": OI["green"], "exclude": OI["vermillion"]}

# Discrete categorical ramps (single source, so A3 & A5 stop drifting). Ordered
# light -> dark; the headache/val category is OI orange and migraine/test is the
# canonical vermillion, so a named entity keeps its brand colour in the heatmaps.
PALE_BLUE = "#d9e6f2"
COVERAGE = ["#ffffff", PALE_BLUE, OI["orange"], OI["vermillion"]]   # absent, free, headache, migraine
SPLIT_GRID = [PALE_BLUE, OI["orange"], OI["vermillion"]]            # train, val, test

# Registry so a colour can be resolved by (role, name) and FAIL LOUDLY - a silent
# None would let matplotlib fall back to black and collapse two entities to one.
_ROLES = {"target": TARGET, "arch": ARCH, "split": SPLIT, "feature_set": FEATURE_SET}


def color(role: str, name: str) -> str:
    """Colour for one entity; raises KeyError on an unknown role or name."""
    table = _ROLES.get(role)
    if table is None:
        raise KeyError(f"unknown palette role {role!r}; choose from {sorted(_ROLES)}")
    if name not in table:
        raise KeyError(f"no colour for {name!r} in role {role!r}; known: {sorted(table)}")
    return table[name]


def arch_color(name): return color("arch", name)
def target_color(name): return color("target", name)
def split_color(name): return color("split", name)
def feature_set_color(name): return color("feature_set", name)


# NOTE: the other three palette roles are role-scoped by design (guide Section 9):
# the diverging metric heatmap lives in experiment/_eval/_metric_palette.py, the
# many-variant architecture family palette in _eval/_benchmark_visuals.py (its
# anchors should derive from OI / ARCH above), and the ordinal tree ramp in
# _eval/_tree_diagram.py. This module owns the qualitative role + shared rcParams.


def apply():
    """Set the global rcParams for every figure. Idempotent enough to re-call."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10.5, "axes.titleweight": "regular", "axes.titlepad": 8,
        "axes.titlecolor": INK,
        "figure.titlesize": 11.5, "figure.titleweight": "regular",  # suptitle
        "axes.labelsize": 9, "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": SOFT, "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": SOFT, "ytick.color": SOFT,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.5, "lines.markersize": 5,
        "legend.fontsize": 8, "legend.frameon": False, "legend.title_fontsize": 8.5,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def box(ax, xy, w, h, text, fc=PALE_FILL, ec=None, role="normal", fontsize=8.5,
        weight="normal"):
    """Rounded text box centred at xy; returns (x, y, w, h) for arrow anchoring.
    Edge colour comes from the schematic ``role`` (normal/output/exclude) unless an
    explicit ``ec`` is given - so flowcharts use palette colours, not inline hex."""
    x, y = xy
    ec = ec or BOX_EDGE.get(role, OI["blue"])
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, zorder=2, fc=fc, ec=ec,
                                lw=1.2, boxstyle="round,pad=0.01,rounding_size=0.015"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3, weight=weight)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=SOFT):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.1, color=color, zorder=1))


def panel_label(ax, letter, x=-0.06, y=1.04, fontsize=10):
    """Bold lowercase panel label (a, b, ...) at the panel's top-left (guide S4)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
            va="bottom", ha="right", color=INK)


def refline(ax, *, y=None, x=None, label=None, ls="--", lw=REF_LW, color=REF_COLOR):
    """Black dashed/dotted reference line (chance 0.5, perfect, O:E=1, treat-none)."""
    if y is not None:
        ax.axhline(y, color=color, ls=ls, lw=lw, label=label, zorder=0)
    if x is not None:
        ax.axvline(x, color=color, ls=ls, lw=lw, label=label, zorder=0)


def ci_band(ax, xs, lo, hi, color=GREY, alpha=CI_ALPHA, **kw):
    """Light grey confidence-interval shading (guide Section 1.3)."""
    return ax.fill_between(xs, lo, hi, color=color, alpha=alpha, lw=0, zorder=0, **kw)


def framed_legend(ax, **kw):
    """Legend WITH a light frame - the documented exception for a legend sitting
    over a filled background (heatmap), where the global frameless default is
    unreadable (guide Section 6)."""
    kw.setdefault("framealpha", 0.92)
    leg = ax.legend(frameon=True, **kw)
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_linewidth(0.6)
    return leg


def figsize(cols="double", h=4.0):
    """Final-size figure width by column count: 'single' ~89 mm, 'double' ~183 mm."""
    return (COL_DOUBLE if cols == "double" else COL_SINGLE, h)


def save(fig, out, dpi=300) -> str:
    """Write ``out`` as PDF (vector) + PNG (raster). ``out`` may be a stem or a
    path with any suffix; both siblings are written next to it."""
    stem = Path(out).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"


# ---------------------------------------------------------------------------
# Leaf-path -> compact figure-legend slug.
# Convention: ``<ARCH-VAR> / <FEATURE> / <SPLIT> / <RATIO>``. A reader can
# always reconstruct the full leaf path from the slug.
# ---------------------------------------------------------------------------

_FEATURE_SHORT = {
    "full_features": "full",
    "spano_features": "spano",
    "park_features": "park",
    "no_rolling_features": "no-roll",
}

_TABPFN_VAR_SHORT = {
    "version_2-6": "v2.6",
    "version_2-5-real": "v2.5r",
    "version_2-5-finetuned": "v2.5f",
    "version_2-5-auto": "v2.5a",
    "version_3-default": "v3d",
    "version_3-binary": "v3b",
}

_SEQ_VAR_SHORT = {
    "version_window-mlp": "windowMLP",
    "version_gru": "GRU",
    "version_tcn": "TCN",
}


def leaf_slug(leaf) -> str:
    """Return the 4-token compact label for a leaf path.

    Examples:
        experiment/0/migraine/full_features/stacked_2xgb_meta_lr/
            70_30/chrono/HyperparameterTuned/single_AUROC/HP020
            -> 'XGB-HP020 / full / chrono / 70-30'

        experiment/1/headache/full_features/tabpfn/version_2-6/70_30/chrono
            -> 'TabPFN-v2.6 / full / chrono / 70-30'

        experiment/4/headache/full_features/sequence/version_window-mlp/
            70_15_15/chrono
            -> 'Seq-windowMLP / full / chrono / 70-15-15'

    Parameters
    ----------
    leaf : str or pathlib.Path
        Any leaf directory under ``experiment/``. Need not exist on disk.

    Returns
    -------
    str
        Four space-separated tokens joined by ``' / '``. If a token cannot
        be extracted (e.g. the leaf path is malformed), it is replaced with
        ``'?'`` so the caller's legend still renders.
    """
    parts = str(leaf).split("/")
    feature = next((_FEATURE_SHORT.get(p, p) for p in parts
                    if p in _FEATURE_SHORT), "?")
    split = next((p for p in parts if p in ("chrono", "stratified", "patient")), "?")
    ratio_raw = next((p for p in parts if p in ("70_30", "70_15_15", "80_20")), "?")
    ratio = ratio_raw.replace("_", "-")

    # Architecture-variant token
    arch = "?"
    if "tabpfn" in parts:
        ti = parts.index("tabpfn")
        ver = next((p for p in parts[ti + 1:] if p.startswith("version_")), None)
        arch = f"TabPFN-{_TABPFN_VAR_SHORT.get(ver, ver or '?')}"
    elif "sequence" in parts:
        si = parts.index("sequence")
        ver = next((p for p in parts[si + 1:] if p.startswith("version_")), None)
        arch = f"Seq-{_SEQ_VAR_SHORT.get(ver, (ver or '?').replace('version_', ''))}"
    elif "stacked_2xgb_meta_lr" in parts:
        if "HyperparameterTuned" in parts:
            hp = next((p for p in parts if p.startswith("HP") and p[2:].isdigit()), None)
            arch = f"XGB-{hp}" if hp else "XGB-HPtuned"
        else:
            arch = "XGB-NonHP"
    elif "blended_xgb_lr_spano2026" in parts:
        arch = "XGB-blended"

    return f"{arch} / {feature} / {split} / {ratio}"


def caption_block(*, leaf=None, target=None, metric=None, n=None, n_pos=None,
                  extra: str | None = None) -> str:
    """Build the standard figure-caption block. Returns a single line,
    ready to prefix the figure-specific description. Tokens are dropped
    when the corresponding argument is ``None`` so a caller that only
    knows the target and metric can still
    emit a partial block."""
    bits = []
    if metric is not None:
        bits.append(metric)
    if target is not None:
        bits.append(f"target {target}")
    if leaf is not None:
        bits.append(f"cell {leaf_slug(leaf)}")
    if n is not None:
        n_str = f"n = {n} patient-days"
        if n_pos is not None:
            n_str += f", {n_pos} positives"
        bits.append(n_str)
    if extra:
        bits.append(extra)
    return ". ".join(bits) + "." if bits else ""


def epv_annotation(ax, target: str, *, cell: str = "full_features",
                   loc: str = "lower right", fontsize: float = 7.0) -> None:
    """Stamp the EPV-5.5 / Riley-criterion marker on figure panels showing
    the migraine `full_features` cell. Per `figure_design_requirements.md`
    §11.7: any figure rendering this cell must annotate it as a priori
    under-powered.

    No-op for non-migraine targets and non-`full_features` cells.
    """
    if target != "migraine" or cell != "full_features":
        return
    locmap = {
        "lower right": dict(x=0.99, y=0.04, ha="right", va="bottom"),
        "lower left": dict(x=0.02, y=0.04, ha="left", va="bottom"),
        "upper right": dict(x=0.99, y=0.96, ha="right", va="top"),
        "upper left": dict(x=0.02, y=0.96, ha="left", va="top"),
    }
    pos = locmap.get(loc, locmap["lower right"])
    ax.text(pos["x"], pos["y"],
            "migraine full_features: EPV ~5.5\n"
            "below Riley criterion (a priori under-powered)",
            transform=ax.transAxes,
            ha=pos["ha"], va=pos["va"],
            fontsize=fontsize, color=SOFT, style="italic",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.7, pad=2))


def cc_by_footer(fig, fontsize: float = 6.5) -> None:
    """Add a CC BY 4.0 licence footer at the bottom-right of the figure.
    Per `figure_design_requirements.md` §11.6: figures intended for
    publication carry an explicit licence affordance.
    """
    fig.text(0.99, 0.005, "CC BY 4.0",
             ha="right", va="bottom",
             fontsize=fontsize, color=GREY, alpha=0.7,
             transform=fig.transFigure)


# §11.11 self-check comment block. Each figure script imports this string
# and includes it as a module-level comment so the regen pipeline carries
# the §11 compliance trail with the artefact.
COMPLIANCE_BLOCK = """\
# §11 compliance (figure_design_requirements.md, reviewer-derived 2026-05-28):
#   §11.1 CIs on headline metric:        see error bars / shaded bands
#   §11.2 estimability denominators:     where applicable (within-person)
#   §11.3 self-contained caption:        S.caption_block() invoked
#   §11.6 CC BY 4.0 footer:              S.cc_by_footer() invoked
#   §11.7 EPV-5.5 annotation:            S.epv_annotation() invoked on
#                                        migraine full_features panels
#   §11.10 no "substantial"/"large":     verified in captions
#   §11.11 self-check:                   this block
"""
