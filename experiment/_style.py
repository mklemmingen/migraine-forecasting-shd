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
Everything a journal-quality diagram in this repo must follow. Full rationale and
verified sources: docs/figure_design_requirements.md. References [bracketed] are
listed at the bottom of this docstring.

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

TARGET = {"headache": OI["blue"], "migraine": OI["vermillion"]}

# Architecture colours; every label alias the figures use maps to one hue.
_ARCH_BASE = {"xgboost": OI["green"], "tabpfn": OI["purple"], "window-mlp": OI["orange"],
              "gru": OI["skyblue"], "tcn": OI["blue"], "pooled": GREY}
ARCH = {
    "XGBoost stack": _ARCH_BASE["xgboost"], "stacked_2xgb_meta_lr": _ARCH_BASE["xgboost"],
    "add0_stacked": _ARCH_BASE["xgboost"],
    "TabPFN": _ARCH_BASE["tabpfn"], "tabpfn": _ARCH_BASE["tabpfn"],
    "add1_tabpfn": _ARCH_BASE["tabpfn"],
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


def apply():
    """Set the global rcParams for every figure. Idempotent enough to re-call."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10.5, "axes.titleweight": "regular", "axes.titlepad": 8,
        "axes.titlecolor": INK,
        "axes.labelsize": 9, "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": SOFT, "axes.linewidth": 0.8,
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


def box(ax, xy, w, h, text, fc="#eef3f8", ec=OI["blue"], fontsize=8.5, weight="normal"):
    """Rounded text box centred at xy; returns (x, y, w, h) for arrow anchoring."""
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, zorder=2, fc=fc, ec=ec,
                                lw=1.2, boxstyle="round,pad=0.01,rounding_size=0.015"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3, weight=weight)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=SOFT):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.1, color=color, zorder=1))


def save(fig, out, dpi=300) -> str:
    """Write ``out`` as PDF (vector) + PNG (raster). ``out`` may be a stem or a
    path with any suffix; both siblings are written next to it."""
    stem = Path(out).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"
