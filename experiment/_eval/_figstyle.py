"""Shared matplotlib styling + saver for journal-quality figure output.

Targets a peer-reviewed journal (Q3-Q4): every figure is emitted both as
a resolution-independent vector PDF (the typesetter's submission asset)
and as a 300-DPI PNG (for the HTML preview and as a raster fallback).

Journal-readiness choices encoded here:
  - ``pdf.fonttype = 42`` / ``ps.fonttype = 42`` embed TrueType fonts
    rather than Type-3 outlines; Type-3 is a frequent submission
    rejection reason and is not editable by typesetters.
  - ``svg.fonttype = 'none'`` keeps text as text in any SVG export.
  - A single sans-serif family and a fixed set of legible point sizes
    keep typography consistent across all figures.
  - Colour is always reinforced by line style and marker shape in the
    benchmark figures (set in _benchmark_visuals), so figures remain
    interpretable in grayscale and for colour-vision deficiency.
"""
from pathlib import Path

import matplotlib

_APPLIED = False


def apply_journal_style():
    """Set global rcParams for journal-quality output. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    _APPLIED = True


def save_journal_figure(fig, png_path, dpi=300):
    """Save ``fig`` as a high-DPI PNG and a vector PDF sibling.

    The PNG (``png_path``) is what the comparison HTML embeds; the PDF
    (same stem, ``.pdf``) is the resolution-independent journal asset.
    Both use a tight bounding box so there is no surplus whitespace
    margin in the submitted figure.
    """
    png_path = Path(png_path)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
