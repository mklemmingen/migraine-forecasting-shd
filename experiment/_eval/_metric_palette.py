"""Metric specifications + ColorBrewer RdBu palette for the
comparison-table HTML.

Each entry in ``COMPARISON_METRICS`` describes one metric column: how
the heatmap should colour it and what counts as a "bad" or "good"
value. ``compute_color`` dispatches on the spec kind to one of four
coloring strategies (max / min / diverge / target).

The palette is ColorBrewer 7-class RdBu, a perceptually balanced
red-white-blue divergent scheme that is the de-facto standard in
scientific cartography for data with a meaningful midpoint (Brewer
1994, 1997; Harrower & Brewer 2003). RdBu is preferred over the
red-green family because:
  1. Red-green palettes fail for the most common forms of colour-vision
     deficiency (deuteranopia, protanopia: ~8% of male readers).
     Crameri et al. 2020 [Nat Commun] explicitly argue against them in
     scientific communication.
  2. Green at high lightness washes out to near-white faster than blue
     does, so cells with similar "good" scores look identical in a
     red-green ramp but distinguishable in a red-blue ramp.

The "warm = bad, cool = good" mapping is preserved.
"""
from typing import NamedTuple, Optional


class MetricSpec(NamedTuple):
    """Color spec for one metric column in the comparison table.

    ``kind`` dispatches to one of four coloring strategies:

    - ``"max"``  : sequential ramp; ``bad`` is treated as fully red,
                   ``good`` as fully teal. Use for metrics whose optimum
                   is the upper anchor and whose lower anchor is
                   "uninformative" (e.g. AUPRC near prevalence).
    - ``"min"``  : sequential ramp with ``bad > good`` (e.g. Brier,
                   ECE10). Same palette, anchors reversed.
    - ``"diverge"``: ``pivot`` is the random-performance value;
                     ``val < pivot`` uses a red-from-white ramp scaled
                     by ``(pivot - val) / (pivot - bad)``, and
                     ``val > pivot`` uses a teal-from-white ramp scaled
                     by ``(val - pivot) / (good - pivot)``. Use for MCC
                     (pivot = 0) and AUROC (pivot = 0.5) so that
                     anti-predictive values (negative MCC, AUROC < 0.5)
                     are visually distinct from random performance.
    - ``"target"``: ``pivot`` is the optimum and the colour is driven
                    by ``|val - pivot|``. ``bad`` is the deviation at
                    which the cell is treated as fully red. Use for
                    CalSlope (pivot = 1.0) where both under- and
                    over-confident miscalibration are equally bad.
    """
    key: str
    label: str
    kind: str
    bad: float
    good: float
    pivot: Optional[float] = None
    cv_alias: Optional[str] = None   # alternate dict key for CV-source results


# All metrics shown in cells. The CV-source MCC is keyed
# "MCC (Cal-Optimal)" instead of "MCC (Optimal)"; the cv_alias field
# lets the per-cell renderer look up the right key when the cell's
# source is CV rather than holdout.
COMPARISON_METRICS = [
    # Discrimination -------------------------------------------------------
    # AUROC: 0.5 = random. Negative-information (Platt-inverted) cells
    # appear as AUROC < 0.5; coloring them with a divergent ramp around
    # the 0.5 pivot lets a reader see "anti-predictive" cells without
    # having to read the digit.
    MetricSpec("AUROC",                "AUROC",       "diverge", bad=0.30, good=0.85, pivot=0.50),
    # AUPRC: baseline = prevalence; on this dataset prevalence ranges
    # from ~5% (migraine) to ~30% (headache). Cross-target comparison is
    # not meaningful; the sequential ramp here is informative for
    # within-target comparison only (see _METRIC_NOTES for the caveat).
    MetricSpec("AUPRC",                "AUPRC",       "max",     bad=0.05, good=0.60),

    # Calibration ----------------------------------------------------------
    # Brier baseline at 5% prevalence is ~0.0475; at 30% it is ~0.21.
    # The 0.25 ceiling captures the worst-case among realistic models.
    MetricSpec("Brier Score",          "Brier ↓",     "min",     bad=0.25, good=0.00),
    # ECE10 below 0.05 is considered well-calibrated; 0.15 is
    # substantial miscalibration.
    MetricSpec("ECE10",                "ECE10 ↓",     "min",     bad=0.15, good=0.00),
    # CalSlope: 1.0 = perfect, <1 = over-confident, >1 = under-confident,
    # sign-flip = Platt inversion. ``target`` kind treats both directions
    # symmetrically; bad=1.0 means |val-1|=1 saturates the red end so
    # CalSlope of 0.0 or 2.0 are equally red, CalSlope of 1.0 is teal.
    # Values outside [0, 2] clamp to red rather than wrapping back.
    MetricSpec("Calibration Slope",    "CalSlope",    "target",  bad=1.00, good=0.00, pivot=1.00),

    # Threshold-derived ----------------------------------------------------
    # MCC pivot = 0 (random); negative MCC indicates inverted prediction
    # and should be visually distinct from MCC near zero. Practical
    # ceiling at ~0.5 keeps the palette stretched across the realistic
    # range rather than reserving deep teal for unreachable scores.
    MetricSpec("MCC (Optimal)",        "MCC",         "diverge",
               bad=-0.30, good=0.50, pivot=0.00, cv_alias="MCC (Cal-Optimal)"),
    MetricSpec("Sensitivity (>=0.5)",  "Sens≥0.5",    "max",     bad=0.20, good=1.00),
    # Accuracy: base-rate-dominated on imbalanced targets. On migraine
    # (~5% positive) a constant-negative predictor reaches ~95%, so the
    # color is misleading without context; the "(base!)" label and the
    # _METRIC_NOTES caveat flag this. Use MCC/AUPRC/Brier as headlines.
    MetricSpec("Accuracy",             "Acc (base!)", "max",     bad=0.70, good=1.00),
    MetricSpec("Precision",            "Prec",        "max",     bad=0.00, good=0.60),
    MetricSpec("Recall",               "Recall",      "max",     bad=0.00, good=0.80),
    MetricSpec("F1",                   "F1",          "max",     bad=0.00, good=0.60),
]


# Hex codes from colorbrewer2.org, 7-class RdBu, ordered worst -> best.
_RDBU_7 = (
    "#b2182b",   # deep red       (worst / most anti-predictive)
    "#ef8a62",   # salmon
    "#fddbc7",   # peach
    "#f7f7f7",   # near-white     (neutral / random performance / target)
    "#d1e5f0",   # pale blue
    "#67a9cf",   # light blue
    "#2166ac",   # deep blue      (best)
)

# t -> hex stop position for divergent metrics: full RdBu spread over
# [-1, +1] with the neutral white anchored at 0.
_DIVERGENT_PALETTE = (
    (-1.00, _RDBU_7[0]),
    (-0.66, _RDBU_7[1]),
    (-0.33, _RDBU_7[2]),
    ( 0.00, _RDBU_7[3]),
    ( 0.33, _RDBU_7[4]),
    ( 0.66, _RDBU_7[5]),
    ( 1.00, _RDBU_7[6]),
)

# t -> hex stop position for sequential metrics (max/min/target): same
# RdBu palette spread over [0, 1] so the visual encoding is consistent
# across all metric kinds in one table.
_SEQUENTIAL_PALETTE = tuple(
    (i / (len(_RDBU_7) - 1), hx) for i, hx in enumerate(_RDBU_7)
)


def _hex_to_rgb(hx):
    """Parse a ``#rrggbb`` string into a (r, g, b) triple of ints."""
    h = hx.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp_palette(t, stops):
    """Piecewise-linear RGB interpolation across ``stops``.

    ``stops`` is a sequence of ``(t, hex)`` tuples sorted by ``t``.
    Values of ``t`` outside the stop range clamp to the nearest
    endpoint. Interpolation is in sRGB; this is acceptable here because
    the ColorBrewer RdBu stops are pre-balanced for perceptual
    uniformity so adjacent-stop interpolation does not drift through
    perceptually non-uniform regions.
    """
    if t <= stops[0][0]:
        return stops[0][1]
    if t >= stops[-1][0]:
        return stops[-1][1]
    for (t0, hx0), (t1, hx1) in zip(stops, stops[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0)
            r0, g0, b0 = _hex_to_rgb(hx0)
            r1, g1, b1 = _hex_to_rgb(hx1)
            r = round(r0 + u * (r1 - r0))
            g = round(g0 + u * (g1 - g0))
            b = round(b0 + u * (b1 - b0))
            return f"#{r:02x}{g:02x}{b:02x}"
    return stops[-1][1]


def _parse_metric_value(val_str):
    """Return the leading float of a metric string, or None if absent."""
    if val_str is None:
        return None
    try:
        return float(str(val_str).split()[0])
    except (ValueError, IndexError):
        return None


def compute_color(val_str, spec):
    """Map a metric string (e.g. '0.519 [...]') to a hex background.

    Dispatches on ``spec.kind`` to the appropriate ramp. Returns a
    light grey for missing values and white for unparseable strings so
    that "no result" is visually distinct from "result outside range".
    """
    val = _parse_metric_value(val_str)
    if val is None:
        return "#e4e4e4" if val_str is None else "#ffffff"

    if spec.kind == "max":
        denom = spec.good - spec.bad
        t = 0.5 if denom == 0 else (val - spec.bad) / denom
        return _lerp_palette(max(0.0, min(1.0, t)), _SEQUENTIAL_PALETTE)

    if spec.kind == "min":
        # bad > good for min-is-best metrics; same palette, anchors swapped.
        denom = spec.good - spec.bad
        t = 0.5 if denom == 0 else (val - spec.bad) / denom
        return _lerp_palette(max(0.0, min(1.0, t)), _SEQUENTIAL_PALETTE)

    if spec.kind == "diverge":
        pivot = spec.pivot
        if val >= pivot:
            denom = spec.good - pivot
            u = 0.0 if denom == 0 else (val - pivot) / denom
        else:
            denom = pivot - spec.bad
            u = 0.0 if denom == 0 else -(pivot - val) / denom
        return _lerp_palette(max(-1.0, min(1.0, u)), _DIVERGENT_PALETTE)

    if spec.kind == "target":
        # bad encodes max acceptable |val - pivot| (saturates at red).
        dist = abs(val - spec.pivot)
        t = 1.0 - (dist / spec.bad if spec.bad > 0 else 0.0)
        return _lerp_palette(max(0.0, min(1.0, t)), _SEQUENTIAL_PALETTE)

    return "#ffffff"


def parse_mean_ci(val_str):
    """Parse the per-leaf bootstrap output ``"mean [lo - hi]"``.

    Returns ``(mean, lo, hi)`` as floats, or ``None`` if the string is
    missing or unparseable. The evaluator templates write the CI as
    1000-iteration bootstrap 2.5/97.5 percentiles, so the bracketed
    range is a 95% CI. For CV-source values the same format is used;
    for the singleton cells (e.g. accuracy that is not bootstrapped)
    the bracket may be absent and only the mean is returned.
    """
    if val_str is None:
        return None
    s = str(val_str)
    try:
        mean = float(s.split()[0])
    except (ValueError, IndexError):
        return None
    lo = hi = None
    lb = s.find("[")
    rb = s.find("]")
    if lb != -1 and rb != -1 and rb > lb:
        inner = s[lb + 1:rb]
        for sep in (" - ", ", ", " to "):
            if sep in inner:
                parts = inner.split(sep)
                if len(parts) == 2:
                    try:
                        lo = float(parts[0].strip())
                        hi = float(parts[1].strip())
                    except ValueError:
                        lo = hi = None
                    break
    return mean, lo, hi
