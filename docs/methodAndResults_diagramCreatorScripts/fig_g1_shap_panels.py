"""Figure 7 (working-notes ID: G1) - SHAP beeswarm panels for both targets.

Body-facing wrapper that combines the per-target SHAP beeswarm renderings
emitted by fig_h2_shap_beeswarm.py (which in turn uses the shared
experiment/_explain/_plots.plot_beeswarm primitive) into a single side-by-side
two-panel figure for the body Figure 7 slot. The per-target PNGs are reused
as-is via PIL so the underlying _explain._plots.plot_beeswarm contract stays
unchanged for the leaf-level insight pass.

Content: panel a (headache headline cell, TabPFN-v2.6 / full / chrono / 70-30)
left, panel b (migraine headline cell, XGB-HP020 / full / chrono / 70-30)
right. Each panel is a per-row SHAP beeswarm with points coloured by feature
value (low blue, high vermillion), the standard SHAP encoding showing direction
and magnitude of each feature's effect on next-day positive-class probability.

The body §3.3 claim that history features carry 77 to 85% of mean absolute
attribution on the full_features cells is visually evident in both panels as
the top rows (migraine_rate_last3 / last7, headache_free_streak,
days_since_last_migraine, migraine_yesterday) for headache and the parallel
top rows (headache_free_streak, days_since_last_migraine,
consecutive_sedentary_days, sleep_variability_7day) for migraine.

Usage: python fig_g1_shap_panels.py
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"

PANEL_A = FIGS / "fig_h2_shap_beeswarm_headache.png"
PANEL_B = FIGS / "fig_h2_shap_beeswarm_migraine.png"
OUT = FIGS / "fig_g1_shap_panels.png"


def main() -> None:
    if not PANEL_A.exists() or not PANEL_B.exists():
        raise SystemExit(
            f"per-target SHAP beeswarm PNGs not on disk; run "
            f"fig_h2_shap_beeswarm.py first to regenerate "
            f"({PANEL_A.name}, {PANEL_B.name})"
        )

    im_a = Image.open(PANEL_A).convert("RGB")
    im_b = Image.open(PANEL_B).convert("RGB")

    # Two-row layout: normalise the panels to a common width so the vertical
    # stack does not gap on one side. Both per-target beeswarms render at
    # 2387 x 1804 px from the shared _plots.plot_beeswarm primitive; if either
    # drifts, the narrower one is scaled up to the wider one's width to keep
    # the row widths flush.
    w = max(im_a.width, im_b.width)
    if im_a.width != w:
        im_a = im_a.resize(
            (w, int(im_a.height * w / im_a.width)), Image.LANCZOS,
        )
    if im_b.width != w:
        im_b = im_b.resize(
            (w, int(im_b.height * w / im_b.width)), Image.LANCZOS,
        )

    combined = Image.new("RGB", (w, im_a.height + im_b.height), "white")
    combined.paste(im_a, (0, 0))
    combined.paste(im_b, (0, im_a.height))
    combined.save(OUT, optimize=True)

    import os
    size_kb = os.path.getsize(OUT) / 1024.0
    print(f"saved {OUT.name}  {combined.width}x{combined.height} px, "
          f"{size_kb:.1f} KB  (panel a = {PANEL_A.name}, panel b = {PANEL_B.name})")


if __name__ == "__main__":
    main()
