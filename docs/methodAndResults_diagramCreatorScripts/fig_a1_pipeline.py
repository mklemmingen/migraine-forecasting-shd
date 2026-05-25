"""Figure A1 - benchmark pipeline schematic.

The linear path from the open diary to the bootstrap evaluation, with the
post-hoc analysis and validation layers (Additions 2/3/5/6 and the
leave-one-site-out external validation) drawing on the fitted models and their
predictions. Schematic only - no data.

Usage: python fig_a1_pipeline.py
"""
from pathlib import Path

import _figstyle as S
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

STAGES = [
    "SHD diary\n63 patients\n4,516 days",
    "Feature\nengineering\n4 feature sets",
    "Evaluation\ndesign\n3 splits x 3 ratios",
    "Architectures\nAdd 0 / 1 / 4\nXGBoost - TabPFN - seq",
    "Bootstrap eval\nAUROC / AUPRC\n+ calibration (CIs)",
]


def main():
    S.apply()
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 5); ax.axis("off")
    xs = [1.0, 3.1, 5.2, 7.3, 9.4]
    boxes = []
    for x, txt in zip(xs, STAGES):
        fc = "#e8eef5" if x != xs[-1] else "#eef5ee"
        boxes.append(S.box(ax, (x, 3.7), 1.85, 1.5, txt, fc=fc))
    for a, b in zip(boxes, boxes[1:]):
        S.arrow(ax, (a[0] + a[2] / 2, a[1]), (b[0] - b[2] / 2, b[1]))
    band = S.box(ax, (5.2, 1.2), 9.0, 1.7,
                 "Analysis & validation layers\n"
                 "Add 2 explainability  -  Add 3 temporal dependence  -  Add 5 within-person\n"
                 "Add 6 clinical value  -  leave-one-site-out external validation",
                 fc="#f6f1e7", ec="#b8860b", fontsize=8.5)
    S.arrow(ax, (boxes[3][0], boxes[3][1] - boxes[3][3] / 2), (4.2, band[1] + band[3] / 2))
    S.arrow(ax, (boxes[4][0], boxes[4][1] - boxes[4][3] / 2), (6.2, band[1] + band[3] / 2))
    ax.set_title("Benchmark pipeline", fontsize=12)
    print("saved", S.save(fig, HERE / "figures" / "fig_a1_pipeline"))


if __name__ == "__main__":
    main()
