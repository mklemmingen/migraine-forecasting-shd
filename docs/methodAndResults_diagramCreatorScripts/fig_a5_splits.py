"""Figure A5 - split-strategy schematic.

How patient-days (rows = patients, columns = days) are assigned to train/val/test
under the four split strategies. Chronological cuts on time; stratified scatters
days randomly - placing a patient's neighbouring days on both sides of the
boundary, which is the history-feature leakage channel quantified in Figure C1;
patient holds out whole patients; site holds out a whole recruitment site
(leave-one-site-out, validation carved from the training site). Illustrative grid.

Usage: python fig_a5_splits.py
"""
from pathlib import Path

import _figstyle as S
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
N_PT, N_DAY = 6, 12
CMAP = ListedColormap(["#d9e6f2", "#fdae6b", "#b2182b"])   # train, val, test
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], CMAP.N)


def _grids():
    rng = np.random.default_rng(1)
    chrono = np.zeros((N_PT, N_DAY), int)
    chrono[:, 8:10] = 1; chrono[:, 10:] = 2            # time cut, same for all patients
    strat = rng.choice([0, 1, 2], size=(N_PT, N_DAY), p=[0.7, 0.15, 0.15])  # scattered
    patient = np.zeros((N_PT, N_DAY), int)
    patient[4] = 1; patient[5] = 2                      # whole patients held out
    site = np.zeros((N_PT, N_DAY), int)
    site[3:] = 2; site[2, 10:] = 1                      # site B = test; val carved from site A
    return {"chronological": chrono, "stratified (leaky)": strat,
            "patient hold-out": patient, "site hold-out": site}


def main():
    S.apply()
    grids = _grids()
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (name, g) in zip(axes, grids.items()):
        ax.imshow(g, cmap=CMAP, norm=NORM, aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("days ->", fontsize=8)
        for i in range(N_PT + 1):
            ax.axhline(i - 0.5, color="white", lw=0.6)
        for j in range(N_DAY + 1):
            ax.axvline(j - 0.5, color="white", lw=0.6)
    axes[0].set_ylabel("patients", fontsize=8)
    fig.legend(handles=[Patch(fc="#d9e6f2", label="train"),
                        Patch(fc="#fdae6b", label="val"),
                        Patch(fc="#b2182b", label="test")],
               loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Split strategies over patient-days", y=1.02, fontsize=12)
    print("saved", S.save(fig, HERE / "figures" / "fig_a5_splits"))


if __name__ == "__main__":
    main()
