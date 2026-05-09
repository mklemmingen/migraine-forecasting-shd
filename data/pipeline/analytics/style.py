"""Shared matplotlib/seaborn configuration and palette for analytics reports."""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Okabe-Ito colour-blind-safe palette
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7"]
POS_COLOR = "#D55E00"   # positive class (migraine event)
NEG_COLOR = "#56B4E9"   # negative class


def apply_theme() -> None:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    sns.set_theme(style="whitegrid", context="paper")


def fig_full(w: float = 10, h: float = 7):
    return plt.subplots(figsize=(w, h))


def fig_small_multiples(w: float = 12, h: float = 9):
    return plt.subplots(figsize=(w, h))
