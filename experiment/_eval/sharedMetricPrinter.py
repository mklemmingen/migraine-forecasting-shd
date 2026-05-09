"""
Defines a metric print to txt contract (how the result.txt look inside) as a two-way.

getContract() can be called to get a line by line view of the strings expected before the numeric values.
-> as a eval script, contract should be kept with header, line strings and footer
-> as the result aggregator, the contract should be used to dynamically find the metric numeric values to use
for any kind of html visualisation of results.

getContract_cv() can be also called to get a line by line view for the more tabular cross validation.

The Contracts should be updated if they do happen to change. document in readme under metrics first, then here,
then in all evals depending.

fyi: the aggregate result runner will fill out N/A under values not found, and will give any user running it a red text
in their terminal where supported.
"""
from typing import NamedTuple

SEPARATOR = "=" * 60
DIVIDER   = "-" * 60
COL_WIDTH = 25  # f"{name:<25}" used by all eval scripts

# Metric names exactly as written in their respective file types.
# Note: MCC label differs between hold-out ("Optimal") and CV ("Cal-Optimal").
METRICS_HOLDOUT = [
    "AUROC",
    "AUPRC",
    "Brier Score",
    "ECE10",
    "MCC (Optimal)",
    "Sensitivity (>=0.5)",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
]

METRICS_CV = [
    "AUROC",
    "AUPRC",
    "Brier Score",
    "ECE10",
    "MCC (Cal-Optimal)",
    "Sensitivity (>=0.5)",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
]


class ContractLine(NamedTuple):
    key:        str   # logical key: metric name or structural role
    prefix:     str   # exact prefix string to match in result files
    structural: bool = False  # True = separator/header; no value to extract


def _getHeaderContract():
    """Lines for the hold-out (70/15/15) result file header."""
    return [
        ContractLine("sep_open",       SEPARATOR,                                    structural=True),
        ContractLine("threshold_mcc",  " -> MCC-Optimal Threshold:          "),
        ContractLine("threshold_sens", " -> Threshold for Sens >= 0.50:     "),
        ContractLine("divider",        DIVIDER,                                       structural=True),
        ContractLine("col_header",     f"{'Metric':<{COL_WIDTH}} | Mean [95% CI]",  structural=True),
        ContractLine("divider",        DIVIDER,                                       structural=True),
    ]


def _getHeaderContract_cv():
    """Lines for the cross-validation result file header."""
    col_header = (
        f"{'Metric':<{COL_WIDTH}} | "
        "F1      F2      F3      F4      F5     | "
        "Mean     Std    "
    )
    return [
        ContractLine("sep_open",   SEPARATOR,  structural=True),
        ContractLine("divider",    DIVIDER,    structural=True),
        ContractLine("col_header", col_header, structural=True),
        ContractLine("divider",    DIVIDER,    structural=True),
    ]


def _getFooterContract():
    return [ContractLine("sep_close", SEPARATOR, structural=True)]


def _getMetricsContract():
    return [
        ContractLine(name, f"{name:<{COL_WIDTH}} | ")
        for name in METRICS_HOLDOUT
    ]


def _getMetricsContract_CV():
    return [
        ContractLine(name, f"{name:<{COL_WIDTH}} | ")
        for name in METRICS_CV
    ]


def getContract():
    """Full contract for hold-out (70/15/15) result files."""
    return _getHeaderContract() + _getMetricsContract() + _getFooterContract()


def getContract_cv():
    """Full contract for cross-validation result files."""
    return _getHeaderContract_cv() + _getMetricsContract_CV() + _getFooterContract()
