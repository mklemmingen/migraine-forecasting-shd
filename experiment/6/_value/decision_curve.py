"""Decision curve analysis - net benefit vs threshold probability.

Net benefit weights true positives against false positives by the odds at the
threshold probability, where the threshold encodes the relative harm of a false
positive versus a false negative for the action under consideration
[vickers2006dca, p. 565; p. 567; vickers2019dca, p. 1]. For next-day migraine
the action is pre-emptive acute medication, whose harm trade-off (an unnecessary
dose vs a missed early-treatment window) the threshold parameterises - so the
clinically plausible band is low thresholds.
"""
import numpy as np

# Clinically plausible threshold band for the pre-emptive-medication action: a
# missed attack is costlier than an unnecessary dose, so low thresholds.
# Decision (docs Section 3.1); reported as a curve over this range, not one cut.
DEFAULT_THRESHOLDS = np.linspace(0.01, 0.50, 50)


def net_benefit(y_true, y_prob, thresholds=DEFAULT_THRESHOLDS):
    """Net benefit of the model across threshold probabilities.

    NB(pt) = TP/n - FP/n * pt/(1-pt)   [vickers2006dca, p. 567]
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    out = np.empty(len(thresholds))
    for i, pt in enumerate(thresholds):
        pred = p >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        out[i] = tp / n - (fp / n) * (pt / (1.0 - pt))
    return out


def treat_all_net_benefit(y_true, thresholds=DEFAULT_THRESHOLDS):
    """Net benefit of the 'treat everyone' default policy."""
    prev = float(np.mean(y_true))
    return prev - (1.0 - prev) * (thresholds / (1.0 - thresholds))


def decision_curve(y_true, y_prob, thresholds=DEFAULT_THRESHOLDS) -> dict:
    """Model, treat-all and treat-none (=0) net-benefit curves over thresholds.

    The model is worth acting on only over the threshold range where its curve
    sits above both references [vickers2019dca, p. 4].
    """
    return {
        "thresholds": np.asarray(thresholds),
        "model": net_benefit(y_true, y_prob, thresholds),
        "treat_all": treat_all_net_benefit(y_true, thresholds),
        "treat_none": np.zeros(len(thresholds)),
    }
