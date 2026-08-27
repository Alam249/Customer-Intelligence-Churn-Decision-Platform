"""Probability calibration and threshold analysis.

A classifier can rank customers well (high ROC-AUC/PR-AUC) while its predicted
probabilities are still poorly calibrated numbers — e.g. a customer scored
"0.80 churn probability" who empirically churns only 55% of the time. That
distinction matters here specifically because Step 9's tuned XGBoost uses
`scale_pos_weight` to correct class imbalance, which is well known to shift
predicted probabilities away from their true frequencies even when it helps
ranking metrics. This module checks that directly rather than assuming the
tuned model's probabilities are usable as-is.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from src.eda import CATEGORICAL, INK, save_figure, set_style

set_style()


def compute_brier_score(y_true, y_proba) -> float:
    """Mean squared error between predicted probability and the binary outcome.

    Lower is better; 0 is perfect, 0.25 is what a constant 0.5 prediction gets
    on a balanced problem. Decomposes into calibration + refinement, which is
    exactly why it's reported alongside — not instead of — the reliability diagram.
    """
    return brier_score_loss(y_true, y_proba)


def plot_calibration_curves(
    curves: dict[str, tuple], n_bins: int = 10, strategy: str = "quantile", name: str = "calibration_curve"
) -> Path:
    """Reliability diagram for one or more probability sets on one axis.

    ``curves`` maps a display label to (y_true, y_proba). ``strategy='quantile'``
    bins by predicted-probability rank rather than fixed width, so each bin has
    a comparable number of test customers even though predictions cluster —
    a fixed-width bin near 0 or 1 can otherwise contain almost no one.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=INK["muted"], linestyle="--", linewidth=1, label="Perfectly calibrated")
    for i, (label, (y_true, y_proba)) in enumerate(curves.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy=strategy)
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            markersize=4,
            color=CATEGORICAL[i % len(CATEGORICAL)],
            linewidth=2,
            label=label,
        )
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed churn frequency (per bin)")
    ax.set_title("Calibration (reliability diagram)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, name)


def threshold_performance_table(y_true, y_proba, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Precision, recall, F1 and predicted-positive rate at each threshold."""
    thresholds = thresholds if thresholds is not None else np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "pct_flagged": pred.mean(),
            }
        )
    return pd.DataFrame(rows).round(4)


def plot_threshold_curves(y_true, y_proba, name: str = "threshold_curves") -> Path:
    """Precision and recall as functions of the decision threshold, on one axis."""
    table = threshold_performance_table(y_true, y_proba, np.arange(0.01, 1.0, 0.01))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(table["threshold"], table["precision"], color=CATEGORICAL[0], linewidth=2, label="Precision")
    ax.plot(table["threshold"], table["recall"], color=CATEGORICAL[1], linewidth=2, label="Recall")
    ax.plot(table["threshold"], table["f1"], color=INK["muted"], linewidth=1.5, linestyle=":", label="F1")
    ax.axvline(0.5, color=INK["muted"], linewidth=1, linestyle="--")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / recall vs. decision threshold")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, name)
