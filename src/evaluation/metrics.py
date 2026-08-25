"""Reusable classification evaluation: metrics + plots.

Shared by every model in this project (Logistic Regression here, the Step 8
comparison, and the Step 9 tuned model) so "how we score a churn model" is
defined once. Plot styling reuses src/eda.py's palette so every chart in the
project — EDA or model evaluation — reads as one visual system.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.eda import CHURN_COLORS, INK, SEQUENTIAL_BLUE, save_figure, set_style

set_style()


def compute_classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Accuracy, precision, recall, F1, ROC-AUC and PR-AUC in one place.

    PR-AUC uses average_precision_score (the exact area under the step-function
    PR curve) rather than trapezoidal auc() over precision_recall_curve's output,
    which is a closer match to how scikit-learn's own summary is computed.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, name: str = "confusion_matrix") -> Path:
    """2x2 confusion matrix with counts AND row-wise percentages labelled directly."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap=plt.cm.Blues, vmin=0)
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (threshold = 0.50)")

    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() / 2 else INK["primary"]
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)", ha="center", va="center",
                     color=text_color, fontsize=11)
    fig.tight_layout()
    return save_figure(fig, name)


def plot_roc_curve(y_true, y_proba, name: str = "roc_curve") -> Path:
    """ROC curve with the diagonal no-skill reference and AUC in the legend."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot(fpr, tpr, color=SEQUENTIAL_BLUE, linewidth=2, label=f"Model (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color=INK["muted"], linestyle="--", linewidth=1, label="No skill (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, name)


def plot_pr_curve(y_true, y_proba, name: str = "pr_curve") -> Path:
    """Precision-Recall curve with the no-skill baseline at the positive class rate.

    The no-skill line for PR is the base rate, NOT 0.5 as in ROC — this is
    exactly why PR-AUC is the more informative metric under class imbalance
    (see the report for the full explanation).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    base_rate = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot(recall, precision, color=CHURN_COLORS[True], linewidth=2, label=f"Model (PR-AUC = {pr_auc:.3f})")
    ax.axhline(base_rate, color=INK["muted"], linestyle="--", linewidth=1,
               label=f"No skill (base rate = {base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, name)


def plot_coefficients(coef_table: pd.DataFrame, top_n: int = 20, name: str = "lr_coefficients") -> Path:
    """Horizontal bar chart of the largest-magnitude Logistic Regression coefficients.

    ``coef_table`` must have columns ['feature', 'coefficient']. Sign is colour-
    coded with the project's fixed churn palette: a positive coefficient pushes
    toward churn (red), negative pushes toward retention (blue) — the same
    convention as every churn-outcome chart since Step 5.
    """
    top = coef_table.reindex(coef_table["coefficient"].abs().sort_values(ascending=False).index).head(top_n)
    top = top.iloc[::-1]  # largest at top of the horizontal bar chart
    colors = [CHURN_COLORS[True] if c > 0 else CHURN_COLORS[False] for c in top["coefficient"]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.axvline(0, color=INK["muted"], linewidth=1)
    ax.set_xlabel("Coefficient (standardised features -> directly comparable)")
    ax.set_title(f"Logistic Regression coefficients (top {len(top)} by |value|)")
    fig.tight_layout()
    return save_figure(fig, name)
