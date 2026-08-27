"""Tests for model-prediction-adjacent logic: risk banding (src/explainability.py)
and classification metrics (src/evaluation/metrics.py).

These are unit-level checks on small hand-computable examples. End-to-end
prediction correctness against the REAL trained models is covered separately
by `tests/test_api.py`, which asserts the live API's prediction for a known
customer matches the value already computed offline by the Step 12 batch
pipeline — a stronger check than this file could do without depending on the
full pipeline having been run first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_classification_metrics
from src.explainability import explain_customer, risk_level_from_probability


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "Low"),
        (0.29, "Low"),
        (0.30, "Medium"),  # exactly at the low cutoff -> Medium (inclusive)
        (0.45, "Medium"),
        (0.59, "Medium"),
        (0.60, "High"),  # exactly at the high cutoff -> High (inclusive)
        (0.99, "High"),
        (1.0, "High"),
    ],
)
def test_risk_level_boundaries(probability, expected):
    assert risk_level_from_probability(probability, low_cutoff=0.30, high_cutoff=0.60) == expected


def test_risk_level_respects_custom_cutoffs():
    """The cutoffs are parameters, not hardcoded — verify a different pair
    of bands actually changes the classification, not just the default."""
    assert risk_level_from_probability(0.5, low_cutoff=0.6, high_cutoff=0.9) == "Low"
    assert risk_level_from_probability(0.5, low_cutoff=0.1, high_cutoff=0.4) == "High"


def test_compute_classification_metrics_matches_hand_computation():
    # 4 samples: 2 correct, 1 false positive, 1 false negative.
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.4, 0.6, 0.1])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    # TP=1 (idx0), FN=1 (idx1), FP=1 (idx2), TN=1 (idx3)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)  # TP / (TP+FP) = 1/2
    assert metrics["recall"] == pytest.approx(0.5)  # TP / (TP+FN) = 1/2
    assert metrics["f1"] == pytest.approx(0.5)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_compute_classification_metrics_perfect_predictions():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.95, 0.90, 0.05, 0.10])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_classification_metrics_handles_no_positive_predictions():
    """precision_score on zero predicted positives divides 0/0 — must return
    0 (via zero_division=0), never raise, since this is a real, valid state
    a model at an extreme threshold can be in (Step 10's threshold sweep
    relies on this not crashing at the tails)."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.4, 0.1, 0.3, 0.05])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_explain_customer_raises_keyerror_for_unknown_id():
    """The lookup happens before any model is touched, so this can be tested
    without a real fitted pipeline — pass `None` for both models and confirm
    the function never gets far enough to need them."""
    df = pd.DataFrame({"customer_id": [1, 2, 3]})
    with pytest.raises(KeyError, match="999"):
        explain_customer(999, df, tuned_pipeline=None, final_model=None)
