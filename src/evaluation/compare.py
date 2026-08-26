"""Reusable multi-model comparison framework.

One function fits and scores any scikit-learn-compatible pipeline the same
way, so the Step 8 model bake-off (and Step 9's tuned-vs-untuned comparison
later) share a single, consistent measurement — not one ad hoc block of code
per model.
"""

from __future__ import annotations

import time

import pandas as pd

from src.evaluation.metrics import compute_classification_metrics


def evaluate_model(
    name: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    already_fitted: bool = False,
) -> dict:
    """Fit (unless already fitted) and score one pipeline; returns train+test
    metrics plus timing, keyed for direct use in a comparison table.

    ``already_fitted=True`` skips refitting — used for the Step 7 baseline,
    which is loaded from disk so this comparison measures the exact model
    already reported, not a coincidentally-similar retrain.
    """
    if already_fitted:
        train_time = float("nan")  # not measured in this run; see baseline_model_report.md
    else:
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    inference_time_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    test_pred = (test_proba >= threshold).astype(int)

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= threshold).astype(int)

    test_metrics = compute_classification_metrics(y_test, test_pred, test_proba)
    train_metrics = compute_classification_metrics(y_train, train_pred, train_proba)

    return {
        "name": name,
        "pipeline": pipeline,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_time_s": train_time,
        "inference_time_ms": inference_time_ms,
        "test_proba": test_proba,
        "test_pred": test_pred,
    }


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """One row per model: test metrics + an overfit gap + timing."""
    rows = []
    for r in results:
        row = {"model": r["name"], **{f"test_{k}": v for k, v in r["test_metrics"].items()}}
        row["overfit_gap_roc_auc"] = r["train_metrics"]["roc_auc"] - r["test_metrics"]["roc_auc"]
        row["train_time_s"] = r["train_time_s"]
        row["inference_time_ms"] = r["inference_time_ms"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(4)
