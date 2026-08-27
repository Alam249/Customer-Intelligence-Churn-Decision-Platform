"""Tests for src/data/quality.py (Step 4's reusable data-quality checks).

These functions decide what gets flagged as a data-quality problem in the
real project report — an off-by-one or wrong-comparison bug here means a
real issue silently goes unreported, or a non-issue gets misreported. Each
test uses a tiny synthetic table where the "right answer" can be verified by
inspection, then checks the function actually finds it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.quality import (
    clean_customer_features,
    constant_or_near_constant_columns,
    correlation_with_target,
    duplicate_report,
    highly_correlated_pairs,
    impossible_value_report,
    iqr_outlier_report,
    missing_value_report,
    target_distribution,
)


def test_missing_value_report_counts_and_excludes_complete_columns():
    df = pd.DataFrame(
        {
            "complete": [1, 2, 3, 4],
            "half_missing": [1, np.nan, 3, np.nan],
            "all_present_zero": [0, 0, 0, 0],
        }
    )
    report = missing_value_report(df)

    assert "complete" not in report.index
    assert "all_present_zero" not in report.index
    assert report.loc["half_missing", "n_missing"] == 2
    assert report.loc["half_missing", "pct_missing"] == pytest.approx(50.0)


def test_duplicate_report_exact_and_subset():
    df = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3],
            "value": [10, 10, 20, 30],
        }
    )
    # Row 0 and row 1 are exact duplicates AND share customer_id=1.
    result = duplicate_report(df, subset=["customer_id"])
    assert result["n_exact_duplicate_rows"] == 1
    assert result["n_duplicate_on_customer_id"] == 1


def test_impossible_value_report_flags_real_violations():
    df = pd.DataFrame({"quantity": [1, 0, -1, 5, 0]})
    result = impossible_value_report(df, {"quantity": "quantity == 0"})
    assert result.loc[0, "violations"] == 2
    assert result.loc[0, "error"] is None


def test_impossible_value_report_surfaces_bad_rule_without_crashing():
    """A rule referencing a nonexistent column must not crash the whole
    report — it should report the error inline so the caller sees exactly
    which rule is broken."""
    df = pd.DataFrame({"quantity": [1, 2, 3]})
    result = impossible_value_report(df, {"quantity": "nonexistent_column < 0"})
    assert result.loc[0, "violations"] is None
    assert result.loc[0, "error"] is not None


def test_iqr_outlier_report_detects_a_planted_outlier():
    # 19 normal values clustered around 10, one obvious outlier at 1000.
    values = [10 + i * 0.1 for i in range(19)] + [1000.0]
    df = pd.DataFrame({"x": values})
    result = iqr_outlier_report(df, ["x"])

    row = result.iloc[0]
    assert row["n_outliers"] == 1
    assert row["upper_fence"] < 1000.0


def test_iqr_outlier_report_finds_nothing_in_uniform_data():
    df = pd.DataFrame({"x": list(range(20))})  # no outliers by construction
    result = iqr_outlier_report(df, ["x"])
    assert result.iloc[0]["n_outliers"] == 0


def test_target_distribution_imbalance_ratio():
    y = pd.Series([1] * 30 + [0] * 10)  # 3:1 imbalance
    result = target_distribution(y)
    assert result.loc[1, "count"] == 30
    assert result.loc[0, "count"] == 10
    assert result.attrs["imbalance_ratio"] == pytest.approx(3.0)


def test_correlation_with_target_flags_strong_correlation_only():
    rng = np.random.RandomState(0)
    n = 200
    target = rng.randint(0, 2, size=n)
    df = pd.DataFrame(
        {
            "target": target,
            "perfectly_correlated": target.astype(float),  # r = 1.0
            "unrelated_noise": rng.normal(size=n),  # r ~ 0
        }
    )
    result = correlation_with_target(df, target="target", flag_threshold=0.6)

    assert result.loc["perfectly_correlated", "flagged_for_review"]
    assert not result.loc["unrelated_noise", "flagged_for_review"]


def test_constant_or_near_constant_columns_detects_single_value_column():
    df = pd.DataFrame(
        {
            "all_same": [5, 5, 5, 5, 5],
            "half_half": [1, 1, 0, 0, 1],
        }
    )
    flagged = constant_or_near_constant_columns(df, threshold=0.99)
    assert flagged == ["all_same"]


def test_highly_correlated_pairs_detects_duplicated_feature():
    n = 100
    x = np.linspace(0, 1, n)
    df = pd.DataFrame(
        {
            "a": x,
            "b": x * 2 + 1,  # perfectly correlated with a
            "c": np.random.RandomState(1).normal(size=n),  # unrelated
        }
    )
    pairs = highly_correlated_pairs(df, columns=["a", "b", "c"], threshold=0.95)
    assert len(pairs) == 1
    assert set(pairs.iloc[0][["feature_a", "feature_b"]]) == {"a", "b"}


def test_clean_customer_features_caps_return_rate_and_preserves_raw():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "return_rate": [0.1, 2.5, 4.27],  # customers 2 and 3 exceed 1.0
        }
    )
    cleaned, log = clean_customer_features(df)

    assert cleaned["return_rate"].tolist() == pytest.approx([0.1, 1.0, 1.0])
    assert cleaned["return_rate_raw"].tolist() == pytest.approx([0.1, 2.5, 4.27])
    assert any(entry["column"] == "return_rate" for entry in log)


def test_clean_customer_features_no_cap_needed_leaves_data_untouched():
    df = pd.DataFrame({"customer_id": [1, 2], "return_rate": [0.1, 0.5]})
    cleaned, log = clean_customer_features(df)
    assert "return_rate_raw" not in cleaned.columns
    assert not any(entry["column"] == "return_rate" for entry in log)


def test_clean_customer_features_adds_missingness_flags():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "avg_interpurchase_days": [10.0, np.nan, 20.0],
            "std_interpurchase_days": [np.nan, np.nan, 5.0],
        }
    )
    cleaned, log = clean_customer_features(df)

    assert cleaned["avg_interpurchase_days_is_missing"].tolist() == [False, True, False]
    assert cleaned["std_interpurchase_days_is_missing"].tolist() == [True, True, False]
    assert len(log) == 2  # one entry per flagged column


def test_clean_customer_features_preserves_row_count_and_order():
    df = pd.DataFrame({"customer_id": [3, 1, 2], "return_rate": [0.1, 2.0, 0.5]})
    cleaned, _ = clean_customer_features(df)
    assert len(cleaned) == len(df)
    assert cleaned["customer_id"].tolist() == [3, 1, 2]
