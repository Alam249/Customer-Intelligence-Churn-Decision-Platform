"""Tests for `CustomerFeatureEngineer` (src/features/engineer.py, Step 6).

This is the most safety-critical piece of logic in the project: if `fit()`
learned thresholds ever leaked into a re-fit on test data, every reported
model metric from Step 7 onward would be silently invalid. The first test
below is written to FAIL if that regression is ever introduced — it doesn't
just check "a value was produced," it checks the value could only have come
from the training threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import CustomerFeatureEngineer


def test_transform_before_fit_raises():
    engineer = CustomerFeatureEngineer()
    df = pd.DataFrame(
        {
            "recency_days": [1],
            "frequency": [1],
            "monetary_total": [1],
            "tenure_days": [1],
            "orders_last_90d": [1],
            "distinct_products": [1],
            "avg_interpurchase_days": [1.0],
            "std_interpurchase_days": [1.0],
        }
    )
    with pytest.raises(RuntimeError, match="before fit"):
        engineer.transform(df)


def test_is_high_value_uses_train_threshold_not_test(raw_customer_df):
    """The core leakage-prevention contract: fit on train, transform test,
    and `is_high_value` must reflect TRAIN's 75th percentile — not one
    recomputed from test's own (deliberately very different) distribution.
    """
    train = raw_customer_df
    train_threshold = train["monetary_total"].quantile(0.75)

    # Test set with a monetary distribution far higher than train's — if
    # `is_high_value` were (incorrectly) computed from test's own quantile,
    # every row here would sit near its own 75th percentile by construction
    # and roughly 25% would be flagged high-value regardless of scale.
    test = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104],
            "recency_days": [10, 20, 30, 40],
            "frequency": [1, 2, 3, 4],
            "monetary_total": [50_000.0, 60_000.0, 70_000.0, 80_000.0],
            "tenure_days": [100, 100, 100, 100],
            "orders_last_90d": [1, 1, 1, 1],
            "distinct_products": [5, 5, 5, 5],
            "avg_interpurchase_days": [30.0, 30.0, 30.0, 30.0],
            "std_interpurchase_days": [5.0, 5.0, 5.0, 5.0],
        }
    )
    test_own_threshold = test["monetary_total"].quantile(0.75)
    assert test_own_threshold != pytest.approx(train_threshold), (
        "test fixture must have a different 75th percentile than train, or this test can't "
        "distinguish 'used train threshold' from 'used test threshold'"
    )

    engineer = CustomerFeatureEngineer()
    engineer.fit(train)
    result = engineer.transform(test)

    # Every test row's monetary_total (50k-80k) is far above train's threshold
    # (computed from train's 50-8000 range) -> every row should be flagged
    # high-value under the TRAIN threshold, which is the correct behaviour.
    assert engineer.high_value_threshold_ == pytest.approx(train_threshold)
    assert (result["is_high_value"] == 1).all()
    expected = (test["monetary_total"] >= engineer.high_value_threshold_).astype(int)
    pd.testing.assert_series_equal(result["is_high_value"], expected, check_names=False)


def test_rfm_scores_bounded_by_fitted_bin_count(raw_customer_df):
    engineer = CustomerFeatureEngineer(rfm_bins=5)
    engineer.fit(raw_customer_df)
    result = engineer.transform(raw_customer_df)

    assert result["recency_score"].min() >= 1
    assert result["recency_score"].max() <= engineer.n_recency_bins_
    assert result["frequency_score"].min() >= 1
    assert result["frequency_score"].max() <= engineer.n_frequency_bins_
    # rfm_score is the sum of three bounded 1..n scores, so it's bounded too.
    assert result["rfm_score"].min() >= 3
    max_possible = engineer.n_recency_bins_ + engineer.n_frequency_bins_ + engineer.n_monetary_bins_
    assert result["rfm_score"].max() <= max_possible


def test_recency_score_direction_is_inverted(raw_customer_df):
    """Lower recency (more recent purchase) is BETTER and must score HIGHER —
    this is the one score in the module with `ascending=False`; a sign flip
    here would silently invert the whole RFM composite's meaning.
    """
    engineer = CustomerFeatureEngineer()
    engineer.fit(raw_customer_df)
    result = engineer.transform(raw_customer_df)

    most_recent = result.loc[raw_customer_df["recency_days"].idxmin()]
    least_recent = result.loc[raw_customer_df["recency_days"].idxmax()]
    assert most_recent["recency_score"] > least_recent["recency_score"]


def test_spend_per_tenure_month_handles_zero_tenure(raw_customer_df):
    """Customer 3 has tenure_days=0 — the denominator must be floored at 1
    day (not 0), so this must produce a finite number, never inf/NaN."""
    engineer = CustomerFeatureEngineer()
    engineer.fit(raw_customer_df)
    result = engineer.transform(raw_customer_df)

    zero_tenure_row = result[raw_customer_df["tenure_days"] == 0].iloc[0]
    assert np.isfinite(zero_tenure_row["spend_per_tenure_month"])
    # tenure floored to 1 day -> denominator = 1/30.44 months
    expected = round(500.0 / (1 / 30.44), 2)
    assert zero_tenure_row["spend_per_tenure_month"] == pytest.approx(expected)


def test_purchase_regularity_cv_is_nan_when_gap_missing(raw_customer_df):
    """Customers 2 and 5 have NaN avg_interpurchase_days (too few orders) —
    the coefficient of variation must stay NaN, not silently become 0 or error."""
    engineer = CustomerFeatureEngineer()
    engineer.fit(raw_customer_df)
    result = engineer.transform(raw_customer_df)

    missing_gap_rows = result[raw_customer_df["avg_interpurchase_days"].isna()]
    assert missing_gap_rows["purchase_regularity_cv"].isna().all()

    present_gap_rows = result[raw_customer_df["avg_interpurchase_days"].notna()]
    assert present_gap_rows["purchase_regularity_cv"].notna().all()


def test_row_wise_derived_features_match_hand_computation(raw_customer_df):
    """orders_ratio_90d and products_per_order are pure row-wise ratios —
    verify against a hand-picked row rather than trusting the implementation."""
    engineer = CustomerFeatureEngineer()
    engineer.fit(raw_customer_df)
    result = engineer.transform(raw_customer_df)

    row = result[raw_customer_df["customer_id"] == 1].iloc[0]
    # customer 1: orders_last_90d=5, frequency=10, distinct_products=40
    assert row["orders_ratio_90d"] == pytest.approx(5 / 10, abs=1e-4)
    assert row["products_per_order"] == pytest.approx(40 / 10, abs=1e-3)


def test_transform_does_not_mutate_input(raw_customer_df):
    """`transform` must return a new frame, never modify the caller's data —
    a fit/transform pipeline stage that mutates its input in place is a classic
    source of hard-to-trace bugs when the same object is reused."""
    original = raw_customer_df.copy(deep=True)
    engineer = CustomerFeatureEngineer()
    engineer.fit(raw_customer_df)
    engineer.transform(raw_customer_df)
    pd.testing.assert_frame_equal(raw_customer_df, original)


def test_transform_preserves_row_count_and_order(raw_customer_df):
    engineer = CustomerFeatureEngineer()
    engineer.fit(raw_customer_df)
    result = engineer.transform(raw_customer_df)
    assert len(result) == len(raw_customer_df)
    pd.testing.assert_series_equal(result["customer_id"], raw_customer_df["customer_id"])
