"""Tests for src/models/retention_priority.py (Step 12's segmentation logic).

`test_assign_segments_quadrants` in particular checks the median-split
boundary handling explicitly — an off-by-one there (`>` vs `>=`) would
silently move customers sitting exactly on the median between "High" and
"Low," changing segment membership without any error.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.retention_priority import (
    assign_segments,
    compare_targeting_strategies,
    compute_retention_priority,
)


def test_compute_retention_priority_is_probability_times_clv():
    df = pd.DataFrame({"customer_id": [1, 2], "churn_probability": [0.5, 0.2], "clv": [1000.0, 500.0]})
    result = compute_retention_priority(df)
    assert result["retention_priority_score"].tolist() == pytest.approx([500.0, 100.0])


def test_compute_retention_priority_does_not_mutate_input():
    df = pd.DataFrame({"customer_id": [1], "churn_probability": [0.5], "clv": [100.0]})
    original = df.copy()
    compute_retention_priority(df)
    pd.testing.assert_frame_equal(df, original)


def test_assign_segments_quadrants_by_hand():
    """5 customers, churn_probability median = 0.5, clv median = 500 — each
    of the 4 quadrants represented, plus the exact-median boundary case."""
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "churn_probability": [0.9, 0.9, 0.1, 0.1, 0.5],  # median = 0.5
            "clv": [900.0, 100.0, 900.0, 100.0, 500.0],  # median = 500.0
        }
    )
    result = assign_segments(df)

    assert result.loc[result["customer_id"] == 1, "segment"].item() == "High risk / High value"
    assert result.loc[result["customer_id"] == 2, "segment"].item() == "High risk / Low value"
    assert result.loc[result["customer_id"] == 3, "segment"].item() == "Low risk / High value"
    assert result.loc[result["customer_id"] == 4, "segment"].item() == "Low risk / Low value"


def test_assign_segments_boundary_is_inclusive_on_high_side():
    """Customer 5 sits EXACTLY on both medians (0.5, 500.0) — the
    implementation uses `>=`, so a value equal to the median counts as
    "high," not "low." This pins that convention down explicitly so a future
    change to `>` doesn't silently reclassify every median-tied customer."""
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "churn_probability": [0.9, 0.9, 0.1, 0.1, 0.5],
            "clv": [900.0, 100.0, 900.0, 100.0, 500.0],
        }
    )
    result = assign_segments(df)
    assert result.loc[result["customer_id"] == 5, "segment"].item() == "High risk / High value"


def test_assign_segments_records_medians_used():
    df = pd.DataFrame({"customer_id": [1, 2], "churn_probability": [0.2, 0.8], "clv": [10.0, 20.0]})
    result = assign_segments(df)
    assert result.attrs["risk_median"] == pytest.approx(df["churn_probability"].median())
    assert result.attrs["value_median"] == pytest.approx(df["clv"].median())


def test_compare_targeting_strategies_on_a_case_with_no_overlap():
    """4 customers: the top-2 by churn probability are exactly the
    LOWEST-value customers, while the top-2 by priority score are the
    HIGHEST-value ones — constructed so the two rankings share zero
    customers, mirroring Step 12's real finding on the actual data.
    """
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "churn_probability": [0.9, 0.8, 0.3, 0.2],  # top-2 by churn alone: 1, 2
            "clv": [10.0, 10.0, 10_000.0, 10_000.0],  # but 3, 4 are far more valuable
        }
    )
    df = compute_retention_priority(df)
    # priority_score = prob * clv: [9, 8, 3000, 2000] -> top-2 by priority: 3, 4
    result = compare_targeting_strategies(df, top_n=2)

    assert result["overlap_pct"] == 0.0
    assert result["churn_only_avg_clv"] == pytest.approx(10.0)
    assert result["priority_avg_clv"] == pytest.approx(10_000.0)
    # churn-only total CLV-at-risk = 0.9*10 + 0.8*10 = 17.0
    assert result["churn_only_total_clv_at_risk"] == pytest.approx(17.0)
    # priority-ranked total CLV-at-risk = 0.3*10000 + 0.2*10000 = 5000.0
    assert result["priority_total_clv_at_risk"] == pytest.approx(5000.0)


def test_compare_targeting_strategies_full_overlap_when_rankings_agree():
    """When churn probability and CLV are positively correlated (so both
    rankings agree on who to target), overlap should be 100%, not 0."""
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "churn_probability": [0.9, 0.8, 0.2, 0.1],
            "clv": [900.0, 800.0, 200.0, 100.0],  # same ranking as churn_probability
        }
    )
    df = compute_retention_priority(df)
    result = compare_targeting_strategies(df, top_n=2)
    assert result["overlap_pct"] == 100.0
