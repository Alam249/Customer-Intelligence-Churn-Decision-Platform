"""Tests for src/llm/tools.py (Step 21's grounding layer).

These tools are what stop the LLM analyst from fabricating a number — a bug
here (a wrong column, a stale computation, a silent double-transform) would
make the LLM confidently wrong while looking grounded. Each test either
cross-checks against a value independently recomputed in the test itself, or
verifies a real structural invariant (e.g. segment counts summing to the
total population) — never just "did it run."
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import PATHS
from src.llm.tools import (
    TOOLS,
    explain_customer_churn,
    get_customer_summary,
    get_drift_status,
    get_model_performance,
    get_population_overview,
    get_segment_profile,
    get_top_customers,
)
from src.models.preprocessing import split_X_y_tree

KNOWN_CUSTOMER_ID = 12346  # used throughout the project's own examples (README, API schema)


def test_get_customer_summary_matches_the_served_population():
    result = get_customer_summary(KNOWN_CUSTOMER_ID)
    assert result["customer_id"] == KNOWN_CUSTOMER_ID
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert result["clv_eur"] is not None and result["clv_eur"] > 0
    assert result["country"] == "United Kingdom"


def test_get_customer_summary_unknown_id_returns_error_not_exception():
    result = get_customer_summary(999_999)
    assert "error" in result
    assert "999999" in result["error"] or "999,999" in result["error"]


def test_explain_customer_churn_agrees_with_get_customer_summary():
    """Both tools ultimately read the same served row — their churn
    probability must agree exactly, or the two tools are quietly drifting
    apart on what "the" answer is.
    """
    summary = get_customer_summary(KNOWN_CUSTOMER_ID)
    explanation = explain_customer_churn(KNOWN_CUSTOMER_ID)
    assert explanation["churn_probability"] == pytest.approx(summary["churn_probability"])
    assert explanation["risk_level"] == summary["risk_level"]
    assert len(explanation["top_risk_factors"]) > 0
    assert isinstance(explanation["narrative"], str) and len(explanation["narrative"]) > 0


def test_explain_customer_churn_unknown_id_returns_error_not_exception():
    result = explain_customer_churn(999_999)
    assert "error" in result


def test_get_population_overview_segment_counts_sum_to_total():
    result = get_population_overview()
    assert sum(result["customers_by_segment"].values()) == result["n_customers"]
    assert sum(result["customers_by_risk_value_quadrant"].values()) == result["n_customers"]
    assert 0.0 <= result["overall_churn_rate"] <= 1.0


def test_get_top_customers_is_actually_sorted_descending():
    result = get_top_customers(n=10, rank_by="clv")
    values = [c["clv"] for c in result["customers"]]
    assert values == sorted(values, reverse=True)
    assert len(values) == 10


def test_get_top_customers_rejects_unknown_rank_by():
    result = get_top_customers(n=5, rank_by="not_a_real_column")
    assert "error" in result


def test_get_top_customers_caps_n_at_fifty():
    result = get_top_customers(n=10_000, rank_by="churn_probability")
    assert result["n"] == 50
    assert len(result["customers"]) == 50


def test_get_model_performance_roc_auc_matches_independent_computation():
    """Recomputed here directly via sklearn (not by calling the tool twice)
    — confirms the tool loads the same real test set and the same deployed
    model, not a stale or mismatched pair.
    """
    test_df = pd.read_parquet(PATHS.data_processed / "test.parquet")
    X_test, y_test = split_X_y_tree(test_df)

    from src.llm.tools import _context

    proba = _context().final_model.predict_proba(X_test)[:, 1]
    expected_auc = roc_auc_score(y_test, proba)

    result = get_model_performance()
    assert result["roc_auc"] == pytest.approx(expected_auc, abs=1e-4)
    assert "865" in result["evaluated_on"] or str(len(test_df)) in result["evaluated_on"]


def test_get_segment_profile_matches_independent_groupby():
    available = get_segment_profile(None)["available_segments"]
    segment = available[0]

    from src.llm.tools import _context

    df = _context().customers
    expected_churn_rate = df.loc[df["segment_name"] == segment, "is_churned"].mean()

    result = get_segment_profile(segment)
    assert result["churn_rate"] == pytest.approx(expected_churn_rate, abs=1e-4)
    assert result["n_customers"] == (df["segment_name"] == segment).sum()


def test_get_segment_profile_unknown_segment_lists_available_ones():
    result = get_segment_profile("Not A Real Segment")
    assert "error" in result
    assert len(result["available_segments"]) >= 1


def test_get_drift_status_severity_count_matches_the_listed_features():
    result = get_drift_status()
    assert result["n_features_major_drift"] == len(result["features_with_major_drift"])
    assert isinstance(result["prediction_drifted"], bool)
    assert 0.0 <= result["prediction_ks_p_value"] <= 1.0


@pytest.mark.slow
def test_get_uplift_summary_is_clearly_labeled_simulated():
    from src.llm.tools import get_uplift_summary

    result = get_uplift_summary()
    assert result["SIMULATED"] is True
    assert "simulat" in result["note"].lower()
    assert len(result["auuc_ranking"]) >= 4  # S-/T-/X-learner + naive + oracle


def test_every_tool_schema_is_valid_json_schema_shape():
    """Each tool's `parameters` must be a well-formed JSON-Schema object —
    both the Anthropic and OpenAI adapters pass this straight through to a
    real API, so a malformed schema would only surface as a live API error.
    """
    for tool in TOOLS:
        assert tool.parameters.get("type") == "object"
        assert "properties" in tool.parameters
        assert callable(tool.function)
        assert tool.name and tool.description
