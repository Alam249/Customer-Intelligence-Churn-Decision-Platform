"""Tests for src/models/clv.py (Step 12 — BG/NBD + Gamma-Gamma).

`test_one_time_buyer_gets_own_transaction_value_not_zero` is a REGRESSION
test for a real bug found and fixed while building Step 12: `lifetimes`
defines `monetary_value` as the average of REPEAT transactions only, which is
structurally 0 for a one-time buyer (frequency=0). An earlier version of
`estimate_clv` used that column directly for everyone, silently valuing every
one-time buyer (32.5% of the population, measured in Step 12) at zero. This
test fails immediately if that regresses.

`bgf`/`ggf` are lightweight stand-ins, not real fitted `lifetimes` models —
this isolates the test to `estimate_clv`'s OWN logic (the one-time-buyer
fallback branch), independent of whether BG/NBD or Gamma-Gamma converge on
tiny synthetic data, which is a separate concern from the bug being guarded
against here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.clv import build_clv_summary, check_independence_assumption, estimate_clv


class _StubBGNBD:
    """Returns a constant expected-purchase count for every customer —
    enough to isolate `estimate_clv`'s value-assignment logic."""

    def conditional_expected_number_of_purchases_up_to_time(self, t, frequency, recency, T):
        return pd.Series(1.0, index=frequency.index)


class _StubGammaGamma:
    """Returns a fixed, obviously-synthetic value distinguishable from any
    one-time buyer's real transaction revenue used in the tests below."""

    def conditional_expected_average_profit(self, frequency, monetary_value):
        return pd.Series(999.0, index=frequency.index)


@pytest.fixture
def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 2, 3],
            "invoice_date": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-03-01", "2011-02-15"]),
            # Customer 1: one purchase (one-time buyer) — the case the bug affected.
            # Customer 2: two purchases (repeat customer).
            # Customer 3: one purchase, a different value, to catch a
            # hardcoded/wrong-customer regression rather than a coincidental match.
            "revenue": [79.48, 100.0, 50.0, 250.00],
        }
    )


def test_one_time_buyer_gets_own_transaction_value_not_zero(transactions):
    summary = build_clv_summary(transactions, cutoff_date="2011-06-09")
    result = estimate_clv(_StubBGNBD(), _StubGammaGamma(), summary, transactions)

    customer_1 = result.loc[result["customer_id"] == 1].iloc[0]
    customer_3 = result.loc[result["customer_id"] == 3].iloc[0]

    assert customer_1["frequency"] == 0, "fixture must produce a genuine one-time buyer"
    # The bug: using summary['monetary_value'] directly would give 0 here.
    assert customer_1["expected_value_per_purchase"] == pytest.approx(79.48)
    assert customer_1["expected_value_per_purchase"] != 0
    assert "own_observed_transaction" in customer_1["value_source"]

    # A second one-time buyer with a different value, to rule out a
    # hardcoded-constant or copy-paste-from-customer-1 regression.
    assert customer_3["expected_value_per_purchase"] == pytest.approx(250.00)


def test_repeat_customer_uses_gamma_gamma_not_own_transaction(transactions):
    summary = build_clv_summary(transactions, cutoff_date="2011-06-09")
    result = estimate_clv(_StubBGNBD(), _StubGammaGamma(), summary, transactions)

    customer_2 = result.loc[result["customer_id"] == 2].iloc[0]
    assert customer_2["frequency"] > 0, "fixture must produce a genuine repeat customer"
    assert customer_2["expected_value_per_purchase"] == pytest.approx(999.0)
    assert customer_2["value_source"] == "gamma_gamma_conditional_expectation"


def test_clv_equals_expected_purchases_times_expected_value(transactions):
    summary = build_clv_summary(transactions, cutoff_date="2011-06-09")
    result = estimate_clv(_StubBGNBD(), _StubGammaGamma(), summary, transactions)

    # Stub bgf always returns 1.0 expected purchases, so clv == value_per_purchase.
    for _, row in result.iterrows():
        assert row["clv"] == pytest.approx(row["expected_value_per_purchase"], abs=0.01)


def test_independence_check_excludes_one_time_buyers(transactions):
    """One-time buyers have no meaningful `monetary_value` (see above) — the
    independence check must exclude them, not silently include a structural
    zero that would bias the correlation toward zero regardless of the real
    relationship among repeat customers.
    """
    summary = build_clv_summary(transactions, cutoff_date="2011-06-09")
    corr = check_independence_assumption(summary)

    repeat_only = summary[summary["frequency"] > 0]
    assert len(repeat_only) == 1  # only customer 2 is a repeat buyer in this fixture
    manual_corr = repeat_only[["frequency", "monetary_value"]].corr().iloc[0, 1]
    # With only one repeat customer, pandas' corr() is NaN (no variance to
    # correlate) — both the function under test and the manual check must
    # agree on that, not silently disagree.
    if pd.isna(manual_corr):
        assert pd.isna(corr)
    else:
        assert corr == pytest.approx(manual_corr)
