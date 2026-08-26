"""Customer Lifetime Value via BG/NBD + Gamma-Gamma.

Why this methodology, not a simpler proxy
------------------------------------------
Online Retail II is exactly the setting BG/NBD ("Buy 'Til You Die") was
designed for: a non-contractual retailer with a full repeat-purchase
transaction history per customer (not a single snapshot). That is a genuine,
checked prerequisite — not assumed — so the full probabilistic approach is
used here instead of a simpler historical-average proxy:

  - **BG/NBD** models two latent processes per customer: how often they buy
    while "alive," and the probability they've silently become "dead" (will
    never buy again) — estimated from `frequency` (repeat purchase count),
    `recency` (age at last purchase), and `T` (age at the observation cutoff).
  - **Gamma-Gamma** models the monetary value of a purchase, assuming
    monetary value is independent of purchase frequency. That assumption is
    checked below, not asserted (see `check_independence_assumption`).

CLV here is computed over the SAME 183-day horizon as the churn label, using
the customer's FULL transaction history up to the cutoff (not the 365-day
churn-eligibility lookback window — CLV wants the whole history, churn
eligibility is a different, narrower question).

A real limitation, found by testing rather than assumed
----------------------------------------------------------
Gamma-Gamma's conditional expectation is undefined/unstable for customers
with zero repeat purchases (mathematically, its formula can return a NEGATIVE
"expected profit" when `frequency=0` — confirmed empirically while building
this module, not a theoretical footnote). This is exactly why the standard
practice is to fit Gamma-Gamma on repeat customers only. One-time buyers
(no repeat purchase observed at all) fall back to their own single observed
transaction value — the one real data point available for them — rather than
a model estimate that would be numerically unstable in this regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sqlalchemy import text
from sqlalchemy.engine import Engine

CLV_HORIZON_MONTHS = 6  # matches the 183-day churn horizon (183 / 30.44 ~= 6.0)


def load_customer_transactions(engine: Engine, cutoff_date: str) -> pd.DataFrame:
    """One row per (customer, invoice date, revenue) using the customer's FULL
    history up to the cutoff — deliberately not the churn model's 365-day
    lookback window, since CLV estimation wants the whole observed history.

    Filters mirror Step 3's `merchandise` definition exactly (SALE invoices,
    genuine product lines, positive quantity/price) for consistency with the
    rest of the project's revenue accounting.
    """
    query = text(
        """
        SELECT i.customer_id, i.invoice_ts::date AS invoice_date,
               SUM(l.line_revenue) AS revenue
        FROM invoices i
        JOIN invoice_lines l USING (invoice_no)
        JOIN products p ON p.stock_code = l.stock_code
        WHERE i.invoice_type = 'SALE'
          AND i.customer_id IS NOT NULL
          AND p.item_type = 'PRODUCT'
          AND l.quantity > 0 AND l.unit_price > 0
          AND i.invoice_ts::date <= CAST(:cutoff_date AS date)
        GROUP BY i.customer_id, i.invoice_ts::date
        """
    )
    return pd.read_sql(query, engine, params={"cutoff_date": cutoff_date})


def build_clv_summary(transactions: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Per-customer (frequency, recency, T, monetary_value) — the BG/NBD input
    format, via `lifetimes`' own well-tested aggregation rather than a
    hand-rolled equivalent.
    """
    return summary_data_from_transaction_data(
        transactions, customer_id_col="customer_id", datetime_col="invoice_date",
        monetary_value_col="revenue", observation_period_end=cutoff_date, freq="D",
    )


def check_independence_assumption(summary: pd.DataFrame) -> float:
    """Pearson correlation between frequency and monetary_value among repeat
    customers — Gamma-Gamma assumes independence. Returns the measured
    correlation so the report states a number, not a claim.
    """
    repeat = summary[summary["frequency"] > 0]
    return float(repeat[["frequency", "monetary_value"]].corr().iloc[0, 1])


def fit_bgnbd(summary: pd.DataFrame, penalizer_coef: float = 0.001) -> BetaGeoFitter:
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    return bgf


def fit_gamma_gamma(summary: pd.DataFrame, penalizer_coef: float = 0.001) -> GammaGammaFitter:
    """Fit on REPEAT customers only (frequency > 0) — see module docstring."""
    repeat = summary[summary["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(repeat["frequency"], repeat["monetary_value"])
    return ggf


def estimate_clv(
    bgf: BetaGeoFitter, ggf: GammaGammaFitter, summary: pd.DataFrame, transactions: pd.DataFrame,
    months: int = CLV_HORIZON_MONTHS,
) -> pd.DataFrame:
    """Expected purchases (all customers, BG/NBD handles frequency=0 natively)
    times expected value per purchase (Gamma-Gamma for repeat customers; the
    customer's own observed transaction value for one-time buyers — see
    module docstring for why Gamma-Gamma cannot be trusted at frequency=0).

    NOTE: `summary['monetary_value']` is NOT the customer's spend for one-time
    buyers — `lifetimes` defines it as the average of REPEAT transactions only
    (excluding the first), which is structurally 0 whenever frequency=0. The
    one-time buyer's actual observed transaction value has to be pulled from
    the raw ``transactions`` table instead; using ``summary['monetary_value']``
    directly for them would silently value every one-time buyer at 0.
    """
    horizon_days = months * 30.44
    out = summary.copy()
    out["expected_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_days, out["frequency"], out["recency"], out["T"]
    )

    is_repeat = out["frequency"] > 0
    out["expected_value_per_purchase"] = np.nan
    out.loc[is_repeat, "expected_value_per_purchase"] = ggf.conditional_expected_average_profit(
        out.loc[is_repeat, "frequency"], out.loc[is_repeat, "monetary_value"]
    )

    # One-time buyers have exactly one transaction row each (frequency=0 means
    # zero REPEAT purchases were observed) — that single row's revenue IS
    # their real, only observed transaction value.
    one_time_value = transactions.groupby("customer_id")["revenue"].sum()
    out.loc[~is_repeat, "expected_value_per_purchase"] = out.index.map(one_time_value).values[~is_repeat.values]

    out["value_source"] = "gamma_gamma_conditional_expectation"
    out.loc[~is_repeat, "value_source"] = "own_observed_transaction (Gamma-Gamma unstable at frequency=0)"

    out["clv"] = (out["expected_purchases"] * out["expected_value_per_purchase"]).round(2)
    return out.reset_index()
