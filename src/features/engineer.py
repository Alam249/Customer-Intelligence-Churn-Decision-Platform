"""Feature engineering for churn prediction.

The 22 core features (RFM, cadence, basket, returns, recent activity) and the
3 data-quality flags were already built in SQL (Step 3) and validated (Step 4) —
they are not recreated here. This module adds a SMALL set of derived features
that combine those columns in ways SQL did not, and — critically — separates
row-wise features (safe to compute at any time) from features that require a
statistic LEARNED from data (which must be fit on the training split only, or
they leak information about the test set into training).

``CustomerFeatureEngineer`` follows the scikit-learn transformer contract
(fit/transform) for exactly that reason: calling ``.fit(train)`` then
``.transform(train)`` / ``.transform(test)`` is the only sequence that keeps
the quantile thresholds below honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DAYS_PER_MONTH = 30.44  # 365.25 / 12 — the same constant used in sql/build_features.sql


def _fit_quantile_edges(series: pd.Series, q: int = 5) -> np.ndarray:
    """Quantile bin edges from a TRAINING series, with open outer bounds.

    ``duplicates='drop'`` means a heavily tied column (e.g. `frequency`, where
    25% of customers sit at the value 1) can legitimately yield fewer than
    ``q`` bins — that is reported by callers, not hidden or forced.
    """
    _, edges = pd.qcut(series, q=q, duplicates="drop", retbins=True)
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf  # so a test-set value outside train's range still scores
    return edges


def _score_by_edges(series: pd.Series, edges: np.ndarray, ascending: bool) -> pd.Series:
    """Bucket a series into the bins defined by ``edges`` and label 1..k.

    ``ascending=True`` -> higher raw value scores higher (frequency, monetary).
    ``ascending=False`` -> lower raw value scores higher (recency: fewer days
    since last purchase is BETTER, so it must map to the top score).
    """
    n_bins = len(edges) - 1
    labels = range(1, n_bins + 1) if ascending else range(n_bins, 0, -1)
    return pd.cut(series, bins=edges, labels=list(labels), include_lowest=True).astype("Int64")


class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds RFM scoring and behavioural ratios on top of the Step 3/4 feature table.

    Parameters
    ----------
    rfm_bins : number of quantile buckets targeted for each of R, F, M (may
        collapse to fewer per column if the column has heavy ties — see above).
    high_value_quantile : the training-set quantile of `monetary_total` above
        which a customer is flagged `is_high_value`.
    """

    def __init__(self, rfm_bins: int = 5, high_value_quantile: float = 0.75):
        self.rfm_bins = rfm_bins
        self.high_value_quantile = high_value_quantile

    def fit(self, X: pd.DataFrame, y=None) -> "CustomerFeatureEngineer":
        self.recency_edges_ = _fit_quantile_edges(X["recency_days"], self.rfm_bins)
        self.frequency_edges_ = _fit_quantile_edges(X["frequency"], self.rfm_bins)
        self.monetary_edges_ = _fit_quantile_edges(X["monetary_total"], self.rfm_bins)
        self.high_value_threshold_ = float(X["monetary_total"].quantile(self.high_value_quantile))
        self.n_recency_bins_ = len(self.recency_edges_) - 1
        self.n_frequency_bins_ = len(self.frequency_edges_) - 1
        self.n_monetary_bins_ = len(self.monetary_edges_) - 1
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "recency_edges_"):
            raise RuntimeError("CustomerFeatureEngineer.transform() called before fit().")

        out = X.copy()

        # --- Row-wise features (pure functions of one row; fit-independent) ---
        month_denominator = np.maximum(out["tenure_days"], 1) / DAYS_PER_MONTH
        out["spend_per_tenure_month"] = (out["monetary_total"] / month_denominator).round(2)

        out["orders_ratio_90d"] = (out["orders_last_90d"] / out["frequency"]).round(4)

        out["products_per_order"] = (out["distinct_products"] / out["frequency"]).round(3)

        avg_gap = out["avg_interpurchase_days"]
        std_gap = out["std_interpurchase_days"]
        out["purchase_regularity_cv"] = np.where(
            (avg_gap.notna()) & (avg_gap > 0), (std_gap / avg_gap).round(3), np.nan
        )

        # --- Features requiring a threshold LEARNED on the training data ---
        out["recency_score"] = _score_by_edges(out["recency_days"], self.recency_edges_, ascending=False)
        out["frequency_score"] = _score_by_edges(out["frequency"], self.frequency_edges_, ascending=True)
        out["monetary_score"] = _score_by_edges(out["monetary_total"], self.monetary_edges_, ascending=True)
        out["rfm_score"] = (
            out["recency_score"].astype("Int64")
            + out["frequency_score"].astype("Int64")
            + out["monetary_score"].astype("Int64")
        )

        out["is_high_value"] = (out["monetary_total"] >= self.high_value_threshold_).astype(int)

        return out

    def get_new_feature_names(self) -> list[str]:
        return [
            "spend_per_tenure_month",
            "orders_ratio_90d",
            "products_per_order",
            "purchase_regularity_cv",
            "recency_score",
            "frequency_score",
            "monetary_score",
            "rfm_score",
            "is_high_value",
        ]
