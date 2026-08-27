"""Shared pytest fixtures.

Fixtures here build small SYNTHETIC data with known, hand-computable values —
deliberately not the real project data. These are unit tests: fast, isolated,
and independent of whether the full pipeline (Steps 3-13) has been run. Only
`tests/test_api.py` needs the real trained artifacts, and says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def raw_customer_df() -> pd.DataFrame:
    """8 customers with hand-chosen values for `CustomerFeatureEngineer` tests:
    a clean spread of recency/frequency/monetary so quantile bucketing has
    something real to split on, one zero-tenure row (division-by-zero guard),
    one zero-frequency-adjacent NaN-gap row (missing-gap guard), and values
    chosen so expected derived-feature outputs can be computed by hand.
    """
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "recency_days": [5, 400, 100, 50, 300, 10, 200, 150],
            "frequency": [10, 1, 3, 8, 2, 12, 4, 5],
            "monetary_total": [5000.0, 50.0, 500.0, 3000.0, 100.0, 8000.0, 700.0, 1200.0],
            "tenure_days": [500, 30, 0, 400, 60, 550, 250, 300],
            "orders_last_90d": [5, 0, 1, 4, 0, 6, 2, 1],
            "distinct_products": [40, 2, 10, 30, 3, 50, 12, 15],
            "avg_interpurchase_days": [20.0, np.nan, 60.0, 25.0, np.nan, 15.0, 50.0, 40.0],
            "std_interpurchase_days": [5.0, np.nan, 10.0, 8.0, np.nan, 3.0, 12.0, 9.0],
        }
    )
