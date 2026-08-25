"""Train/test split strategy for the churn model.

Why a STRATIFIED RANDOM split, not a time-based one
----------------------------------------------------
The modelling table is a single cross-sectional snapshot: one row per customer,
each describing that customer's state as of ONE fixed cutoff (2011-06-09). There
is no per-row time axis to split on — every row shares the same cutoff, so
"earlier rows vs. later rows" does not exist within this table.

A genuine time-based (walk-forward) split would require multiple such snapshots
at different cutoffs and training on the earlier one(s) to predict the later
one — a valid technique (`sql/build_features.sql` supports arbitrary cutoffs
precisely so this can be done later, e.g. for the drift/monitoring work in
Step 19), but it is a different, larger exercise: those snapshots share most of
their customer base, so "time-based" there means "same customers, later
life-stage" rather than "unseen customers," which is a different validation
question than the one Step 7-11 are asking (does the model generalise to
customers it hasn't scored before).

For a single-snapshot model — which is what gets deployed and queried by the
API in Step 14 — a stratified random split is the correct and standard choice:
it matches how the model is actually used (score the whole customer base at
one point in time) and it is stratified on `is_churned` so the mild 42.5/57.5
imbalance (Step 4/5) is preserved in both halves.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_customer_split(
    df: pd.DataFrame,
    target: str = "is_churned",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split at the customer level.

    One row per customer in this table, so splitting rows IS splitting
    customers — no separate grouping key is needed to avoid a customer
    appearing in both halves.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
