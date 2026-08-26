"""Preprocessing for the churn models — Logistic Regression baseline (Step 7)
and the tree-based models compared in Step 8.

A linear model is far more sensitive to two things a tree-based model mostly
shrugs off: **collinear inputs** (inflated, unstable coefficients) and
**skewed inputs** (a few extreme values dominate the fit). Both were measured
on the training split, not assumed — the numbers behind every linear-model
exclusion are in `reports/baseline_model_report.md`. Tree-based models get the
FULL feature set instead, including everything excluded from the linear model,
because they are not sensitive to either problem.

Target column: `is_churned`. Identifier: `customer_id` (never a feature).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

TARGET = "is_churned"
IDENTIFIER = "customer_id"

# Continuous/count features. Imputed (median) then Yeo-Johnson power-transformed
# (handles the zeros many of these columns contain, unlike a plain log) plus
# standardised — a single principled step instead of hand-picking log1p per
# column, since skew here ranges from 0.4 to 49 depending on the feature.
NUMERIC_FEATURES = [
    "recency_days", "frequency", "monetary_total", "monetary_avg_order", "tenure_days",
    "avg_interpurchase_days", "std_interpurchase_days", "purchase_rate_per_month",
    "total_items", "avg_items_per_order", "distinct_products", "avg_unit_price",
    "return_invoices", "return_value", "return_rate",
    "orders_last_30d", "orders_last_90d", "spend_last_90d", "spend_ratio_90d",
    "spend_per_tenure_month", "products_per_order", "purchase_regularity_cv",
    "rfm_score",
]

# Already 0/1 — passed through unscaled. Mixing scaled continuous with raw
# binary columns in one linear model is standard practice; each coefficient is
# still interpreted per its own column's units.
BOOLEAN_FEATURES = [
    "is_uk", "is_high_value",
    "avg_interpurchase_days_is_missing", "std_interpurchase_days_is_missing",
]

# Excluded from the LINEAR model specifically, each for a measured reason.
# Step 8's tree-based models are not sensitive to collinearity and may reuse
# the full feature set including these.
EXCLUDED_WITH_REASON = {
    "country_name": "Step 5 EDA: is_uk alone is not significant (p=0.44); the full country "
                     "breakdown is only significant via small-sample categories (some <25 "
                     "customers) that risk overfitting a linear model through rare-category noise.",
    "active_days": "r=0.958 with frequency (measured on train) — near-duplicate information, "
                    "flagged in Step 5's EDA modelling implications.",
    "recency_score": "r=0.961 with recency_days — a discretised copy of a feature already included.",
    "frequency_score": "r=0.892 with rfm_score, plus a discretised copy of frequency.",
    "monetary_score": "r=0.878 with rfm_score, plus a discretised copy of monetary_total. "
                       "rfm_score is kept as the single composite; its three raw inputs "
                       "(recency_days, frequency, monetary_total) are kept instead of their "
                       "discretised *_score versions to avoid representing the same signal three ways.",
    "return_rate_raw": "Differs from return_rate (kept) only for the 9 customers capped in Step 4 "
                        "— redundant by construction, not merely by correlation.",
    "orders_ratio_90d": "r=0.978 with spend_ratio_90d (kept) — measured on train, the single "
                         "highest pairwise correlation among all candidate features.",
}


def get_feature_columns() -> tuple[list[str], list[str]]:
    """Returns (numeric_features, boolean_features) for the linear baseline."""
    return NUMERIC_FEATURES, BOOLEAN_FEATURES


def build_linear_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for the Logistic Regression baseline.

    Fit only on the training split (via the enclosing sklearn Pipeline's
    ordinary fit/transform contract) — the imputation medians and power-
    transform parameters are learned from train and applied, not refit, on test.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
        ]
    )


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Feature names in the order the ColumnTransformer emits them — needed to
    label Logistic Regression coefficients meaningfully rather than by index.
    """
    return NUMERIC_FEATURES + BOOLEAN_FEATURES


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the linear-model feature columns and target from a train/test frame."""
    X = df[NUMERIC_FEATURES + BOOLEAN_FEATURES]
    y = df[TARGET].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Tree-based models (Step 8): the FULL feature set, including everything
# excluded above for the linear model specifically.
# ---------------------------------------------------------------------------

TREE_CATEGORICAL_FEATURES = ["country_name"]

TREE_NUMERIC_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + list(EXCLUDED_WITH_REASON.keys())
TREE_NUMERIC_FEATURES.remove("country_name")  # the one exclusion that is categorical, not numeric


def build_tree_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for Random Forest / XGBoost.

    No scaling or power transform — trees split on raw thresholds, so monotonic
    transforms of a single column change nothing about what the model learns.
    Median imputation is still needed: scikit-learn's RandomForestClassifier
    cannot accept NaN natively (XGBoost can, but a shared preprocessor keeps
    the comparison apples-to-apples between the two).
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), TREE_NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), TREE_CATEGORICAL_FEATURES),
        ]
    )


def split_X_y_tree(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the full tree-model feature columns and target from a train/test frame."""
    X = df[TREE_NUMERIC_FEATURES + TREE_CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)
    return X, y


def get_tree_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Feature names in the order the tree ColumnTransformer emits them —
    the one-hot categorical block expands to one column per observed country.
    """
    ohe: OneHotEncoder = preprocessor.named_transformers_["categorical"]
    country_names = [f"country_{c}" for c in ohe.categories_[0]]
    return TREE_NUMERIC_FEATURES + country_names
