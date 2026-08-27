"""Shared model/data loading for anything that SERVES predictions.

Both the FastAPI service (Step 14) and the Streamlit dashboard (Step 15) need
the exact same thing: the fitted feature engineer, the pre-calibration tuned
model (for SHAP), the calibrated final model (for the displayed probability),
a cached SHAP explainer, and the full customer table with CLV/segment already
joined in. That is built here, ONCE, so the API and dashboard can never
quietly compute two different versions of "the customer table" — the same
principle Step 11 already established for `explain_customer()`.

Scope, stated plainly
----------------------
Both consumers serve predictions for the customers already present in the
project's historical feature table (Online Retail II, features computed as of
the 2011-06-09 cutoff — Steps 3-13). Neither accepts arbitrary new-customer
feature payloads. A production system serving genuinely new customers would
need a live feature-computation pipeline (the SQL in `sql/build_features.sql`
generalises to that), which is a separate, larger engineering task outside
this project's scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import shap

from src.config import PATHS
from src.explainability import build_explainer
from src.utils.logging import get_logger

logger = get_logger(__name__)

VALIDATED_FEATURES_PATH = PATHS.data_processed / "customer_features_2011-06-09_h183_validated.parquet"
FEATURE_ENGINEER_PATH = PATHS.models / "feature_engineer.joblib"
TUNED_MODEL_PATH = PATHS.models / "xgboost_tuned.joblib"
FINAL_MODEL_PATH = PATHS.models / "final_churn_model.joblib"
RETENTION_PRIORITY_PATH = PATHS.reports / "retention_priority_list.csv"
SEGMENTS_PATH = PATHS.reports / "customer_segments.csv"
TEST_SET_PATH = PATHS.data_processed / "test.parquet"

REQUIRED_ARTIFACTS = (
    VALIDATED_FEATURES_PATH,
    FEATURE_ENGINEER_PATH,
    TUNED_MODEL_PATH,
    FINAL_MODEL_PATH,
    RETENTION_PRIORITY_PATH,
    SEGMENTS_PATH,
)


@dataclass
class ServingContext:
    """Everything needed to score a customer or explain a prediction."""

    customers: pd.DataFrame
    tuned_pipeline: object
    final_model: object
    explainer: shap.TreeExplainer

    def get_customer_row(self, customer_id: int) -> pd.DataFrame:
        """One-row DataFrame for `customer_id`. Raises KeyError if absent."""
        row = self.customers.loc[self.customers["customer_id"] == customer_id]
        if row.empty:
            raise KeyError(customer_id)
        return row


def check_missing_artifacts() -> list[Path]:
    """Required files that don't exist yet — empty if everything is present."""
    return [p for p in REQUIRED_ARTIFACTS if not p.is_file()]


def load_serving_context() -> ServingContext:
    """Load models and build the served customer table. Raises
    FileNotFoundError with an actionable message if a prerequisite Step
    hasn't been run, or a re-raised, contextualised RuntimeError if a present
    file fails to load (corrupted or built with an incompatible library
    version) — failing loudly either way, never silently serving a stale or
    fabricated prediction.
    """
    missing = check_missing_artifacts()
    if missing:
        raise FileNotFoundError(
            "Required artefact(s) not found: "
            + ", ".join(str(p.relative_to(PATHS.root)) for p in missing)
            + ". Run the project pipeline first (README: Steps 6, 9, 10, 12, 13)."
        )

    try:
        logger.info("Loading feature engineer and models...")
        engineer = joblib.load(FEATURE_ENGINEER_PATH)
        tuned_pipeline = joblib.load(TUNED_MODEL_PATH)
        final_model = joblib.load(FINAL_MODEL_PATH)

        logger.info("Building the served customer table...")
        features = pd.read_parquet(VALIDATED_FEATURES_PATH).drop(columns=["cutoff_date"])
        features = engineer.transform(features)

        # `segment` here is Step 12's risk/value quadrant (e.g. "Low risk /
        # High value"); renamed on read to `risk_value_quadrant` so it's never
        # confused with Step 13's K-Means `segment_name` (e.g. "Champions") —
        # both are real, distinct outputs this project produced, kept distinct.
        priority = pd.read_csv(RETENTION_PRIORITY_PATH)[
            ["customer_id", "churn_probability", "clv", "retention_priority_score", "segment"]
        ].rename(columns={"segment": "risk_value_quadrant"})
        segments = pd.read_csv(SEGMENTS_PATH)[["customer_id", "segment_name"]]
    except Exception as exc:
        # All these files passed the existence check above, so a failure
        # here means a PRESENT file is corrupted, truncated, or was built
        # with an incompatible library version — a materially different,
        # more actionable diagnosis than "file not found."
        raise RuntimeError(
            f"An artefact exists but failed to load ({type(exc).__name__}: {exc}). "
            "It may be corrupted or built with a different library version than the one "
            "installed (see requirements.txt) — try regenerating it via the project's "
            "pipeline scripts (README: Steps 6, 9, 10, 12, 13)."
        ) from exc

    # `validate` turns a future duplicate customer_id in either CSV into a
    # loud MergeError instead of silently fanning out into multiple rows.
    merged = features.merge(priority, on="customer_id", how="left", validate="one_to_one").merge(
        segments, on="customer_id", how="left", validate="one_to_one"
    )

    unmatched = (
        merged[
            ["churn_probability", "clv", "retention_priority_score", "risk_value_quadrant", "segment_name"]
        ]
        .isna()
        .any(axis=1)
    )
    if unmatched.any():
        logger.warning(
            "%d customers have no CLV/segment match (Step 12/13 join) — "
            "they can still be scored for churn but value/segment fields will be null.",
            int(unmatched.sum()),
        )

    explainer = build_explainer(tuned_pipeline)
    logger.info("Loaded %d customers and both models.", len(merged))

    return ServingContext(
        customers=merged,
        tuned_pipeline=tuned_pipeline,
        final_model=final_model,
        explainer=explainer,
    )
