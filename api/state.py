"""Application state: every model and data artefact loaded exactly once.

FastAPI's lifespan hook (see `api/main.py`) populates this module's `state`
singleton when the process starts. No request handler ever loads, fits, or
re-reads a file — that is the entire point of "load the model only once."

Scope, stated plainly
----------------------
This API serves predictions for the {n} customers already present in the
project's historical feature table (Online Retail II, features computed as of
the 2011-06-09 cutoff — Steps 3-13). It does NOT accept arbitrary new-customer
feature payloads. A production system serving genuinely new customers would
need a live feature-computation pipeline (the SQL in `sql/build_features.sql`
generalises to that), which is a separate, larger engineering task outside
this project's scope. Scoring by `customer_id` against a precomputed table is
the honest, correctly-scoped design for what this project actually has.
"""

from __future__ import annotations

import joblib
import pandas as pd

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

REQUIRED_ARTIFACTS = (
    VALIDATED_FEATURES_PATH, FEATURE_ENGINEER_PATH, TUNED_MODEL_PATH,
    FINAL_MODEL_PATH, RETENTION_PRIORITY_PATH, SEGMENTS_PATH,
)


class AppState:
    """Holds every loaded artefact the API needs. One instance, populated once."""

    def __init__(self) -> None:
        self.customers: pd.DataFrame | None = None
        self.tuned_pipeline = None
        self.final_model = None
        self.explainer = None
        self.loaded = False

    def load(self) -> None:
        """Load models and build the served customer table. Raises
        FileNotFoundError with an actionable message if a prerequisite Step
        hasn't been run, or a re-raised, contextualised error if a present
        file fails to load (corrupted or built with an incompatible library
        version) — failing loudly at startup either way, never silently
        serving a stale or fabricated prediction.
        """
        missing = [p for p in REQUIRED_ARTIFACTS if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required artefact(s) not found: "
                + ", ".join(str(p.relative_to(PATHS.root)) for p in missing)
                + ". Run the project pipeline first (README: Steps 6, 9, 10, 12, 13)."
            )

        try:
            logger.info("Loading feature engineer and models...")
            engineer = joblib.load(FEATURE_ENGINEER_PATH)
            self.tuned_pipeline = joblib.load(TUNED_MODEL_PATH)
            self.final_model = joblib.load(FINAL_MODEL_PATH)

            logger.info("Building the served customer table...")
            features = pd.read_parquet(VALIDATED_FEATURES_PATH).drop(columns=["cutoff_date"])
            features = engineer.transform(features)

            priority = pd.read_csv(RETENTION_PRIORITY_PATH)[["customer_id", "clv", "retention_priority_score"]]
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
        # loud startup MergeError instead of silently fanning out into
        # multiple rows per customer — the same "fail loudly" contract as the
        # missing-file check above, extended to a data-integrity failure.
        merged = features.merge(
            priority, on="customer_id", how="left", validate="one_to_one"
        ).merge(segments, on="customer_id", how="left", validate="one_to_one")

        unmatched = merged[["clv", "retention_priority_score", "segment_name"]].isna().any(axis=1)
        if unmatched.any():
            logger.warning(
                "%d customers have no CLV/segment match (Step 12/13 join) — "
                "they can still be scored for churn but value/segment fields will be null.",
                int(unmatched.sum()),
            )

        self.customers = merged
        self.explainer = build_explainer(self.tuned_pipeline)
        self.loaded = True
        logger.info("Loaded %d customers and both models.", len(self.customers))

    def get_customer_row(self, customer_id: int) -> pd.DataFrame:
        """One-row DataFrame for `customer_id`. Raises KeyError if absent —
        callers (the routers) translate that into an HTTP 404.
        """
        row = self.customers.loc[self.customers["customer_id"] == customer_id]
        if row.empty:
            raise KeyError(customer_id)
        return row


state = AppState()
