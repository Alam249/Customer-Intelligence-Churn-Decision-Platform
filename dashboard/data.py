"""Cached data/model loading shared by every dashboard page.

Every page imports from here rather than loading files itself — models and
the customer table are loaded exactly once per Streamlit server process
(`st.cache_resource`), and the (cheap, pure) derived tables are cached too
(`st.cache_data`) so switching pages doesn't repeat work needlessly.

Every number and figure downstream of this module is a real, live-computed
project output: predictions come from `models/final_churn_model.joblib`
applied to the actual test split or the actual full customer table, never a
hardcoded number copied from a report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS  # noqa: E402
from src.evaluation.metrics import compute_classification_metrics  # noqa: E402
from src.models.preprocessing import split_X_y, split_X_y_tree  # noqa: E402
from src.monitoring import DriftAnalysis, compute_drift_analysis  # noqa: E402
from src.serving import ServingContext, check_missing_artifacts, load_serving_context  # noqa: E402
from src.uplift import UpliftAnalysis, compute_uplift_analysis  # noqa: E402

DRIFT_REFERENCE_PATH = PATHS.data_processed / "train.parquet"
DRIFT_CURRENT_RAW_PATH = PATHS.data_processed / "customer_features_2011-03-09_h91.parquet"

MODEL_FILES = {
    "Logistic Regression (Step 7)": (PATHS.models / "baseline_logistic_regression.joblib", "linear"),
    "Random Forest (Step 8)": (PATHS.models / "random_forest.joblib", "tree"),
    "XGBoost, untuned (Step 8)": (PATHS.models / "xgboost.joblib", "tree"),
    "XGBoost, tuned (Step 9)": (PATHS.models / "xgboost_tuned.joblib", "tree"),
    "Final model, calibrated (Step 10)": (PATHS.models / "final_churn_model.joblib", "tree"),
}


@st.cache_resource(show_spinner="Loading models and customer data...")
def get_context() -> ServingContext:
    return load_serving_context()


def missing_artifacts() -> list[str]:
    return [str(p.relative_to(PATHS.root)) for p in check_missing_artifacts()]


@st.cache_data(show_spinner="Scoring every model against the held-out test set...")
def get_model_comparison() -> pd.DataFrame:
    """Live metrics for every saved model, computed against the actual test
    split right now — not copied from Steps 7-10's markdown reports, so this
    can never silently go stale relative to what's actually on disk.
    """
    test_df = pd.read_parquet(PATHS.data_processed / "test.parquet")
    X_test_linear, y_test = split_X_y(test_df)
    X_test_tree, _ = split_X_y_tree(test_df)

    rows = []
    for name, (path, kind) in MODEL_FILES.items():
        if not path.is_file():
            continue
        model = joblib.load(path)
        X_test = X_test_linear if kind == "linear" else X_test_tree
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = compute_classification_metrics(y_test, pred, proba)
        rows.append({"model": name, **metrics})

    return pd.DataFrame(rows).set_index("model").round(4)


@st.cache_data
def get_test_predictions() -> pd.DataFrame:
    """The final calibrated model's predictions on the real test split, for
    the interactive threshold explorer — actual out-of-sample probabilities,
    not the full (partly in-sample) population used elsewhere in the dashboard.
    """
    test_df = pd.read_parquet(PATHS.data_processed / "test.parquet")
    X_test, y_test = split_X_y_tree(test_df)
    final_model = get_context().final_model
    proba = final_model.predict_proba(X_test)[:, 1]
    return pd.DataFrame(
        {
            "customer_id": test_df["customer_id"].values,
            "is_churned": y_test.values,
            "churn_probability": proba,
        }
    )


def risk_band(probability: float, low: float = 0.30, high: float = 0.60) -> str:
    if probability >= high:
        return "High"
    if probability >= low:
        return "Medium"
    return "Low"


@st.cache_data
def get_customers_with_risk() -> pd.DataFrame:
    """The full population table with a `risk_band` column added — computed
    once and cached, reused by every page that needs it.
    """
    df = get_context().customers.copy()
    df["risk_band"] = df["churn_probability"].apply(risk_band)
    return df


def drift_missing_artifacts() -> list[str]:
    """Files Step 19's monitoring page needs beyond the standard serving
    artefacts — the real March 2011 snapshot used as the "current" population.
    """
    return [
        str(p.relative_to(PATHS.root))
        for p in (DRIFT_REFERENCE_PATH, DRIFT_CURRENT_RAW_PATH)
        if not p.is_file()
    ]


@st.cache_data(show_spinner="Computing drift analysis against the real March 2011 snapshot...")
def get_drift_analysis() -> DriftAnalysis:
    """Step 19's reference-vs-current drift comparison, computed live via the
    same `compute_drift_analysis` function `scripts/run_drift_monitoring.py`
    uses — never a copy of that script's report pasted into the dashboard.
    """
    context = get_context()
    reference = pd.read_parquet(DRIFT_REFERENCE_PATH)
    current_raw = pd.read_parquet(DRIFT_CURRENT_RAW_PATH).drop(columns=["cutoff_date"])
    engineer = joblib.load(PATHS.models / "feature_engineer.joblib")
    return compute_drift_analysis(reference, current_raw, engineer, context.final_model)


@st.cache_data(show_spinner="Simulating retention campaign and cross-fitting uplift models (~30s)...")
def get_uplift_analysis() -> UpliftAnalysis:
    """Step 20's simulated-campaign uplift analysis, computed live via the
    same `compute_uplift_analysis` function `scripts/run_uplift_modeling.py`
    uses. Every treatment-effect number here is SIMULATED, not measured —
    see `src/uplift.py`'s module docstring for why.
    """
    return compute_uplift_analysis(get_context().customers)
