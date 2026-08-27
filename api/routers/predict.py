"""POST /predict and POST /predict/explain.

No prediction here is fabricated: every response comes from the calibrated
final model (Step 10) loaded once at startup and applied to the customer's
real, precomputed feature row (Step 6). `/predict/explain` calls the exact
same `explain_customer` function from Step 11 — the explanation logic is
written once, in `src/explainability.py`, not reimplemented here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.prediction import ErrorResponse, ExplainResponse, PredictRequest, PredictResponse
from api.state import state
from src.explainability import (
    DEFAULT_RISK_HIGH_CUTOFF,
    DEFAULT_RISK_LOW_CUTOFF,
    explain_customer,
    risk_level_from_probability,
)
from src.models.preprocessing import TREE_CATEGORICAL_FEATURES, TREE_NUMERIC_FEATURES
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["prediction"])

FEATURE_COLUMNS = TREE_NUMERIC_FEATURES + TREE_CATEGORICAL_FEATURES

# Documented error responses shared by both endpoints below, so the OpenAPI
# spec reflects the 404/500 outcomes both routes can genuinely return (both
# exercised in tests/test_api.py) — not just the 200/422 FastAPI adds by default.
COMMON_ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Customer ID not found in the served table."},
    500: {"model": ErrorResponse, "description": "Prediction or explanation failed. See server logs."},
}


def _lookup_customer_or_404(customer_id: int):
    try:
        return state.get_customer_row(customer_id)
    except KeyError as exc:
        n = len(state.customers) if state.customers is not None else 0
        raise HTTPException(
            status_code=404,
            detail=(
                f"Customer {customer_id} not found. This API scores the {n:,} customers present "
                "in the project's historical feature table (Online Retail II, IDs roughly "
                "12346-18287) — it does not accept arbitrary new-customer data. "
                "See api/state.py for why."
            ),
        ) from exc


def _optional_float(row, column: str) -> float | None:
    value = row[column].iloc[0]
    return None if value != value else float(value)  # NaN != NaN


def _optional_str(row, column: str) -> str | None:
    value = row[column].iloc[0]
    return None if value != value else str(value)


@router.post("/predict", response_model=PredictResponse, responses=COMMON_ERROR_RESPONSES)
def predict(request: PredictRequest) -> PredictResponse:
    """Churn probability, risk band, CLV, and retention priority for one customer."""
    row = _lookup_customer_or_404(request.customer_id)

    X = row[FEATURE_COLUMNS]
    try:
        probability = float(state.final_model.predict_proba(X)[:, 1][0])
    except Exception as exc:  # model/preprocessing failure on a specific row
        logger.exception("Prediction failed for customer %d", request.customer_id)
        raise HTTPException(status_code=500, detail="Prediction failed. See server logs.") from exc

    risk = risk_level_from_probability(probability, DEFAULT_RISK_LOW_CUTOFF, DEFAULT_RISK_HIGH_CUTOFF)

    return PredictResponse(
        customer_id=request.customer_id,
        churn_probability=round(probability, 4),
        risk_level=risk,
        estimated_customer_value=_optional_float(row, "clv"),
        retention_priority=_optional_float(row, "retention_priority_score"),
        segment=_optional_str(row, "segment_name"),
    )


@router.post("/predict/explain", response_model=ExplainResponse, responses=COMMON_ERROR_RESPONSES)
def predict_explain(request: PredictRequest) -> ExplainResponse:
    """Everything `/predict` returns, plus the top SHAP factors driving the score."""
    row = _lookup_customer_or_404(request.customer_id)

    try:
        result = explain_customer(
            request.customer_id, state.customers, state.tuned_pipeline, state.final_model,
            risk_low_cutoff=DEFAULT_RISK_LOW_CUTOFF, risk_high_cutoff=DEFAULT_RISK_HIGH_CUTOFF,
            save_plot=False, explainer=state.explainer,
        )
    except Exception as exc:
        logger.exception("Explanation failed for customer %d", request.customer_id)
        raise HTTPException(status_code=500, detail="Explanation failed. See server logs.") from exc

    return ExplainResponse(
        customer_id=request.customer_id,
        churn_probability=round(result["churn_probability"], 4),
        risk_level=result["risk_level"],
        estimated_customer_value=_optional_float(row, "clv"),
        retention_priority=_optional_float(row, "retention_priority_score"),
        segment=_optional_str(row, "segment_name"),
        top_risk_factors=result["top_risk_factors"],
        top_protective_factors=result["top_protective_factors"],
        narrative=result["narrative"],
    )
