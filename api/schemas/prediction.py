"""Pydantic request/response schemas for the prediction endpoints.

These are the API's actual input validation — a malformed request never
reaches model code; FastAPI turns a schema violation into a 422 response
with a field-level explanation automatically.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Real IDs in this dataset run ~12346-18287. `le` bounds it well above
    # that (rather than tight to the real range) so the constraint is a
    # sanity ceiling against an obvious client bug (e.g. sending a row index
    # by mistake), not a maintenance burden if the served ID range shifts —
    # and, being a Field constraint rather than custom validator code, it
    # shows up in the generated JSON Schema for any client/codegen to see.
    customer_id: int = Field(
        ..., gt=0, le=999_999,
        description="Customer ID from the Online Retail II dataset (e.g. 12346; real IDs run ~12346-18287).",
        examples=[12346],
    )


class PredictResponse(BaseModel):
    customer_id: int
    churn_probability: float = Field(..., ge=0, le=1, description="Calibrated probability (Step 10).")
    risk_level: Literal["Low", "Medium", "High"]
    estimated_customer_value: float | None = Field(
        None, description="6-month CLV estimate in EUR (Step 12, BG/NBD + Gamma-Gamma)."
    )
    retention_priority: float | None = Field(
        None, description="churn_probability * estimated_customer_value (Step 12)."
    )
    segment: str | None = Field(None, description="K-Means behavioural segment name (Step 13).")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": 12346,
                "churn_probability": 0.3736,
                "risk_level": "Medium",
                "estimated_customer_value": 32668.27,
                "retention_priority": 12205.12,
                "segment": "Champions (loyal, high value)",
            }
        }
    }


class SHAPFactor(BaseModel):
    feature: str
    value: float
    shap_value: float = Field(..., description="Log-odds contribution; + toward churn, - toward retention.")


class ExplainResponse(PredictResponse):
    top_risk_factors: list[SHAPFactor]
    top_protective_factors: list[SHAPFactor]
    narrative: str = Field(..., description="Plain-English summary suitable for a non-technical stakeholder.")

    # Overrides PredictResponse's example rather than inheriting it unchanged
    # — the inherited example would otherwise omit this model's own required
    # fields (top_risk_factors, top_protective_factors, narrative) from the
    # documented /predict/explain response.
    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": 12346,
                "churn_probability": 0.3736,
                "risk_level": "Medium",
                "estimated_customer_value": 32668.27,
                "retention_priority": 12205.12,
                "segment": "Champions (loyal, high value)",
                "top_risk_factors": [
                    {"feature": "purchase_rate_per_month", "value": 0.197, "shap_value": 0.112},
                ],
                "top_protective_factors": [
                    {"feature": "rfm_score", "value": 10.0, "shap_value": -0.208},
                ],
                "narrative": (
                    "Customer 12346: 37.4% predicted churn probability (Medium risk). "
                    "Top factors increasing risk: purchase_rate_per_month = 0.197 (+0.11). "
                    "Top factors reducing risk: rfm_score = 10 (-0.21)."
                ),
            }
        }
    }


class ErrorResponse(BaseModel):
    """Shape of every error body this API returns (404s, 500s, and FastAPI's
    own 422 validation errors) — documented so a consumer doesn't have to
    guess from the source or trial-and-error.
    """
    detail: str = Field(..., description="Human-readable explanation of what went wrong.")


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(..., description="Always 'ok' if the process is responding at all.")
    models_loaded: bool = Field(..., description="Whether startup finished loading models and data successfully.")
    n_customers: int = Field(..., description="Number of customers available to score via /predict.")
