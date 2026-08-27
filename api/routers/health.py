"""GET /health — liveness/readiness check."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas.prediction import HealthResponse
from api.state import state

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether startup finished loading models and how many customers are servable."""
    return HealthResponse(
        status="ok",
        models_loaded=state.loaded,
        n_customers=len(state.customers) if state.customers is not None else 0,
    )
