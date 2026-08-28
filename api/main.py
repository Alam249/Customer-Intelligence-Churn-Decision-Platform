"""Customer Intelligence & Churn Decision Platform — prediction API.

Run locally:
    uvicorn api.main:app --reload --port 8000

Swagger UI:  http://127.0.0.1:8000/docs
ReDoc:       http://127.0.0.1:8000/redoc

Scope: no authentication or rate limiting
-------------------------------------------
Deliberately out of scope for this portfolio demo, not an oversight: every
endpoint here serves only already-public UCI Online Retail II customer IDs
with no real PII, and is meant to run locally or in a controlled demo
environment. Before this service accepted traffic from an untrusted network
it would need an API-key dependency (or a gateway) and a rate limiter (e.g.
`slowapi`) in front of `/predict/explain`, the more expensive endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from api.routers import analyst, health, predict
from api.state import state
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading models and customer feature table (once)...")
    state.load()
    logger.info("Startup complete — ready to serve.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Customer Intelligence & Churn Decision Platform API",
    description=(
        "Serves churn-probability predictions, SHAP explanations, customer lifetime value, and "
        "retention-priority scores for the Online Retail II customer base (Steps 6-13), plus a "
        "tool-calling LLM analyst endpoint grounded in the same real models and reports (Step 21)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(analyst.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Every unexpected error becomes a clean JSON message, never a raw
    traceback leaked to the client — full detail still goes to the server log.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. See server logs for details."},
    )
