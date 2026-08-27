"""Application state: every model and data artefact loaded exactly once.

FastAPI's lifespan hook (see `api/main.py`) populates this module's `state`
singleton when the process starts. No request handler ever loads, fits, or
re-reads a file — that is the entire point of "load the model only once."

The actual loading logic lives in `src/serving.py`, shared with the Streamlit
dashboard (Step 15) — this module is a thin adapter exposing it as the
module-level singleton the routers import.
"""

from __future__ import annotations

import pandas as pd

from src.serving import ServingContext, load_serving_context


class AppState:
    """Holds the loaded `ServingContext`. One instance, populated once."""

    def __init__(self) -> None:
        self._context: ServingContext | None = None

    def load(self) -> None:
        self._context = load_serving_context()

    @property
    def loaded(self) -> bool:
        return self._context is not None

    @property
    def customers(self) -> pd.DataFrame | None:
        return self._context.customers if self._context else None

    @property
    def tuned_pipeline(self):
        return self._context.tuned_pipeline if self._context else None

    @property
    def final_model(self):
        return self._context.final_model if self._context else None

    @property
    def explainer(self):
        return self._context.explainer if self._context else None

    def get_customer_row(self, customer_id: int) -> pd.DataFrame:
        """One-row DataFrame for `customer_id`. Raises KeyError if absent —
        callers (the routers) translate that into an HTTP 404.
        """
        if self._context is None:
            raise KeyError(customer_id)
        return self._context.get_customer_row(customer_id)


state = AppState()
