# Customer Intelligence & Churn Decision Platform
#
# Multi-stage build: one shared `base` stage installs dependencies and copies
# the application code ONCE; `api` and `dashboard` are thin final stages that
# differ only in which process they run. This avoids installing the same
# ~1.5GB dependency stack (xgboost, shap, mlflow, streamlit...) twice.
#
# Deliberate simplification, stated rather than hidden: this Dockerfile
# installs the FULL requirements.txt (including pipeline-only packages like
# sqlalchemy/psycopg2/optuna/mlflow/jupyter) into the serving images, rather
# than maintaining a second, leaner requirements-serve.txt. A production
# system at larger scale would likely split those to shrink image size; here
# it also means the SAME `api` image can run the training/pipeline scripts
# via `docker compose run`, which is why Step 17 doesn't need a third image.
#
# Models, data, and reports are NOT copied into the image (see
# docker-compose.yml) — they're mounted as runtime volumes, so retraining a
# model or rebuilding a feature table never requires rebuilding the image.

FROM python:3.10-slim AS base

# libgomp1: XGBoost's Linux wheel needs the OpenMP runtime at import time
# (the same requirement that needed `brew install libomp` on macOS in
# requirements.txt — this is the Debian equivalent).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code only — data/models/reports arrive as volumes at runtime.
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY src/ ./src/
COPY sql/ ./sql/
COPY config/ ./config/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg

# Non-root: standard container hardening. The dashboard's mounted volumes
# are read-only (it never writes); the api service's are read-write because
# the same image doubles as the pipeline runner (see docker-compose.yml).
RUN useradd --create-home --uid 1000 appuser
USER appuser


FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS dashboard
# STREAMLIT_BROWSER_GATHER_USAGE_STATS=false + --server.headless: without
# these, `streamlit run` in a container with no TTY can hang on first launch
# waiting for an interactive telemetry-opt-in prompt on stdin that will never
# come, and otherwise tries (and fails) to open a browser on the container.
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/Home.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
