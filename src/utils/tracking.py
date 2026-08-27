"""MLflow experiment tracking setup.

One place configures the tracking store and experiment; every training/
evaluation script (Steps 7-10) calls `init_experiment()` instead of each
reimplementing `mlflow.set_tracking_uri`/`set_experiment` itself.

Why SQLite, not the plain filesystem store
---------------------------------------------
MLflow 3.x puts the pure filesystem backend (`file:./mlruns`) into
maintenance mode with reduced features (confirmed directly: it raises on
`set_experiment` unless `MLFLOW_ALLOW_FILE_STORE=true` is set). The Model
Registry used to register the final model (Step 16's explicit requirement)
needs a database-backed store anyway, so this project uses a local SQLite
file (`mlflow.db` at the repo root) — still fully local, no separate
tracking server process, just the currently-supported way to get one.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from src.config import CONFIG, PATHS


def _resolve_sqlite_uri(tracking_uri: str) -> str:
    """A relative `sqlite:///mlflow.db` is resolved against the repo root —
    not whatever directory the script happens to be launched from — so every
    script logs to the SAME database regardless of cwd.
    """
    if not tracking_uri.startswith("sqlite:///") or tracking_uri.startswith("sqlite:////"):
        # Already absolute (four slashes) or a non-sqlite URI (e.g. a remote
        # tracking server) — use it exactly as configured.
        return tracking_uri
    relative_path = tracking_uri.removeprefix("sqlite:///")
    absolute_path = PATHS.root / relative_path
    return f"sqlite:///{absolute_path}"


def init_experiment() -> str:
    """Point MLflow at this project's tracking store and experiment.

    Returns the experiment name for convenience (e.g. to print/log it).
    """
    mlflow_cfg = CONFIG.get("mlflow", {})
    tracking_uri = _resolve_sqlite_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    experiment_name = mlflow_cfg.get("experiment_name", "customer-churn-prediction")
    artifact_location = mlflow_cfg.get("artifact_location", "mlartifacts")

    mlflow.set_tracking_uri(tracking_uri)

    artifact_path = Path(artifact_location)
    if not artifact_path.is_absolute():
        artifact_path = PATHS.root / artifact_path
    # get_experiment_by_name / create_experiment rather than set_experiment
    # directly, so the artifact_location can be specified on first creation —
    # set_experiment alone can't set it after the experiment already exists.
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        mlflow.create_experiment(experiment_name, artifact_location=f"file:{artifact_path}")
    mlflow.set_experiment(experiment_name)
    return experiment_name


def get_registered_model_name() -> str:
    return CONFIG.get("mlflow", {}).get("registered_model_name", "churn-classifier")
