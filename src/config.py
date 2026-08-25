"""Configuration loading for the Customer Intelligence Platform.

One place decides where the repository root is and what the project settings are,
so no module has to guess with relative paths.

Usage
-----
    from src.config import CONFIG, PATHS

    df = pd.read_csv(PATHS.data_raw / CONFIG["data"]["raw_file"])
    seed = CONFIG["project"]["random_seed"]
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote_plus

import yaml
from dotenv import load_dotenv

# Repository root = parent of the directory holding this file (src/).
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH: Path = ROOT_DIR / "config" / "config.yaml"

# Load .env if present. override=False so real environment variables
# (e.g. those injected by Docker Compose) always win over the local file.
load_dotenv(ROOT_DIR / ".env", override=False)


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read the YAML configuration file.

    Raises
    ------
    FileNotFoundError
        If the config file is missing — failing loudly is better than
        silently running on defaults that nobody reviewed.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Expected config/config.yaml at the repository root."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file {config_path} did not parse into a mapping.")

    return config


def _build_paths(config: dict[str, Any]) -> SimpleNamespace:
    """Turn the relative paths in config into absolute Path objects."""
    resolved = {key: ROOT_DIR / value for key, value in config.get("paths", {}).items()}
    resolved["root"] = ROOT_DIR
    return SimpleNamespace(**resolved)


_REQUIRED_DB_VARS = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")


def _require_db_env() -> dict[str, str]:
    """Read database settings from the environment, failing clearly if unset."""
    missing = [var for var in _REQUIRED_DB_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing required database environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the values."
        )
    return {
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.environ["POSTGRES_DB"],
    }


def get_database_url() -> str:
    """SQLAlchemy connection URL, for pandas/SQLAlchemy consumers.

    Credentials never live in the repository; only their variable names do.
    Note this string *contains the password* — never log it. For subprocesses
    use :func:`get_psql_env` instead, which keeps the password out of argv.
    """
    driver = CONFIG.get("database", {}).get("driver", "postgresql+psycopg2")
    p = _require_db_env()
    return f"{driver}://{p['user']}:{quote_plus(p['password'])}@{p['host']}:{p['port']}/{p['database']}"


def get_psql_env() -> dict[str, str]:
    """Environment overlay carrying libpq connection settings for `psql`.

    Passing credentials this way keeps the password out of the process argument
    list (visible to any user via `ps`) and out of psql's error messages.
    """
    p = _require_db_env()
    return {
        "PGUSER": p["user"],
        "PGPASSWORD": p["password"],
        "PGHOST": p["host"],
        "PGPORT": p["port"],
        "PGDATABASE": p["database"],
    }


CONFIG: dict[str, Any] = load_config()
PATHS: SimpleNamespace = _build_paths(CONFIG)
RANDOM_SEED: int = CONFIG.get("project", {}).get("random_seed", 42)
