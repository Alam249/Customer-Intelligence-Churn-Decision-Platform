"""End-to-end data pipeline: raw CSV -> PostgreSQL -> analytical feature table.

Reproducible entry point for Step 3. Every stage is idempotent — rerunning the
script from scratch produces the same database state.

    python scripts/run_pipeline.py                 # full rebuild
    python scripts/run_pipeline.py --skip-build    # reuse data/interim CSVs
    python scripts/run_pipeline.py --cutoff 2011-03-09 --horizon 91

Requires a running PostgreSQL server and a populated .env (see .env.example).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG, PATHS, get_database_url, get_psql_env  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def find_psql() -> str:
    """Locate the psql client, including the Homebrew keg-only install path."""
    found = shutil.which("psql")
    if found:
        return found

    for candidate in (
        "/opt/homebrew/opt/postgresql@16/bin/psql",
        "/usr/local/opt/postgresql@16/bin/psql",
        "/Library/PostgreSQL/16/bin/psql",
    ):
        if Path(candidate).is_file():
            return candidate

    raise FileNotFoundError(
        "psql not found on PATH. Install PostgreSQL (macOS: brew install postgresql@16) "
        "or add its bin directory to PATH."
    )


def run_sql(script: str, psql: str, pg_env: dict[str, str], variables: dict[str, str] | None = None) -> None:
    """Execute a .sql file with ON_ERROR_STOP so failures are never silent.

    Connection details arrive through the environment, so the password never
    appears in argv or in psql's error output.
    """
    path = PATHS.sql / script
    if not path.is_file():
        raise FileNotFoundError(f"SQL script not found: {path}")

    cmd = [psql, "-v", "ON_ERROR_STOP=1", "-q"]
    for key, value in (variables or {}).items():
        cmd += ["-v", f"{key}={value}"]
    cmd += ["-f", str(path)]

    logger.info("Running %s", script)
    # cwd is the repo root so the \copy paths in load_data.sql resolve.
    result = subprocess.run(
        cmd, cwd=PATHS.root, capture_output=True, text=True, env={**os.environ, **pg_env}
    )

    if result.stdout.strip():
        print(result.stdout)
    if result.returncode != 0:
        logger.error("%s failed:\n%s", script, result.stderr.strip())
        raise RuntimeError(f"SQL script {script} exited with code {result.returncode}")
    if result.stderr.strip():
        logger.debug("psql stderr: %s", result.stderr.strip())


def export_feature_table(url: str, cutoff: str, horizon: int) -> Path:
    """Write the feature + label table for ONE label definition to data/processed.

    The filter on both cutoff_date and horizon_days is essential: `churn_labels`
    is designed to hold several coexisting label definitions, so an unfiltered
    join silently concatenates them and produces duplicate customer_ids under
    contradictory targets. The filename records the definition for the same reason.
    """
    import pandas as pd
    from sqlalchemy import create_engine, text

    query = text(
        """
        SELECT f.*, l.is_churned
        FROM customer_features f
        JOIN churn_labels l USING (customer_id, cutoff_date)
        WHERE f.cutoff_date  = :cutoff
          AND l.horizon_days = :horizon
        ORDER BY f.customer_id
        """
    )
    engine = create_engine(url)
    try:
        df = pd.read_sql(query, engine, params={"cutoff": cutoff, "horizon": horizon})
    finally:
        engine.dispose()

    if df.empty:
        raise RuntimeError(f"No feature rows found for cutoff={cutoff} horizon={horizon}.")
    if df["customer_id"].duplicated().any():
        raise RuntimeError("Duplicate customer_id in the exported table — label filter is wrong.")

    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    target = PATHS.data_processed / f"customer_features_{cutoff}_h{horizon}.parquet"
    df.to_parquet(target, index=False)

    logger.info(
        "Exported %d customers x %d columns -> %s (churn rate %.2f%%)",
        len(df),
        df.shape[1],
        target.relative_to(PATHS.root),
        df["is_churned"].mean() * 100,
    )
    return target


def main() -> int:
    churn_cfg = CONFIG["churn_definition"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default=churn_cfg["cutoff_date"], help="Observation cutoff date")
    parser.add_argument("--horizon", type=int, default=churn_cfg["horizon_days"], help="Label window (days)")
    parser.add_argument(
        "--lookback", type=int, default=churn_cfg["eligibility_lookback_days"], help="Eligibility window (days)"
    )
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing data/interim CSVs")
    parser.add_argument("--skip-load", action="store_true", help="Skip schema creation and data load")
    args = parser.parse_args()

    try:
        psql = find_psql()
        pg_env = get_psql_env()
        url = get_database_url()
    except (FileNotFoundError, EnvironmentError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Cutoff=%s  horizon=%dd  lookback=%dd", args.cutoff, args.horizon, args.lookback)

    try:
        if not args.skip_build:
            from src.data.build_relational import build_all

            build_all()
        else:
            logger.info("Skipping raw->relational build (--skip-build)")

        if not args.skip_load:
            run_sql("schema.sql", psql, pg_env)
            run_sql("load_data.sql", psql, pg_env)
        else:
            logger.info("Skipping schema + load (--skip-load)")

        run_sql(
            "build_features.sql",
            psql,
            pg_env,
            {
                "cutoff_date": f"'{args.cutoff}'",
                "horizon_days": str(args.horizon),
                "lookback_days": str(args.lookback),
            },
        )
        run_sql("validation.sql", psql, pg_env, {"cutoff_date": f"'{args.cutoff}'"})

        export_feature_table(url, args.cutoff, args.horizon)
    except (RuntimeError, FileNotFoundError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
