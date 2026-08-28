"""Smoke test for Step 1: confirms configuration, paths and logging all work.

Run from the repository root:
    python scripts/verify_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG, PATHS, RANDOM_SEED  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    logger.info("Project: %s", CONFIG["project"]["name"])
    logger.info("Repository root: %s", PATHS.root)
    logger.info("Random seed: %d", RANDOM_SEED)

    expected_dirs = [
        PATHS.data_raw,
        PATHS.data_interim,
        PATHS.data_processed,
        PATHS.models,
        PATHS.figures,
        PATHS.sql,
    ]
    missing = [str(p) for p in expected_dirs if not p.is_dir()]
    if missing:
        logger.error("Missing expected directories: %s", ", ".join(missing))
        return 1
    logger.info("All %d expected directories present.", len(expected_dirs))

    raw_file = PATHS.data_raw / CONFIG["data"]["raw_file"]
    if raw_file.is_file():
        size_mb = raw_file.stat().st_size / 1_000_000
        logger.info("Raw dataset found: %s (%.1f MB)", raw_file.name, size_mb)
    else:
        logger.warning("Raw dataset not found at %s — see BUILD_LOG.md for download steps.", raw_file)

    logger.info("Setup verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
