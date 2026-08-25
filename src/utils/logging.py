"""Project-wide logging setup.

Every script and module gets its logger from :func:`get_logger` so that output
format, level and file destination are configured in exactly one place.

Usage
-----
    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Loaded %d rows", len(df))
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from src.config import CONFIG, PATHS

_CONFIGURED = False


def _configure_root() -> None:
    """Attach handlers to the root logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_cfg = CONFIG.get("logging", {})
    # Environment variable wins so containers/CI can turn up verbosity
    # without editing the config file.
    level_name = os.getenv("LOG_LEVEL", log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(
        fmt=log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        datefmt=log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S"),
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_cfg.get("log_to_file", True):
        try:
            PATHS.logs.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                PATHS.logs / log_cfg.get("filename", "pipeline.log"),
                maxBytes=5_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            # A read-only filesystem must not take down the pipeline;
            # console logging is enough to keep running.
            root.warning("File logging disabled (%s)", exc)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name`` (pass ``__name__``)."""
    _configure_root()
    return logging.getLogger(name)
