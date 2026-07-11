"""Uniform logging: console always, plus a log file inside each run directory."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logger


def add_run_file_handler(run_dir: str | Path) -> Path:
    """Mirror all log output into <run_dir>/run.log; returns the log path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.getLogger().addHandler(handler)
    return log_path
