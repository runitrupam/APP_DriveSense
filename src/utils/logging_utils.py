"""Logging helpers writing to console and an optional per-run file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "deepsense",
    log_file: str | Path | None = None,
    level: str | int = "INFO",
) -> logging.Logger:
    """Create or reconfigure a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configured[name] = logger
    return logger


def get_logger(name: str = "deepsense") -> logging.Logger:
    """Return a configured logger, creating a console-only one if needed."""
    if name in _configured:
        return _configured[name]
    return setup_logger(name)
