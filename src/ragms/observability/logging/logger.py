"""Helpers for building structured loggers without duplicate handlers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from ragms.observability.logging.json_formatter import JsonFormatter


def get_trace_logger(
    *,
    name: str = "ragms.trace",
    log_file: str | Path | None = None,
    level: str | int = "INFO",
) -> logging.Logger:
    """Return a structured logger configured for trace-adjacent warnings."""

    logger = logging.getLogger(name)
    logger.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    target_log_file = None if log_file is None else Path(log_file)
    handler = _find_existing_handler(logger, target_log_file)
    if handler is None:
        handler = _build_handler(target_log_file)
        logger.handlers = [
            existing
            for existing in logger.handlers
            if not getattr(existing, "_ragms_managed", False)
        ]
        logger.addHandler(handler)
    return logger


def _build_handler(log_file: Path | None) -> logging.Handler:
    if log_file is None:
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        setattr(handler, "_ragms_log_file", str(log_file))
    setattr(handler, "_ragms_managed", True)
    handler.setFormatter(JsonFormatter())
    return handler


def _find_existing_handler(logger: logging.Logger, log_file: Path | None) -> logging.Handler | None:
    expected = None if log_file is None else str(log_file)
    for handler in logger.handlers:
        if not getattr(handler, "_ragms_managed", False):
            continue
        if getattr(handler, "_ragms_log_file", None) == expected:
            return handler
        if expected is None and isinstance(handler, logging.StreamHandler):
            return handler
    return None
