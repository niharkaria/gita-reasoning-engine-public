"""Structured logging configuration.

Why this file exists:
    Every pipeline stage (ingestion, retrieval, reasoning, generation) needs
    to emit logs that are actually queryable later — "which query triggered
    this retrieval failure" is a search you want to run against structured
    JSON, not grep against free-text prints. This module configures
    `structlog` once, at process startup, so every module can just do
    `logger = get_logger(__name__)` and get consistent, structured output.

Design notes:
    - JSON renderer in production/staging, human-readable console renderer
      in development (readable while coding, machine-parseable when deployed).
    - Configured once via `configure_logging()`, called from the API startup
      hook and from Kaggle notebook entrypoints.
"""

import logging
import sys
from typing import cast

import structlog

from gita_engine.core.config import get_settings


def configure_logging() -> None:
    """Configure structlog + stdlib logging based on current settings."""
    settings = get_settings()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.app_env in ("staging", "production")
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given module name."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
