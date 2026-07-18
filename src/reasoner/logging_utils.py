"""Structured logging setup. One place to control format/level so every
module just does `logger = logging.getLogger(__name__)` and inherits it."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger("reasoner")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root.handlers:
        return  # avoid duplicate handlers if called more than once (e.g. in tests)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
