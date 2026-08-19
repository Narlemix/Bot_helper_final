from __future__ import annotations

import logging
import os
import sys

from .log_buffer import BufferHandler

_CONFIGURED = False


def configure_logging() -> None:
    """Настраивает логгер hrbot (stdout + буфер для /admin/logs). Идемпотентно."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )

    logger = logging.getLogger("hrbot")
    logger.setLevel(level)
    logger.addHandler(stdout_handler)
    logger.addHandler(BufferHandler())
    logger.propagate = False

    _CONFIGURED = True
