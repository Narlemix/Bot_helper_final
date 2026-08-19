"""Central logging setup so every module logs the same way, in one place.

Deliberately writes to **stdout**, not a file: on a PaaS platform like Railway
(and in plain `docker compose logs`), the "Logs" tab is just the container's
stdout/stderr — there's no log file to create, rotate, or ship anywhere.
Whoever has access to the hosting dashboard can watch routing decisions live
without touching the code or SSH-ing into anything.

All bot loggers live under the "hrbot" namespace (`hrbot.classifier`,
`hrbot.dialog`, ...) so `LOG_LEVEL` in `.env` controls all of them at once,
and so their lines are easy to grep for separately from uvicorn's own
request-access logs.
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent — safe to call from main.py even if imported more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )

    logger = logging.getLogger("hrbot")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False  # don't also hand lines to uvicorn's root logger (avoids duplicate lines)

    _CONFIGURED = True
