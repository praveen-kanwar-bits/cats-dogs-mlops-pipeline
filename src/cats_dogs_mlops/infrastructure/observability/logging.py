from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_json(logger: logging.Logger, payload: dict) -> None:
    record = dict(payload)
    record.setdefault("timestamp", datetime.now(UTC).isoformat())
    record.setdefault("level", "INFO")
    record.setdefault("logger", logger.name)
    logger.info(json.dumps(record, default=str))
