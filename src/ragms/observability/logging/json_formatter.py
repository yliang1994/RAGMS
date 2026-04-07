"""Stable JSON formatter for structured application and trace-adjacent logs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any

from ragms.core.trace_collector.trace_utils import sanitize_metadata, serialize_exception


_STANDARD_LOG_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    """Format log records into a stable JSON structure."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the record into one JSON object per line."""

        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload: dict[str, Any] = {
            "timestamp": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_KEYS and not key.startswith("_")
        }
        if extras:
            payload["context"] = sanitize_metadata(extras)

        if record.exc_info:
            payload["exception"] = serialize_exception(record.exc_info[1])
        elif record.exc_text:
            payload["exception"] = {"type": "Exception", "message": str(record.exc_text)}

        return json.dumps(payload, ensure_ascii=False)
