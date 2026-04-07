from __future__ import annotations

import json
import logging

from ragms.observability.logging import JsonFormatter


def test_json_formatter_emits_stable_fields_and_redacts_sensitive_context() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="ragms.trace",
        level=logging.WARNING,
        pathname=__file__,
        lineno=10,
        msg="write failed",
        args=(),
        exc_info=None,
    )
    record.trace_id = "trace-1"
    record.api_key = "sk-secret"
    record.collection = "docs"

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "ragms.trace"
    assert payload["message"] == "write failed"
    assert payload["context"]["trace_id"] == "trace-1"
    assert payload["context"]["api_key"] == "[REDACTED]"
    assert payload["context"]["collection"] == "docs"
    assert payload["timestamp"].endswith("Z")


def test_json_formatter_serializes_exception_payload() -> None:
    formatter = JsonFormatter()
    try:
        raise RuntimeError("Bearer secret-token")
    except RuntimeError:
        record = logging.LogRecord(
            name="ragms.trace",
            level=logging.ERROR,
            pathname=__file__,
            lineno=20,
            msg="trace write error",
            args=(),
            exc_info=True,
        )
        record.exc_info = __import__("sys").exc_info()

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "ERROR"
    assert payload["exception"]["type"] == "RuntimeError"
    assert payload["exception"]["message"] == "Bearer [REDACTED]"
