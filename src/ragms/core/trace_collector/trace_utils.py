"""Utilities for producing JSON-safe, redacted trace payloads."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any


REDACTED = "[REDACTED]"
_MAX_STRING_LENGTH = 240
_MAX_SUMMARY_KEYS = 10
_MAX_SUMMARY_PREVIEW = 6
_MAX_SEQUENCE_SAMPLE = 3
_MAX_METADATA_DEPTH = 4
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "token",
    "secret",
    "password",
    "access_token",
    "refresh_token",
)
_TOKEN_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)([^\s,;]+)"),
    re.compile(r"sk-[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(api[_ -]?key|token|password|secret)\s*[:=]\s*([^\s,;]+)"),
)


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def format_timestamp(value: datetime) -> str:
    """Render a UTC ISO-8601 timestamp with millisecond precision."""

    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def elapsed_ms(started_at: datetime, finished_at: datetime) -> int:
    """Return elapsed milliseconds between two timestamps."""

    return max(0, int(round((finished_at - started_at).total_seconds() * 1000)))


def serialize_exception(error: BaseException | None) -> dict[str, Any] | None:
    """Normalize an exception into a JSON-safe, redacted structure."""

    if error is None:
        return None

    payload: dict[str, Any] = {
        "type": error.__class__.__name__,
        "module": error.__class__.__module__,
        "message": _redact_text(str(error)),
    }
    cause = error.__cause__ or error.__context__
    if cause is not None and cause is not error:
        payload["cause"] = {
            "type": cause.__class__.__name__,
            "message": _redact_text(str(cause)),
        }
    return payload


def build_input_summary(payload: Any) -> Any:
    """Build a compact JSON-safe summary for stage inputs."""

    return _summarize_value(payload, depth=0, key_hint=None)


def build_output_summary(payload: Any) -> Any:
    """Build a compact JSON-safe summary for stage outputs."""

    return _summarize_value(payload, depth=0, key_hint=None)


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize a metadata mapping without collapsing it into a summary wrapper."""

    normalized = dict(metadata or {})
    return {
        str(key): _sanitize_json_value(value, depth=0, key_hint=str(key))
        for key, value in normalized.items()
    }


def _sanitize_json_value(value: Any, *, depth: int, key_hint: str | None) -> Any:
    if key_hint is not None and _is_sensitive_key(key_hint):
        return REDACTED

    value = _coerce_supported_value(value)
    if depth >= _MAX_METADATA_DEPTH:
        return _terminal_value(value)

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_string(_redact_text(value))
    if isinstance(value, dict):
        return {
            str(key): _sanitize_json_value(item, depth=depth + 1, key_hint=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_json_value(item, depth=depth + 1, key_hint=None)
            for item in value[:_MAX_SUMMARY_KEYS]
        ]
    return _terminal_value(value)


def _summarize_value(value: Any, *, depth: int, key_hint: str | None) -> Any:
    if key_hint is not None and _is_sensitive_key(key_hint):
        return REDACTED

    value = _coerce_supported_value(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_string(_redact_text(value))
    if isinstance(value, dict):
        keys = [str(key) for key in value.keys()]
        preview = {
            str(key): _summarize_value(value[key], depth=depth + 1, key_hint=str(key))
            for key in list(value.keys())[:_MAX_SUMMARY_PREVIEW]
        }
        return {
            "kind": "mapping",
            "size": len(value),
            "keys": keys[:_MAX_SUMMARY_KEYS],
            "preview": preview,
        }
    if isinstance(value, list):
        return {
            "kind": "sequence",
            "count": len(value),
            "sample": [
                _summarize_value(item, depth=depth + 1, key_hint=None)
                for item in value[:_MAX_SEQUENCE_SAMPLE]
            ],
        }
    return _terminal_value(value)


def _coerce_supported_value(value: Any) -> Any:
    if isinstance(value, BaseException):
        return serialize_exception(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value, key=str)
    return value


def _terminal_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {"kind": "mapping", "size": len(value)}
    if isinstance(value, list):
        return {"kind": "sequence", "count": len(value)}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_string(_redact_text(value))
    return _truncate_string(_redact_text(str(value)))


def _truncate_string(value: str) -> str:
    if len(value) <= _MAX_STRING_LENGTH:
        return value
    return f"{value[:_MAX_STRING_LENGTH]}..."


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _TOKEN_PATTERNS:
        if pattern.pattern.startswith("(?i)(bearer"):
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        elif pattern.pattern.startswith("(?i)(api"):
            redacted = pattern.sub(r"\1=[REDACTED]", redacted)
        else:
            redacted = pattern.sub(REDACTED, redacted)
    return redacted


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)
