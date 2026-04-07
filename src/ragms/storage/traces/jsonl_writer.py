"""Append-only JSONL writer for complete trace records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class JsonlTraceWriter:
    """Persist one complete trace record per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, trace: Mapping[str, Any] | Any) -> Path:
        """Append one complete JSON-serializable trace line."""

        payload = _normalize_payload(trace)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return self.path


def _normalize_payload(trace: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(trace, "to_dict"):
        payload = trace.to_dict()
    elif isinstance(trace, Mapping):
        payload = dict(trace)
    else:  # pragma: no cover - defensive guard
        raise TypeError("trace must be a mapping or expose to_dict()")
    if "trace_id" not in payload:
        raise ValueError("trace payload must include trace_id")
    return payload
