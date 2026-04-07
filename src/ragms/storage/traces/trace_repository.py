"""Repository for append-only trace persistence and tolerant readback."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from ragms.storage.traces.jsonl_writer import JsonlTraceWriter
from ragms.observability.logging.logger import get_trace_logger
from ragms.core.trace_collector.trace_utils import serialize_exception


class TraceRepository:
    """Persist and query complete traces stored as JSON Lines."""

    REQUIRED_FIELDS = {
        "trace_id",
        "trace_type",
        "status",
        "started_at",
        "finished_at",
        "duration_ms",
        "collection",
        "metadata",
        "error",
        "stages",
    }

    def __init__(
        self,
        traces_file: str | Path,
        *,
        writer: JsonlTraceWriter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.traces_file = Path(traces_file)
        self.writer = writer or JsonlTraceWriter(self.traces_file)
        self.logger = logger or get_trace_logger(name="ragms.trace_repository")

    def append(self, trace: Mapping[str, Any] | Any) -> bool:
        """Append one trace record, warning instead of raising on write failures."""

        try:
            self.writer.write(trace)
        except Exception as exc:
            trace_id = None
            if isinstance(trace, Mapping):
                trace_id = trace.get("trace_id")
            elif hasattr(trace, "trace_id"):
                trace_id = getattr(trace, "trace_id")
            self.logger.warning(
                "Failed to append trace record",
                extra={
                    "trace_id": trace_id,
                    "error": serialize_exception(exc),
                    "traces_file": str(self.traces_file),
                },
            )
            return False
        return True

    def get_by_trace_id(self, trace_id: str) -> dict[str, Any] | None:
        """Return the matching trace detail or None when absent."""

        matched: dict[str, Any] | None = None
        for payload in self._iter_valid_traces():
            if str(payload.get("trace_id")) == trace_id:
                matched = payload
        return matched

    def list_traces(
        self,
        *,
        trace_type: str | None = None,
        status: str | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List traces filtered by type, status, and collection."""

        matched: list[dict[str, Any]] = []
        for payload in self._iter_valid_traces():
            if trace_type is not None and payload.get("trace_type") != trace_type:
                continue
            if status is not None and payload.get("status") != status:
                continue
            if collection is not None and payload.get("collection") != collection:
                continue
            matched.append(payload)

        matched.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
        if limit is not None:
            return matched[:limit]
        return matched

    def _iter_valid_traces(self) -> list[dict[str, Any]]:
        if not self.traces_file.exists():
            return []

        valid: list[dict[str, Any]] = []
        with self.traces_file.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._warn_invalid_line(
                        line_number=line_number,
                        reason=f"json_decode_error:{exc.msg}",
                    )
                    continue
                if not isinstance(payload, dict):
                    self._warn_invalid_line(
                        line_number=line_number,
                        reason="trace_line_not_mapping",
                    )
                    continue
                if not self.REQUIRED_FIELDS.issubset(payload):
                    self._warn_invalid_line(
                        line_number=line_number,
                        reason="trace_line_missing_required_fields",
                    )
                    continue
                valid.append(payload)
        return valid

    def _warn_invalid_line(self, *, line_number: int, reason: str) -> None:
        self.logger.warning(
            "Skipping invalid trace line",
            extra={
                "line_number": line_number,
                "reason": reason,
                "traces_file": str(self.traces_file),
            },
        )
