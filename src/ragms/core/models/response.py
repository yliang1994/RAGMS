"""Structured response contracts shared by query, MCP, and evaluation flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryResponsePayload:
    """Stable query response structure that can be serialized into reports."""

    query: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str | None = None
    fallback_applied: bool = False
    fallback_reason: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe response payload."""

        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [dict(item) for item in self.citations],
            "retrieved_chunks": [dict(item) for item in self.retrieved_chunks],
            "trace_id": self.trace_id,
            "fallback_applied": self.fallback_applied,
            "fallback_reason": self.fallback_reason,
            "config_snapshot": dict(self.config_snapshot),
        }
