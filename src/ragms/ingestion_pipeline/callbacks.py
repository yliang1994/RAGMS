"""Pipeline callback protocol and event payloads for ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageEvent:
    """Stage boundary event emitted before and after each ingestion step."""

    stage: str
    source_path: str
    index: int
    total: int
    status: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressEvent:
    """Coarse-grained progress update for dashboard or logging hooks."""

    source_path: str
    completed_stages: int
    total_stages: int
    current_stage: str
    status: str


class PipelineCallback:
    """Default no-op callback base used by ingestion pipeline hooks."""

    def on_stage_start(self, event: StageEvent) -> None:
        """Handle a stage start event."""

    def on_stage_end(self, event: StageEvent) -> None:
        """Handle a stage completion or failure event."""

    def on_progress(self, event: ProgressEvent) -> None:
        """Handle a coarse-grained pipeline progress update."""
