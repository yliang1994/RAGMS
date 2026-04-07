"""Stage recorder responsible for turning runtime facts into stable stage traces."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .trace_schema import StageTrace
from .trace_utils import (
    build_input_summary,
    build_output_summary,
    elapsed_ms,
    format_timestamp,
    sanitize_metadata,
    serialize_exception,
)


class StageRecorder:
    """Build stable, JSON-safe stage traces from raw runtime inputs and outputs."""

    def record_stage(
        self,
        *,
        stage_name: str,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        input_payload: Any = None,
        output_payload: Any = None,
        metadata: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> StageTrace:
        """Normalize one stage result into the shared stage schema."""

        return StageTrace(
            stage_name=stage_name,
            status=status,
            started_at=format_timestamp(started_at),
            finished_at=format_timestamp(finished_at),
            elapsed_ms=elapsed_ms(started_at, finished_at),
            input_summary=build_input_summary(input_payload),
            output_summary=build_output_summary(output_payload),
            metadata=sanitize_metadata(metadata),
            error=serialize_exception(error),
        )
