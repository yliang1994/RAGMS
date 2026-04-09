from __future__ import annotations

from pathlib import Path

from ragms.mcp_server.tools.traces import handle_get_trace_detail, serialize_trace_detail
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings


class StubTraceService:
    def __init__(self, payload: dict[str, object] | None = None, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[str] = []

    def get_trace_detail(self, trace_id: str) -> dict[str, object]:
        self.calls.append(trace_id)
        if self.error is not None:
            raise self.error
        assert self.payload is not None
        return dict(self.payload)


def _runtime(tmp_path: Path) -> ServiceContainer:
    settings = AppSettings.model_validate({"environment": "test"})
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.dashboard.traces_file = tmp_path / "logs" / "traces.jsonl"
    return ServiceContainer(settings=settings, services={})


def test_serialize_trace_detail_preserves_stage_payloads() -> None:
    detail = serialize_trace_detail(
        {
            "trace_id": "trace-1",
            "trace_type": "query",
            "status": "succeeded",
            "duration_ms": 15,
            "stages": [
                {
                    "stage_name": "query_processing",
                    "status": "succeeded",
                    "input_summary": {"query": "what is rag"},
                    "output_summary": {"normalized_query": "what is rag"},
                    "metadata": {"method": "normalize"},
                    "error": None,
                }
            ],
        }
    )

    assert detail["trace_id"] == "trace-1"
    assert detail["stages"][0]["stage_name"] == "query_processing"
    assert detail["stages"][0]["metadata"]["method"] == "normalize"


def test_handle_get_trace_detail_wraps_trace_payload(tmp_path: Path) -> None:
    service = StubTraceService(
        {
            "trace_id": "trace-1",
            "trace_type": "ingestion",
            "status": "succeeded",
            "duration_ms": 25,
            "stages": [{"stage_name": "file_integrity", "metadata": {}, "error": None}],
            "metadata": {},
            "error": None,
        }
    )

    result = handle_get_trace_detail(
        trace_id="trace-1",
        runtime=_runtime(tmp_path),
        trace_service=service,
    )

    assert service.calls == ["trace-1"]
    assert result.isError is False
    assert result.structuredContent["trace_id"] == "trace-1"
    assert result.structuredContent["stages"][0]["stage_name"] == "file_integrity"


def test_handle_get_trace_detail_returns_error_for_missing_trace(tmp_path: Path) -> None:
    from ragms.core.management.trace_service import TraceNotFoundError

    service = StubTraceService(error=TraceNotFoundError("Trace not found: missing"))

    result = handle_get_trace_detail(
        trace_id="missing",
        runtime=_runtime(tmp_path),
        trace_service=service,
    )

    assert result.isError is True
    assert result.structuredContent["error"]["code"] == -32602
    assert result.structuredContent["error"]["message"] == "Trace not found: missing"
