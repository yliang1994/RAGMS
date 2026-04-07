from __future__ import annotations

import json
from pathlib import Path

from ragms.storage.traces import TraceRepository


def _trace_payload(
    *,
    trace_id: str,
    trace_type: str,
    status: str,
    collection: str | None,
    started_at: str,
) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "trace_type": trace_type,
        "status": status,
        "started_at": started_at,
        "finished_at": started_at,
        "duration_ms": 12,
        "collection": collection,
        "metadata": {},
        "error": None,
        "stages": [],
    }


def test_trace_repository_appends_and_reads_back_by_trace_id(tmp_path: Path) -> None:
    repository = TraceRepository(tmp_path / "logs" / "traces.jsonl")
    payload = _trace_payload(
        trace_id="trace-1",
        trace_type="query",
        status="succeeded",
        collection="docs",
        started_at="2026-04-07T10:00:00.000Z",
    )

    assert repository.append(payload) is True
    assert repository.get_by_trace_id("trace-1") == payload


def test_trace_repository_filters_and_skips_invalid_lines(tmp_path: Path) -> None:
    traces_file = tmp_path / "logs" / "traces.jsonl"
    traces_file.parent.mkdir(parents=True, exist_ok=True)
    traces_file.write_text(
        "\n".join(
            [
                json.dumps(
                    _trace_payload(
                        trace_id="trace-old",
                        trace_type="query",
                        status="failed",
                        collection="docs",
                        started_at="2026-04-07T09:00:00.000Z",
                    )
                ),
                "{bad json",
                json.dumps({"trace_id": "broken"}),
                json.dumps(
                    _trace_payload(
                        trace_id="trace-new",
                        trace_type="ingestion",
                        status="skipped",
                        collection="reports",
                        started_at="2026-04-07T11:00:00.000Z",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repository = TraceRepository(traces_file)

    listed = repository.list_traces()
    ingestion_only = repository.list_traces(trace_type="ingestion")
    docs_failed = repository.list_traces(status="failed", collection="docs")

    assert [item["trace_id"] for item in listed] == ["trace-new", "trace-old"]
    assert [item["trace_id"] for item in ingestion_only] == ["trace-new"]
    assert [item["trace_id"] for item in docs_failed] == ["trace-old"]
    assert repository.get_by_trace_id("missing") is None
