from __future__ import annotations

import json
from pathlib import Path

from ragms.storage.traces import JsonlTraceWriter


def test_jsonl_trace_writer_appends_complete_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "traces.jsonl"
    writer = JsonlTraceWriter(path)

    writer.write({"trace_id": "trace-1", "trace_type": "query", "status": "succeeded"})
    writer.write({"trace_id": "trace-2", "trace_type": "ingestion", "status": "failed"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["trace_id"] == "trace-1"
    assert json.loads(lines[1])["trace_type"] == "ingestion"
