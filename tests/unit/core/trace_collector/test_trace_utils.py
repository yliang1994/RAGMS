from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragms.core.trace_collector.trace_utils import (
    build_input_summary,
    build_output_summary,
    serialize_exception,
)


@dataclass
class _Payload:
    chunk_id: str
    score: float


def test_serialize_exception_redacts_sensitive_tokens() -> None:
    try:
        raise RuntimeError("api_key=secret-value and Bearer abc.def")
    except RuntimeError as exc:
        payload = serialize_exception(exc)

    assert payload is not None
    assert payload["type"] == "RuntimeError"
    assert payload["message"] == "api_key=[REDACTED] and Bearer [REDACTED]"


def test_build_summaries_truncate_large_values_and_normalize_non_json_objects() -> None:
    payload = {
        "path": Path("/tmp/demo.pdf"),
        "embedding": [0.1] * 8,
        "api_key": "sk-secret",
        "blob": b"123456",
        "records": [_Payload(chunk_id="chunk-1", score=0.95)],
        "content": "x" * 400,
    }

    input_summary = build_input_summary(payload)
    output_summary = build_output_summary(payload)

    assert input_summary["kind"] == "mapping"
    assert input_summary["preview"]["path"] == "/tmp/demo.pdf"
    assert input_summary["preview"]["api_key"] == "[REDACTED]"
    assert input_summary["preview"]["embedding"]["kind"] == "sequence"
    assert input_summary["preview"]["embedding"]["count"] == 8
    assert input_summary["preview"]["blob"]["preview"]["length"] == 6
    assert input_summary["preview"]["records"]["sample"][0]["preview"]["chunk_id"] == "chunk-1"
    assert input_summary["preview"]["content"].endswith("...")
    assert output_summary == input_summary
