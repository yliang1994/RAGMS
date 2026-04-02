from __future__ import annotations

import pytest

from ragms.ingestion_pipeline.transform import TransformPipeline, TransformStepError


class RecordingStep:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, object]] = []

    def transform(self, chunks, *, context=None):
        self.calls.append(
            {
                "chunks": [dict(chunk) for chunk in chunks],
                "context": dict(context or {}),
            }
        )
        updated = []
        for chunk in chunks:
            item = dict(chunk)
            metadata = dict(item.get("metadata") or {})
            metadata[f"step_{self.name}"] = True
            item["metadata"] = metadata
            item["content"] = f"{item['content']}|{self.name}"
            updated.append(item)
        return updated


class ExplodingStep:
    def __init__(self, name: str = "explode") -> None:
        self.name = name

    def transform(self, chunks, *, context=None):
        del chunks, context
        raise ValueError(self.name)


def _base_chunks():
    return [
        {
            "content": "alpha",
            "chunk_id": "chunk-1",
            "metadata": {"document_id": "doc-1"},
        }
    ]


def test_transform_pipeline_runs_steps_in_order() -> None:
    smart = RecordingStep("smart")
    metadata = RecordingStep("meta")
    caption = RecordingStep("caption")
    pipeline = TransformPipeline(
        smart_chunk_builder=smart,
        metadata_injector=metadata,
        image_captioner=caption,
    )

    result = pipeline.run(_base_chunks(), context={"source_path": "docs/a.md"})

    assert result[0]["content"] == "alpha|smart|meta|caption"
    assert result[0]["metadata"]["step_smart"] is True
    assert result[0]["metadata"]["step_meta"] is True
    assert result[0]["metadata"]["step_caption"] is True
    assert smart.calls[0]["context"]["source_path"] == "docs/a.md"
    assert metadata.calls[0]["chunks"][0]["content"] == "alpha|smart"
    assert caption.calls[0]["chunks"][0]["content"] == "alpha|smart|meta"


def test_transform_pipeline_skips_missing_steps() -> None:
    pipeline = TransformPipeline()

    result = pipeline.run(_base_chunks())

    assert result == _base_chunks()


def test_transform_pipeline_degrades_when_a_step_fails_in_fail_open_mode() -> None:
    smart = RecordingStep("smart")
    pipeline = TransformPipeline(
        smart_chunk_builder=smart,
        metadata_injector=ExplodingStep("metadata failed"),
        image_captioner=RecordingStep("caption"),
        fail_open=True,
    )

    result = pipeline.run(_base_chunks(), context={"collection": "demo"})

    assert result[0]["content"] == "alpha|smart|caption"
    assert result[0]["metadata"]["transform_warnings"] == ["metadata_injector:metadata failed"]
    assert result[0]["metadata"]["step_caption"] is True


def test_transform_pipeline_raises_when_fail_open_is_disabled() -> None:
    pipeline = TransformPipeline(
        smart_chunk_builder=ExplodingStep("smart failed"),
        fail_open=False,
    )

    with pytest.raises(TransformStepError, match="Transform step failed: smart_chunk_builder"):
        pipeline.run(_base_chunks())
