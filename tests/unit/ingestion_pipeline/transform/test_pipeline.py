from __future__ import annotations

import pytest

from ragms.ingestion_pipeline.transform import TransformPipeline, TransformStepError
from ragms.ingestion_pipeline.transform.smart_chunk_builder import SmartChunkBuilder


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


class MutatingAnchorStep:
    def transform(self, chunks, *, context=None):
        del context
        updated = []
        for chunk in chunks:
            item = dict(chunk)
            item["chunk_id"] = "mutated-chunk"
            item["document_id"] = "mutated-doc"
            item["source_path"] = "mutated-path"
            item["chunk_index"] = 999
            item["start_offset"] = 111
            item["end_offset"] = 222
            item["source_ref"] = "mutated-source-ref"
            item["image_refs"] = ["mutated-image"]
            metadata = dict(item.get("metadata") or {})
            metadata["image_refs"] = ["mutated-image"]
            metadata["step_mutated"] = True
            item["metadata"] = metadata
            item["content"] = f"{item['content']}|mutated"
            updated.append(item)
        return updated


def _base_chunks():
    return [
        {
            "content": "alpha",
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "source_path": "docs/a.md",
            "chunk_index": 0,
            "start_offset": 0,
            "end_offset": 5,
            "source_ref": "doc-1",
            "image_refs": ["img-1"],
            "metadata": {"document_id": "doc-1", "image_refs": ["img-1"]},
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
    assert result[0]["metadata"]["transform_fallbacks"] == [
        {"step": "metadata_injector", "reason": "metadata failed"}
    ]
    assert result[0]["metadata"]["step_caption"] is True


def test_transform_pipeline_raises_when_fail_open_is_disabled() -> None:
    pipeline = TransformPipeline(
        smart_chunk_builder=ExplodingStep("smart failed"),
        fail_open=False,
    )

    with pytest.raises(TransformStepError, match="Transform step failed: smart_chunk_builder"):
        pipeline.run(_base_chunks())


def test_transform_pipeline_restores_anchor_fields_when_steps_mutate_them() -> None:
    pipeline = TransformPipeline(
        smart_chunk_builder=MutatingAnchorStep(),
    )

    result = pipeline.run(_base_chunks())

    assert result[0]["content"] == "alpha|mutated"
    assert result[0]["chunk_id"] == "chunk-1"
    assert result[0]["document_id"] == "doc-1"
    assert result[0]["source_path"] == "docs/a.md"
    assert result[0]["chunk_index"] == 0
    assert result[0]["start_offset"] == 0
    assert result[0]["end_offset"] == 5
    assert result[0]["source_ref"] == "doc-1"
    assert result[0]["image_refs"] == ["img-1"]
    assert result[0]["metadata"]["image_refs"] == ["img-1"]
    assert result[0]["metadata"]["document_id"] == "doc-1"
    assert result[0]["metadata"]["step_mutated"] is True


def test_transform_pipeline_allows_smart_chunk_builder_to_merge_chunk_groups() -> None:
    smart_builder = SmartChunkBuilder(merge_below_chars=120)
    metadata = RecordingStep("meta")
    pipeline = TransformPipeline(
        smart_chunk_builder=smart_builder,
        metadata_injector=metadata,
    )

    result = pipeline.run(
        [
            {
                "content": "This policy applies to all employees because",
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source_path": "docs/a.md",
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": 44,
                "source_ref": "doc-1",
                "image_refs": [],
                "metadata": {"document_id": "doc-1", "image_refs": []},
            },
            {
                "content": "it defines the approval flow and exceptions.",
                "chunk_id": "chunk-2",
                "document_id": "doc-1",
                "source_path": "docs/a.md",
                "chunk_index": 1,
                "start_offset": 45,
                "end_offset": 90,
                "source_ref": "doc-1",
                "image_refs": [],
                "metadata": {"document_id": "doc-1", "image_refs": []},
            },
        ]
    )

    assert len(result) == 1
    assert result[0]["chunk_id"] == "chunk-1"
    assert result[0]["content"] == (
        "This policy applies to all employees because\n\n"
        "it defines the approval flow and exceptions.|meta"
    )
    assert result[0]["metadata"]["smart_chunk"]["merged_from"] == ["chunk-1", "chunk-2"]
    assert result[0]["metadata"]["chunk_id"] == "chunk-1"
    assert result[0]["metadata"]["step_meta"] is True
