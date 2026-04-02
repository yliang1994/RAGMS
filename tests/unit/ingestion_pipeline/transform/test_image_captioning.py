from __future__ import annotations

import sqlite3
from pathlib import Path

from ragms.ingestion_pipeline.transform import generate_image_caption, inject_image_caption
from ragms.storage.sqlite.repositories import ProcessingCacheRepository
from tests.fakes.fake_vision_llm import FakeVisionLLM


def _chunk(
    content: str,
    *,
    image_refs: list[str] | None = None,
    images: list[dict[str, object]] | None = None,
    image_occurrences: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "content": content,
        "source_path": "docs/report.md",
        "chunk_index": 0,
        "start_offset": 0,
        "end_offset": len(content),
        "image_refs": list(image_refs or []),
        "metadata": {
            "images": list(images or []),
            "image_occurrences": list(image_occurrences or []),
        },
    }


def _repository() -> ProcessingCacheRepository:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    return ProcessingCacheRepository(connection)


def test_inject_image_caption_skips_chunks_without_images() -> None:
    vision_llm = FakeVisionLLM()
    chunk = _chunk("Plain text without images.")

    result = inject_image_caption([chunk], vision_llm=vision_llm)

    assert result == [chunk]
    assert vision_llm.calls == []


def test_inject_image_caption_injects_single_image_caption_by_occurrence(tmp_path: Path) -> None:
    image_path = tmp_path / "chart-1.png"
    image_path.write_bytes(b"fake-png")
    content = "Overview [IMAGE: chart-1] explains the trend."
    chunk = _chunk(
        content,
        image_refs=["chart-1"],
        images=[{"id": "chart-1", "path": str(image_path), "page": 1}],
        image_occurrences=[
            {
                "image_id": "chart-1",
                "text_offset": 9,
                "text_length": len("[IMAGE: chart-1]"),
            }
        ],
    )

    result = inject_image_caption([chunk], vision_llm=FakeVisionLLM())

    assert "[Image caption: caption:chart-1" in result[0]["content"]
    assert result[0]["metadata"]["image_captions"][0]["image_id"] == "chart-1"
    assert result[0]["metadata"]["image_captions"][0]["cached"] is False
    assert result[0]["metadata"]["caption_prompt_version"] == "image_caption_v1"
    assert "has_unprocessed_images" not in result[0]["metadata"]


def test_inject_image_caption_processes_multiple_images_and_reuses_cache(tmp_path: Path) -> None:
    first_image = tmp_path / "chart-1.png"
    second_image = tmp_path / "chart-2.png"
    first_image.write_bytes(b"same-image")
    second_image.write_bytes(b"same-image")
    repository = _repository()
    vision_llm = FakeVisionLLM()
    first_chunk = _chunk(
        "A [IMAGE: chart-1]",
        image_refs=["chart-1"],
        images=[{"id": "chart-1", "path": str(first_image)}],
        image_occurrences=[
            {"image_id": "chart-1", "text_offset": 2, "text_length": len("[IMAGE: chart-1]")}
        ],
    )
    second_chunk = _chunk(
        "B [IMAGE: chart-2]",
        image_refs=["chart-2"],
        images=[{"id": "chart-2", "path": str(second_image)}],
        image_occurrences=[
            {"image_id": "chart-2", "text_offset": 2, "text_length": len("[IMAGE: chart-2]")}
        ],
    )

    result = inject_image_caption(
        [first_chunk, second_chunk],
        vision_llm=vision_llm,
        cache_repository=repository,
        model="fake-vision",
        prompt_version="caption-v2",
    )

    assert len(vision_llm.calls) == 1
    assert result[0]["metadata"]["image_captions"][0]["cached"] is False
    assert result[1]["metadata"]["image_captions"][0]["cached"] is True
    cached = repository.get_caption(
        image_hash=result[0]["metadata"]["image_captions"][0]["image_hash"],
        model="fake-vision",
        prompt_version="caption-v2",
    )
    assert cached is not None
    assert cached["payload"] == "caption:chart-1 context=A [IMAGE: chart-1]"


def test_inject_image_caption_marks_failures_without_blocking_document(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"
    chunk = _chunk(
        "See [IMAGE: chart-1]",
        image_refs=["chart-1"],
        images=[{"id": "chart-1", "path": str(missing_path)}],
        image_occurrences=[
            {"image_id": "chart-1", "text_offset": 4, "text_length": len("[IMAGE: chart-1]")}
        ],
    )

    result = inject_image_caption([chunk], vision_llm=FakeVisionLLM())

    assert result[0]["content"] == "See [IMAGE: chart-1]"
    assert result[0]["image_refs"] == ["chart-1"]
    assert result[0]["metadata"]["has_unprocessed_images"] is True
    assert result[0]["metadata"]["image_caption_failures"][0]["image_id"] == "chart-1"


def test_generate_image_caption_reuses_cached_entry(tmp_path: Path) -> None:
    image_path = tmp_path / "chart-1.png"
    image_path.write_bytes(b"same-image")
    repository = _repository()
    vision_llm = FakeVisionLLM()

    first = generate_image_caption(
        image_path,
        vision_llm=vision_llm,
        cache_repository=repository,
        model="fake-vision",
        prompt_version="caption-v1",
    )
    second = generate_image_caption(
        image_path,
        vision_llm=vision_llm,
        cache_repository=repository,
        model="fake-vision",
        prompt_version="caption-v1",
    )

    assert first["cached"] is False
    assert second["cached"] is True
    assert len(vision_llm.calls) == 1
