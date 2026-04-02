from __future__ import annotations

from ragms.ingestion_pipeline.chunking import ChunkingPipeline
from ragms.libs.providers.splitters.recursive_character_splitter import RecursiveCharacterSplitter


def _build_document() -> dict[str, object]:
    return {
        "content": (
            "# Title\n\n"
            "Alpha paragraph.\n\n"
            "[IMAGE: chart-1]\n"
            "Chart discussion stays nearby.\n\n"
            "Beta paragraph continues with extra details for another chunk.\n\n"
            "[IMAGE: chart-2]\n"
            "Closing notes."
        ),
        "source_path": "docs/report.md",
        "metadata": {
            "document_id": "doc_123",
            "source_sha256": "a" * 64,
            "heading_outline": ["Title"],
            "images": [
                {"image_id": "chart-1", "page": 1},
                {"image_id": "chart-2", "page": 2},
            ],
        },
    }


def test_chunking_pipeline_builds_stable_chunks_with_offsets_and_ids() -> None:
    pipeline = ChunkingPipeline(
        RecursiveCharacterSplitter(
            chunk_size=90,
            chunk_overlap=10,
        )
    )
    document = _build_document()

    first_run = pipeline.run(document)
    second_run = pipeline.run(document)

    assert len(first_run) >= 2
    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]
    assert first_run[0].document_id == "doc_123"
    assert first_run[0].source_path == "docs/report.md"
    assert first_run[0].start_offset == 0
    assert all(chunk.end_offset > chunk.start_offset for chunk in first_run)


def test_chunking_pipeline_preserves_image_refs_from_placeholder_markers() -> None:
    pipeline = ChunkingPipeline(
        RecursiveCharacterSplitter(
            chunk_size=90,
            chunk_overlap=10,
        )
    )

    chunks = pipeline.run(_build_document())

    image_ref_sets = [chunk.image_refs for chunk in chunks]
    assert any("chart-1" in refs for refs in image_ref_sets)
    assert any("chart-2" in refs for refs in image_ref_sets)
    assert all(chunk.metadata["image_refs"] == chunk.image_refs for chunk in chunks)


def test_chunk_build_id_changes_when_chunk_boundaries_or_content_change() -> None:
    left = _build_document()
    right = _build_document()
    right["content"] = str(right["content"]).replace("Closing notes.", "Different ending.")

    pipeline = ChunkingPipeline(RecursiveCharacterSplitter(chunk_size=120, chunk_overlap=0))
    left_chunks = pipeline.run(left)
    right_chunks = pipeline.run(right)

    assert [chunk.chunk_id for chunk in left_chunks] != [chunk.chunk_id for chunk in right_chunks]
