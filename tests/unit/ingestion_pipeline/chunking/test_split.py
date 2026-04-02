from __future__ import annotations

from ragms.ingestion_pipeline.chunking import ChunkingPipeline
from ragms.libs.providers.splitters.recursive_character_splitter import RecursiveCharacterSplitter


def _build_document() -> dict[str, object]:
    content = (
        "# Title\n\n"
        "Alpha paragraph.\n\n"
        "[IMAGE: chart-1]\n"
        "Chart discussion stays nearby.\n\n"
        "Beta paragraph continues with extra details for another chunk.\n\n"
        "[IMAGE: chart-2]\n"
        "Closing notes.\n\n"
        "[IMAGE: chart-1]\n"
        "Repeated chart reference."
    )
    return {
        "content": content,
        "source_path": "docs/report.md",
        "metadata": {
            "document_id": "doc_123",
            "source_sha256": "a" * 64,
            "heading_outline": ["Title"],
            "images": [
                {"id": "chart-1", "path": "data/images/demo/chart-1.png", "page": 1, "position": {"x": 10}},
                {"id": "chart-2", "path": "data/images/demo/chart-2.png", "page": 2, "position": {"x": 20}},
            ],
            "image_occurrences": [
                {
                    "image_id": "chart-1",
                    "text_offset": content.index("[IMAGE: chart-1]"),
                    "text_length": len("[IMAGE: chart-1]"),
                    "page": 1,
                    "position": {"x": 10},
                },
                {
                    "image_id": "chart-2",
                    "text_offset": content.index("[IMAGE: chart-2]"),
                    "text_length": len("[IMAGE: chart-2]"),
                    "page": 2,
                    "position": {"x": 20},
                },
                {
                    "image_id": "chart-1",
                    "text_offset": content.rindex("[IMAGE: chart-1]"),
                    "text_length": len("[IMAGE: chart-1]"),
                    "page": 3,
                    "position": {"x": 30},
                },
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
    assert first_run[0].source_ref == "doc_123"
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
    assert all(
        sorted({item["image_id"] for item in chunk.metadata.get("image_occurrences", [])}) == sorted(chunk.image_refs)
        for chunk in chunks
        if chunk.image_refs
    )
    assert all("images" not in chunk.metadata for chunk in chunks if not chunk.image_refs)


def test_chunking_pipeline_slices_image_occurrences_to_chunk_boundaries() -> None:
    pipeline = ChunkingPipeline(
        RecursiveCharacterSplitter(
            chunk_size=90,
            chunk_overlap=10,
        )
    )

    chunks = pipeline.run(_build_document())

    chunks_with_occurrences = [chunk for chunk in chunks if chunk.metadata.get("image_occurrences")]

    assert len(chunks_with_occurrences) >= 2
    all_occurrence_refs = [
        occurrence["image_id"]
        for chunk in chunks_with_occurrences
        for occurrence in chunk.metadata["image_occurrences"]
    ]
    assert all_occurrence_refs.count("chart-1") >= 2
    assert "chart-2" in all_occurrence_refs
    for chunk in chunks_with_occurrences:
        for occurrence in chunk.metadata["image_occurrences"]:
            occurrence_start = occurrence["text_offset"]
            occurrence_end = occurrence_start + occurrence["text_length"]
            assert chunk.start_offset <= occurrence_start < chunk.end_offset
            assert occurrence_end <= chunk.end_offset
        if "chart-1" in chunk.image_refs:
            assert all(image["id"] in chunk.image_refs for image in chunk.metadata.get("images", []))


def test_chunk_build_id_changes_when_chunk_boundaries_or_content_change() -> None:
    left = _build_document()
    right = _build_document()
    right["content"] = str(right["content"]).replace("Closing notes.", "Different ending.")

    pipeline = ChunkingPipeline(RecursiveCharacterSplitter(chunk_size=120, chunk_overlap=0))
    left_chunks = pipeline.run(left)
    right_chunks = pipeline.run(right)

    assert [chunk.chunk_id for chunk in left_chunks] != [chunk.chunk_id for chunk in right_chunks]
