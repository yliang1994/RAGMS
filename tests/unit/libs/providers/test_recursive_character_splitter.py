from __future__ import annotations

import pytest

from ragms.libs.providers.splitters.recursive_character_splitter import (
    RecursiveCharacterSplitter,
)


def test_recursive_character_splitter_creates_stable_chunks_from_markdown() -> None:
    document = {
        "content": "# Title\n\nAlpha paragraph.\n\nBeta paragraph with extra detail.\n\nGamma paragraph.",
        "source_path": "docs/sample.md",
        "metadata": {"document_id": "doc_123", "page": 1, "heading_outline": ["Title"]},
    }
    splitter = RecursiveCharacterSplitter(chunk_size=35, chunk_overlap=8)

    chunks = splitter.split(document)

    assert len(chunks) >= 3
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0]["source"] == "docs/sample.md"
    assert chunks[0]["start_offset"] == 0
    assert chunks[0]["metadata"]["document_id"] == "doc_123"
    assert all(chunk["end_offset"] > chunk["start_offset"] for chunk in chunks)
    assert all(chunk["end_offset"] - chunk["start_offset"] == len(chunk["content"]) for chunk in chunks)


def test_recursive_character_splitter_respects_runtime_chunk_size_and_overlap() -> None:
    document = {
        "content": "0123456789" * 12,
        "source_path": "docs/long.txt",
        "metadata": {},
    }
    splitter = RecursiveCharacterSplitter(chunk_size=60, chunk_overlap=5)

    chunks = splitter.split(document, chunk_size=30, chunk_overlap=10)

    assert len(chunks) >= 4
    assert all(len(chunk["content"]) <= 30 for chunk in chunks)
    assert chunks[1]["start_offset"] < chunks[0]["end_offset"]
    assert chunks[2]["start_offset"] < chunks[1]["end_offset"]
    assert chunks[1]["content"][:10] == chunks[0]["content"][-10:]


def test_recursive_character_splitter_returns_empty_list_for_empty_content() -> None:
    splitter = RecursiveCharacterSplitter()

    assert splitter.split({"content": "", "metadata": {}}) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (20, -1),
        (20, 20),
    ],
)
def test_recursive_character_splitter_rejects_invalid_sizes(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
