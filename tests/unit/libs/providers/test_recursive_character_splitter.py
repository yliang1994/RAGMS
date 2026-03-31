from __future__ import annotations

from ragms.libs.providers.splitters.recursive_character_splitter import RecursiveCharacterSplitter


def test_recursive_character_splitter_splits_text_into_expected_windows() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=5, chunk_overlap=2)

    chunks = splitter.split({"content": "abcdefghij", "metadata": {"source": "demo"}})

    assert [chunk["text"] for chunk in chunks] == ["abcde", "defgh", "ghij"]
    assert chunks[0]["metadata"]["start_offset"] == 0
    assert chunks[1]["metadata"]["start_offset"] == 3
    assert chunks[2]["metadata"]["end_offset"] == 10


def test_recursive_character_splitter_preserves_overlap_length() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=6, chunk_overlap=2)

    chunks = splitter.split({"content": "abcdefghijkl"})

    assert chunks[0]["text"][-2:] == chunks[1]["text"][:2]


def test_recursive_character_splitter_returns_single_chunk_for_short_text() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=5)

    chunks = splitter.split({"content": "short text"})

    assert len(chunks) == 1
    assert chunks[0]["text"] == "short text"
