from __future__ import annotations

from pathlib import Path

from ragms.ingestion_pipeline.storage import ChunkRecord
from ragms.ingestion_pipeline.storage.bm25_indexing import BM25StorageWriter
from ragms.storage.indexes import BM25Indexer


def _record(
    *,
    chunk_id: str,
    content_hash: str,
    tokens: list[str],
    term_frequencies: dict[str, int],
) -> ChunkRecord:
    document_length = sum(term_frequencies.values())
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc-1",
        content="sample",
        metadata={"chunk_title": "Sample"},
        dense_vector=[0.1, 0.2],
        sparse_vector={
            "tokens": tokens,
            "term_frequencies": term_frequencies,
            "term_weights": {
                token: round(count / document_length, 6)
                for token, count in term_frequencies.items()
            },
            "document_length": document_length,
            "unique_terms": len(term_frequencies),
        },
        content_hash=content_hash,
        source_path="docs/sample.pdf",
        chunk_index=0,
        image_refs=[],
    )


def test_bm25_storage_writer_indexes_and_persists_sparse_state(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path / "indexes", collection="demo")
    writer = BM25StorageWriter(indexer)

    result = writer.index(
        [
            _record(
                chunk_id="chunk-1",
                content_hash="hash-1",
                tokens=["alpha", "beta"],
                term_frequencies={"alpha": 1, "beta": 1},
            ),
            _record(
                chunk_id="chunk-2",
                content_hash="hash-2",
                tokens=["beta", "gamma"],
                term_frequencies={"beta": 2, "gamma": 1},
            ),
        ]
    )

    reloaded = BM25Indexer(index_dir=tmp_path / "indexes", collection="demo").snapshot()

    assert result["indexed_count"] == 2
    assert reloaded["document_count"] == 2
    assert reloaded["inverted_index"]["beta"] == ["chunk-1", "chunk-2"]
    assert reloaded["idf"]["alpha"] > reloaded["idf"]["beta"]
    assert reloaded["average_document_length"] == 2.5


def test_bm25_indexer_updates_existing_chunk_incrementally(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path / "indexes", collection="demo")
    writer = BM25StorageWriter(indexer)
    writer.index(
        [
            _record(
                chunk_id="chunk-1",
                content_hash="hash-1",
                tokens=["alpha"],
                term_frequencies={"alpha": 1},
            )
        ]
    )

    writer.index(
        [
            _record(
                chunk_id="chunk-1",
                content_hash="hash-2",
                tokens=["beta"],
                term_frequencies={"beta": 1},
            )
        ]
    )

    snapshot = indexer.snapshot()
    assert snapshot["document_count"] == 1
    assert "alpha" not in snapshot["inverted_index"]
    assert snapshot["inverted_index"]["beta"] == ["chunk-1"]
