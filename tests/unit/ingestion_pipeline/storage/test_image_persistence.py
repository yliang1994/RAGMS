from __future__ import annotations

import sqlite3
from pathlib import Path

from ragms.ingestion_pipeline.storage import ChunkRecord
from ragms.ingestion_pipeline.storage.image_persistence import ImageStorageWriter
from ragms.storage.images import ImageStorage
from ragms.storage.sqlite.repositories import ImagesRepository


def _repository() -> ImagesRepository:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    return ImagesRepository(connection)


def _record(image_path: Path) -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="A [IMAGE: chart-1]",
        metadata={
            "images": [
                {
                    "id": "chart-1",
                    "path": str(image_path),
                    "page": 1,
                    "position": {"x": 10, "y": 20},
                }
            ]
        },
        dense_vector=[0.1, 0.2],
        sparse_vector={"tokens": [], "term_frequencies": {}, "term_weights": {}, "document_length": 0, "unique_terms": 0},
        content_hash="hash-1",
        source_path="docs/report.pdf",
        chunk_index=0,
        image_refs=["chart-1"],
    )


def test_image_storage_writer_persists_files_and_repository_rows(tmp_path: Path) -> None:
    source_image = tmp_path / "chart-1.png"
    source_image.write_bytes(b"png-data")
    repository = _repository()
    writer = ImageStorageWriter(
        image_storage=ImageStorage(root_dir=tmp_path / "images"),
        repository=repository,
        collection="demo",
    )

    result = writer.save_all([_record(source_image)])

    rows = repository.list_by_document_id("doc-1")
    assert result["stored_count"] == 1
    assert result["file_count"] == 1
    assert len(rows) == 1
    assert Path(rows[0]["file_path"]).is_file()
    assert rows[0]["chunk_id"] == "chunk-1"
    assert rows[0]["page"] == 1
    assert rows[0]["position"] == {"x": 10, "y": 20}
    assert str(rows[0]["file_path"]).endswith("demo/chart-1.png")


def test_image_storage_writer_is_idempotent_for_repeated_same_image(tmp_path: Path) -> None:
    source_image = tmp_path / "chart-1.png"
    source_image.write_bytes(b"same-image")
    repository = _repository()
    writer = ImageStorageWriter(
        image_storage=ImageStorage(root_dir=tmp_path / "images"),
        repository=repository,
        collection="demo",
    )
    record = _record(source_image)

    first = writer.save_all([record])
    second = writer.save_all([record])

    rows = repository.list_by_document_id("doc-1")
    stored_files = list((tmp_path / "images" / "demo").glob("*"))
    assert first["stored_count"] == 1
    assert second["stored_count"] == 1
    assert len(rows) == 1
    assert len(stored_files) == 1
