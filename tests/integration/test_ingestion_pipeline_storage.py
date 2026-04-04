from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ragms.ingestion_pipeline import IngestionPipeline
from ragms.ingestion_pipeline.chunking import ChunkingPipeline
from ragms.ingestion_pipeline.embedding import DenseEncoder, SparseEncoder
from ragms.ingestion_pipeline.file_integrity import FileIntegrity
from ragms.ingestion_pipeline.lifecycle import DocumentRegistry
from ragms.ingestion_pipeline.storage import StoragePipeline, VectorUpsert
from ragms.ingestion_pipeline.storage.bm25_indexing import BM25StorageWriter
from ragms.ingestion_pipeline.storage.image_persistence import ImageStorageWriter
from ragms.ingestion_pipeline.transform import TransformPipeline
from ragms.libs.providers.splitters.recursive_character_splitter import RecursiveCharacterSplitter
from ragms.storage.images import ImageStorage
from ragms.storage.indexes import BM25Indexer
from ragms.storage.sqlite.repositories import (
    DocumentsRepository,
    ImagesRepository,
    IngestionHistoryRepository,
)
from ragms.storage.sqlite.schema import initialize_metadata_schema
from tests.fakes import FakeEmbedding, FakeVectorStore


class ImageAwareLoader:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    def load(self, source_path: str | Path, *, metadata: dict[str, object] | None = None) -> list[dict[str, object]]:
        content = "Overview [IMAGE: chart-1] explains the trend."
        source = Path(source_path)
        return [
            {
                "content": content,
                "source_path": str(source),
                "metadata": {
                    **dict(metadata or {}),
                    "document_id": "doc-storage",
                    "source_sha256": "sha-storage",
                    "images": [
                        {
                            "id": "chart-1",
                            "path": str(self.image_path),
                            "page": 1,
                            "position": {"x": 5, "y": 8},
                        }
                    ],
                    "image_occurrences": [
                        {
                            "image_id": "chart-1",
                            "text_offset": 9,
                            "text_length": len("[IMAGE: chart-1]"),
                            "page": 1,
                            "position": {"x": 5, "y": 8},
                        }
                    ],
                },
            }
        ]


@pytest.mark.integration
def test_ingestion_pipeline_persists_vectors_bm25_and_images(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text("placeholder", encoding="utf-8")
    image = tmp_path / "chart-1.png"
    image.write_bytes(b"fake-image")

    connection = initialize_metadata_schema(tmp_path / "metadata.db")
    pipeline = IngestionPipeline(
        loader=ImageAwareLoader(image),
        splitter=RecursiveCharacterSplitter(chunk_size=80, chunk_overlap=0),
        transform=TransformPipeline(),
        embedding=FakeEmbedding(dimension=4),
        vector_store=FakeVectorStore(),
        file_integrity=FileIntegrity(IngestionHistoryRepository(connection)),
        document_registry=DocumentRegistry(DocumentsRepository(connection)),
        chunking_pipeline=ChunkingPipeline(RecursiveCharacterSplitter(chunk_size=80, chunk_overlap=0)),
        dense_encoder=DenseEncoder(FakeEmbedding(dimension=4)),
        sparse_encoder=SparseEncoder(enable_jieba=False),
        storage_pipeline=StoragePipeline(vector_upsert=VectorUpsert(FakeVectorStore())),
        bm25_writer=BM25StorageWriter(
            BM25Indexer(index_dir=tmp_path / "indexes", collection="demo")
        ),
        image_storage_writer=ImageStorageWriter(
            image_storage=ImageStorage(root_dir=tmp_path / "images"),
            repository=ImagesRepository(connection),
            collection="demo",
        ),
    )

    result = pipeline.run(source, metadata={"collection": "demo"})

    assert result["status"] == "completed"
    assert result["lifecycle"]["final_status"] == "indexed"
    assert len(result["stored_ids"]) == 1
    assert result["bm25"]["indexed_count"] == 1
    assert result["stored_images"]["stored_count"] == 1
    assert (tmp_path / "indexes" / "demo.json").is_file()
    persisted_image = tmp_path / "images" / "demo" / "chart-1.png"
    assert persisted_image.is_file()
