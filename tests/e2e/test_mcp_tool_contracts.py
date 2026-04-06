from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp import types

from ragms.core.query_engine import ResponseBuilder
from ragms.storage.images.image_storage import ImageStorage
from ragms.storage.sqlite.repositories.images import ImagesRepository


def test_query_tool_contract_supports_text_and_image_content(tmp_path: Path) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"diagram-bytes")

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    repository = ImagesRepository(connection)
    repository.upsert_image(
        image_id="img-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        file_path=str(image_path),
        source_path="docs/rag.pdf",
        image_hash="hash",
        page=1,
        position={"x": 1},
    )
    builder = ResponseBuilder(
        images_repository=repository,
        image_storage=ImageStorage(root_dir=tmp_path),
    )

    content = builder.build_multimodal_contents(
        markdown="RAG answer [1]",
        retrieved_chunks=[
            {
                "chunk_id": "chunk-1",
                "metadata": {"image_refs": ["img-1"]},
            }
        ],
    )

    assert content[0] == types.TextContent(type="text", text="RAG answer [1]")
    assert content[1].type == "image"
    assert content[1].mimeType == "image/png"


def test_query_tool_contract_degrades_to_text_when_image_file_missing(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    repository = ImagesRepository(connection)
    repository.upsert_image(
        image_id="img-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        file_path=str(tmp_path / "missing.png"),
        source_path="docs/rag.pdf",
        image_hash="hash",
        page=1,
        position={"x": 1},
    )
    builder = ResponseBuilder(
        images_repository=repository,
        image_storage=ImageStorage(root_dir=tmp_path),
    )

    content = builder.build_multimodal_contents(
        markdown="RAG answer [1]",
        retrieved_chunks=[
            {
                "chunk_id": "chunk-1",
                "metadata": {"image_refs": ["img-1"]},
            }
        ],
    )

    assert len(content) == 1
    assert content[0].type == "text"
