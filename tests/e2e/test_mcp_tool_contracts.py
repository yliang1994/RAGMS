from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp import types

from ragms.core.query_engine import ResponseBuilder
from ragms.storage.images.image_storage import ImageStorage
from ragms.storage.sqlite.repositories.images import ImagesRepository


def assert_mcp_response_contract(
    result: types.CallToolResult,
    *,
    expect_error: bool = False,
    require_image: bool = False,
) -> None:
    """Assert the shared shape of a successful or failed MCP tool result."""

    assert result.isError is expect_error
    assert result.content
    assert result.content[0].type == "text"
    if expect_error:
        assert result.structuredContent["error"]["code"] < 0
        return

    assert result.structuredContent is not None
    if require_image:
        assert any(item.type == "image" for item in result.content)


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

    assert_mcp_response_contract(
        types.CallToolResult(
            content=content,
            structuredContent={"trace_id": "trace-1", "citations": [{"index": 1}]},
            isError=False,
        ),
        require_image=True,
    )


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

    assert_mcp_response_contract(
        types.CallToolResult(
            content=content,
            structuredContent={"trace_id": "trace-empty", "citations": []},
            isError=False,
        ),
    )


def test_query_tool_contract_supports_structured_error_payload() -> None:
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text="Internal error")],
        structuredContent={"error": {"code": -32603, "message": "Internal error"}},
        isError=True,
    )

    assert_mcp_response_contract(result, expect_error=True)
