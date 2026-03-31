from __future__ import annotations

from pathlib import Path

import pytest

from ragms.libs.providers.loaders.markitdown_loader import MarkItDownLoader


def test_markitdown_loader_loads_supported_document(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text("# Title\n\nRAGMS bootstrap content.\n", encoding="utf-8")

    loader = MarkItDownLoader(extract_images=False)
    document = loader.load(str(source), collection="demo")

    assert document["source"] == str(source.resolve())
    assert "RAGMS bootstrap content." in document["content"]
    assert document["metadata"]["loader"] == "markitdown"
    assert document["metadata"]["source_type"] == "md"
    assert document["metadata"]["collection"] == "demo"


def test_markitdown_loader_rejects_empty_source() -> None:
    loader = MarkItDownLoader()

    with pytest.raises(ValueError, match="source must not be empty"):
        loader.load("")


def test_markitdown_loader_rejects_unsupported_file_type(tmp_path: Path) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"binary-content")

    loader = MarkItDownLoader()

    with pytest.raises(ValueError, match="unsupported file type"):
        loader.load(str(source))
