from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ragms.libs.providers.loaders.markitdown_loader import (
    DocumentLoadError,
    MarkItDownLoader,
)


@dataclass
class FakeConvertResult:
    text_content: str
    title: str | None = None
    images: list[dict[str, object]] | None = None
    image_occurrences: list[dict[str, object]] | None = None


class FakeMarkItDownConverter:
    def __init__(self, result: FakeConvertResult) -> None:
        self.result = result
        self.calls: list[str] = []

    def convert(self, source_path: str) -> FakeConvertResult:
        self.calls.append(source_path)
        return self.result


def test_markitdown_loader_loads_markdown_into_canonical_document(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.md"
    source_path.write_text("# Title\n\nHello RagMS.\n", encoding="utf-8")

    loader = MarkItDownLoader()

    documents = loader.load(source_path, metadata={"collection": "demo"})

    assert len(documents) == 1
    document = documents[0]
    assert document["content"] == "# Title\n\nHello RagMS.\n"
    assert document["source_path"] == str(source_path)
    assert document["metadata"]["collection"] == "demo"
    assert document["metadata"]["doc_type"] == "markdown"
    assert document["metadata"]["title"] == "sample"
    assert document["metadata"]["page"] == 1
    assert document["metadata"]["heading_outline"] == []
    assert document["metadata"]["images"] == []
    assert document["metadata"]["image_occurrences"] == []
    assert document["metadata"]["document_id"].startswith("doc_")
    assert len(document["metadata"]["source_sha256"]) == 64
    assert document["metadata"]["source_path"] == str(source_path)


def test_markitdown_loader_uses_converter_for_pdf_inputs(tmp_path: Path) -> None:
    source_path = tmp_path / "report.pdf"
    source_path.write_bytes(b"%PDF-sample")
    converter = FakeMarkItDownConverter(
        FakeConvertResult(
            text_content="# Report\n\nConverted body",
            title="Quarterly Report",
        )
    )

    loader = MarkItDownLoader(converter=converter)

    documents = loader.load(source_path)

    assert converter.calls == [str(source_path)]
    assert documents[0]["content"] == "# Report\n\nConverted body"
    assert documents[0]["metadata"]["doc_type"] == "pdf"
    assert documents[0]["metadata"]["title"] == "Quarterly Report"


def test_markitdown_loader_normalizes_multimodal_image_metadata_and_occurrences(tmp_path: Path) -> None:
    source_path = tmp_path / "report.pdf"
    source_path.write_bytes(b"%PDF-sample")
    converter = FakeMarkItDownConverter(
        FakeConvertResult(
            text_content=(
                "# Report\n\n"
                "Intro section.\n"
                "[IMAGE: chart-1]\n"
                "More detail.\n"
                "[IMAGE: chart-2]\n"
            ),
            title="Quarterly Report",
            images=[
                {"image_id": "chart-1", "path": "data/images/demo/chart-1.png", "page": 2},
                {"id": "chart-2", "path": "data/images/demo/chart-2.png", "page": 3, "position": {"x": 12}},
            ],
        )
    )
    loader = MarkItDownLoader(converter=converter)

    documents = loader.load(source_path)

    assert len(documents) == 1
    metadata = documents[0]["metadata"]
    assert metadata["images"] == [
        {"id": "chart-1", "path": "data/images/demo/chart-1.png", "page": 2, "position": {}},
        {"id": "chart-2", "path": "data/images/demo/chart-2.png", "page": 3, "position": {"x": 12}},
    ]
    assert metadata["image_occurrences"] == [
        {
            "image_id": "chart-1",
            "text_offset": documents[0]["content"].index("[IMAGE: chart-1]"),
            "text_length": len("[IMAGE: chart-1]"),
            "page": None,
            "position": {},
        },
        {
            "image_id": "chart-2",
            "text_offset": documents[0]["content"].index("[IMAGE: chart-2]"),
            "text_length": len("[IMAGE: chart-2]"),
            "page": None,
            "position": {},
        },
    ]


def test_markitdown_loader_rejects_unsupported_file_type(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.bin"
    source_path.write_bytes(b"\x00\x01")

    loader = MarkItDownLoader()

    with pytest.raises(DocumentLoadError, match="Unsupported source file type: .bin"):
        loader.load(source_path)


def test_markitdown_loader_rejects_empty_input(tmp_path: Path) -> None:
    source_path = tmp_path / "empty.md"
    source_path.write_text("", encoding="utf-8")

    loader = MarkItDownLoader()

    with pytest.raises(DocumentLoadError, match="Source file is empty"):
        loader.load(source_path)
