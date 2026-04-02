"""LangChain-backed recursive character splitter provider."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter as LangChainSplitter

from ragms.libs.abstractions import BaseSplitter

DEFAULT_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n- ",
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    ". ",
    "! ",
    "? ",
    " ",
    "",
]


class RecursiveCharacterSplitter(BaseSplitter):
    """Split canonical documents into deterministic chunk records."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap
        self.separators = list(separators or DEFAULT_SEPARATORS)
        self.keep_separator = keep_separator

    def split(
        self,
        document: dict[str, Any],
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[dict[str, Any]]:
        """Split a canonical document into deterministic chunk records."""

        text = str(document.get("content", ""))
        if not text:
            return []

        resolved_chunk_size = chunk_size or self.default_chunk_size
        resolved_overlap = (
            self.default_chunk_overlap if chunk_overlap is None else chunk_overlap
        )
        if resolved_chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if resolved_overlap < 0 or resolved_overlap >= resolved_chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

        metadata = dict(document.get("metadata") or {})
        source = document.get("source") or document.get("source_path") or metadata.get("source")
        splitter = LangChainSplitter(
            separators=self.separators,
            keep_separator=self.keep_separator,
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_overlap,
            add_start_index=True,
        )
        split_documents = splitter.create_documents([text], metadatas=[metadata])
        chunks: list[dict[str, Any]] = []

        for chunk_index, chunk_document in enumerate(split_documents):
            chunk_text = chunk_document.page_content
            start = int(chunk_document.metadata.get("start_index", 0))
            end = start + len(chunk_text)
            chunk_metadata = dict(metadata)
            chunk_metadata.update(
                {
                    key: value
                    for key, value in chunk_document.metadata.items()
                    if key != "start_index"
                }
            )
            chunks.append(
                {
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "start_offset": start,
                    "end_offset": end,
                    "source": source,
                    "metadata": chunk_metadata,
                }
            )

        return chunks
