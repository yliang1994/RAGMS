"""Chroma vector store provider backed by chromadb."""

from __future__ import annotations

from typing import Any

from chromadb import Client
from chromadb.api import ClientAPI
from chromadb.config import Settings

from ragms.libs.abstractions import BaseVectorStore
from ragms.runtime.exceptions import RagMSError


class VectorStoreProviderError(RagMSError):
    """Raised when the vector store provider cannot complete a request."""


class ChromaStore(BaseVectorStore):
    """Persist vector records and query them through chromadb."""

    def __init__(
        self,
        *,
        collection: str = "default",
        persist_directory: str | None = None,
        client: ClientAPI | None = None,
    ) -> None:
        self.collection = collection
        self.persist_directory = persist_directory
        self._client = client or self._build_client()
        self._collection = self._client.get_or_create_collection(
            name=self.collection,
            embedding_function=None,
        )

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add or update vector entries and return their ids."""

        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")

        resolved_documents = documents or [""] * len(ids)
        resolved_metadatas = metadatas or [{} for _ in ids]
        if len(resolved_documents) != len(ids) or len(resolved_metadatas) != len(ids):
            raise ValueError("documents and metadatas must align with ids")
        chroma_metadatas = _normalize_metadatas_for_chroma(resolved_metadatas)

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=resolved_documents,
                metadatas=chroma_metadatas,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise VectorStoreProviderError("Chroma add request failed") from exc
        return ids

    def query(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k nearest matches for a query vector."""

        if top_k <= 0:
            return []
        if self._collection.count() == 0:
            return []

        try:
            result = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise VectorStoreProviderError("Chroma query request failed") from exc

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        matches: list[dict[str, Any]] = []
        for index, item_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 0.0
            matches.append(
                {
                    "id": item_id,
                    "score": 1.0 / (1.0 + distance),
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": _restore_metadata(metadatas[index] if index < len(metadatas) else {}),
                }
            )
        return matches

    def delete(self, ids: list[str]) -> int:
        """Delete entries by id and return the deletion count."""

        if not ids:
            return 0
        existing = self._collection.get(ids=ids, include=[])
        existing_ids = existing.get("ids") or []
        if not existing_ids:
            return 0
        try:
            result = self._collection.delete(ids=ids)
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise VectorStoreProviderError("Chroma delete request failed") from exc

        deleted = result.get("deleted") if isinstance(result, dict) else None
        return int(deleted) if deleted is not None else len(existing_ids)

    def _build_client(self) -> ClientAPI:
        """Create the underlying chromadb client."""

        settings = Settings(
            is_persistent=self.persist_directory is not None,
            persist_directory=self.persist_directory or "./chroma",
            anonymized_telemetry=False,
        )
        return Client(settings)


_EMPTY_METADATA_MARKER = "_ragms_empty_metadata"


def _normalize_metadatas_for_chroma(
    metadatas: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Convert empty metadata dicts into a Chroma-compatible payload."""

    if not metadatas:
        return None
    if all(not metadata for metadata in metadatas):
        return None
    normalized: list[dict[str, Any]] = []
    for metadata in metadatas:
        if metadata:
            normalized.append(dict(metadata))
        else:
            normalized.append({_EMPTY_METADATA_MARKER: True})
    return normalized


def _restore_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Strip internal placeholder metadata added only for Chroma compatibility."""

    restored = dict(metadata or {})
    restored.pop(_EMPTY_METADATA_MARKER, None)
    return restored
