"""Persistent BM25-style sparse index built from precomputed term statistics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class BM25Indexer:
    """Persist sparse document statistics and derived BM25 lookup state."""

    def __init__(
        self,
        *,
        index_dir: str | Path = "data/indexes/sparse",
        collection: str = "default",
    ) -> None:
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.collection = collection
        self.index_path = self.index_dir / f"{self.collection}.json"
        self._state = self._load_state()

    def index_document(self, record: Any) -> dict[str, Any]:
        """Insert or update one document entry and persist the derived index."""

        chunk_id = str(record.chunk_id)
        sparse_vector = dict(record.sparse_vector)
        content_hash = str(record.content_hash)
        existing = self._state["documents"].get(chunk_id)
        if existing is not None and str(existing.get("content_hash")) == content_hash:
            return dict(existing)

        self._state["documents"][chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": str(record.document_id),
            "content_hash": content_hash,
            "tokens": list(sparse_vector.get("tokens") or []),
            "term_frequencies": dict(sparse_vector.get("term_frequencies") or {}),
            "term_weights": dict(sparse_vector.get("term_weights") or {}),
            "document_length": int(sparse_vector.get("document_length", 0) or 0),
            "unique_terms": int(sparse_vector.get("unique_terms", 0) or 0),
        }
        self._rebuild_derived_state()
        self._persist()
        return dict(self._state["documents"][chunk_id])

    def snapshot(self) -> dict[str, Any]:
        """Return the current persisted index state."""

        return json.loads(json.dumps(self._state, ensure_ascii=False, sort_keys=True))

    def _load_state(self) -> dict[str, Any]:
        """Load a persisted index state or initialize an empty one."""

        if self.index_path.is_file():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {
            "collection": self.collection,
            "documents": {},
            "idf": {},
            "inverted_index": {},
            "document_count": 0,
            "average_document_length": 0.0,
        }

    def _rebuild_derived_state(self) -> None:
        """Recompute inverted lists, IDF values, and corpus statistics."""

        documents: dict[str, dict[str, Any]] = self._state["documents"]
        document_count = len(documents)
        total_length = sum(int(item.get("document_length", 0) or 0) for item in documents.values())
        average_document_length = (
            round(total_length / document_count, 6) if document_count else 0.0
        )

        postings: dict[str, list[str]] = {}
        for chunk_id, item in documents.items():
            for token, frequency in sorted((item.get("term_frequencies") or {}).items()):
                if int(frequency) <= 0:
                    continue
                postings.setdefault(str(token), []).append(chunk_id)
        for token in postings:
            postings[token] = sorted(postings[token])

        idf = {
            token: round(math.log(((document_count - len(doc_ids) + 0.5) / (len(doc_ids) + 0.5)) + 1.0), 6)
            for token, doc_ids in sorted(postings.items())
        }

        self._state["collection"] = self.collection
        self._state["inverted_index"] = postings
        self._state["idf"] = idf
        self._state["document_count"] = document_count
        self._state["average_document_length"] = average_document_length

    def _persist(self) -> None:
        """Persist the current index state atomically."""

        self.index_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.index_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(self._state, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.index_path)
