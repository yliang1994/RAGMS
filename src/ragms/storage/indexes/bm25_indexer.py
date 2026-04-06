"""Persistent BM25-style sparse index built from precomputed term statistics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class BM25Indexer:
    """Persist sparse document statistics and derived BM25 lookup state."""

    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

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
            "content": str(getattr(record, "content", "")),
            "metadata": dict(getattr(record, "metadata", {}) or {}),
            "content_hash": content_hash,
            "source_path": str(getattr(record, "source_path", "")),
            "chunk_index": int(getattr(record, "chunk_index", 0) or 0),
            "image_refs": list(getattr(record, "image_refs", []) or []),
            "tokens": list(sparse_vector.get("tokens") or []),
            "term_frequencies": dict(sparse_vector.get("term_frequencies") or {}),
            "term_weights": dict(sparse_vector.get("term_weights") or {}),
            "document_length": int(sparse_vector.get("document_length", 0) or 0),
            "unique_terms": int(sparse_vector.get("unique_terms", 0) or 0),
        }
        self._rebuild_derived_state()
        self._persist()
        return dict(self._state["documents"][chunk_id])

    def search(
        self,
        query_terms: list[str] | str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the sparse index once with a BM25-style scoring pass."""

        if top_k <= 0:
            return []

        normalized_terms = self._normalize_query_terms(query_terms)
        if not normalized_terms:
            return []

        filters = dict(filters or {})
        requested_collection = filters.pop("collection", self.collection)
        if str(requested_collection).strip() != self.collection:
            return []

        documents = self._state.get("documents", {})
        if not documents:
            return []

        average_document_length = float(self._state.get("average_document_length", 0.0) or 0.0)
        idf = dict(self._state.get("idf") or {})
        matches: list[dict[str, Any]] = []

        for chunk_id, payload in documents.items():
            if not self._matches_filters(payload, filters):
                continue
            score = self._score_document(
                payload,
                query_terms=normalized_terms,
                average_document_length=average_document_length,
                idf=idf,
            )
            if score <= 0:
                continue
            metadata = dict(payload.get("metadata") or {})
            metadata.setdefault("document_id", str(payload.get("document_id", "")))
            metadata.setdefault("source_path", str(payload.get("source_path", "")))
            metadata.setdefault("chunk_index", int(payload.get("chunk_index", 0) or 0))
            metadata.setdefault("image_refs", list(payload.get("image_refs") or []))
            metadata.setdefault("collection", self.collection)
            matches.append(
                {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "document_id": str(payload.get("document_id", "")),
                    "document": str(payload.get("content", "")),
                    "content": str(payload.get("content", "")),
                    "metadata": metadata,
                    "score": round(score, 6),
                }
            )

        matches.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return matches[:top_k]

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

    @staticmethod
    def _normalize_query_terms(query_terms: list[str] | str) -> list[str]:
        if isinstance(query_terms, str):
            raw_terms = query_terms.split()
        else:
            raw_terms = list(query_terms)

        normalized: list[str] = []
        seen: set[str] = set()
        for term in raw_terms:
            normalized_term = str(term).strip().lower()
            if not normalized_term or normalized_term in seen:
                continue
            seen.add(normalized_term)
            normalized.append(normalized_term)
        return normalized

    @classmethod
    def _score_document(
        cls,
        payload: dict[str, Any],
        *,
        query_terms: list[str],
        average_document_length: float,
        idf: dict[str, float],
    ) -> float:
        term_frequencies = {
            str(token): float(value)
            for token, value in dict(payload.get("term_frequencies") or {}).items()
        }
        document_length = float(payload.get("document_length", 0) or 0)
        if document_length <= 0:
            return 0.0

        denominator_base = cls.DEFAULT_K1 * (
            1 - cls.DEFAULT_B + cls.DEFAULT_B * (document_length / (average_document_length or 1.0))
        )
        score = 0.0
        for term in query_terms:
            term_frequency = term_frequencies.get(term, 0.0)
            if term_frequency <= 0:
                continue
            term_idf = float(idf.get(term, 0.0))
            numerator = term_frequency * (cls.DEFAULT_K1 + 1.0)
            denominator = term_frequency + denominator_base
            score += term_idf * (numerator / denominator)
        return score

    @staticmethod
    def _matches_filters(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
        if not filters:
            return True

        metadata = dict(payload.get("metadata") or {})
        for key, expected in filters.items():
            actual = metadata.get(key, payload.get(key))
            if isinstance(expected, list):
                if actual not in expected:
                    return False
                continue
            if actual != expected:
                return False
        return True
