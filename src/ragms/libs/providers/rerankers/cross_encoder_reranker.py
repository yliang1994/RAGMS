"""Cross-encoder reranker backed by sentence-transformers."""

from __future__ import annotations

from threading import Lock
from typing import Any

from sentence_transformers import CrossEncoder

from ragms.libs.abstractions import BaseReranker
from ragms.runtime.exceptions import RagMSError


class RerankerProviderError(RagMSError):
    """Raised when a reranker provider cannot score candidate inputs."""


class CrossEncoderReranker(BaseReranker):
    """Rank candidates by relevance using a lazily loaded CrossEncoder model."""

    _MODEL_CACHE: dict[tuple[str, str | None, str | None], Any] = {}
    _MODEL_CACHE_LOCK = Lock()

    def __init__(
        self,
        *,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_folder: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_candidates: int | None = None,
    ) -> None:
        self.model = model.strip()
        self.cache_folder = cache_folder
        self.device = device
        self.batch_size = batch_size
        self.max_candidates = max_candidates

    def rerank(
        self,
        query: str,
        candidates: list[str | dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates sorted by descending cross-encoder relevance score."""

        if not candidates:
            return []
        if not self.model:
            raise RerankerProviderError("Cross-encoder model must not be empty")

        limited_candidates = candidates[: self.max_candidates] if self.max_candidates else list(candidates)
        sentence_pairs = [
            [query, _coerce_candidate_text(candidate)]
            for candidate in limited_candidates
        ]

        try:
            scores = self._get_model().predict(
                sentence_pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise RerankerProviderError("Cross-encoder rerank request failed") from exc

        normalized_scores = _normalize_scores(scores)
        if len(normalized_scores) != len(limited_candidates):
            raise RerankerProviderError("Cross-encoder returned an unexpected score count")

        ranked = sorted(
            (
                {
                    "document": candidate,
                    "score": score,
                }
                for candidate, score in zip(limited_candidates, normalized_scores, strict=True)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked[:top_k] if top_k is not None else ranked

    def _get_model(self) -> Any:
        """Return a lazily initialized and process-cached CrossEncoder instance."""

        cache_key = (self.model, self.cache_folder, self.device)
        with self._MODEL_CACHE_LOCK:
            if cache_key not in self._MODEL_CACHE:
                self._MODEL_CACHE[cache_key] = CrossEncoder(
                    self.model,
                    cache_folder=self.cache_folder,
                    device=self.device,
                )
            return self._MODEL_CACHE[cache_key]


def _coerce_candidate_text(candidate: str | dict[str, Any]) -> str:
    if isinstance(candidate, str):
        return candidate
    if "text" in candidate:
        return str(candidate["text"])
    if "document" in candidate:
        value = candidate["document"]
        if isinstance(value, dict) and "text" in value:
            return str(value["text"])
        return str(value)
    if "content" in candidate:
        return str(candidate["content"])
    return str(candidate)


def _normalize_scores(raw_scores: Any) -> list[float]:
    """Normalize score containers returned by sentence-transformers."""

    if raw_scores is None:
        raise RerankerProviderError("Cross-encoder returned no scores")
    if hasattr(raw_scores, "tolist"):
        values = raw_scores.tolist()
    else:
        values = list(raw_scores)
    return [float(score) for score in values]
