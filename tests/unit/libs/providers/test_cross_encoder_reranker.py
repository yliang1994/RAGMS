from __future__ import annotations

import pytest

from ragms.libs.providers.rerankers.cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankerProviderError,
)


class FakeCrossEncoder:
    init_calls: list[dict[str, object]] = []
    predict_calls: list[dict[str, object]] = []
    scores: list[float] = []
    raise_error: Exception | None = None

    def __init__(
        self,
        model_name_or_path: str,
        *,
        cache_folder: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.cache_folder = cache_folder
        self.device = device
        FakeCrossEncoder.init_calls.append(
            {
                "model": model_name_or_path,
                "cache_folder": cache_folder,
                "device": device,
            }
        )

    def predict(
        self,
        sentences: list[list[str]],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> list[float]:
        FakeCrossEncoder.predict_calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        if FakeCrossEncoder.raise_error:
            raise FakeCrossEncoder.raise_error
        return list(FakeCrossEncoder.scores)


@pytest.fixture(autouse=True)
def reset_cross_encoder_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeCrossEncoder.init_calls = []
    FakeCrossEncoder.predict_calls = []
    FakeCrossEncoder.scores = []
    FakeCrossEncoder.raise_error = None
    CrossEncoderReranker._MODEL_CACHE.clear()
    monkeypatch.setattr(
        "ragms.libs.providers.rerankers.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoder,
    )


def test_cross_encoder_reranker_sorts_candidates_by_score() -> None:
    FakeCrossEncoder.scores = [0.2, 0.9, 0.5]
    reranker = CrossEncoderReranker(model="cross-test", batch_size=4)

    ranked = reranker.rerank(
        "what is rag",
        [
            {"text": "low relevance"},
            {"text": "high relevance"},
            {"text": "mid relevance"},
        ],
    )

    assert [item["score"] for item in ranked] == [0.9, 0.5, 0.2]
    assert ranked[0]["document"]["text"] == "high relevance"
    assert FakeCrossEncoder.init_calls == [
        {"model": "cross-test", "cache_folder": None, "device": None}
    ]
    assert FakeCrossEncoder.predict_calls[0]["batch_size"] == 4


def test_cross_encoder_reranker_returns_empty_list_for_empty_candidates() -> None:
    reranker = CrossEncoderReranker()

    assert reranker.rerank("query", []) == []
    assert FakeCrossEncoder.init_calls == []


def test_cross_encoder_reranker_respects_top_k_and_max_candidates() -> None:
    FakeCrossEncoder.scores = [0.4, 0.8]
    reranker = CrossEncoderReranker(max_candidates=2)

    ranked = reranker.rerank(
        "query",
        ["doc one", "doc two", "doc three"],
        top_k=1,
    )

    assert ranked == [{"document": "doc two", "score": 0.8}]
    assert len(FakeCrossEncoder.predict_calls[0]["sentences"]) == 2


def test_cross_encoder_reranker_reuses_model_instance_within_process() -> None:
    FakeCrossEncoder.scores = [0.5]
    first = CrossEncoderReranker(model="cross-test", cache_folder="/tmp/hf-cache")
    second = CrossEncoderReranker(model="cross-test", cache_folder="/tmp/hf-cache")

    first.rerank("query", ["first"])
    second.rerank("query", ["second"])

    assert len(FakeCrossEncoder.init_calls) == 1
    assert len(FakeCrossEncoder.predict_calls) == 2


def test_cross_encoder_reranker_raises_on_missing_model_name() -> None:
    reranker = CrossEncoderReranker(model="   ")

    with pytest.raises(RerankerProviderError, match="Cross-encoder model must not be empty"):
        reranker.rerank("query", ["candidate"])


def test_cross_encoder_reranker_raises_on_unexpected_score_count() -> None:
    FakeCrossEncoder.scores = [0.5]
    reranker = CrossEncoderReranker()

    with pytest.raises(RerankerProviderError, match="unexpected score count"):
        reranker.rerank("query", ["first", "second"])


def test_cross_encoder_reranker_maps_provider_failures() -> None:
    FakeCrossEncoder.raise_error = RuntimeError("model unavailable")
    reranker = CrossEncoderReranker()

    with pytest.raises(RerankerProviderError, match="Cross-encoder rerank request failed"):
        reranker.rerank("query", ["candidate"])
