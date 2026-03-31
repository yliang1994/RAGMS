from __future__ import annotations

from tests.fakes.fake_embedding import FakeEmbedding
from tests.fakes.fake_evaluator import FakeEvaluator
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_reranker import FakeReranker
from tests.fakes.fake_vector_store import FakeVectorStore
from tests.fakes.fake_vision_llm import FakeVisionLLM


def test_fake_llm_is_deterministic(fake_llm: FakeLLM) -> None:
    result = fake_llm.chat([{"role": "user", "content": "hello"}], temperature=0)

    assert result["content"] == "fake-llm-response"
    assert fake_llm.calls


def test_fake_vision_llm_returns_caption(fake_vision_llm: FakeVisionLLM) -> None:
    result = fake_vision_llm.caption("image.png", prompt="describe")

    assert result["caption"] == "fake-image-caption"
    assert fake_vision_llm.calls[0]["image_ref"] == "image.png"


def test_fake_embedding_generates_vectors(fake_embedding: FakeEmbedding) -> None:
    vectors = fake_embedding.embed(["alpha", "beta"])

    assert len(vectors) == 2
    assert len(vectors[0]) == fake_embedding.dimensions
    assert vectors[0] != vectors[1]


def test_fake_vector_store_supports_upsert_and_query(
    fake_vector_store: FakeVectorStore,
    sample_documents: list[dict[str, object]],
) -> None:
    upserted = fake_vector_store.upsert(sample_documents)
    results = fake_vector_store.query("local modular retrieval", top_k=1)

    assert upserted == 2
    assert results[0]["id"] == "doc-1"


def test_fake_reranker_orders_highest_score_first(fake_reranker: FakeReranker) -> None:
    ranked = fake_reranker.rerank(
        "ragms",
        [
            {"id": "b", "score": 0.4, "text": "short"},
            {"id": "a", "score": 0.8, "text": "longer candidate"},
        ],
        top_k=1,
    )

    assert ranked == [{"id": "a", "score": 0.8, "text": "longer candidate"}]


def test_fake_evaluator_reports_match_ratio(fake_evaluator: FakeEvaluator) -> None:
    report = fake_evaluator.evaluate(
        [
            {"expected": "yes", "actual": "yes"},
            {"expected": "no", "actual": "yes"},
        ]
    )

    assert report["total"] == 2
    assert report["passed"] == 1
    assert report["score"] == 0.5
