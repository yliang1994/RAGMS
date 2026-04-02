from __future__ import annotations

from tests.fakes import (
    FakeEmbedding,
    FakeEvaluator,
    FakeLLM,
    FakeReranker,
    FakeVectorStore,
    FakeVisionLLM,
)


def test_fake_llm_supports_queued_generation_and_streaming() -> None:
    llm = FakeLLM(["hello world"])

    assert llm.generate("ignored") == "hello world"
    assert list(llm.stream("stream this")) == ["fake-llm-response:stream", "this"]


def test_fake_vision_llm_supports_single_and_batch_captioning() -> None:
    vision_llm = FakeVisionLLM()

    assert vision_llm.caption("images/chart.png", context="sales") == "caption:chart context=sales"
    assert vision_llm.caption_batch(["a.png", "b.png"]) == ["caption:a", "caption:b"]


def test_fake_embedding_is_deterministic() -> None:
    embedding = FakeEmbedding(dimension=4)

    assert embedding.embed_query("hello") == embedding.embed_query("hello")
    assert len(embedding.embed_documents(["a", "b"])) == 2


def test_fake_vector_store_queries_and_deletes_vectors() -> None:
    store = FakeVectorStore()
    ids = store.add(
        ["doc-1", "doc-2"],
        [[1.0, 0.0], [0.5, 0.5]],
        documents=["alpha", "beta"],
    )

    results = store.query([1.0, 0.0], top_k=1)

    assert ids == ["doc-1", "doc-2"]
    assert results[0]["id"] == "doc-1"
    assert store.delete(["doc-1", "missing"]) == 1


def test_fake_reranker_scores_candidates_by_overlap() -> None:
    reranker = FakeReranker()

    ranked = reranker.rerank("hybrid search", ["hybrid search result", "other result"], top_k=1)

    assert ranked[0]["document"] == "hybrid search result"
    assert ranked[0]["score"] > 0


def test_fake_evaluator_reports_stable_metrics() -> None:
    evaluator = FakeEvaluator(base_score=0.9)

    metrics = evaluator.evaluate(["answer"], ["reference"])

    assert metrics == {"score": 0.9, "coverage": 1.0, "prediction_count": 1.0}
