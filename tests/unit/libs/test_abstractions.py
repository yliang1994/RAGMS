from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ragms.libs.abstractions import (
    BaseEmbedding,
    BaseEvaluator,
    BaseLLM,
    BaseLoader,
    BaseReranker,
    BaseSplitter,
    BaseTransform,
    BaseVectorStore,
    BaseVisionLLM,
)


class ConcreteLoader(BaseLoader):
    def load(
        self,
        source_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [{"source_path": str(source_path), "metadata": metadata or {}}]


class ConcreteSplitter(BaseSplitter):
    def split(
        self,
        document: dict[str, Any],
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "content": document["content"],
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        ]


class ConcreteTransform(BaseTransform):
    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [{**chunk, "context": context or {}} for chunk in chunks]


class ConcreteLLM(BaseLLM):
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        return f"{system_prompt or 'none'}:{prompt}"

    def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        yield self.generate(prompt, system_prompt=system_prompt)


class ConcreteVisionLLM(BaseVisionLLM):
    def caption(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        context: str | None = None,
    ) -> str:
        suffix = f":{context}" if context else ""
        return f"{Path(image_path).stem}{suffix}"

    def caption_batch(
        self,
        image_paths: list[str | Path],
        *,
        prompt: str | None = None,
        context: str | None = None,
    ) -> list[str]:
        return [self.caption(image_path, prompt=prompt, context=context) for image_path in image_paths]


class ConcreteEmbedding(BaseEmbedding):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class ConcreteReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[str | dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        ranked = [
            {"document": candidate, "score": float(index)}
            for index, candidate in enumerate(reversed(candidates), start=1)
        ]
        return ranked[:top_k] if top_k is not None else ranked


class ConcreteVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        documents = documents or [""] * len(ids)
        metadatas = metadatas or [{} for _ in ids]
        for index, item_id in enumerate(ids):
            self._items[item_id] = {
                "id": item_id,
                "vector": vectors[index],
                "document": documents[index],
                "metadata": metadatas[index],
            }
        return ids

    def query(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del query_vector
        matches = list(self._items.values())
        if filters:
            matches = [item for item in matches if all(item["metadata"].get(key) == value for key, value in filters.items())]
        return matches[:top_k]

    def delete(self, ids: list[str]) -> int:
        deleted = 0
        for item_id in ids:
            if item_id in self._items:
                deleted += 1
                del self._items[item_id]
        return deleted


class ConcreteEvaluator(BaseEvaluator):
    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        del metadata
        references = references or []
        return {
            "prediction_count": float(len(predictions)),
            "reference_count": float(len(references)),
        }


@pytest.mark.parametrize(
    ("abstract_type", "missing_method"),
    [
        (BaseLoader, "load"),
        (BaseSplitter, "split"),
        (BaseTransform, "transform"),
        (BaseLLM, "generate"),
        (BaseVisionLLM, "caption"),
        (BaseEmbedding, "embed_documents"),
        (BaseReranker, "rerank"),
        (BaseVectorStore, "add"),
        (BaseEvaluator, "evaluate"),
    ],
)
def test_abstract_bases_cannot_be_instantiated(abstract_type: type[object], missing_method: str) -> None:
    with pytest.raises(TypeError, match=missing_method):
        abstract_type()  # type: ignore[abstract]


@pytest.mark.parametrize(
    ("callable_obj", "expected_parameters"),
    [
        (BaseLoader.load, ["self", "source_path", "metadata"]),
        (BaseSplitter.split, ["self", "document", "chunk_size", "chunk_overlap"]),
        (BaseTransform.transform, ["self", "chunks", "context"]),
        (BaseLLM.generate, ["self", "prompt", "system_prompt"]),
        (BaseLLM.stream, ["self", "prompt", "system_prompt"]),
        (BaseVisionLLM.caption, ["self", "image_path", "prompt", "context"]),
        (BaseVisionLLM.caption_batch, ["self", "image_paths", "prompt", "context"]),
        (BaseEmbedding.embed_documents, ["self", "texts"]),
        (BaseEmbedding.embed_query, ["self", "text"]),
        (BaseReranker.rerank, ["self", "query", "candidates", "top_k"]),
        (BaseVectorStore.add, ["self", "ids", "vectors", "documents", "metadatas"]),
        (BaseVectorStore.query, ["self", "query_vector", "top_k", "filters"]),
        (BaseVectorStore.delete, ["self", "ids"]),
        (BaseEvaluator.evaluate, ["self", "predictions", "references", "metadata"]),
    ],
)
def test_abstract_method_signatures_are_stable(
    callable_obj: object,
    expected_parameters: list[str],
) -> None:
    signature = inspect.signature(callable_obj)

    assert list(signature.parameters) == expected_parameters
    assert signature.return_annotation is not inspect.Signature.empty


def test_loader_splitter_and_transform_contracts_are_composable() -> None:
    loader = ConcreteLoader()
    splitter = ConcreteSplitter()
    transform = ConcreteTransform()

    documents = loader.load("sample.pdf", metadata={"page": 1})
    chunks = splitter.split({"content": "hello ragms"}, chunk_size=200, chunk_overlap=20)
    transformed = transform.transform(chunks, context={"document_count": len(documents)})

    assert documents[0]["source_path"] == "sample.pdf"
    assert chunks[0]["chunk_size"] == 200
    assert transformed[0]["context"]["document_count"] == 1


def test_llm_and_vision_contracts_return_expected_shapes() -> None:
    llm = ConcreteLLM()
    vision_llm = ConcreteVisionLLM()

    assert llm.generate("hello", system_prompt="system") == "system:hello"
    assert list(llm.stream("hello")) == ["none:hello"]
    assert vision_llm.caption("image/chart.png", context="sales") == "chart:sales"
    assert vision_llm.caption_batch(["a.png", "b.png"]) == ["a", "b"]


def test_embedding_reranker_vector_store_and_evaluator_contracts_are_usable() -> None:
    embedding = ConcreteEmbedding()
    reranker = ConcreteReranker()
    vector_store = ConcreteVectorStore()
    evaluator = ConcreteEvaluator()

    vectors = embedding.embed_documents(["alpha", "beta"])
    assert embedding.embed_query("alpha") == [5.0]

    added_ids = vector_store.add(
        ["doc-1", "doc-2"],
        vectors,
        documents=["alpha", "beta"],
        metadatas=[{"source": "a"}, {"source": "b"}],
    )
    matches = vector_store.query([5.0], filters={"source": "a"})
    ranked = reranker.rerank("alpha", [match["document"] for match in matches], top_k=1)
    metrics = evaluator.evaluate(["alpha"], ["alpha"])

    assert added_ids == ["doc-1", "doc-2"]
    assert matches[0]["id"] == "doc-1"
    assert ranked[0]["document"] == "alpha"
    assert metrics["prediction_count"] == 1.0
    assert vector_store.delete(["doc-1", "missing"]) == 1
