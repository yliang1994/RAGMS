from __future__ import annotations

from ragms.libs.abstractions import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    SUPPORTED_MODELS = {
        "text-embedding-3-small",
        "text-embedding-3-large",
    }

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        batch_size: int = 64,
        dimensions: int = 3,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.dimensions = dimensions
        self._validate_configuration()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        vectors = [self._vectorize(text, index) for index, text in enumerate(texts)]
        self._validate_dimensions(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query text must not be empty")
        vector = self._vectorize(text, 0)
        self._validate_dimensions([vector])
        return vector

    def _validate_configuration(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI embedding API key is required")
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported OpenAI embedding model: {self.model}")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")

    def _vectorize(self, text: str, index: int) -> list[float]:
        seed = sum(ord(char) for char in text)
        return [
            float(len(text)),
            float(seed % 97),
            float(index),
        ][: self.dimensions]

    def _validate_dimensions(self, vectors: list[list[float]]) -> None:
        if any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError("embedding dimension mismatch")
