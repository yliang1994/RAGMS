from __future__ import annotations

from ragms.libs.abstractions import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, *, model: str, api_key: str | None = None, batch_size: int = 64) -> None:
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(index), 1.0] for index, text in enumerate(texts)]

