from __future__ import annotations

from dataclasses import dataclass, field

from ragms.libs.abstractions.base_embedding import BaseEmbedding


@dataclass
class FakeEmbedding(BaseEmbedding):
    dimensions: int = 4
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vectorize(text) for text in texts]

    def _vectorize(self, text: str) -> list[float]:
        seed = sum(ord(char) for char in text)
        return [float((seed + index) % 17) for index in range(self.dimensions)]
