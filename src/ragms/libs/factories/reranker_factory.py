from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseReranker
from ragms.libs.providers.rerankers.cross_encoder_reranker import CrossEncoderReranker
from ragms.libs.providers.rerankers.llm_reranker import LLMReranker


class RerankerFactory:
    _REGISTRY = {
        "cross_encoder": CrossEncoderReranker,
        "llm": LLMReranker,
    }

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseReranker | None:
        enabled = config.get("enabled", True)
        mode = config.get("mode") or config.get("backend")
        if not enabled or mode == "none":
            return None
        model = config.get("model")
        if not mode or not model:
            raise ValueError("Reranker config requires mode and model when enabled")
        try:
            reranker_class = cls._REGISTRY[mode]
        except KeyError as exc:
            raise ValueError(f"Unknown reranker mode: {mode}") from exc
        return reranker_class(model=model)

