from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseVisionLLM
from ragms.libs.providers.vision_llms.gpt4o_vision_llm import GPT4OVisionLLM
from ragms.libs.providers.vision_llms.qwen_vl_llm import QwenVLLM


class VisionLLMFactory:
    _REGISTRY = {
        "openai": GPT4OVisionLLM,
        "qwen": QwenVLLM,
    }

    @classmethod
    def create(cls, config: dict[str, Any], *, document_language: str | None = None, deployment_env: str | None = None) -> BaseVisionLLM:
        resolved = dict(config)

        if resolved.get("provider") is None:
            resolved["provider"] = cls._choose_provider(document_language=document_language, deployment_env=deployment_env)

        provider = resolved.get("provider")
        model = resolved.get("model")
        if not provider or not model:
            raise ValueError("Vision LLM config requires provider and model")
        try:
            vision_class = cls._REGISTRY[provider]
        except KeyError as exc:
            raise ValueError(f"Unknown vision llm provider: {provider}") from exc
        return vision_class(model=model, api_key=resolved.get("api_key"))

    @staticmethod
    def _choose_provider(*, document_language: str | None, deployment_env: str | None) -> str:
        if deployment_env in {"cn", "domestic"}:
            return "qwen"
        if document_language in {"zh", "zh-cn", "zh-hans"}:
            return "qwen"
        return "openai"

