from __future__ import annotations

import pytest

from ragms.libs.factories.vision_llm_factory import VisionLLMFactory
from ragms.libs.providers.vision_llms.gpt4o_vision_llm import GPT4oVisionLLM
from ragms.libs.providers.vision_llms.qwen_vl_llm import QwenVLLLM
from ragms.runtime.exceptions import RagMSError


def test_vision_llm_factory_routes_by_document_language() -> None:
    vision_llm = VisionLLMFactory.create(
        {
            "provider": "auto",
            "model": "vision-router",
            "language_providers": {"zh": "qwen_vl", "en": "gpt4o"},
        },
        document_language="zh-CN",
    )

    assert isinstance(vision_llm, QwenVLLLM)
    assert vision_llm.model == "vision-router"


def test_vision_llm_factory_routes_by_deployment_environment() -> None:
    vision_llm = VisionLLMFactory.create(
        {
            "provider": "auto",
            "environment_providers": {"local_cn": "qwen_vl", "production": "gpt4o"},
        },
        deployment_environment="local_cn",
    )

    assert isinstance(vision_llm, QwenVLLLM)


def test_vision_llm_factory_honors_explicit_provider() -> None:
    vision_llm = VisionLLMFactory.create({"provider": "gpt4o", "model": "gpt-4o-mini"})

    assert isinstance(vision_llm, GPT4oVisionLLM)
    assert vision_llm.model == "gpt-4o-mini"


def test_vision_llm_factory_rejects_missing_provider_in_mapping() -> None:
    with pytest.raises(RagMSError, match="Missing vision llm provider in configuration"):
        VisionLLMFactory.create({"model": "vision-router"})


def test_vision_llm_factory_rejects_unknown_provider() -> None:
    with pytest.raises(RagMSError, match="Unknown vision llm provider: unknown"):
        VisionLLMFactory.create({"provider": "unknown"})
