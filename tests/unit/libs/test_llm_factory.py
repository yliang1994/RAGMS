from __future__ import annotations

import pytest

from ragms.libs.factories.llm_factory import LLMFactory
from ragms.libs.providers.llm.openai_llm import OpenAILLM
from ragms.libs.providers.llm.qwen_llm import QwenLLM
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings, LLMSettings


def test_llm_factory_uses_provider_from_app_settings() -> None:
    settings = AppSettings(llm=LLMSettings(provider="qwen", model="qwen-turbo", api_key="secret"))

    llm = LLMFactory.create(settings)

    assert isinstance(llm, QwenLLM)
    assert llm.model == "qwen-turbo"
    assert llm.api_key == "secret"


def test_llm_factory_accepts_explicit_mapping() -> None:
    llm = LLMFactory.create({"provider": "openai", "model": "gpt-test", "base_url": "https://api"})

    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gpt-test"
    assert llm.base_url == "https://api"


def test_llm_factory_rejects_missing_provider_in_mapping() -> None:
    with pytest.raises(RagMSError, match="Missing llm provider in configuration"):
        LLMFactory.create({"model": "gpt-test"})


def test_llm_factory_rejects_unknown_provider() -> None:
    with pytest.raises(RagMSError, match="Unknown llm provider: unknown"):
        LLMFactory.create({"provider": "unknown", "model": "m"})
