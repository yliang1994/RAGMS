from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from ragms.libs.providers.llm.openai_llm import LLMProviderError
from ragms.libs.providers.llm.qwen_llm import QwenLLM


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.response: object | None = None
        self.stream_response: object | None = None
        self.error: Exception | None = None
        self.stream_error: Exception | None = None

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if kwargs.get("stream"):
            if self.stream_error:
                raise self.stream_error
            return self.stream_response
        if self.error:
            raise self.error
        return self.response


class FakeOpenAIClient:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _build_response(text: str) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _build_chunk(text: str) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


def _build_rate_limit_error(message: str) -> RateLimitError:
    request = httpx.Request("POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(message, response=response, body=None)


def test_qwen_llm_uses_openai_compatible_gateway_defaults() -> None:
    completions = FakeChatCompletions()
    completions.response = _build_response("ni hao")
    llm = QwenLLM(api_key="test-key", client=FakeOpenAIClient(completions))

    response = llm.generate("Say hello", system_prompt="Use Chinese")

    assert response == "ni hao"
    assert llm.model == "qwen-plus"
    assert llm.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert completions.calls[0]["messages"][0]["content"] == "Use Chinese"


def test_qwen_llm_stream_yields_incremental_chunks() -> None:
    completions = FakeChatCompletions()
    completions.stream_response = iter([_build_chunk("你好"), _build_chunk("，世界")])
    llm = QwenLLM(api_key="test-key", client=FakeOpenAIClient(completions))

    assert list(llm.stream("Say hello")) == ["你好", "，世界"]


def test_qwen_llm_maps_rate_limit_failures() -> None:
    completions = FakeChatCompletions()
    completions.error = _build_rate_limit_error("too many requests")
    llm = QwenLLM(api_key="test-key", client=FakeOpenAIClient(completions))

    with pytest.raises(LLMProviderError, match="Qwen rate limit exceeded"):
        llm.generate("Say hello")


def test_qwen_llm_maps_generic_gateway_failures() -> None:
    completions = FakeChatCompletions()
    completions.stream_error = RuntimeError("gateway down")
    llm = QwenLLM(api_key="test-key", client=FakeOpenAIClient(completions))

    with pytest.raises(LLMProviderError, match="Qwen request failed"):
        list(llm.stream("Say hello"))
