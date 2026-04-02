from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, BadRequestError

from ragms.libs.providers.llm.openai_llm import LLMProviderError, OpenAILLM


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


def _build_auth_error(message: str) -> AuthenticationError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return AuthenticationError(message, response=response, body=None)


def _build_bad_request_error(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body=None)


def test_openai_llm_generate_returns_completion_text() -> None:
    completions = FakeChatCompletions()
    completions.response = _build_response("hello from openai")
    llm = OpenAILLM(
        model="gpt-4.1-mini",
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    response = llm.generate("Say hello", system_prompt="Be concise")

    assert response == "hello from openai"
    assert completions.calls[0]["model"] == "gpt-4.1-mini"
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Say hello"},
    ]


def test_openai_llm_stream_yields_incremental_chunks() -> None:
    completions = FakeChatCompletions()
    completions.stream_response = iter(
        [_build_chunk("hello "), _build_chunk("world"), _build_chunk("")]
    )
    llm = OpenAILLM(
        model="gpt-4.1-mini",
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    response_chunks = list(llm.stream("Say hello"))

    assert response_chunks == ["hello ", "world"]
    assert completions.calls[0]["stream"] is True


def test_openai_llm_rejects_missing_api_key() -> None:
    llm = OpenAILLM(model="gpt-4.1-mini")

    with pytest.raises(LLMProviderError, match="OpenAI api_key is required"):
        llm.generate("Say hello")


def test_openai_llm_rejects_empty_model() -> None:
    llm = OpenAILLM(model="   ", api_key="test-key")

    with pytest.raises(LLMProviderError, match="OpenAI model must not be empty"):
        llm.generate("Say hello")


def test_openai_llm_maps_authentication_failure() -> None:
    completions = FakeChatCompletions()
    completions.error = _build_auth_error("bad key")
    llm = OpenAILLM(
        model="gpt-4.1-mini",
        api_key="bad-key",
        client=FakeOpenAIClient(completions),
    )

    with pytest.raises(LLMProviderError, match="OpenAI authentication failed"):
        llm.generate("Say hello")


def test_openai_llm_maps_invalid_model_failure() -> None:
    completions = FakeChatCompletions()
    completions.error = _build_bad_request_error("model not found")
    llm = OpenAILLM(
        model="bad-model",
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    with pytest.raises(LLMProviderError, match="OpenAI rejected model or request: bad-model"):
        llm.generate("Say hello")


def test_openai_llm_maps_upstream_stream_failure() -> None:
    completions = FakeChatCompletions()
    completions.stream_error = RuntimeError("upstream down")
    llm = OpenAILLM(
        model="gpt-4.1-mini",
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    with pytest.raises(LLMProviderError, match="OpenAI request failed"):
        list(llm.stream("Say hello"))
