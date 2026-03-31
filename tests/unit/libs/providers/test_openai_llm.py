from __future__ import annotations

import pytest

from ragms.libs.providers.llm.openai_llm import OpenAILLM


def test_openai_llm_generate_returns_sync_response() -> None:
    llm = OpenAILLM(model="gpt-4.1-mini", api_key="test-key", temperature=0.1)

    response = llm.generate([{"role": "user", "content": "hello world"}])

    assert response["provider"] == "openai"
    assert response["model"] == "gpt-4.1-mini"
    assert response["content"] == "openai-response: hello world"
    assert response["usage"]["prompt_tokens"] == 1


def test_openai_llm_stream_yields_token_deltas() -> None:
    llm = OpenAILLM(model="gpt-4.1-mini", api_key="test-key")

    chunks = list(llm.stream([{"role": "user", "content": "hello world"}]))

    assert [chunk["delta"] for chunk in chunks] == ["openai-response:", "hello", "world"]


def test_openai_llm_rejects_missing_api_key_and_invalid_model() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        OpenAILLM(model="gpt-4.1-mini", api_key=None)

    with pytest.raises(ValueError, match="Unsupported OpenAI model"):
        OpenAILLM(model="invalid-model", api_key="test-key")


def test_openai_llm_raises_for_invalid_messages() -> None:
    llm = OpenAILLM(model="gpt-4.1-mini", api_key="test-key")

    with pytest.raises(ValueError, match="messages must not be empty"):
        llm.generate([])

    with pytest.raises(RuntimeError, match="Upstream response error"):
        llm.generate([{"role": "user"}])
