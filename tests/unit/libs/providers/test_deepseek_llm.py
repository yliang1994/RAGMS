from __future__ import annotations

import pytest

from ragms.libs.providers.llm.deepseek_llm import DeepSeekLLM


def test_deepseek_llm_generate_and_stream_support_gateway_params() -> None:
    llm = DeepSeekLLM(
        model="deepseek-chat",
        api_key="test-key",
        temperature=0.2,
        base_url="https://gateway.example.com/v1",
    )

    response = llm.generate([{"role": "user", "content": "hello"}])
    chunks = list(llm.stream([{"role": "user", "content": "hello"}]))

    assert response["content"] == "deepseek-response: hello"
    assert response["base_url"] == "https://gateway.example.com/v1"
    assert [chunk["delta"] for chunk in chunks] == ["deepseek-response:", "hello"]


def test_deepseek_llm_handles_invalid_config_and_failure_cases() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        DeepSeekLLM(model="deepseek-chat", api_key=None)

    with pytest.raises(ValueError, match="Unsupported DeepSeek model"):
        DeepSeekLLM(model="bad-model", api_key="test-key")

    llm = DeepSeekLLM(model="deepseek-chat", api_key="test-key")
    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        llm.generate([{"role": "user", "content": "hi"}], simulate_rate_limit=True)

    with pytest.raises(RuntimeError, match="upstream request failed"):
        llm.generate([{"role": "user", "content": "hi"}], simulate_upstream_failure=True)
