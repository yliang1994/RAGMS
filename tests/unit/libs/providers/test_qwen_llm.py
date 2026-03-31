from __future__ import annotations

import pytest

from ragms.libs.providers.llm.qwen_llm import QwenLLM


def test_qwen_llm_generate_and_stream_support_gateway_params() -> None:
    llm = QwenLLM(
        model="qwen-max",
        api_key="test-key",
        temperature=0.2,
        base_url="https://gateway.example.com/v1",
    )

    response = llm.generate([{"role": "user", "content": "你好"}])
    chunks = list(llm.stream([{"role": "user", "content": "你好"}]))

    assert response["content"] == "qwen-response: 你好"
    assert response["base_url"] == "https://gateway.example.com/v1"
    assert [chunk["delta"] for chunk in chunks] == ["qwen-response:", "你好"]


def test_qwen_llm_handles_invalid_config_and_failure_cases() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        QwenLLM(model="qwen-max", api_key=None)

    with pytest.raises(ValueError, match="Unsupported Qwen model"):
        QwenLLM(model="bad-model", api_key="test-key")

    llm = QwenLLM(model="qwen-max", api_key="test-key")
    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        llm.generate([{"role": "user", "content": "hi"}], simulate_rate_limit=True)

    with pytest.raises(RuntimeError, match="upstream request failed"):
        llm.generate([{"role": "user", "content": "hi"}], simulate_upstream_failure=True)

