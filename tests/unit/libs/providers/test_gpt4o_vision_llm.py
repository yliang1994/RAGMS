from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError

from ragms.libs.providers.vision_llms.gpt4o_vision_llm import (
    GPT4oVisionLLM,
    VisionProviderError,
)


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses: list[object] = []
        self.error: Exception | None = None

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if self.error:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError("No fake response queued")


class FakeOpenAIClient:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _build_response(text: str) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _build_auth_error(message: str) -> AuthenticationError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return AuthenticationError(message, response=response, body=None)


def _write_png(path: Path) -> Path:
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D49484452000000010000000108060000001F15C489"
            "0000000A49444154789C6360000002000154A24F5D00000000"
            "49454E44AE426082"
        )
    )
    return path


def test_gpt4o_vision_llm_captions_single_image_with_context(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path / "chart.png")
    completions = FakeChatCompletions()
    completions.responses.append(_build_response("A sales chart trending upward."))
    llm = GPT4oVisionLLM(
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    caption = llm.caption(image_path, prompt="Describe the chart", context="Quarterly revenue")

    assert caption == "A sales chart trending upward."
    content = completions.calls[0]["messages"][0]["content"]
    assert content[0]["text"] == "Describe the chart\nContext: Quarterly revenue"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_gpt4o_vision_llm_captions_multiple_images(tmp_path: Path) -> None:
    first = _write_png(tmp_path / "first.png")
    second = _write_png(tmp_path / "second.png")
    completions = FakeChatCompletions()
    completions.responses.extend([
        _build_response("first caption"),
        _build_response("second caption"),
    ])
    llm = GPT4oVisionLLM(
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    captions = llm.caption_batch([first, second], prompt="Summarize")

    assert captions == ["first caption", "second caption"]
    assert len(completions.calls) == 2


def test_gpt4o_vision_llm_rejects_unsupported_image_type(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.bin"
    image_path.write_bytes(b"\x00\x01")
    llm = GPT4oVisionLLM(api_key="test-key")

    with pytest.raises(VisionProviderError, match="Unsupported image type for captioning: .bin"):
        llm.caption(image_path)


def test_gpt4o_vision_llm_maps_authentication_failures(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path / "chart.png")
    completions = FakeChatCompletions()
    completions.error = _build_auth_error("bad key")
    llm = GPT4oVisionLLM(
        api_key="bad-key",
        client=FakeOpenAIClient(completions),
    )

    with pytest.raises(VisionProviderError, match="GPT-4o Vision authentication failed"):
        llm.caption(image_path)
