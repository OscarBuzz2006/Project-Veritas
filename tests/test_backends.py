"""Unit tests for model backend adapters, secret redaction, and replay lookup."""

from __future__ import annotations

import json
from typing import Any

import pytest

from apart_epistemics.backends import (
    AnthropicBackend,
    BackendConfigError,
    GeminiBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
    OpenRouterBackend,
    create_backend,
)


def test_openrouter_backend_enforces_pinning_no_online_and_safe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(BackendConfigError, match="OPENROUTER_API_KEY"):
        OpenRouterBackend(model="openai/gpt-6-luna")

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter-value")
    captured: dict[str, Any] = {}

    def fake_transport(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "choices": [{"message": {"content": "Final estimate: 4200"}}],
            "provider": "OpenAI",
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "reasoning_tokens": 8},
        }

    backend = OpenRouterBackend(
        model="openai/gpt-6-luna:online",
        pin_provider="OpenAI",
        reasoning_effort="low",
        transport=fake_transport,
    )
    resp = backend.generate_chat(
        system_prompt="You are an assistant.",
        user_prompt="Estimate something.",
        temperature=0.8,
        top_p=0.95,
        max_tokens=200,
    )
    assert resp["text"] == "Final estimate: 4200"
    assert resp["provider"] == "OpenAI"
    assert resp["reasoning_tokens"] == 8
    assert ":online" not in captured["payload"]["model"]
    assert captured["payload"]["provider"] == {"order": ["OpenAI"], "allow_fallbacks": False}
    assert captured["payload"]["reasoning"] == {"effort": "low"}
    assert captured["payload"]["temperature"] == pytest.approx(0.8)
    assert captured["payload"]["top_p"] == pytest.approx(0.95)
    assert "dummy-openrouter-value" not in json.dumps(resp)


@pytest.mark.parametrize(
    ("backend_name", "env_var", "secret_val"),
    [
        ("openrouter", "OPENROUTER_API_KEY", "or-secret-xyz-123"),
        ("openai", "OPENAI_API_KEY", "oa-secret-xyz-123"),
        ("anthropic", "ANTHROPIC_API_KEY", "ant-secret-xyz-123"),
        ("gemini", "GEMINI_API_KEY", "gem-secret-xyz-123"),
    ],
)
def test_http_adapters_redact_secrets_from_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    env_var: str,
    secret_val: str,
) -> None:
    monkeypatch.setenv(env_var, secret_val)

    def failing_transport(_url: str, _headers: dict[str, str], _payload: dict[str, Any]) -> Any:
        raise RuntimeError(f"upstream failed with token {secret_val}")

    backend = create_backend(backend_name, transport=failing_transport)
    with pytest.raises(RuntimeError, match=r"\[REDACTED\]") as exc_info:
        backend.generate_chat("sys", "usr")
    assert secret_val not in str(exc_info.value)


def test_ollama_backend_serializes_temperature_and_handles_connection_failure() -> None:
    captured: dict[str, Any] = {}

    def fake_transport(
        url: str, _headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": '{"stance_prob": 0.65, "confidence": 0.72, "topics": []}'}}

    ollama = OllamaBackend(model="llama3.1:8b", transport=fake_transport)
    out = ollama.generate_chat("sys", "usr", temperature=0.0, top_p=0.0, max_tokens=128)
    assert "0.65" in out["text"]
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["options"]["temperature"] == 0.0
    assert captured["payload"]["options"]["top_p"] == 0.0

    failing = OllamaBackend(model="llama3.1:8b", base_url="http://127.0.0.1:59999")
    with pytest.raises(RuntimeError, match="Ollama"):
        failing.generate_chat("sys", "usr", temperature=0.7, top_p=0.95, max_tokens=32)


def test_direct_vendor_adapters_and_factory_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(BackendConfigError):
        OpenAICompatibleBackend(model="gpt-4o-mini")
    with pytest.raises(BackendConfigError):
        AnthropicBackend(model="claude-3-7-sonnet-latest")
    with pytest.raises(BackendConfigError):
        GeminiBackend(model="gemini-2.5-flash")
    with pytest.raises(BackendConfigError, match="jsonl_path"):
        create_backend("replay")
    with pytest.raises(BackendConfigError, match="Unknown backend"):
        create_backend("unsupported_vendor")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-ant-value")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-value")

    def openai_tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "Final estimate: 55"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def ant_tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": [{"text": "Final estimate: 89"}],
            "usage": {"input_tokens": 12, "output_tokens": 6},
        }

    def gem_tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidates": [{"content": {"parts": [{"text": "Final estimate: 144"}]}}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4},
        }

    assert (
        create_backend("openai", transport=openai_tx)
        .generate_numeric_reply(system_prompt="s", user_prompt="u")["raw_reply"]
        .endswith("55")
    )
    assert (
        create_backend("anthropic", transport=ant_tx)
        .generate_numeric_reply(system_prompt="s", user_prompt="u")["raw_reply"]
        .endswith("89")
    )
    assert (
        create_backend("gemini", transport=gem_tx)
        .generate_numeric_reply(system_prompt="s", user_prompt="u")["raw_reply"]
        .endswith("144")
    )
