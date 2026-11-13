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
    assert captured["payload"]["provider"] == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
        "require_parameters": True,  # sampling was set explicitly, so it must be honoured
    }
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


def _capturing_backend(monkeypatch: pytest.MonkeyPatch, name: str, captured: dict[str, Any]) -> Any:
    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(env_var, "dummy-value")

    def tx(_u: str, _h: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return {
            "choices": [{"message": {"content": "Final estimate: 5"}}],
            "message": {"content": "Final estimate: 5"},
            "content": [{"text": "Final estimate: 5"}],
            "candidates": [{"content": {"parts": [{"text": "Final estimate: 5"}]}}],
        }

    return create_backend(name, transport=tx)


def _sampling_keys(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name == "ollama":
        return payload["options"]
    if name == "gemini":
        return payload["generationConfig"]
    return payload


@pytest.mark.parametrize("name", ["openrouter", "openai", "anthropic", "gemini", "ollama"])
def test_sampling_params_are_omitted_unless_set(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    captured: dict[str, Any] = {}
    backend = _capturing_backend(monkeypatch, name, captured)
    sampling = ("temperature", "top_p", "topP")

    # Numeric path with defaults: nothing about sampling may be sent.
    backend.generate_numeric_reply(system_prompt="s", user_prompt="u")
    assert not any(k in _sampling_keys(name, captured["payload"]) for k in sampling)

    # Explicit settings (e.g. the T=0 ablation) are sent, including zero.
    backend.generate_chat("s", "u", temperature=0.0, top_p=0.9)
    sent = _sampling_keys(name, captured["payload"])
    assert sent["temperature"] == 0.0
    assert sent["topP" if name == "gemini" else "top_p"] == pytest.approx(0.9)

    # Only one of the two set: the other stays out.
    backend.generate_chat("s", "u", temperature=0.3)
    sent = _sampling_keys(name, captured["payload"])
    assert sent["temperature"] == pytest.approx(0.3)
    assert "top_p" not in sent
    assert "topP" not in sent


def test_openrouter_requires_parameters_only_when_sampling_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-value")
    captured: dict[str, Any] = {}

    def tx(_u: str, _h: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "Final estimate: 5"}}]}

    pinned = OpenRouterBackend(model="m", pin_provider="OpenAI", transport=tx)
    pinned.generate_numeric_reply(system_prompt="s", user_prompt="u")
    assert captured["payload"]["provider"] == {"order": ["OpenAI"], "allow_fallbacks": False}

    pinned.generate_chat("s", "u", temperature=0.0)
    assert captured["payload"]["provider"] == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }

    unpinned = OpenRouterBackend(model="m", reasoning_effort="", transport=tx)
    unpinned.generate_numeric_reply(system_prompt="s", user_prompt="u")
    assert "provider" not in captured["payload"]
    unpinned.generate_chat("s", "u", top_p=0.9)
    assert captured["payload"]["provider"] == {"require_parameters": True}


def test_numeric_reply_defaults_to_1024_max_tokens_and_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    backend = _capturing_backend(monkeypatch, "openrouter", captured)
    backend.generate_numeric_reply(system_prompt="s", user_prompt="u")
    assert captured["payload"]["max_tokens"] == 1024
    backend.generate_numeric_reply(system_prompt="s", user_prompt="u", max_tokens=2048)
    assert captured["payload"]["max_tokens"] == 2048


@pytest.mark.parametrize(
    ("name", "response", "expected"),
    [
        (
            "openrouter",
            {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]},
            "length",
        ),
        ("openai", {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}, "stop"),
        ("anthropic", {"content": [{"text": "x"}], "stop_reason": "max_tokens"}, "max_tokens"),
        (
            "gemini",
            {"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "MAX_TOKENS"}]},
            "MAX_TOKENS",
        ),
        ("ollama", {"message": {"content": "x"}, "done_reason": "length"}, "length"),
        ("openai", {"choices": [{"message": {"content": "x"}}]}, ""),
    ],
)
def test_finish_reason_is_reported_by_every_adapter(
    monkeypatch: pytest.MonkeyPatch, name: str, response: dict[str, Any], expected: str
) -> None:
    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(env_var, "dummy-value")
    backend = create_backend(name, transport=lambda _u, _h, _p: response)
    assert backend.generate_chat("s", "u")["finish_reason"] == expected
    assert backend.generate_numeric_reply(system_prompt="s", user_prompt="u")["finish_reason"] == (
        expected
    )


def test_create_backend_passes_pin_provider_to_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-value")
    captured: dict[str, Any] = {}

    def tx(_u: str, _h: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "Final estimate: 5"}}]}

    backend = create_backend("openrouter", pin_provider="OpenAI", transport=tx)
    backend.generate_numeric_reply(system_prompt="s", user_prompt="u")
    assert captured["payload"]["provider"] == {"order": ["OpenAI"], "allow_fallbacks": False}


@pytest.mark.parametrize(
    ("usage", "expected_reasoning"),
    [
        ({"completion_tokens_details": {"reasoning_tokens": 40}}, 40),  # OpenAI format
        ({"reasoning_tokens": 8}, 8),  # flat fallback
        ({"completion_tokens_details": {"reasoning_tokens": 40}, "reasoning_tokens": 8}, 40),
        ({"completion_tokens_details": {"reasoning_tokens": 0}, "reasoning_tokens": 8}, 0),
        ({"completion_tokens_details": None}, 0),
        ({}, 0),
    ],
)
def test_reasoning_tokens_read_from_nested_usage_with_flat_fallback(
    usage: dict[str, Any], expected_reasoning: int
) -> None:
    from apart_epistemics.backends.base import extract_openai_chat_fields

    data = {
        "choices": [{"message": {"content": "x"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 100, **usage},
    }
    out = extract_openai_chat_fields(data, "p", "m")
    assert out["reasoning_tokens"] == expected_reasoning
    assert out["output_tokens"] == 100


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"cost": 9.47e-05}, 9.47e-05),
        ({"cost": 0}, 0.0),  # a free call is a real, known cost of zero
        ({"cost": None}, None),
        ({"cost": "n/a"}, None),
        ({"cost": -1}, None),
        ({}, None),
    ],
)
def test_provider_reported_cost_is_extracted_when_valid(
    usage: dict[str, Any], expected: float | None
) -> None:
    from apart_epistemics.backends.base import extract_openai_chat_fields

    data = {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 1, **usage}}
    assert extract_openai_chat_fields(data, "p", "m")["cost_usd"] == expected


def test_numeric_reply_passes_cost_through_and_other_adapters_report_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(env_var, "dummy-value")
    router = create_backend(
        "openrouter",
        transport=lambda _u, _h, _p: {
            "choices": [{"message": {"content": "Final estimate: 5"}}],
            "usage": {"cost": 0.0123},
        },
    )
    assert router.generate_numeric_reply(system_prompt="s", user_prompt="u")["cost_usd"] == 0.0123

    claude = create_backend(
        "anthropic",
        transport=lambda _u, _h, _p: {
            "content": [{"text": "Final estimate: 5"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    assert claude.generate_numeric_reply(system_prompt="s", user_prompt="u")["cost_usd"] is None
