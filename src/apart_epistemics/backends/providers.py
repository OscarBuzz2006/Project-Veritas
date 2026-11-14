"""Live HTTP LLM provider adapters (OpenRouter, Ollama, OpenAI, Anthropic, Gemini)."""

from __future__ import annotations

import os
import random
import re
import time
from typing import Any

from apart_epistemics.backends.base import (
    DEFAULT_NUMERIC_MAX_TOKENS,
    BackendConfigError,
    TransportFn,
    _HTTPChatBackend,
    extract_adaption_response_fields,
    extract_openai_chat_fields,
)


class OpenRouterBackend(_HTTPChatBackend):
    """OpenRouter client with :online stripping, provider pinning, and safe error handling."""

    def __init__(
        self,
        model: str,
        pin_provider: str | None = None,
        reasoning_effort: str = "low",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        transport: TransportFn | None = None,
    ) -> None:
        key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not key:
            raise BackendConfigError(
                "OPENROUTER_API_KEY is not set. Export it in your environment before live runs."
            )
        self._api_key = key
        self._secret_to_redact = key
        # Strip ":online" suffix so web retrieval cannot contaminate pure model priors.
        self.model = re.sub(r":online$", "", model.strip())
        self.pin_provider = pin_provider
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url.rstrip("/")
        self._transport = transport

    def for_model(self, model: str) -> OpenRouterBackend:
        return super().for_model(re.sub(r":online$", "", model.strip()))

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(max_tokens),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        provider_prefs: dict[str, Any] = {}
        if self.pin_provider:
            provider_prefs.update({"order": [self.pin_provider], "allow_fallbacks": False})
        if temperature is not None or top_p is not None:
            # OpenRouter silently drops unsupported parameters unless told otherwise. A run that
            # sets sampling on purpose must fail loudly, not log a setting that never applied.
            provider_prefs["require_parameters"] = True
        if provider_prefs:
            payload["provider"] = provider_prefs

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vikramwish/apart-epistemics",
            "X-Title": "Epistemic Prism",
        }
        data = self._post(
            f"{self.base_url}/chat/completions",
            headers,
            payload,
            f"OpenRouter request failed for '{self.model}'",
        )
        return extract_openai_chat_fields(
            data,
            default_provider=self.pin_provider or "openrouter",
            default_model=self.model,
        )


class OllamaBackend(_HTTPChatBackend):
    """Local Ollama adapter calling /api/chat without spending cloud tokens."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str | None = None,
        transport: TransportFn | None = None,
    ) -> None:
        self.model = model
        resolved = base_url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        self.base_url = resolved.rstrip("/")
        self._transport = transport

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"num_predict": int(max_tokens)},
        }
        if temperature is not None:
            payload["options"]["temperature"] = float(temperature)
        if top_p is not None:
            payload["options"]["top_p"] = float(top_p)
        data = self._post(
            url,
            {"Content-Type": "application/json"},
            payload,
            f"Ollama request failed for '{self.model}' at {self.base_url}",
        )
        msg = data.get("message") or {}
        return {
            "text": str(msg.get("content", "")),
            "finish_reason": str(data.get("done_reason") or ""),
            "provider": "ollama_local",
            "model_slug": str(data.get("model") or self.model),
            "input_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(data.get("eval_count", 0) or 0),
            "reasoning_tokens": 0,
        }


class OpenAICompatibleBackend(_HTTPChatBackend):
    """Direct OpenAI / Groq / Mistral chat-completions adapter."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = "openai",
        transport: TransportFn | None = None,
    ) -> None:
        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise BackendConfigError(
                f"{api_key_env} is not set. Export it before using {provider_name} backend."
            )
        self._api_key = key
        self._secret_to_redact = key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name
        self._transport = transport

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(max_tokens),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        data = self._post(
            f"{self.base_url}/chat/completions",
            headers,
            payload,
            f"{self.provider_name} request failed",
        )
        return extract_openai_chat_fields(data, self.provider_name, self.model)


class AnthropicBackend(_HTTPChatBackend):
    """Direct Anthropic Messages API adapter."""

    def __init__(
        self,
        model: str = "claude-3-7-sonnet-latest",
        api_key: str | None = None,
        transport: TransportFn | None = None,
    ) -> None:
        key = (api_key or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
        if not key:
            raise BackendConfigError("ANTHROPIC_API_KEY is not set.")
        self._api_key = key
        self._secret_to_redact = key
        self.model = model
        self._transport = transport

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": int(max_tokens),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            headers,
            payload,
            "Anthropic request failed",
        )
        blocks = data.get("content") or []
        text = " ".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()
        usage = data.get("usage") or {}
        return {
            "text": text,
            "finish_reason": str(data.get("stop_reason") or ""),
            "provider": "anthropic",
            "model_slug": str(data.get("model") or self.model),
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "reasoning_tokens": 0,
        }


class GeminiBackend(_HTTPChatBackend):
    """Direct Google Gemini generateContent API adapter."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        transport: TransportFn | None = None,
    ) -> None:
        key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        if not key:
            raise BackendConfigError("GEMINI_API_KEY is not set.")
        self._api_key = key
        self._secret_to_redact = key
        self.model = model
        self._transport = transport

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": int(max_tokens)},
        }
        if temperature is not None:
            payload["generationConfig"]["temperature"] = float(temperature)
        if top_p is not None:
            payload["generationConfig"]["topP"] = float(top_p)
        data = self._post(
            url,
            {"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            payload,
            "Gemini request failed",
        )
        cands = data.get("candidates") or []
        parts = cands[0].get("content", {}).get("parts", []) if cands else []
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        usage = data.get("usageMetadata") or {}
        return {
            "text": text,
            "finish_reason": str((cands[0].get("finishReason") if cands else "") or ""),
            "provider": "google_gemini",
            "model_slug": self.model,
            "input_tokens": int(usage.get("promptTokenCount", 0) or 0),
            "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "reasoning_tokens": 0,
        }


class AdaptionBackend(_HTTPChatBackend):
    """Direct Adaption Labs Responses API adapter (`https://api.prod.adaptionlabs.ai/api/v1`)."""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str = "https://api.prod.adaptionlabs.ai/api/v1",
        min_output_tokens: int = 8192,
        transport: TransportFn | None = None,
    ) -> None:
        raw_key = api_key if api_key is not None else os.environ.get("ADAPTION_API_KEY", "")
        key = raw_key.strip().strip("\"'")
        if not key:
            raise BackendConfigError(
                "ADAPTION_API_KEY is not set. Export it before using adaption backend."
            )
        self._api_key = key
        self._secret_to_redact = key
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.min_output_tokens = int(min_output_tokens)
        self.max_retries = 10
        if transport is None:
            self.retry_sleep = lambda delay: time.sleep(
                min(65.0, max(5.0, delay)) + random.uniform(0.5, 4.0)
            )
        self._transport = transport

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        effective_input = (
            f"{user_prompt} /no_think"
            if self.model.startswith("qwen")
            and "/think" not in user_prompt
            and "/no_think" not in user_prompt
            else user_prompt
        )
        effective_max = max(
            int(max_tokens), 1024 if int(max_tokens) < 512 else self.min_output_tokens
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": effective_input,
            "max_output_tokens": effective_max,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        # Adaption rejects top_p=0.0 with HTTP 400; temperature=0.0 already gives greedy decoding.
        if top_p is not None and float(top_p) > 0.0:
            payload["top_p"] = float(top_p)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Adaption/Python 0.19.0",
        }
        data = self._post(
            f"{self.base_url}/responses",
            headers,
            payload,
            f"Adaption request failed for '{self.model}'",
        )
        return extract_adaption_response_fields(data, default_model=self.model)

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return super().generate_numeric_reply(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max(int(max_tokens), self.min_output_tokens),
            **kwargs,
        )
