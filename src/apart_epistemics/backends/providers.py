"""Live HTTP LLM provider adapters (OpenRouter, Ollama, OpenAI, Anthropic, Gemini)."""

from __future__ import annotations

import os
import re
from typing import Any

from apart_epistemics.backends.base import (
    BackendConfigError,
    TransportFn,
    _HTTPChatBackend,
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
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.pin_provider:
            payload["provider"] = {"order": [self.pin_provider], "allow_fallbacks": False}

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
        temperature: float = 0.8,
        top_p: float = 0.95,
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
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "num_predict": int(max_tokens),
            },
        }
        data = self._post(
            url,
            {"Content-Type": "application/json"},
            payload,
            f"Ollama request failed for '{self.model}' at {self.base_url}",
        )
        msg = data.get("message") or {}
        return {
            "text": str(msg.get("content", "")),
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
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
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
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
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
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "topP": float(top_p),
                "maxOutputTokens": int(max_tokens),
            },
        }
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
            "provider": "google_gemini",
            "model_slug": self.model,
            "input_tokens": int(usage.get("promptTokenCount", 0) or 0),
            "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "reasoning_tokens": 0,
        }
