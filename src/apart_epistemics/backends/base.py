"""Shared HTTP transport, exceptions, secret redaction, and chat adapter base class."""

from __future__ import annotations

import copy
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

TransportFn = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


class BackendConfigError(Exception):
    """Raised when required provider credentials or configuration are missing."""


class BudgetExceededError(Exception):
    """Raised when estimated or actual API spend exceeds the configured budget cap."""


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 45.0,
) -> dict[str, Any]:
    """Execute a synchronous JSON POST request using Python's standard library."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise RuntimeError("Unexpected non-object JSON response from backend.")
            return parsed
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
        except OSError:
            detail = str(exc.reason)
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Request error calling {url}: {exc}") from exc


def extract_openai_chat_fields(
    data: dict[str, Any],
    default_provider: str,
    default_model: str,
) -> dict[str, Any]:
    """Normalize an OpenAI-compatible chat-completions JSON response."""
    choices = data.get("choices") or []
    text = ""
    if choices and isinstance(choices, list):
        msg = choices[0].get("message") or {}
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            ).strip()
        else:
            text = str(content)
    usage = data.get("usage") or {}
    return {
        "text": text,
        "provider": str(data.get("provider") or default_provider),
        "model_slug": str(data.get("model") or default_model),
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
    }


class _HTTPChatBackend(ABC):
    """Base adapter providing safe transport execution, secret redaction, and numeric wrapper."""

    model: str
    _transport: TransportFn | None
    _secret_to_redact: str | None = None

    budget_ledger: Any = None
    request_log: Any = None

    def for_model(self, model: str) -> _HTTPChatBackend:
        """Clone routing settings while sharing the run budget and request journal."""
        routed = copy.copy(self)
        routed.model = model
        return routed

    def _post(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        error_prefix: str,
    ) -> dict[str, Any]:
        if self.budget_ledger is not None:
            self.budget_ledger.reserve()
        try:
            result = (
                self._transport(url, headers, payload)
                if self._transport is not None
                else post_json(url, headers, payload)
            )
            if self.request_log is not None:
                entry = json.dumps({"model": self.model, "request": payload, "response": result})
                if self._secret_to_redact:
                    entry = entry.replace(self._secret_to_redact, "[REDACTED]")
                with self.request_log.open("a", encoding="utf-8") as fh:
                    fh.write(entry + "\n")
                    fh.flush()
            return result
        except Exception as exc:
            msg = str(exc)
            if self._secret_to_redact:
                msg = msg.replace(self._secret_to_redact, "[REDACTED]")
            raise RuntimeError(f"{error_prefix}: {msg}") from None

    @abstractmethod
    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Execute one chat completion turn and return normalized text and usage fields."""

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 256,
        **_: Any,
    ) -> dict[str, Any]:
        """Run one quantitative estimation query reusing `generate_chat`."""
        resp = self.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return {
            "raw_reply": resp["text"],
            "model_slug": resp["model_slug"],
            "provider": resp["provider"],
            "reasoning_tokens": resp["reasoning_tokens"],
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
        }
