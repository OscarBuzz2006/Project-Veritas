"""Shared HTTP transport, exceptions, secret redaction, and chat adapter base class."""

from __future__ import annotations

import copy
import json
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

# Reasoning models may spend hidden reasoning tokens out of this budget, so keep it generous.
DEFAULT_NUMERIC_MAX_TOKENS = 1024

TransportFn = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


class BackendConfigError(Exception):
    """Raised when required provider credentials or configuration are missing."""


class BudgetExceededError(Exception):
    """Raised when estimated or actual API spend exceeds the configured budget cap."""


class TransientBackendError(RuntimeError):
    """A timeout, connection error, HTTP 429 or 5xx: worth retrying with backoff."""


# Serializes appends to the shared provider request journal when calls run in threads.
_REQUEST_LOG_LOCK = threading.Lock()


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 90.0,
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
        error_cls = TransientBackendError if exc.code == 429 or exc.code >= 500 else RuntimeError
        raise error_cls(f"HTTP {exc.code} from {url}: {detail}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Request error calling {url}: {exc}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransientBackendError(f"Request error calling {url}: {exc}") from exc


def extract_openai_chat_fields(
    data: dict[str, Any],
    default_provider: str,
    default_model: str,
) -> dict[str, Any]:
    """Normalize an OpenAI-compatible chat-completions JSON response."""
    choices = data.get("choices") or []
    text = ""
    finish_reason = ""
    if choices and isinstance(choices, list):
        finish_reason = str(choices[0].get("finish_reason") or "")
        msg = choices[0].get("message") or {}
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            ).strip()
        else:
            text = str(content)
    usage = data.get("usage") or {}
    # OpenAI-format APIs report reasoning tokens under completion_tokens_details (and already
    # include them in completion_tokens); some gateways put them at the top level of usage.
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    if reasoning is None:
        reasoning = usage.get("reasoning_tokens", 0)
    # OpenRouter reports the actual charge per call; other providers do not.
    raw_cost = usage.get("cost")
    cost_usd = float(raw_cost) if isinstance(raw_cost, int | float) and raw_cost >= 0 else None
    return {
        "text": text,
        "finish_reason": finish_reason,
        "cost_usd": cost_usd,
        "provider": str(data.get("provider") or default_provider),
        "model_slug": str(data.get("model") or default_model),
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "reasoning_tokens": int(reasoning or 0),
    }


def extract_adaption_response_fields(
    data: dict[str, Any],
    default_model: str,
) -> dict[str, Any]:
    """Normalize an Adaption Responses API JSON payload."""
    if data.get("error"):
        raise TransientBackendError(f"Adaption upstream error: {data['error']}")
    text = str(data.get("output_text") or "")
    if not text:
        parts = [
            str(content.get("text", ""))
            for item in (data.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "message"
            for content in (item.get("content") or [])
            if isinstance(content, dict) and content.get("type") == "output_text"
        ]
        text = "".join(parts)
    status = str(data.get("status") or "")
    if status == "completed":
        finish_reason = "stop"
    elif status == "incomplete":
        finish_reason = "length"
    else:
        finish_reason = status or "stop"
    usage = data.get("usage") or {}
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if reasoning is None:
        reasoning = usage.get("reasoning_tokens", 0)
    raw_cost = usage.get("cost")
    cost_usd = float(raw_cost) if isinstance(raw_cost, int | float) and raw_cost >= 0 else None
    return {
        "text": text,
        "finish_reason": finish_reason,
        "cost_usd": cost_usd,
        "provider": "adaption",
        "model_slug": str(data.get("model") or default_model),
        "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(reasoning or 0),
    }


class _HTTPChatBackend(ABC):
    """Base adapter providing safe transport execution, secret redaction, and numeric wrapper."""

    model: str
    _transport: TransportFn | None
    _secret_to_redact: str | None = None

    budget_ledger: Any = None
    request_log: Any = None
    # Retries after the first attempt on TransientBackendError, with exponential backoff.
    max_retries: int = 3
    retry_sleep: Callable[[float], None] = staticmethod(time.sleep)

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
        attempt = 0
        while True:
            # Every HTTP attempt, retries included, reserves budget before it is sent.
            if self.budget_ledger is not None:
                self.budget_ledger.reserve()
            try:
                result = (
                    self._transport(url, headers, payload)
                    if self._transport is not None
                    else post_json(url, headers, payload)
                )
                if self.request_log is not None:
                    entry = json.dumps(
                        {"model": self.model, "request": payload, "response": result}
                    )
                    if self._secret_to_redact:
                        entry = entry.replace(self._secret_to_redact, "[REDACTED]")
                    with _REQUEST_LOG_LOCK, self.request_log.open("a", encoding="utf-8") as fh:
                        fh.write(entry + "\n")
                        fh.flush()
                return result
            except Exception as exc:
                if isinstance(exc, TransientBackendError) and attempt < self.max_retries:
                    self.retry_sleep(min(30.0, 2.0**attempt))
                    attempt += 1
                    continue
                msg = str(exc)
                if self._secret_to_redact:
                    msg = msg.replace(self._secret_to_redact, "[REDACTED]")
                raise RuntimeError(f"{error_prefix}: {msg}") from None

    @abstractmethod
    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Execute one chat completion turn and return normalized text and usage fields."""

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
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
            "finish_reason": resp.get("finish_reason", ""),
            "cost_usd": resp.get("cost_usd"),
            "reasoning_tokens": resp["reasoning_tokens"],
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
        }
