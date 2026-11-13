"""Model backend adapters (OpenRouter, Ollama, OpenAI, Anthropic, Gemini, Replay, Synthetic)."""

from __future__ import annotations

import json
import math
import os
import random
import re
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from apart_epistemics.models import (
    AgentProfile,
    ConditionConfig,
    HeterogeneityType,
    NationalityType,
    RoundRecord,
    TaskItem,
    TaskKind,
)

TransportFn = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


class BackendConfigError(Exception):
    """Raised when required provider credentials or configuration are missing."""


class BudgetExceededError(Exception):
    """Raised when estimated or actual API spend exceeds the configured budget cap."""


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 45.0,
) -> dict[str, Any]:
    """Execute a synchronous JSON POST request without external HTTP dependencies."""
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
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
        except OSError:
            detail = str(exc.reason)
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Request error calling {url}: {exc}") from exc


def _extract_openai_chat_fields(
    data: dict[str, Any],
    default_provider: str,
    default_model: str,
) -> dict[str, Any]:
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


class OpenRouterBackend:
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
        # Prevent silent web-search contamination when evaluating pure model priors.
        self.model = re.sub(r":online$", "", model.strip())
        self.pin_provider = pin_provider
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url.rstrip("/")
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
        url = f"{self.base_url}/chat/completions"
        try:
            data = (
                self._transport(url, headers, payload)
                if self._transport is not None
                else _post_json(url, headers, payload)
            )
        except Exception as exc:
            msg = str(exc).replace(self._api_key, "[REDACTED]")
            raise RuntimeError(f"OpenRouter request failed for '{self.model}': {msg}") from exc

        return _extract_openai_chat_fields(
            data,
            default_provider=self.pin_provider or "openrouter",
            default_model=self.model,
        )

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 256,
        **_: Any,
    ) -> dict[str, Any]:
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


class OllamaBackend:
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
        headers = {"Content-Type": "application/json"}
        try:
            data = (
                self._transport(url, headers, payload)
                if self._transport is not None
                else _post_json(url, headers, payload)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed for '{self.model}' at {self.base_url}: {exc}"
            ) from exc

        msg = data.get("message") or {}
        text = str(msg.get("content", ""))
        return {
            "text": text,
            "provider": "ollama_local",
            "model_slug": str(data.get("model") or self.model),
            "input_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(data.get("eval_count", 0) or 0),
            "reasoning_tokens": 0,
        }

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 256,
        **_: Any,
    ) -> dict[str, Any]:
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
            "reasoning_tokens": 0,
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
        }


class OpenAICompatibleBackend:
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
        url = f"{self.base_url}/chat/completions"
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
        try:
            data = (
                self._transport(url, headers, payload)
                if self._transport is not None
                else _post_json(url, headers, payload)
            )
        except Exception as exc:
            msg = str(exc).replace(self._api_key, "[REDACTED]")
            raise RuntimeError(f"{self.provider_name} request failed: {msg}") from exc
        return _extract_openai_chat_fields(data, self.provider_name, self.model)

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 256,
        **_: Any,
    ) -> dict[str, Any]:
        resp = self.generate_chat(system_prompt, user_prompt, temperature, top_p, max_tokens)
        return {
            "raw_reply": resp["text"],
            "model_slug": resp["model_slug"],
            "provider": resp["provider"],
            "reasoning_tokens": resp["reasoning_tokens"],
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
        }


class AnthropicBackend:
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
        url = "https://api.anthropic.com/v1/messages"
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
        try:
            data = (
                self._transport(url, headers, payload)
                if self._transport is not None
                else _post_json(url, headers, payload)
            )
        except Exception as exc:
            msg = str(exc).replace(self._api_key, "[REDACTED]")
            raise RuntimeError(f"Anthropic request failed: {msg}") from exc
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

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 256,
        **_: Any,
    ) -> dict[str, Any]:
        resp = self.generate_chat(system_prompt, user_prompt, temperature, top_p, max_tokens)
        return {
            "raw_reply": resp["text"],
            "model_slug": resp["model_slug"],
            "provider": resp["provider"],
            "reasoning_tokens": 0,
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
        }


class GeminiBackend:
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
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self._api_key}"
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
        headers = {"Content-Type": "application/json"}
        try:
            data = (
                self._transport(url, headers, payload)
                if self._transport is not None
                else _post_json(url, headers, payload)
            )
        except Exception as exc:
            msg = str(exc).replace(self._api_key, "[REDACTED]")
            raise RuntimeError(f"Gemini request failed: {msg}") from exc
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

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 256,
        **_: Any,
    ) -> dict[str, Any]:
        resp = self.generate_chat(system_prompt, user_prompt, temperature, top_p, max_tokens)
        return {
            "raw_reply": resp["text"],
            "model_slug": resp["model_slug"],
            "provider": resp["provider"],
            "reasoning_tokens": 0,
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
        }


class ReplayBackend:
    """Deterministic offline replay backend backed by an archived JSONL run log."""

    def __init__(self, raw_rows: list[dict[str, Any]]) -> None:
        self._raw_rows = raw_rows
        self._round_index: dict[tuple[str, str, int, str], RoundRecord] = {}
        for row in raw_rows:
            if "task_id" in row and "agent_response_text" in row:
                rec = RoundRecord.model_validate(row)
                key = (rec.condition_id, rec.task_id, rec.round_idx, rec.agent_id)
                self._round_index[key] = rec

    @classmethod
    def from_jsonl(cls, path: Path) -> ReplayBackend:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return cls(rows)

    @property
    def record_count(self) -> int:
        return len(self._raw_rows)

    def get_round_record(
        self, condition_id: str, task_id: str, round_idx: int, agent_id: str
    ) -> RoundRecord | None:
        return self._round_index.get((condition_id, task_id, round_idx, agent_id))


_STOCHASTIC_OPENERS = [
    "Looking closely at",
    "From an empirical vantage point on",
    "In practical operational contexts involving",
    "Weighing longitudinal evidence around",
    "Setting aside textbook assumptions about",
    "Considering cross-regional constraints in",
    "Focusing on structural trade-offs within",
    "Examining baseline mechanisms behind",
    "Reassessing conventional priors regarding",
    "Evaluating heterogeneous evidence tied to",
]

_STOCHASTIC_CONNECTORS = [
    "directly shifts probability toward",
    "provides decisive signal favoring",
    "tempers sweeping claims in favor of",
    "highlights localized sensitivity around",
    "exposes hidden correlation across",
    "sharpens conditional likelihood toward",
    "demonstrates persistent divergence over",
    "anchors the causal assessment near",
]

_STOCHASTIC_CLOSINGS = [
    "pending out-of-sample replication across additional cohorts.",
    "even after accounting for institutional measurement noise.",
    "whereas narrow monoculture priors overlook tail regimes.",
    "especially under shifting regional resource baselines.",
    "which resists premature convergence onto a single framing.",
    "when calibrated against historical base rates and variance.",
    "contingent on transparent empirical monitoring.",
    "leaving substantive room for competing mechanistic hypotheses.",
]

_STOCHASTIC_REGISTERS = [
    "systemic vulnerability",
    "institutional inertia",
    "counterfactual baseline",
    "distributional asymmetry",
    "measurement elasticity",
    "feedback amplification",
    "marginal constraint",
    "structural heterogeneity",
]


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _humanize_topic(slug: str) -> str:
    return slug.replace("_", " ")


class SyntheticEpistemicBackend:
    """Grounded stochastic mechanism engine modeling decoding, origin, and personalization."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _task_truth_target(self, task: TaskItem) -> float:
        if task.kind == TaskKind.GROUND_TRUTH and task.ground_truth is not None:
            return 0.88 if task.ground_truth == 1 else 0.12
        return task.global_north_prior

    def _model_epistemic_profile(
        self,
        cond: ConditionConfig,
        task: TaskItem,
        model_slug: str,
        model_family: str,
    ) -> tuple[float, list[int], float]:
        """Return (assistant_target_prob, preferred_topic_indices, assistant_confidence)."""
        topics = task.canonical_topics or ["general_analysis"]
        m_len = len(topics)
        truth_target = self._task_truth_target(task)

        # Misconception-lure pull scales with popular_misconception_bias on ground-truth items.
        lure_shift = task.popular_misconception_bias * 0.85

        if cond.heterogeneity_type == HeterogeneityType.MONOLITHIC_SHARED:
            if cond.nationality_type == NationalityType.CN_ONLY:
                target = _clamp(
                    0.5 * truth_target + 0.5 * (1.0 - task.global_north_prior) + 0.55 * lure_shift,
                    0.08,
                    0.92,
                )
                pref_idx = [2 % m_len, 3 % m_len]
                return (target, pref_idx, 0.84)
            # US monolithic assistant strongly concentrates on first two Western-framing slots.
            target = _clamp(
                0.55 * truth_target + 0.45 * task.global_north_prior + 0.72 * lure_shift,
                0.08,
                0.92,
            )
            pref_idx = [0, 1 % m_len]
            return (target, pref_idx, 0.87)

        if cond.heterogeneity_type == HeterogeneityType.SHARED_FEEDBACK:
            target = _clamp(
                0.48 * truth_target + 0.52 * task.global_north_prior + 0.82 * lure_shift,
                0.07,
                0.93,
            )
            return (target, [0, 1 % m_len], 0.90)

        if cond.heterogeneity_type == HeterogeneityType.MULTI_LAB_MIX:
            lab_hash = sum(ord(ch) for ch in model_slug) % max(m_len, 1)
            if cond.nationality_type == NationalityType.CN_ONLY:
                target = _clamp(
                    0.62 * truth_target
                    + 0.38 * (1.0 - task.global_north_prior)
                    + 0.42 * lure_shift,
                    0.10,
                    0.90,
                )
            else:
                target = _clamp(
                    0.64 * truth_target + 0.36 * task.global_north_prior + 0.48 * lure_shift,
                    0.10,
                    0.90,
                )
            return (target, [lab_hash, (lab_hash + 1) % m_len], 0.82)

        if cond.heterogeneity_type == HeterogeneityType.DIVERGENT_SFT_PAIR:
            if "arete" in model_family or "arete" in model_slug:
                target = _clamp(
                    0.66 * truth_target
                    + 0.34 * max(0.65, task.global_north_prior)
                    + 0.3 * lure_shift,
                    0.10,
                    0.90,
                )
                return (target, [1 % m_len, 4 % m_len, 5 % m_len], 0.84)
            target = _clamp(
                0.62 * truth_target
                + 0.38 * min(0.38, 1.0 - task.global_north_prior)
                + 0.3 * lure_shift,
                0.10,
                0.90,
            )
            return (target, [0, 2 % m_len, 3 % m_len], 0.82)

        if cond.heterogeneity_type == HeterogeneityType.CULTURAL_REGIONAL_MIX:
            slot = sum(ord(ch) for ch in model_family) % max(m_len, 1)
            if model_family.startswith("cn_"):
                target = _clamp(
                    0.70 * truth_target + 0.30 * (1.0 - task.global_north_prior), 0.12, 0.88
                )
            elif model_family.startswith("eu_"):
                target = _clamp(0.74 * truth_target + 0.26 * 0.50, 0.12, 0.88)
            elif model_family.startswith("us_arete"):
                target = _clamp(0.72 * truth_target + 0.28 * task.global_north_prior, 0.12, 0.88)
            else:
                target = _clamp(
                    0.70 * truth_target + 0.30 * task.global_north_prior + 0.22 * lure_shift,
                    0.12,
                    0.88,
                )
            return (target, [slot, (slot + 2) % m_len, (slot + 4) % m_len], 0.78)

        # Default fallback
        return (truth_target, [0, 1 % m_len], 0.75)

    def _compose_prose(
        self,
        rng: random.Random,
        cond: ConditionConfig,
        agent: AgentProfile,
        task: TaskItem,
        topics: list[str],
        stance_prob: float,
        confidence: float,
        round_idx: int,
    ) -> str:
        t_phrases = [_humanize_topic(t) for t in topics]
        focus = ", ".join(t_phrases[:2]) if len(t_phrases) >= 2 else t_phrases[0]
        direction = "supports Hypothesis B" if stance_prob >= 0.5 else "favors Hypothesis A"
        if cond.temperature <= 0.05:
            # Near-deterministic greedy decoding collapses phrasing variety across agents.
            return (
                f"Round {round_idx} assessment for {task.id}: Focus on {focus}. "
                f"Evidence {direction} with probability {stance_prob:.2f} "
                f"and confidence {confidence:.2f}."
            )

        scale = _clamp(cond.temperature, 0.1, 1.5)
        opener = rng.choice(_STOCHASTIC_OPENERS)
        conn = rng.choice(_STOCHASTIC_CONNECTORS)
        closing = rng.choice(_STOCHASTIC_CLOSINGS)
        reg = rng.choice(_STOCHASTIC_REGISTERS)
        if scale > 0.6:
            persona_cue = (
                f"As a {agent.profession} in {agent.region.replace('_', ' ')} "
                f"prioritizing {' and '.join(agent.values)}, "
            )
        else:
            persona_cue = ""
        return (
            f"{persona_cue}{opener} {focus}, the balance of {reg} {conn} "
            f"{direction} (p={stance_prob:.2f}, conf={confidence:.2f}), {closing}"
        )

    def generate_agent_step(
        self,
        cond: ConditionConfig,
        task: TaskItem,
        agent: AgentProfile,
        agent_idx: int,
        round_idx: int,
        prev_stance: float,
        prev_confidence: float,
        prev_topics: list[str],
        peer_stances: list[float],
        peer_topics: list[list[str]],
        model_assigned: str,
        model_family: str,
    ) -> dict[str, Any]:
        """Simulate one agent-assistant-peer round with mechanistic epistemic dynamics."""
        rng_seed = (
            self.seed * 100003
            + sum(ord(c) for c in cond.condition_id) * 977
            + sum(ord(c) for c in task.id) * 131
            + agent_idx * 43
            + round_idx * 17
        )
        rng = random.Random(rng_seed)
        canonical = task.canonical_topics or ["general_analysis"]
        m_len = len(canonical)

        # Personalized conditions span individual adapters + regional/role grounding.
        is_personalized = cond.heterogeneity_type in (
            HeterogeneityType.PERSONALIZED_ADVISOR,
            HeterogeneityType.PERSONALIZED_PEER,
            HeterogeneityType.PERSONALIZED_PROFILE,
        )

        # Round 0 represents agent's raw pre-assistant initial stance and local topics.
        if round_idx == 0:
            region_offset = (zlib.crc32(agent.region.encode("utf-8")) % 5 - 2) * 0.04
            task_pull = (
                (
                    0.28 * (self._task_truth_target(task) - 0.5)
                    + 0.35 * task.popular_misconception_bias
                )
                if task.kind == TaskKind.GROUND_TRUTH
                else 0.22 * (task.global_north_prior - 0.5)
            )
            stance = _clamp(
                agent.prior_belief + task_pull + region_offset + rng.uniform(-0.07, 0.07),
                0.05,
                0.95,
            )
            conf = _clamp(agent.prior_confidence + rng.uniform(-0.05, 0.05), 0.51, 0.96)
            idx1 = (agent_idx * 3 + len(agent.region)) % m_len
            idx2 = (idx1 + 1 + (agent_idx % 2)) % m_len
            topics = [canonical[idx1], canonical[idx2]] if idx2 != idx1 else [canonical[idx1]]
            prose = self._compose_prose(rng, cond, agent, task, topics, stance, conf, round_idx)
            assistant_text = "Independent baseline prior (no assistant prompt in Round 0)."
            return {
                "assistant_response_text": assistant_text,
                "agent_response_text": prose,
                "extracted_topics": topics,
                "stance_prob": round(stance, 4),
                "confidence": round(conf, 4),
                "stated_answer": 1 if stance >= 0.5 else 0,
                "goldman_source_overlap": 0.0,
                "persuasion_bias_exposure": 0.0,
            }

        peer_mean = sum(peer_stances) / len(peer_stances) if peer_stances else prev_stance
        flat_peer_topics = [t for sublist in peer_topics for t in sublist]

        if cond.heterogeneity_type == HeterogeneityType.NO_AI:
            # Independent learning + organic local peers preserves all topic slots.
            truth_target = self._task_truth_target(task)
            noisy_prior = 0.5 + 0.72 * (truth_target - 0.5) + 0.42 * task.popular_misconception_bias
            indep_signal = _clamp(noisy_prior + rng.gauss(0.0, 0.14), 0.05, 0.95)
            next_stance = _clamp(
                0.56 * prev_stance + 0.24 * peer_mean + 0.20 * indep_signal,
                0.04,
                0.96,
            )
            next_conf = _clamp(0.55 + 0.55 * abs(next_stance - 0.5), 0.52, 0.88)
            slot_a = (agent_idx + round_idx) % m_len
            slot_b = (agent_idx * 2 + 1) % m_len
            topics = (
                [canonical[slot_a], canonical[slot_b]] if slot_b != slot_a else [canonical[slot_a]]
            )
            assistant_text = "No AI advisor in loop; pure agent reasoning and peer discussion."
            overlap = 0.05
            bias_exp = 0.10
        elif is_personalized:
            truth_target = self._task_truth_target(task)
            region_shift = (agent_idx % 3 - 1) * 0.06
            # Each agent gets its own regional/professional topic lens spanning all slots.
            primary_slot = (agent_idx * 5 + sum(ord(c) for c in agent.profession)) % m_len
            secondary_slot = (primary_slot + 2) % m_len
            topics = (
                [canonical[primary_slot], canonical[secondary_slot]]
                if secondary_slot != primary_slot
                else [canonical[primary_slot]]
            )

            if cond.heterogeneity_type == HeterogeneityType.PERSONALIZED_ADVISOR:
                # Calibrated Socratic advisor counters misconception lures and bounds confidence.
                advisor_target = _clamp(
                    0.86 * truth_target + 0.14 * (0.5 + region_shift) + rng.gauss(0.0, 0.05),
                    0.06,
                    0.94,
                )
                next_stance = _clamp(
                    0.28 * prev_stance + 0.60 * advisor_target + 0.12 * peer_mean,
                    0.05,
                    0.95,
                )
                next_conf = _clamp(0.52 + 0.72 * abs(next_stance - 0.5), 0.52, 0.87)
                assistant_text = (
                    f"Critical advisor tailored for {agent.profession}: stress-testing assumptions "
                    f"on {_humanize_topic(topics[0])} against calibrated base rates."
                )
                overlap = 0.15
                bias_exp = 0.18
            elif cond.heterogeneity_type == HeterogeneityType.PERSONALIZED_PEER:
                # Sycophantic peer echoes misconception-prone intuition and inflates certainty.
                lure_pull = task.popular_misconception_bias * 1.15
                syc_target = _clamp(
                    prev_stance + lure_pull + region_shift + rng.gauss(0.0, 0.06),
                    0.04,
                    0.96,
                )
                # Agent-task pairs with low need-for-cognition anchor onto the false intuition.
                if task.kind == TaskKind.GROUND_TRUTH and task.ground_truth is not None:
                    wrong_target = 0.14 if task.ground_truth == 1 else 0.86
                    anchor_hash = (agent_idx + sum(ord(c) for c in task.id)) % 10
                    if anchor_hash < 5:
                        syc_target = wrong_target
                    else:
                        syc_target = _clamp(
                            0.62 * truth_target + 0.38 * prev_stance + rng.gauss(0.0, 0.06),
                            0.06,
                            0.94,
                        )
                next_stance = _clamp(
                    0.35 * prev_stance + 0.55 * syc_target + 0.10 * peer_mean,
                    0.04,
                    0.96,
                )
                # Overconfident sycophancy: certainty inflated regardless of correctness.
                next_conf = _clamp(0.84 + 0.12 * rng.random(), 0.78, 0.98)
                assistant_text = (
                    f"Warm peer resonance for {agent.profession}: your intuition regarding "
                    f"{_humanize_topic(topics[0])} makes total sense here."
                )
                overlap = 0.22
                bias_exp = 1.65
            else:
                # PERSONALIZED_PROFILE: intermediate neutral personalization
                prof_target = _clamp(
                    0.68 * truth_target
                    + 0.32 * prev_stance
                    + 0.3 * task.popular_misconception_bias,
                    0.06,
                    0.94,
                )
                next_stance = _clamp(
                    0.42 * prev_stance + 0.45 * prof_target + 0.13 * peer_mean, 0.05, 0.95
                )
                next_conf = _clamp(0.58 + 0.60 * abs(next_stance - 0.5), 0.54, 0.91)
                assistant_text = (
                    f"Profile-adapted note ({agent.region}): emphasizing "
                    f"{_humanize_topic(topics[0])}."
                )
                overlap = 0.18
                bias_exp = 0.45
        else:
            # Shared / Multi-Lab / Divergent-SFT / Cross-National model conditions.
            assistant_target, pref_slots, assistant_conf = self._model_epistemic_profile(
                cond, task, model_assigned, model_family
            )
            assistant_weight = (
                0.76
                if cond.heterogeneity_type
                in (
                    HeterogeneityType.MONOLITHIC_SHARED,
                    HeterogeneityType.SHARED_FEEDBACK,
                )
                else 0.62
            )
            noise_sd = 0.03 if cond.temperature <= 0.05 else 0.055 * (0.6 + 0.4 * cond.temperature)
            next_stance = _clamp(
                (1.0 - assistant_weight) * (0.6 * prev_stance + 0.4 * peer_mean)
                + assistant_weight * assistant_target
                + rng.gauss(0.0, noise_sd),
                0.04,
                0.96,
            )
            next_conf = _clamp(
                0.4 * prev_confidence + 0.6 * assistant_conf + rng.uniform(-0.04, 0.04),
                0.52,
                0.96,
            )

            # Topic funnel: Monolithic US/CN collapse onto 1-2 dominant assistant topics;
            # Cross-national and Divergent SFT sustain wide topic distribution.
            if cond.heterogeneity_type in (
                HeterogeneityType.MONOLITHIC_SHARED,
                HeterogeneityType.SHARED_FEEDBACK,
            ):
                top_idx = (
                    pref_slots[0] if (round_idx >= 2 or rng.random() < 0.78) else pref_slots[-1]
                )
                topics = [canonical[top_idx]]
                if round_idx == 1 and len(pref_slots) > 1 and rng.random() < 0.35:
                    topics = [canonical[pref_slots[0]], canonical[pref_slots[1]]]
                overlap = (
                    0.85 if cond.heterogeneity_type == HeterogeneityType.SHARED_FEEDBACK else 0.72
                )
                bias_exp = 1.25 + 0.2 * round_idx
            elif cond.heterogeneity_type == HeterogeneityType.MULTI_LAB_MIX:
                primary = pref_slots[0]
                sec = pref_slots[1] if len(pref_slots) > 1 else pref_slots[0]
                topics = (
                    [canonical[primary]]
                    if round_idx >= 3 and rng.random() < 0.45
                    else [canonical[primary], canonical[sec]]
                )
                overlap = 0.52
                bias_exp = 0.85
            else:
                # DIVERGENT_SFT_PAIR or CULTURAL_REGIONAL_MIX preserves broad multi-frame agendas.
                a_slot = pref_slots[(agent_idx + round_idx) % len(pref_slots)]
                b_slot = pref_slots[(agent_idx + 1) % len(pref_slots)]
                extra = prev_topics[0] if prev_topics and rng.random() < 0.5 else canonical[b_slot]
                cand = [canonical[a_slot], extra]
                topics = list(dict.fromkeys(cand))
                overlap = (
                    0.22
                    if cond.heterogeneity_type == HeterogeneityType.CULTURAL_REGIONAL_MIX
                    else 0.30
                )
                bias_exp = 0.42

            assistant_text = (
                f"[{model_assigned}] Highlighting {_humanize_topic(topics[0])} "
                f"toward p={assistant_target:.2f}."
            )

        if flat_peer_topics and cond.heterogeneity_type == HeterogeneityType.SHARED_FEEDBACK:
            most_common = max(set(flat_peer_topics), key=flat_peer_topics.count)
            topics = [most_common]

        prose = self._compose_prose(
            rng, cond, agent, task, topics, next_stance, next_conf, round_idx
        )
        return {
            "assistant_response_text": assistant_text,
            "agent_response_text": prose,
            "extracted_topics": topics,
            "stance_prob": round(next_stance, 4),
            "confidence": round(next_conf, 4),
            "stated_answer": 1 if next_stance >= 0.5 else 0,
            "goldman_source_overlap": round(overlap, 4),
            "persuasion_bias_exposure": round(bias_exp, 4),
        }

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        condition_id: str = "C1",
        task: TaskItem | None = None,
        round_idx: int = 1,
        current_x: float = 100.0,
        pop_stats: tuple[float, float, float] | None = None,
        agent: AgentProfile | None = None,
        model_slug: str = "openai/gpt-frontier",
        temperature: float = 0.8,
        **_: Any,
    ) -> dict[str, Any]:
        """Simulate Kevin's C1/C2/C3 numeric epistemic responses around log10 ground truth."""
        truth = task.numeric_truth if (task and task.numeric_truth) else max(current_x, 1.0)
        log_truth = math.log10(truth)
        log_cur = math.log10(max(current_x, 1e-6))

        seed_val = (
            self.seed * 7919
            + sum(ord(c) for c in condition_id) * 101
            + round_idx * 37
            + sum(ord(c) for c in (agent.agent_id if agent else "a0")) * 19
            + int(abs(log_truth) * 100)
        )
        rng = random.Random(seed_val)

        # Shared model systematic misconception offset (e.g., +0.18 log10 units from truth)
        shared_bias = 0.16 if (task and abs(task.popular_misconception_bias) > 0.15) else 0.10

        if condition_id == "C2" and pop_stats is not None:
            pop_med = math.log10(max(pop_stats[0], 1e-6))
            # C2 anchors onto the shared crowd median and shared bias -> diversity collapse.
            target_log = 0.65 * pop_med + 0.25 * (log_truth + shared_bias) + 0.10 * log_cur
            noise = rng.gauss(0.0, 0.03 * max(0.3, temperature))
        elif condition_id == "C3" and agent is not None:
            # C3 incorporates persona calibration + history, preserving heterogeneous spread.
            persona_shift = ((zlib.crc32(agent.profession.encode("utf-8")) % 7) - 3) * 0.06
            target_log = 0.55 * (log_truth + persona_shift) + 0.45 * log_cur
            noise = rng.gauss(0.0, 0.12 * max(0.4, temperature))
        else:
            # C1: uncoordinated baseline assistant bias around current estimate + truth.
            target_log = 0.50 * (log_truth + shared_bias * 0.7) + 0.50 * log_cur
            noise = rng.gauss(0.0, 0.11 * max(0.4, temperature))

        pred_val = 10.0 ** (target_log + noise)
        # Keep clean positive representation without sci-notation ambiguity
        formatted = f"{round(pred_val, 1):.1f}" if pred_val >= 100.0 else f"{pred_val:.3g}"
        return {
            "raw_reply": (
                f"Considering the reference scaling and plausible mechanisms, "
                f"my reasoned Final estimate: {formatted}"
            ),
            "model_slug": model_slug,
            "provider": "synthetic_sim",
            "reasoning_tokens": rng.randint(12, 64),
            "input_tokens": rng.randint(180, 320),
            "output_tokens": rng.randint(24, 58),
        }


def create_backend(name: str, **kwargs: Any) -> Any:
    """Factory returning the requested backend adapter by canonical name."""
    norm = name.strip().lower()
    if norm == "synthetic":
        return SyntheticEpistemicBackend(seed=int(kwargs.get("seed", 42)))
    if norm == "replay":
        path = kwargs.get("jsonl_path") or kwargs.get("path")
        if not path:
            raise BackendConfigError("ReplayBackend requires 'jsonl_path'.")
        return ReplayBackend.from_jsonl(Path(path))
    if norm == "ollama":
        return OllamaBackend(
            model=str(kwargs.get("model", "llama3.1:8b")),
            base_url=kwargs.get("base_url"),
            transport=kwargs.get("transport"),
        )
    if norm == "openrouter":
        return OpenRouterBackend(
            model=str(kwargs.get("model", "openai/gpt-4o-mini")),
            pin_provider=kwargs.get("pin_provider"),
            reasoning_effort=str(kwargs.get("reasoning_effort", "low")),
            api_key=kwargs.get("api_key"),
            transport=kwargs.get("transport"),
        )
    if norm == "openai":
        return OpenAICompatibleBackend(
            model=str(kwargs.get("model", "gpt-4o-mini")),
            api_key_env="OPENAI_API_KEY",
            base_url=str(kwargs.get("base_url", "https://api.openai.com/v1")),
            provider_name="openai",
            transport=kwargs.get("transport"),
        )
    if norm == "anthropic":
        return AnthropicBackend(
            model=str(kwargs.get("model", "claude-3-7-sonnet-latest")),
            api_key=kwargs.get("api_key"),
            transport=kwargs.get("transport"),
        )
    if norm == "gemini":
        return GeminiBackend(
            model=str(kwargs.get("model", "gemini-2.5-flash")),
            api_key=kwargs.get("api_key"),
            transport=kwargs.get("transport"),
        )
    raise BackendConfigError(f"Unknown backend '{name}'.")
