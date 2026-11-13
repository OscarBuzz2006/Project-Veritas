"""Robust parsers for numeric estimates (Kevin's core) and structured JSON beliefs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from apart_epistemics.metrics import clamp

_MULTIPLIERS = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

_NUMBER_PATTERN = re.compile(
    r"(?P<sign>[-+])?\s*\$?\s*~?\s*"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:\s*[eE]\s*(?P<exp>[+-]?\d+))?"
    r"(?:\s*(?P<unit>thousand|million|billion|trillion|mn|bn|[kmbt])\b)?",
    re.IGNORECASE,
)

_FINAL_PREFIX_PATTERN = re.compile(
    r"(?:final\s+estimate|final\s+answer|estimate)\s*[:=]\s*(?P<tail>[^\n]+)",
    re.IGNORECASE,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class NumericParseResult:
    """Result of parsing a numeric estimate from an assistant reply."""

    value: float | None
    log10_value: float | None
    miss: bool
    matched_span: str = ""


def _parse_single_number_segment(segment: str) -> tuple[float | None, str]:
    match = _NUMBER_PATTERN.search(segment)
    if not match:
        return None, ""
    sign = -1.0 if match.group("sign") == "-" else 1.0
    raw_num = match.group("num").replace(",", "")
    try:
        base = float(raw_num)
        if match.group("exp") is not None:
            base *= 10.0 ** int(match.group("exp"))
    except (ValueError, OverflowError):
        return None, ""
    unit = (match.group("unit") or "").lower()
    if unit in _MULTIPLIERS:
        base *= _MULTIPLIERS[unit]
    return sign * base, match.group(0).strip()


def parse_numeric_estimate(text: str, require_positive: bool = True) -> NumericParseResult:
    """Extract numeric estimate handling commas, unit suffixes, scientific notation, and misses."""
    if not text or not text.strip():
        return NumericParseResult(value=None, log10_value=None, miss=True)

    prefix_matches = list(_FINAL_PREFIX_PATTERN.finditer(text))
    if prefix_matches:
        tail = prefix_matches[-1].group("tail").strip()
        val, span = _parse_single_number_segment(tail)
    else:
        matches = list(_NUMBER_PATTERN.finditer(text))
        val, span = (None, "")
        if matches:
            val, span = _parse_single_number_segment(matches[-1].group(0))

    if val is None or not math.isfinite(val):
        return NumericParseResult(value=None, log10_value=None, miss=True)
    if require_positive and val <= 0.0:
        return NumericParseResult(value=None, log10_value=None, miss=True, matched_span=span)

    log_val = math.log10(val) if val > 0.0 else None
    return NumericParseResult(value=val, log10_value=log_val, miss=False, matched_span=span)


def parse_structured_belief(text: str) -> dict[str, Any]:
    """Parse JSON or fenced JSON belief outputs, falling back cleanly on free-form prose."""
    cleaned = text.strip()
    fence_match = _FENCE_RE.search(cleaned)
    candidate = fence_match.group(1) if fence_match else cleaned
    if not candidate.startswith("{"):
        brace_match = _BRACE_RE.search(candidate)
        candidate = brace_match.group(0) if brace_match else ""

    if candidate:
        try:
            payload, _end = json.JSONDecoder().raw_decode(candidate)
            if isinstance(payload, dict):
                stance = float(payload.get("stance_prob", 0.5))
                conf = float(payload.get("confidence", max(stance, 1.0 - stance)))
                if not math.isfinite(stance) or not math.isfinite(conf):
                    raise ValueError("Non-finite belief")
                topics = payload.get("topics", [])
                if not isinstance(topics, list):
                    topics = []
                return {
                    "parse_status": "parsed",
                    "stance_prob": clamp(stance, 0.0, 1.0),
                    "confidence": clamp(conf, 0.5, 1.0),
                    "rationale": str(payload.get("rationale", cleaned)).strip(),
                    "topics": [str(t).strip() for t in topics if str(t).strip()],
                }
        except (ValueError, TypeError):
            pass

    return {
        "parse_status": "invalid",
        "stance_prob": float("nan"),
        "confidence": 0.5,
        "rationale": cleaned,
        "topics": [],
    }
