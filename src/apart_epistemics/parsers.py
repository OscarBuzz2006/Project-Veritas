"""Robust parsers for numeric estimates (Kevin's core) and structured JSON beliefs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from apart_epistemics.metrics import clamp

# Only spelled-out multipliers are honoured. Single letters are never multipliers because
# "m" is metres (10,935 m), not million.
_MULTIPLIERS = {
    "thousand": 1e3,
    "mn": 1e6,
    "million": 1e6,
    "bn": 1e9,
    "billion": 1e9,
    "trillion": 1e12,
}

_NUMBER_PATTERN = re.compile(
    r"(?P<sign>[-+])?\s*\$?\s*~?\s*"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:\s*[eE]\s*(?P<exp>[+-]?\d+)"
    r"|\s*[\u00d7xX*]\s*10\s*\^\s*\{?\s*(?P<pow>[+-]?\d+)\s*\}?)?"
    r"(?:\s*(?P<unit>thousand|million|billion|trillion|mn|bn)\b)?",
    re.IGNORECASE,
)

# A range such as "2,840-2,900" or "10 to 20" right after the first number is ambiguous.
_RANGE_AFTER_NUMBER = re.compile(r"\s*(?:[-\u2013\u2014~]|to\b)\s*\$?\s*~?\d", re.IGNORECASE)
# "12.5k" / "3b" / "2t": an attached single-letter multiplier we refuse to guess at.
_AMBIGUOUS_SUFFIX = re.compile(r"[kbt]\b", re.IGNORECASE)

_FINAL_PREFIX_PATTERN = re.compile(
    r"(?:final\s+estimate|final\s+answer)\s*[:=]\s*(?P<tail>[^\n]+)",
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
    after = segment[match.end() :]
    if _RANGE_AFTER_NUMBER.match(after) or (
        match.group("unit") is None and _AMBIGUOUS_SUFFIX.match(after)
    ):
        return None, match.group(0).strip()
    try:
        base = float(raw_num)
        exponent = match.group("exp") if match.group("exp") is not None else match.group("pow")
        if exponent is not None:
            base *= 10.0 ** int(exponent)
    except (ValueError, OverflowError):
        return None, ""
    unit = (match.group("unit") or "").lower()
    if unit in _MULTIPLIERS:
        base *= _MULTIPLIERS[unit]
    return sign * base, match.group(0).strip()


def parse_numeric_estimate(text: str, require_positive: bool = True) -> NumericParseResult:
    """Extract the number after a 'Final estimate:' prefix, or report a miss.

    Handles commas, spelled-out multipliers, e-notation and 'a x 10^n'. Replies with no
    prefix, a range, or an ambiguous single-letter suffix are misses, never guesses.
    """
    if not text or not text.strip():
        return NumericParseResult(value=None, log10_value=None, miss=True)

    prefix_matches = list(_FINAL_PREFIX_PATTERN.finditer(text))
    if not prefix_matches:
        return NumericParseResult(value=None, log10_value=None, miss=True)
    val, span = _parse_single_number_segment(prefix_matches[-1].group("tail").strip())

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
