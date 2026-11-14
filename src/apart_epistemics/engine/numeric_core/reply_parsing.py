"""Turn a raw model reply into a usable number, or a "miss" if it cannot be trusted."""

from __future__ import annotations

import math

from apart_epistemics.parsers import NumericParseResult, parse_numeric_estimate

# A parsed value further than this (in log10 units) from the agent's current estimate is
# treated as a parse miss, e.g. a unit slip such as metres read as millions of metres.
MAX_LOG10_JUMP = 3.0


def parse_reply(raw_reply: str, cur_x: float) -> NumericParseResult:
    """Parse a reply, turning implausibly distant values into misses."""
    parsed = parse_numeric_estimate(raw_reply, require_positive=True)
    if parsed.miss or parsed.log10_value is None:
        return parsed

    # Distance between the model's answer and the agent's current belief, in powers of ten.
    # A gap of 3 means the answer is 1,000 times bigger or smaller, which is almost always a
    # unit mistake rather than a real estimate, so we throw it away instead of trusting it.
    distance = abs(parsed.log10_value - math.log10(cur_x))
    if distance > MAX_LOG10_JUMP:
        return NumericParseResult(
            value=None, log10_value=None, miss=True, matched_span=parsed.matched_span
        )
    return parsed
