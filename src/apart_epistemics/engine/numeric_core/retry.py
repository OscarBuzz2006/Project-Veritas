"""Ask the backend once, and once more with a format reminder if the reply is unusable."""

from __future__ import annotations

from typing import Any

from apart_epistemics.backends.base import DEFAULT_NUMERIC_MAX_TOKENS
from apart_epistemics.engine.numeric_core.pricing import (
    SpendTracker,
    TokenPricing,
    estimate_call_cost_usd,
)
from apart_epistemics.engine.numeric_core.reply_parsing import parse_reply
from apart_epistemics.models import AgentProfile, TaskItem

FORMAT_REMINDER = "Respond with a single positive number prefixed by 'Final estimate: '."


def query_with_retry(
    backend: Any,
    sys_p: str,
    usr_p: str,
    cond_id: str,
    task: TaskItem,
    round_idx: int,
    cur_x: float,
    pop_stats: tuple[float, float, float],
    agent: AgentProfile,
    tracker: SpendTracker,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
    pricing: TokenPricing | None = None,
) -> tuple[dict[str, Any], Any]:
    """Query backend once, retrying with an explicit format reminder on parse failure.

    Returns the backend response (with `cost_usd` covering every attempt) and the parse result.
    """
    # The real cost reported for each attempt, in order. None means the provider did not say.
    costs: list[float | None] = []

    def call(user_prompt: str) -> tuple[dict[str, Any], Any]:
        resp = backend.generate_numeric_reply(
            system_prompt=sys_p,
            user_prompt=user_prompt,
            condition_id=cond_id,
            task=task,
            round_idx=round_idx,
            current_x=cur_x,
            pop_stats=pop_stats,
            agent=agent,
            max_tokens=max_tokens,
        )
        tracker.add(estimate_call_cost_usd(resp, pricing))
        costs.append(resp.get("cost_usd"))
        return resp, parse_reply(str(resp.get("raw_reply", "")), cur_x)

    resp, parsed = call(usr_p)
    if parsed.miss:
        # The first answer was unusable. Make sure there is still budget, then ask once more.
        tracker.check("Budget cap exhausted during retry")
        resp, parsed = call(f"{usr_p}\n\n{FORMAT_REMINDER}")

    return {**resp, "cost_usd": _total_cost(costs)}, parsed


def _total_cost(costs: list[float | None]) -> float | None:
    """The row's cost covers every attempt; it is unknown if any attempt did not report one."""
    if any(c is None for c in costs):
        return None
    return sum(c for c in costs if c is not None)
