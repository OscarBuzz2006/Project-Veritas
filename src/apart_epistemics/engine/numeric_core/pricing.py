"""Cost estimates for model calls and a running total checked against the budget."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from apart_epistemics.backends import BudgetExceededError


@dataclass(frozen=True)
class TokenPricing:
    """USD per 1M tokens. Set per model; the defaults are a rough placeholder, not a price list."""

    input_per_m: float = 0.5
    output_per_m: float = 1.5


def estimate_call_cost_usd(resp: dict[str, Any], pricing: TokenPricing | None = None) -> float:
    """Cost of one call (or one saved record): the provider's real charge when reported.

    Otherwise estimated from tokens. `output_tokens` (completion tokens) already includes hidden
    reasoning tokens, so `reasoning_tokens` is a breakdown and is not added again here.
    """
    real = resp.get("cost_usd")
    if real is not None:
        return float(real)
    price = pricing or TokenPricing()
    in_tok = int(resp.get("input_tokens", 0) or 0)
    out_tok = int(resp.get("output_tokens", 0) or 0)
    return max(1e-5, (in_tok * price.input_per_m + out_tok * price.output_per_m) / 1_000_000.0)


class SpendTracker:
    """Thread-safe running cost estimate checked against the budget before each call.

    Several worker threads share one tracker, so every read and write of the running total
    happens while holding a lock. Without it, two threads could both pass the budget check
    and overspend.
    """

    def __init__(self, budget_usd: float, spent_usd: float = 0.0) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self._lock = threading.Lock()

    def check(self, message: str = "Budget cap exhausted") -> None:
        """Raise if no money is left to spend."""
        with self._lock:
            if self.budget_usd <= 0.0 or self.spent_usd >= self.budget_usd:
                raise BudgetExceededError(
                    f"{message} (spent={self.spent_usd:.4f}, cap={self.budget_usd:.4f})."
                )

    def add(self, cost_usd: float) -> None:
        """Record money that has been spent."""
        with self._lock:
            self.spent_usd += cost_usd
