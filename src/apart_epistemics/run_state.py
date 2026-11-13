"""Validated run manifests and persistent, conservative request allowances."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any


def backend_identity(backend: Any) -> dict[str, Any]:
    """Describe reproducibility settings without credentials or transport objects."""
    fields = ("model", "seed", "base_url", "pin_provider", "reasoning_effort", "provider_name")
    return {
        "class": type(backend).__name__,
        **{key: getattr(backend, key) for key in fields if hasattr(backend, key)},
    }


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def validate_manifest(path: Path, manifest: dict[str, Any], records_path: Path) -> None:
    """Refuse legacy or incompatible records before opening them for mutation."""
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(
                "Run configuration differs from the saved manifest; use a new output directory."
            )
    elif records_path.exists() and records_path.stat().st_size:
        raise RuntimeError("Existing records have no run manifest; use a new output directory.")
    else:
        write_json_atomic(path, manifest)


class BudgetLedger:
    """Reserve a caller-specified upper cost allowance before every HTTP attempt.

    Reservations are never refunded: failed or interrupted requests may be billed.
    This is an authorization ledger, not a claim about provider billing accuracy.
    """

    def __init__(self, path: Path, cap: float, per_request: float = 1.0):
        if not math.isfinite(cap) or cap < 0 or not math.isfinite(per_request) or per_request <= 0:
            raise RuntimeError(
                "Budget must be finite and nonnegative; request allowance must be positive."
            )
        self.path, self.cap, self.per_request = path, cap, per_request
        self._lock = threading.Lock()  # reserve() is called from worker threads
        state = (
            json.loads(path.read_text()) if path.exists() else {"reserved_usd": 0.0, "requests": 0}
        )
        self.reserved = float(state["reserved_usd"])
        self.requests = int(state["requests"])
        if not math.isfinite(self.reserved) or self.reserved < 0 or self.requests < 0:
            raise RuntimeError("Invalid saved budget ledger.")

    def reserve(self) -> None:
        from apart_epistemics.backends.base import BudgetExceededError

        with self._lock:
            if self.reserved + self.per_request > self.cap + 1e-12:
                raise BudgetExceededError(
                    f"Request allowance exceeds budget (reserved={self.reserved:g}, "
                    f"next={self.per_request:g}, cap={self.cap:g})."
                )
            self.reserved += self.per_request
            self.requests += 1
            write_json_atomic(
                self.path,
                {
                    "reserved_usd": self.reserved,
                    "requests": self.requests,
                    "cap_usd": self.cap,
                    "per_request_allowance_usd": self.per_request,
                    "accounting": "conservative request allowances, not actual billing",
                },
            )


def configure_budget(backend: Any, out_dir: Path, cap: float, per_request: float = 1.0) -> None:
    from apart_epistemics.backends.base import _HTTPChatBackend

    if isinstance(backend, _HTTPChatBackend):
        out_dir.mkdir(parents=True, exist_ok=True)
        backend.budget_ledger = BudgetLedger(out_dir / "budget.json", cap, per_request)
        backend.request_log = out_dir / "provider_requests.jsonl"
