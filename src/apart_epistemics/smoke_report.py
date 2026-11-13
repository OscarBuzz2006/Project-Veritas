"""Check a live numeric smoke run against its pass criteria (no network, no spend)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apart_epistemics.models import NumericEstimateRecord

TRUNCATED = {"length", "max_tokens", "MAX_TOKENS"}


def _load(path: Path) -> list[NumericEstimateRecord]:
    return [
        NumericEstimateRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_smoke_report(
    run_dir: Path,
    expected_model: str | None = None,
    expected_provider: str | None = None,
    max_miss_rate: float = 0.02,
    price_in: float = 0.5,
    price_out: float = 1.5,
    sample: int = 10,
) -> dict[str, Any]:
    """Summarize `numeric_runs.jsonl`; checks marked `manual` need a person."""
    records = _load(run_dir / "numeric_runs.jsonl")
    if not records:
        raise ValueError(f"No numeric records in {run_dir}.")
    n = len(records)
    misses = sum(1 for r in records if r.miss)
    models = sorted({r.model_slug for r in records})
    providers = sorted({r.provider for r in records})
    truncated = sum(1 for r in records if r.finish_reason in TRUNCATED)
    no_reason = sum(1 for r in records if not r.finish_reason)
    reasoning = sum(1 for r in records if r.reasoning_tokens > 0)
    in_tok = sum(r.input_tokens for r in records)
    out_tok = sum(r.output_tokens for r in records)
    est_cost = (in_tok * price_in + out_tok * price_out) / 1_000_000

    ledger: dict[str, Any] = {}
    if (run_dir / "budget.json").exists():
        ledger = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))

    by_cond: dict[str, list[int]] = {}
    for r in records:
        by_cond.setdefault(r.condition_id, []).append(r.input_tokens)
    mean_input = {c: round(sum(v) / len(v), 1) for c, v in sorted(by_cond.items())}

    step = max(1, n // sample)
    picked = records[::step][:sample]

    checks = {
        "miss_rate_near_zero": {
            "pass": misses / n <= max_miss_rate,
            "detail": f"{misses}/{n} = {misses / n:.3f} (limit {max_miss_rate})",
        },
        "model_and_provider_as_expected": {
            "pass": (expected_model is None or models == [expected_model])
            and (expected_provider is None or providers == [expected_provider]),
            "detail": f"models={models} providers={providers}",
        },
        "reasoning_tokens_recorded": {
            "pass": reasoning > 0,
            "detail": f"{reasoning}/{n} rows have reasoning_tokens > 0",
        },
        "no_truncated_replies": {
            "pass": truncated == 0,
            "detail": f"{truncated}/{n} truncated; {no_reason} rows have no finish_reason",
        },
        "cost_known_per_call": {
            "pass": "manual",
            "detail": (
                f"repo estimate {est_cost:.4f} USD for {n} rows "
                f"({est_cost / n:.6f}/row); compare with the OpenRouter dashboard"
                + (f"; ledger reserved {ledger.get('reserved_usd')}" if ledger else "")
            ),
        },
        "ten_replies_read_by_a_person": {"pass": "manual", "detail": "see sample_rows"},
        "c3_prompt_size": {
            "pass": "manual",
            "detail": f"mean input tokens by condition: {mean_input}",
        },
    }
    return {
        "n_rows": n,
        "all_automatic_checks_pass": all(
            c["pass"] is True for c in checks.values() if c["pass"] != "manual"
        ),
        "checks": checks,
        "sample_rows": [
            {
                "condition": r.condition_id,
                "question": r.question_id,
                "round": r.round_idx,
                "parsed_value": r.parsed_value,
                "miss": r.miss,
                "reply": r.raw_reply[:300],
            }
            for r in picked
        ],
    }
