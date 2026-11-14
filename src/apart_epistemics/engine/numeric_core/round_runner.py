"""Run one round of one cell: ask every agent, then move each agent's belief."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TextIO

from apart_epistemics.engine.numeric_core.pricing import SpendTracker, TokenPricing
from apart_epistemics.engine.numeric_core.prompts import build_numeric_prompts
from apart_epistemics.engine.numeric_core.record_store import RecordKey
from apart_epistemics.engine.numeric_core.retry import query_with_retry
from apart_epistemics.models import AgentProfile, NumericEstimateRecord, TaskItem

# Conditions whose prompt includes the crowd's median and spread.
CONDITIONS_WITH_CROWD_SUMMARY = ("C2", "C3")


@dataclass(frozen=True)
class RoundContext:
    """Everything that is fixed for one (condition, question, replicate, round) cell."""

    backend: Any
    agents: list[AgentProfile]
    cond_id: str
    task: TaskItem
    round_idx: int
    pop_stats: tuple[float, float, float]
    current_logs: list[float]
    histories: list[list[tuple[str, str]]]
    tracker: SpendTracker
    max_tokens: int
    pricing: TokenPricing | None
    abort: threading.Event
    current_date: str = ""


def query_agent(ctx: RoundContext, a_idx: int) -> tuple[str, Any]:
    """Query one agent from round-start state; never raises, so a pool can collect outcomes.

    Returns ("ok", (response, parsed)), ("error", the exception) or ("skipped", None).
    """
    if ctx.abort.is_set():
        # Another agent already failed, so stop spending money on the rest of this round.
        return ("skipped", None)
    try:
        ctx.tracker.check()
        cur_x = 10.0 ** ctx.current_logs[a_idx]
        sys_p, usr_p = build_numeric_prompts(
            condition_id=ctx.cond_id,
            agent=ctx.agents[a_idx],
            question=ctx.task.question,
            current_x=cur_x,
            pop_stats=ctx.pop_stats if ctx.cond_id in CONDITIONS_WITH_CROWD_SUMMARY else None,
            history=ctx.histories[a_idx],
            unit=ctx.task.unit,
            current_date=ctx.current_date,
        )
        result = query_with_retry(
            ctx.backend,
            sys_p,
            usr_p,
            ctx.cond_id,
            ctx.task,
            ctx.round_idx,
            cur_x,
            ctx.pop_stats,
            ctx.agents[a_idx],
            ctx.tracker,
            ctx.max_tokens,
            ctx.pricing,
        )
        return ("ok", result)
    except Exception as exc:
        ctx.abort.set()
        return ("error", exc)


def make_record(
    spec: dict[str, Any],
    ctx: RoundContext,
    rep: int,
    a_idx: int,
    resp: dict[str, Any],
    parsed: Any,
    alpha: float,
) -> NumericEstimateRecord:
    """Apply the damped update for one answered query and return its record."""
    before_log = ctx.current_logs[a_idx]
    if parsed.miss or parsed.log10_value is None:
        # No usable answer: the agent keeps their belief unchanged.
        after_log, asst_log, parsed_val, is_miss = before_log, None, None, True
    else:
        asst_log, parsed_val, is_miss = parsed.log10_value, parsed.value, False
        # The update rule: move a fraction `alpha` of the way from the agent's old belief to
        # the model's answer, working in log10 so that "twice as big" counts the same everywhere.
        after_log = (1.0 - alpha) * before_log + alpha * asst_log
    return NumericEstimateRecord(
        run_spec=spec,
        condition_id=ctx.cond_id,
        question_id=ctx.task.id,
        replicate=rep,
        round_idx=ctx.round_idx,
        agent_id=ctx.agents[a_idx].agent_id,
        model_slug=str(resp.get("model_slug", "synthetic")),
        provider=str(resp.get("provider", "synthetic_sim")),
        raw_reply=str(resp.get("raw_reply", "")),
        parsed_value=parsed_val,
        assistant_log10=asst_log,
        agent_log10_before=before_log,
        agent_log10_after=after_log,
        pop_median_log10=math.log10(ctx.pop_stats[0]),
        miss=is_miss,
        reasoning_tokens=int(resp.get("reasoning_tokens", 0) or 0),
        input_tokens=int(resp.get("input_tokens", 0) or 0),
        output_tokens=int(resp.get("output_tokens", 0) or 0),
        finish_reason=str(resp.get("finish_reason", "") or ""),
        cost_usd=resp.get("cost_usd"),
    )


def run_round(
    ctx: RoundContext,
    spec: dict[str, Any],
    rep: int,
    alpha: float,
    workers: int,
    existing: dict[RecordKey, NumericEstimateRecord],
    fh: TextIO,
) -> list[NumericEstimateRecord]:
    """Run one round: query pending agents (concurrently if asked), then apply in agent order."""

    def key_for(a_idx: int) -> RecordKey:
        return (ctx.cond_id, ctx.task.id, rep, ctx.round_idx, ctx.agents[a_idx].agent_id)

    # Agents already saved by an earlier (interrupted) run are not asked again.
    pending = [i for i in range(len(ctx.agents)) if key_for(i) not in existing]
    outcomes = dict(zip(pending, _ask_agents(ctx, pending, workers), strict=True))

    round_records: list[NumericEstimateRecord] = []
    first_error: Exception | None = None
    # Apply results in agent order, whatever order the threads finished in, so the output
    # file is identical to a sequential run.
    for a_idx in range(len(ctx.agents)):
        key = key_for(a_idx)
        if key in existing:
            round_records.append(_replay_saved_record(ctx, a_idx, existing[key]))
            continue
        status, payload = outcomes[a_idx]
        if status == "error":
            first_error = first_error or payload
        elif status == "ok":
            resp, parsed = payload
            round_records.append(_apply_answer(ctx, spec, rep, a_idx, resp, parsed, alpha, fh))
            existing[key] = round_records[-1]

    # Calls that finished are saved above (they were paid for and keep the run resumable);
    # only then surface the first failure so the run stops cleanly.
    if first_error is not None:
        raise first_error
    return round_records


def _ask_agents(ctx: RoundContext, pending: list[int], workers: int) -> list[tuple[str, Any]]:
    """Ask the pending agents, in parallel when more than one worker is allowed."""

    def ask(a_idx: int) -> tuple[str, Any]:
        return query_agent(ctx, a_idx)

    if workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(ask, pending))
    return [ask(i) for i in pending]


def _replay_saved_record(
    ctx: RoundContext, a_idx: int, rec: NumericEstimateRecord
) -> NumericEstimateRecord:
    """Rebuild the in-memory state for an agent whose answer was saved on a previous run."""
    ctx.histories[a_idx].append(
        (f"My current estimate is {10.0**rec.agent_log10_before:g}.", rec.raw_reply)
    )
    ctx.current_logs[a_idx] = rec.agent_log10_after
    return rec


def _apply_answer(
    ctx: RoundContext,
    spec: dict[str, Any],
    rep: int,
    a_idx: int,
    resp: dict[str, Any],
    parsed: Any,
    alpha: float,
    fh: TextIO,
) -> NumericEstimateRecord:
    """Move the agent's belief, remember the exchange, and save the row to disk straight away."""
    cur_x = 10.0 ** ctx.current_logs[a_idx]
    rec = make_record(spec, ctx, rep, a_idx, resp, parsed, alpha)
    ctx.current_logs[a_idx] = rec.agent_log10_after
    ctx.histories[a_idx].append((f"My current estimate is {cur_x:g}.", rec.raw_reply))
    fh.write(rec.model_dump_json() + "\n")
    fh.flush()
    return rec
