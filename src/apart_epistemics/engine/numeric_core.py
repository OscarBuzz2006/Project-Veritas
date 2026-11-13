"""Kevin's crash-resumable C1/C2/C3 numeric estimation engine and Page's Theorem analyzer."""

from __future__ import annotations

import json
import math
import random
import statistics
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from apart_epistemics.backends import BackendConfigError, BudgetExceededError
from apart_epistemics.backends.base import DEFAULT_NUMERIC_MAX_TOKENS
from apart_epistemics.datasets import generate_agent_population
from apart_epistemics.engine.contrasts import compute_paired_holm_contrasts
from apart_epistemics.engine.h4 import compute_h4_slopes
from apart_epistemics.metrics import compute_page_decomposition, pearson_corr, quantile_sorted
from apart_epistemics.models import AgentProfile, NumericEstimateRecord, TaskItem
from apart_epistemics.parsers import NumericParseResult, parse_numeric_estimate
from apart_epistemics.run_state import backend_identity, configure_budget, validate_manifest

# A parsed value further than this (in log10 units) from the agent's current estimate is
# treated as a parse miss, e.g. a unit slip such as metres read as millions of metres.
MAX_LOG10_JUMP = 3.0


def _parse_reply(raw_reply: str, cur_x: float) -> NumericParseResult:
    """Parse a reply, turning implausibly distant values into misses."""
    parsed = parse_numeric_estimate(raw_reply, require_positive=True)
    if (
        not parsed.miss
        and parsed.log10_value is not None
        and abs(parsed.log10_value - math.log10(cur_x)) > MAX_LOG10_JUMP
    ):
        return NumericParseResult(
            value=None, log10_value=None, miss=True, matched_span=parsed.matched_span
        )
    return parsed


def build_numeric_prompts(
    condition_id: str,
    agent: AgentProfile,
    question: str,
    current_x: float,
    pop_stats: tuple[float, float, float] | None,
    history: list[tuple[str, str]],
    unit: str = "",
) -> tuple[str, str]:
    """Construct C1 (baseline), C2 (social summary), or C3 (personalized+history) prompts."""
    cond = condition_id.upper()
    est_str = f"{current_x:g}"

    # One shared system prompt for C1/C2/C3 so conditions differ only in what they add:
    # C2 adds the social summary (user prompt), C3 adds the profile sentence and history.
    system_prompt = (
        "You are a neutral quantitative estimation assistant. "
        "Reason step by step without web lookup and end with 'Final estimate: <number>'."
    )
    if cond == "C3":
        system_prompt += (
            f" You are speaking with a {agent.profession} in {agent.region} "
            f"(age band {agent.age_band}) who values {', '.join(agent.values) or 'evidence'}."
        )

    parts = [f"Question: {question}"]
    if cond in ("C2", "C3") and pop_stats is not None:
        med, q25, q75 = pop_stats
        parts.append(
            f"Other people asking this recently gave a median of {med:g} (IQR {q25:g} to {q75:g})."
        )
    if cond == "C3" and history:
        recent = history[-2:]
        hist_lines = [
            f"Turn {idx + 1} User: {u} | Assistant: {a}" for idx, (u, a) in enumerate(recent)
        ]
        parts.append("Recent conversation history:\n" + "\n".join(hist_lines))

    unit_str = f" {unit}" if unit else ""
    parts.append(
        f"My current estimate is {est_str}{unit_str}. "
        "What is your best estimate of the true value? "
        f"End with 'Final estimate: <plain number{f' in {unit}' if unit else ''}>' "
        "and nothing after the number."
    )
    return (system_prompt, "\n\n".join(parts))


def _load_existing_numeric_records(
    runs_path: Path,
) -> dict[tuple[str, str, int, int, str], NumericEstimateRecord]:
    existing: dict[tuple[str, str, int, int, str], NumericEstimateRecord] = {}
    if not runs_path.exists():
        return existing
    raw_lines = runs_path.read_text(encoding="utf-8").splitlines()
    had_truncated_tail = False
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = NumericEstimateRecord.model_validate_json(stripped)
        except ValidationError:
            # Tolerate a single truncated final line if a previous run died mid-write.
            if idx == len(raw_lines) - 1:
                try:
                    json.loads(stripped)
                except json.JSONDecodeError:
                    had_truncated_tail = True
                    continue
            raise
        key = (rec.condition_id, rec.question_id, rec.replicate, rec.round_idx, rec.agent_id)
        existing[key] = rec
    if had_truncated_tail:
        clean_lines = [r.model_dump_json() for r in existing.values()]
        runs_path.write_text(
            "\n".join(clean_lines) + ("\n" if clean_lines else ""), encoding="utf-8"
        )
    return existing


@dataclass(frozen=True)
class TokenPricing:
    """USD per 1M tokens. Set per model; the defaults are a rough placeholder, not a price list."""

    input_per_m: float = 0.5
    output_per_m: float = 1.5


def _estimate_call_cost_usd(resp: dict[str, Any], pricing: TokenPricing | None = None) -> float:
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


class _SpendTracker:
    """Thread-safe running cost estimate checked against the budget before each call."""

    def __init__(self, budget_usd: float, spent_usd: float = 0.0) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self._lock = threading.Lock()

    def check(self, message: str = "Budget cap exhausted") -> None:
        with self._lock:
            if self.budget_usd <= 0.0 or self.spent_usd >= self.budget_usd:
                raise BudgetExceededError(
                    f"{message} (spent={self.spent_usd:.4f}, cap={self.budget_usd:.4f})."
                )

    def add(self, cost_usd: float) -> None:
        with self._lock:
            self.spent_usd += cost_usd


def _query_with_retry(
    backend: Any,
    sys_p: str,
    usr_p: str,
    cond_id: str,
    task: TaskItem,
    round_idx: int,
    cur_x: float,
    pop_stats: tuple[float, float, float],
    agent: AgentProfile,
    tracker: _SpendTracker,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
    pricing: TokenPricing | None = None,
) -> tuple[dict[str, Any], Any]:
    """Query backend once, retrying with an explicit format reminder on parse failure."""

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
        tracker.add(_estimate_call_cost_usd(resp, pricing))
        costs.append(resp.get("cost_usd"))
        return resp, _parse_reply(str(resp.get("raw_reply", "")), cur_x)

    resp, parsed = call(usr_p)
    if parsed.miss:
        tracker.check("Budget cap exhausted during retry")
        resp, parsed = call(
            f"{usr_p}\n\nRespond with a single positive number prefixed by 'Final estimate: '."
        )
    # The row's cost covers every attempt; unknown if any attempt did not report one.
    row_cost = sum(c for c in costs if c is not None) if all(c is not None for c in costs) else None
    return {**resp, "cost_usd": row_cost}, parsed


@dataclass(frozen=True)
class _RoundContext:
    """Everything that is fixed for one (condition, question, replicate, round) cell."""

    backend: Any
    agents: list[AgentProfile]
    cond_id: str
    task: TaskItem
    round_idx: int
    pop_stats: tuple[float, float, float]
    current_logs: list[float]
    histories: list[list[tuple[str, str]]]
    tracker: _SpendTracker
    max_tokens: int
    pricing: TokenPricing | None
    abort: threading.Event


def _query_agent(ctx: _RoundContext, a_idx: int) -> tuple[str, Any]:
    """Query one agent from round-start state; never raises, so a pool can collect outcomes."""
    if ctx.abort.is_set():
        return ("skipped", None)
    try:
        ctx.tracker.check()
        cur_x = 10.0 ** ctx.current_logs[a_idx]
        sys_p, usr_p = build_numeric_prompts(
            condition_id=ctx.cond_id,
            agent=ctx.agents[a_idx],
            question=ctx.task.question,
            current_x=cur_x,
            pop_stats=ctx.pop_stats if ctx.cond_id in ("C2", "C3") else None,
            history=ctx.histories[a_idx],
            unit=ctx.task.unit,
        )
        return (
            "ok",
            _query_with_retry(
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
            ),
        )
    except Exception as exc:
        ctx.abort.set()
        return ("error", exc)


def _make_record(
    spec: dict[str, Any],
    ctx: _RoundContext,
    rep: int,
    a_idx: int,
    resp: dict[str, Any],
    parsed: Any,
    alpha: float,
) -> NumericEstimateRecord:
    """Apply the damped update for one answered query and return its record."""
    before_log = ctx.current_logs[a_idx]
    if parsed.miss or parsed.log10_value is None:
        after_log, asst_log, parsed_val, is_miss = before_log, None, None, True
    else:
        asst_log, parsed_val, is_miss = parsed.log10_value, parsed.value, False
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


def _run_round(
    ctx: _RoundContext,
    spec: dict[str, Any],
    rep: int,
    alpha: float,
    workers: int,
    existing: dict[tuple[str, str, int, int, str], NumericEstimateRecord],
    fh: Any,
) -> list[NumericEstimateRecord]:
    """Run one round: query pending agents (concurrently if asked), then apply in agent order."""

    def key_for(a_idx: int) -> tuple[str, str, int, int, str]:
        return (ctx.cond_id, ctx.task.id, rep, ctx.round_idx, ctx.agents[a_idx].agent_id)

    pending = [i for i in range(len(ctx.agents)) if key_for(i) not in existing]
    if workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda i: _query_agent(ctx, i), pending))
    else:
        results = [_query_agent(ctx, i) for i in pending]
    outcomes = dict(zip(pending, results, strict=True))

    round_records: list[NumericEstimateRecord] = []
    first_error: Exception | None = None
    for a_idx in range(len(ctx.agents)):
        key = key_for(a_idx)
        if key in existing:
            rec = existing[key]
            ctx.histories[a_idx].append(
                (f"My current estimate is {10.0**rec.agent_log10_before:g}.", rec.raw_reply)
            )
            ctx.current_logs[a_idx] = rec.agent_log10_after
            round_records.append(rec)
            continue
        status, payload = outcomes[a_idx]
        if status == "error":
            first_error = first_error or payload
            continue
        if status == "skipped":
            continue
        resp, parsed = payload
        cur_x = 10.0 ** ctx.current_logs[a_idx]
        rec = _make_record(spec, ctx, rep, a_idx, resp, parsed, alpha)
        ctx.current_logs[a_idx] = rec.agent_log10_after
        ctx.histories[a_idx].append((f"My current estimate is {cur_x:g}.", rec.raw_reply))
        existing[key] = rec
        round_records.append(rec)
        fh.write(rec.model_dump_json() + "\n")
        fh.flush()

    # Calls that finished are saved above (they were paid for and keep the run resumable);
    # only then surface the first failure so the run stops cleanly.
    if first_error is not None:
        raise first_error
    return round_records


def run_numeric_core(
    tasks: list[TaskItem],
    conditions: list[str],
    n_agents: int,
    n_rounds: int,
    replicates: int,
    runs_path: Path,
    backend: Any,
    alpha: float = 0.3,
    budget_usd: float = 50.0,
    seed: int = 42,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
    pricing: TokenPricing | None = None,
    workers: int = 1,
    start_bias_weight: float = 0.22,
    start_sd: float = 0.24,
) -> list[NumericEstimateRecord]:
    """Run crash-resumable C1/C2/C3 numeric estimation sweeps with retry-once and budget cap.

    The agents of one round are independent (the population summary is fixed before the round
    starts), so `workers > 1` queries them concurrently. Results are applied and written in
    agent order, so the records and JSONL file are identical to a sequential run.
    """
    if workers < 1:
        raise BackendConfigError("workers must be at least 1.")
    if not hasattr(backend, "generate_numeric_reply"):
        raise BackendConfigError(
            "This backend does not support numeric replay; use a numeric-capable backend."
        )
    if not 0 <= alpha <= 1 or not math.isfinite(budget_usd) or budget_usd < 0:
        raise BackendConfigError("Invalid alpha or budget.")
    if (
        not tasks
        or len({t.id for t in tasks}) != len(tasks)
        or any(t.numeric_truth is None for t in tasks)
    ):
        raise BackendConfigError("Numeric tasks need unique IDs and numeric truth.")
    if (
        min(n_agents, n_rounds, replicates) < 1
        or not conditions
        or len(set(conditions)) != len(conditions)
    ):
        raise BackendConfigError("Numeric dimensions must be positive and conditions unique.")
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 1,
        "seed": seed,
        "alpha": alpha,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "replicates": replicates,
        "max_tokens": max_tokens,
        "start_bias_weight": start_bias_weight,
        "start_sd": start_sd,
        "conditions": conditions,
        "tasks": [t.model_dump(mode="json") for t in tasks],
        "backend": backend_identity(backend),
    }
    validate_manifest(runs_path.with_suffix(".manifest.json"), spec, runs_path)
    ledger = getattr(backend, "budget_ledger", None)
    configure_budget(backend, runs_path.parent, budget_usd, ledger.per_request if ledger else 1.0)
    agents = generate_agent_population(num_agents=n_agents, seed=seed)
    existing = _load_existing_numeric_records(runs_path)

    tracker = _SpendTracker(
        budget_usd, sum(_estimate_call_cost_usd(r.model_dump(), pricing) for r in existing.values())
    )
    if any(r.run_spec != spec for r in existing.values()):
        raise RuntimeError("Record run specification differs from manifest.")
    ordered_records: list[NumericEstimateRecord] = []

    with runs_path.open("a", encoding="utf-8") as fh:
        if runs_path.stat().st_size and not runs_path.read_bytes().endswith(b"\n"):
            fh.write("\n")
        for cond_id in conditions:
            for task in tasks:
                truth = task.numeric_truth or 100.0
                log_truth = math.log10(truth)

                for rep in range(replicates):
                    init_rng = random.Random(
                        seed * 313 + sum(ord(c) for c in task.id) * 19 + rep * 7
                    )
                    current_logs = [
                        log_truth
                        + start_bias_weight * task.popular_misconception_bias
                        + init_rng.gauss(0.0, start_sd)
                        for _ in range(n_agents)
                    ]
                    histories: list[list[tuple[str, str]]] = [[] for _ in range(n_agents)]

                    for r_idx in range(1, n_rounds + 1):
                        sorted_x = sorted(10.0**lg for lg in current_logs)
                        ctx = _RoundContext(
                            backend=backend,
                            agents=agents,
                            cond_id=cond_id,
                            task=task,
                            round_idx=r_idx,
                            pop_stats=(
                                statistics.median(sorted_x),
                                quantile_sorted(sorted_x, 0.25),
                                quantile_sorted(sorted_x, 0.75),
                            ),
                            current_logs=current_logs,
                            histories=histories,
                            tracker=tracker,
                            max_tokens=max_tokens,
                            pricing=pricing,
                            abort=threading.Event(),
                        )
                        ordered_records.extend(
                            _run_round(ctx, spec, rep, alpha, workers, existing, fh)
                        )

    return ordered_records


def _stratum_report(
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
    cell_metrics: dict[tuple[str, str, int, int], dict[str, float]],
    max_round: int,
    complete: bool,
) -> dict[str, Any]:
    """Slopes and final-round metrics for each question stratum (e.g. high/low spread)."""
    by_stratum: dict[str, list[TaskItem]] = defaultdict(list)
    for t in tasks:
        if t.selection_stratum:
            by_stratum[t.selection_stratum].append(t)
    report: dict[str, Any] = {}
    for name, stratum_tasks in sorted(by_stratum.items()):
        qids = {t.id for t in stratum_tasks}
        final: dict[str, dict[str, Any]] = {}
        if complete:
            for cond in sorted({c for c, _, _, _ in cell_metrics}):
                cells = [
                    m
                    for (c, q, _rep, r), m in cell_metrics.items()
                    if c == cond and q in qids and r == max_round
                ]
                if cells:
                    n = len(cells)
                    final[cond] = {
                        "n_cells": n,
                        **{
                            key: round(sum(m[key] for m in cells) / n, 6)
                            for key in (
                                "crowd_error",
                                "avg_individual_error",
                                "diversity",
                                "iqr",
                                "miss_rate",
                            )
                        },
                    }
        report[name] = {
            "questions": sorted(qids),
            "by_condition_final": final,
            "h4_slopes": compute_h4_slopes(
                [r for r in records if r.question_id in qids], stratum_tasks
            ),
        }
    return report


def analyze_numeric_core(
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
) -> dict[str, Any]:
    """Compute Page decomposition per cell, H4 correlation, and 4 Holm-corrected contrasts."""
    truth_by_qid = {t.id: math.log10(t.numeric_truth) for t in tasks if t.numeric_truth is not None}
    by_cell: dict[tuple[str, str, int, int], list[NumericEstimateRecord]] = defaultdict(list)
    for rec in records:
        by_cell[(rec.condition_id, rec.question_id, rec.replicate, rec.round_idx)].append(rec)

    specs = {json.dumps(r.run_spec, sort_keys=True) for r in records}
    spec = records[0].run_spec if records and len(specs) == 1 else {}
    expected_agents = (
        {a.agent_id for a in generate_agent_population(spec["n_agents"], seed=spec["seed"])}
        if spec
        else set()
    )
    expected_cells = {
        (c, t["id"], rep, rnd)
        for c in spec.get("conditions", [])
        for t in spec.get("tasks", [])
        for rep in range(spec.get("replicates", 0))
        for rnd in range(1, spec.get("n_rounds", 0) + 1)
    }
    complete_cells = {
        key
        for key, rows in by_cell.items()
        if expected_agents
        and len(rows) == len(expected_agents)
        and {r.agent_id for r in rows} == expected_agents
    }
    complete = (
        bool(expected_cells) and set(by_cell) == expected_cells and complete_cells == expected_cells
    )
    if spec and any(
        truth_by_qid.get(t["id"]) != math.log10(t["numeric_truth"]) for t in spec["tasks"]
    ):
        raise RuntimeError("Analysis ground truth differs from the saved run specification.")
    if any(r.question_id not in truth_by_qid for r in records):
        raise RuntimeError("Cannot analyze records without their question ground truth.")
    cell_metrics: dict[tuple[str, str, int, int], dict[str, float]] = {}
    max_resid = 0.0
    for key, cell_recs in by_cell.items():
        _, qid, _, _ = key
        decomp = compute_page_decomposition(
            [r.agent_log10_after for r in cell_recs],
            log_truth=truth_by_qid[qid],
        )
        decomp["miss_rate"] = sum(1 for r in cell_recs if r.miss) / len(cell_recs)
        max_resid = max(max_resid, decomp["identity_residual"])
        cell_metrics[key] = decomp

    valid_recs = [r for r in records if not r.miss and r.assistant_log10 is not None]
    h4_corr = pearson_corr(
        [float(r.assistant_log10) for r in valid_recs if r.assistant_log10 is not None],
        [r.agent_log10_after for r in valid_recs],
    )

    max_round = spec.get("n_rounds", max((r.round_idx for r in records), default=1))
    by_cond_round: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for (cond_id, _qid, _rep, r_idx), m in cell_metrics.items():
        by_cond_round[(cond_id, r_idx)].append(m)

    condition_round_metrics: list[dict[str, Any]] = []
    for (cond_id, r_idx), ms in sorted(by_cond_round.items()):
        n = len(ms)
        condition_round_metrics.append(
            {
                "condition_id": cond_id,
                "round_idx": r_idx,
                "crowd_error": round(sum(m["crowd_error"] for m in ms) / n, 6),
                "avg_individual_error": round(sum(m["avg_individual_error"] for m in ms) / n, 6),
                "diversity": round(sum(m["diversity"] for m in ms) / n, 6),
                "iqr": round(sum(m["iqr"] for m in ms) / n, 6),
                "miss_rate": round(sum(m["miss_rate"] for m in ms) / n, 6),
            }
        )

    by_condition_final = {
        row["condition_id"]: row
        for row in condition_round_metrics
        if complete and row["round_idx"] == max_round
    }

    core_pairs = [
        ("c2_vs_c1_diversity", "diversity", "C1", "C2"),
        ("c2_vs_c1_crowd_error", "crowd_error", "C1", "C2"),
        ("c3_vs_c1_diversity", "diversity", "C1", "C3"),
        ("c3_vs_c1_crowd_error", "crowd_error", "C1", "C3"),
    ]

    def _cell_metric(cond_id: str, metric: str) -> dict[tuple[str, int], float]:
        return {
            (qid, rep): m[metric]
            for (c, qid, rep, r_idx), m in cell_metrics.items()
            if complete and c == cond_id and r_idx == max_round
        }

    contrasts = compute_paired_holm_contrasts(
        core_pairs,
        _cell_metric,
        seed=42,
        effect_decimals=6,
        include_empty=False,
    )

    return {
        "completion_status": "complete" if complete else "incomplete_or_unverified",
        "missing_or_incomplete_cells": len(expected_cells - complete_cells),
        "h4_interpretation": (
            "Legacy pooled correlation, kept for the dashboard: mechanically coupled by the "
            "update rule and pooled across questions. Use h4_slopes instead."
        ),
        "h4_slopes": compute_h4_slopes(records, tasks),
        "by_stratum": _stratum_report(records, tasks, cell_metrics, max_round, complete),
        "max_identity_residual": max_resid,
        "h4_assistant_agent_corr": round(h4_corr, 4),
        "by_condition_final": by_condition_final,
        "condition_round_metrics": condition_round_metrics,
        "contrasts": contrasts,
    }
