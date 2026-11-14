"""Run the whole C1/C2/C3 sweep: check inputs, start the agents, loop over every cell."""

from __future__ import annotations

import math
import random
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apart_epistemics.backends import BackendConfigError
from apart_epistemics.backends.base import DEFAULT_NUMERIC_MAX_TOKENS
from apart_epistemics.datasets import generate_agent_population
from apart_epistemics.engine.numeric_core.pricing import (
    SpendTracker,
    TokenPricing,
    estimate_call_cost_usd,
)
from apart_epistemics.engine.numeric_core.record_store import RecordKey, load_existing_records
from apart_epistemics.engine.numeric_core.round_runner import RoundContext, run_round
from apart_epistemics.metrics import quantile_sorted
from apart_epistemics.models import AgentProfile, NumericEstimateRecord, TaskItem
from apart_epistemics.run_state import backend_identity, configure_budget, validate_manifest


@dataclass(frozen=True)
class SweepSettings:
    """The knobs that stay the same for every cell of one sweep."""

    seed: int
    alpha: float
    n_agents: int
    n_rounds: int
    replicates: int
    max_tokens: int
    pricing: TokenPricing | None
    workers: int
    start_bias_weight: float
    start_sd: float
    current_date: str


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
    current_date: str = "",
) -> list[NumericEstimateRecord]:
    """Run crash-resumable C1/C2/C3 numeric estimation sweeps with retry-once and budget cap.

    The agents of one round are independent (the population summary is fixed before the round
    starts), so `workers > 1` queries them concurrently. Results are applied and written in
    agent order, so the records and JSONL file are identical to a sequential run.
    """
    _validate_inputs(
        tasks, conditions, n_agents, n_rounds, replicates, backend, alpha, budget_usd, workers
    )
    settings = SweepSettings(
        seed=seed,
        alpha=alpha,
        n_agents=n_agents,
        n_rounds=n_rounds,
        replicates=replicates,
        max_tokens=max_tokens,
        pricing=pricing,
        workers=workers,
        start_bias_weight=start_bias_weight,
        start_sd=start_sd,
        current_date=current_date,
    )
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    spec = _build_run_spec(settings, tasks, conditions, backend)

    # Refuse to continue a half-finished run that was started with different settings.
    validate_manifest(runs_path.with_suffix(".manifest.json"), spec, runs_path)
    ledger = getattr(backend, "budget_ledger", None)
    configure_budget(backend, runs_path.parent, budget_usd, ledger.per_request if ledger else 1.0)

    agents = generate_agent_population(num_agents=n_agents, seed=seed)
    existing = load_existing_records(runs_path)
    # Money already spent by the saved rows counts against this run's budget.
    tracker = SpendTracker(
        budget_usd, sum(estimate_call_cost_usd(r.model_dump(), pricing) for r in existing.values())
    )
    if any(r.run_spec != spec for r in existing.values()):
        raise RuntimeError("Record run specification differs from manifest.")

    return _run_all_cells(
        settings, spec, tasks, conditions, agents, backend, runs_path, existing, tracker
    )


def _validate_inputs(
    tasks: list[TaskItem],
    conditions: list[str],
    n_agents: int,
    n_rounds: int,
    replicates: int,
    backend: Any,
    alpha: float,
    budget_usd: float,
    workers: int,
) -> None:
    """Reject bad settings before any money is spent."""
    if workers < 1:
        raise BackendConfigError("workers must be at least 1.")
    if not hasattr(backend, "generate_numeric_reply"):
        raise BackendConfigError(
            "This backend does not support numeric replay; use a numeric-capable backend."
        )
    if not 0 <= alpha <= 1 or not math.isfinite(budget_usd) or budget_usd < 0:
        raise BackendConfigError("Invalid alpha or budget.")
    # Every question needs a unique id and a known true answer to score against.
    has_unique_ids = len({t.id for t in tasks}) == len(tasks)
    if not tasks or not has_unique_ids or any(t.numeric_truth is None for t in tasks):
        raise BackendConfigError("Numeric tasks need unique IDs and numeric truth.")
    if (
        min(n_agents, n_rounds, replicates) < 1
        or not conditions
        or len(set(conditions)) != len(conditions)
    ):
        raise BackendConfigError("Numeric dimensions must be positive and conditions unique.")


def _build_run_spec(
    settings: SweepSettings, tasks: list[TaskItem], conditions: list[str], backend: Any
) -> dict[str, Any]:
    """Every setting that affects results, saved with the run so later runs can be compared."""
    return {
        "version": 1,
        "seed": settings.seed,
        "alpha": settings.alpha,
        "n_agents": settings.n_agents,
        "n_rounds": settings.n_rounds,
        "replicates": settings.replicates,
        "max_tokens": settings.max_tokens,
        "start_bias_weight": settings.start_bias_weight,
        "start_sd": settings.start_sd,
        "current_date": settings.current_date,
        "conditions": conditions,
        "tasks": [t.model_dump(mode="json") for t in tasks],
        "backend": backend_identity(backend),
    }


def initial_beliefs(task: TaskItem, rep: int, settings: SweepSettings) -> list[float]:
    """Each agent's starting belief (log10), scattered around the true value.

    The scatter has two parts: a shared push toward the popular wrong answer (scaled by
    `start_bias_weight`) and random personal noise (`start_sd`). The random generator is
    seeded from the question and replicate only, so every condition starts from the same
    beliefs and the conditions can be compared fairly.
    """
    truth = task.numeric_truth or 100.0
    log_truth = math.log10(truth)
    init_rng = random.Random(settings.seed * 313 + sum(ord(c) for c in task.id) * 19 + rep * 7)
    return [
        log_truth
        + settings.start_bias_weight * task.popular_misconception_bias
        + init_rng.gauss(0.0, settings.start_sd)
        for _ in range(settings.n_agents)
    ]


def crowd_summary(current_logs: list[float]) -> tuple[float, float, float]:
    """The median and quartiles of everyone's current belief, in original (not log) units."""
    sorted_x = sorted(10.0**lg for lg in current_logs)
    return (
        statistics.median(sorted_x),
        quantile_sorted(sorted_x, 0.25),
        quantile_sorted(sorted_x, 0.75),
    )


def _run_all_cells(
    settings: SweepSettings,
    spec: dict[str, Any],
    tasks: list[TaskItem],
    conditions: list[str],
    agents: list[AgentProfile],
    backend: Any,
    runs_path: Path,
    existing: dict[RecordKey, NumericEstimateRecord],
    tracker: SpendTracker,
) -> list[NumericEstimateRecord]:
    """Loop over condition, question and replicate; inside each, play every round in turn."""
    ordered_records: list[NumericEstimateRecord] = []
    with runs_path.open("a", encoding="utf-8") as fh:
        # If a crash left the file without a final newline, add one so the next row starts
        # on its own line.
        if runs_path.stat().st_size and not runs_path.read_bytes().endswith(b"\n"):
            fh.write("\n")
        for cond_id in conditions:
            for task in tasks:
                for rep in range(settings.replicates):
                    current_logs = initial_beliefs(task, rep, settings)
                    histories: list[list[tuple[str, str]]] = [[] for _ in range(len(agents))]
                    for r_idx in range(1, settings.n_rounds + 1):
                        # The crowd summary is fixed at the start of the round, so agents
                        # within a round do not influence each other.
                        ctx = RoundContext(
                            backend=backend,
                            agents=agents,
                            cond_id=cond_id,
                            task=task,
                            round_idx=r_idx,
                            pop_stats=crowd_summary(current_logs),
                            current_logs=current_logs,
                            histories=histories,
                            tracker=tracker,
                            max_tokens=settings.max_tokens,
                            pricing=settings.pricing,
                            abort=threading.Event(),
                            current_date=settings.current_date,
                        )
                        ordered_records.extend(
                            run_round(
                                ctx, spec, rep, settings.alpha, settings.workers, existing, fh
                            )
                        )
    return ordered_records
