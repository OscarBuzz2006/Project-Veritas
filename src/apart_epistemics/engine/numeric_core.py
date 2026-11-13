"""Kevin's crash-resumable C1/C2/C3 numeric estimation engine and Page's Theorem analyzer."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from apart_epistemics.backends import BackendConfigError, BudgetExceededError
from apart_epistemics.datasets import generate_agent_population
from apart_epistemics.engine.contrasts import compute_paired_holm_contrasts
from apart_epistemics.metrics import compute_page_decomposition, pearson_corr, quantile_sorted
from apart_epistemics.models import AgentProfile, NumericEstimateRecord, TaskItem
from apart_epistemics.parsers import parse_numeric_estimate
from apart_epistemics.run_state import backend_identity, configure_budget, validate_manifest


def build_numeric_prompts(
    condition_id: str,
    agent: AgentProfile,
    question: str,
    current_x: float,
    pop_stats: tuple[float, float, float] | None,
    history: list[tuple[str, str]],
) -> tuple[str, str]:
    """Construct C1 (baseline), C2 (social summary), or C3 (personalized+history) prompts."""
    cond = condition_id.upper()
    est_str = f"{current_x:g}"

    if cond == "C3":
        system_prompt = (
            f"You are an analytical estimation advisor speaking with a {agent.profession} "
            f"in {agent.region} (age band {agent.age_band}) who values "
            f"{', '.join(agent.values) or 'evidence'}. "
            f"Help them reason accurately and end cleanly with 'Final estimate: <number>'."
        )
    else:
        system_prompt = (
            "You are a neutral quantitative estimation assistant. "
            "Reason step by step without web lookup and end with 'Final estimate: <number>'."
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

    parts.append(f"My current estimate is {est_str}. What should my updated numerical estimate be?")
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


def _estimate_call_cost_usd(resp: dict[str, Any]) -> float:
    in_tok = int(resp.get("input_tokens", 0) or 0)
    out_tok = int(resp.get("output_tokens", 0) or 0) + int(resp.get("reasoning_tokens", 0) or 0)
    return max(1e-5, (in_tok * 0.5 + out_tok * 1.5) / 1_000_000.0)


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
    spent_usd: float,
    budget_usd: float,
) -> tuple[dict[str, Any], Any, float]:
    """Query backend once, retrying with an explicit format reminder on parse failure."""
    resp = backend.generate_numeric_reply(
        system_prompt=sys_p,
        user_prompt=usr_p,
        condition_id=cond_id,
        task=task,
        round_idx=round_idx,
        current_x=cur_x,
        pop_stats=pop_stats,
        agent=agent,
    )
    spent_usd += _estimate_call_cost_usd(resp)
    raw_reply = str(resp.get("raw_reply", ""))
    parsed = parse_numeric_estimate(raw_reply, require_positive=True)

    if parsed.miss:
        if spent_usd >= budget_usd:
            raise BudgetExceededError("Budget cap exhausted during retry.")
        retry_usr = (
            f"{usr_p}\n\nRespond with a single positive number prefixed by 'Final estimate: '."
        )
        resp = backend.generate_numeric_reply(
            system_prompt=sys_p,
            user_prompt=retry_usr,
            condition_id=cond_id,
            task=task,
            round_idx=round_idx,
            current_x=cur_x,
            pop_stats=pop_stats,
            agent=agent,
        )
        spent_usd += _estimate_call_cost_usd(resp)
        raw_reply = str(resp.get("raw_reply", ""))
        parsed = parse_numeric_estimate(raw_reply, require_positive=True)

    return resp, parsed, spent_usd


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
) -> list[NumericEstimateRecord]:
    """Run crash-resumable C1/C2/C3 numeric estimation sweeps with retry-once and budget cap."""
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
        "conditions": conditions,
        "tasks": [t.model_dump(mode="json") for t in tasks],
        "backend": backend_identity(backend),
    }
    validate_manifest(runs_path.with_suffix(".manifest.json"), spec, runs_path)
    ledger = getattr(backend, "budget_ledger", None)
    configure_budget(backend, runs_path.parent, budget_usd, ledger.per_request if ledger else 1.0)
    agents = generate_agent_population(num_agents=n_agents, seed=seed)
    existing = _load_existing_numeric_records(runs_path)

    spent_usd = sum(_estimate_call_cost_usd(r.model_dump()) for r in existing.values())
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
                        + 0.22 * task.popular_misconception_bias
                        + init_rng.gauss(0.0, 0.24)
                        for _ in range(n_agents)
                    ]
                    histories: list[list[tuple[str, str]]] = [[] for _ in range(n_agents)]

                    for r_idx in range(1, n_rounds + 1):
                        sorted_x = sorted(10.0**lg for lg in current_logs)
                        pop_stats = (
                            statistics.median(sorted_x),
                            quantile_sorted(sorted_x, 0.25),
                            quantile_sorted(sorted_x, 0.75),
                        )

                        for a_idx, agent in enumerate(agents):
                            key = (cond_id, task.id, rep, r_idx, agent.agent_id)
                            if key in existing:
                                rec = existing[key]
                                current_logs[a_idx] = rec.agent_log10_after
                                histories[a_idx].append(
                                    (
                                        f"My current estimate is {10.0**rec.agent_log10_before:g}.",
                                        rec.raw_reply,
                                    )
                                )
                                ordered_records.append(rec)
                                continue

                            if budget_usd <= 0.0 or spent_usd >= budget_usd:
                                raise BudgetExceededError(
                                    f"Budget cap exhausted (spent={spent_usd:.4f}, "
                                    f"cap={budget_usd:.4f})."
                                )

                            before_log = current_logs[a_idx]
                            cur_x = 10.0**before_log
                            sys_p, usr_p = build_numeric_prompts(
                                condition_id=cond_id,
                                agent=agent,
                                question=task.question,
                                current_x=cur_x,
                                pop_stats=pop_stats if cond_id in ("C2", "C3") else None,
                                history=histories[a_idx],
                            )
                            resp, parsed, spent_usd = _query_with_retry(
                                backend,
                                sys_p,
                                usr_p,
                                cond_id,
                                task,
                                r_idx,
                                cur_x,
                                pop_stats,
                                agent,
                                spent_usd,
                                budget_usd,
                            )
                            raw_reply = str(resp.get("raw_reply", ""))

                            if parsed.miss or parsed.log10_value is None:
                                after_log = before_log
                                asst_log = None
                                parsed_val = None
                                is_miss = True
                            else:
                                asst_log = parsed.log10_value
                                parsed_val = parsed.value
                                after_log = (1.0 - alpha) * before_log + alpha * asst_log
                                is_miss = False

                            current_logs[a_idx] = after_log
                            histories[a_idx].append(
                                (f"My current estimate is {cur_x:g}.", raw_reply)
                            )

                            rec = NumericEstimateRecord(
                                run_spec=spec,
                                condition_id=cond_id,
                                question_id=task.id,
                                replicate=rep,
                                round_idx=r_idx,
                                agent_id=agent.agent_id,
                                model_slug=str(resp.get("model_slug", "synthetic")),
                                provider=str(resp.get("provider", "synthetic_sim")),
                                raw_reply=raw_reply,
                                parsed_value=parsed_val,
                                assistant_log10=asst_log,
                                agent_log10_before=before_log,
                                agent_log10_after=after_log,
                                miss=is_miss,
                                reasoning_tokens=int(resp.get("reasoning_tokens", 0) or 0),
                                input_tokens=int(resp.get("input_tokens", 0) or 0),
                                output_tokens=int(resp.get("output_tokens", 0) or 0),
                            )
                            existing[key] = rec
                            ordered_records.append(rec)
                            fh.write(rec.model_dump_json() + "\n")
                            fh.flush()

    return ordered_records


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
        "h4_interpretation": "Descriptive correlation, mechanically coupled by update; not causal.",
        "max_identity_residual": max_resid,
        "h4_assistant_agent_corr": round(h4_corr, 4),
        "by_condition_final": by_condition_final,
        "condition_round_metrics": condition_round_metrics,
        "contrasts": contrasts,
    }
