"""Cold-start question selection: keep the candidates a model gets most wrong.

If the model under test already knows an answer, every condition converges on the truth and the
experiment shows nothing. This asks each candidate question cold (no user estimate, no crowd,
no history) several times, measures error and spread on a log10 scale, and keeps the questions
with the largest error. The rule is recorded in the output so it can be reported.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from apart_epistemics.backends.base import DEFAULT_NUMERIC_MAX_TOKENS
from apart_epistemics.models import TaskItem
from apart_epistemics.parsers import parse_numeric_estimate

SELECTION_RULE = (
    "Each candidate was asked {repeats} times with no user estimate, crowd statistic or history. "
    "Error is the mean absolute log10 distance of the parsed answers from the documented truth. "
    "The {keep} candidates with the largest error were kept; ties broke by question id. "
    "Candidates with no parseable answer were excluded."
)


def validate_candidates(tasks: list[TaskItem]) -> None:
    """Refuse candidates that cannot be audited or cited."""
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Candidate question ids must be unique.")
    for t in tasks:
        if t.numeric_truth is None:
            raise ValueError(f"{t.id}: numeric_truth is required.")
        if not t.unit.strip():
            raise ValueError(f"{t.id}: unit is required.")
        if len(t.source_note.strip()) < 15:
            raise ValueError(f"{t.id}: source_note must name a specific, citable source and date.")


def cold_prompts(task: TaskItem) -> tuple[str, str]:
    """Same system prompt as the experiment's C1; the user turn has no current estimate."""
    system = (
        "You are a neutral quantitative estimation assistant. "
        "Reason step by step without web lookup and end with 'Final estimate: <number>'."
    )
    user = (
        f"Question: {task.question}\n\n"
        f"What is your best estimate of the true value? End with "
        f"'Final estimate: <plain number in {task.unit}>' and nothing after the number."
    )
    return system, user


def run_cold_selection(
    candidates: list[TaskItem],
    backend: Any,
    repeats: int = 3,
    keep: int = 10,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
) -> dict[str, Any]:
    """Query every candidate cold `repeats` times and pick the `keep` with the largest error."""
    if repeats < 1 or keep < 1:
        raise ValueError("repeats and keep must be at least 1.")
    validate_candidates(candidates)
    if keep > len(candidates):
        raise ValueError(f"Cannot keep {keep} of only {len(candidates)} candidates.")

    rows: list[dict[str, Any]] = []
    for task in candidates:
        system, user = cold_prompts(task)
        log_truth = math.log10(task.numeric_truth or 1.0)
        answers: list[float] = []
        replies: list[dict[str, Any]] = []
        for rep in range(repeats):
            resp = backend.generate_numeric_reply(
                system_prompt=system,
                user_prompt=user,
                condition_id="COLD",
                task=task,
                round_idx=rep + 1,
                current_x=task.numeric_truth,
                max_tokens=max_tokens,
            )
            parsed = parse_numeric_estimate(str(resp.get("raw_reply", "")), require_positive=True)
            replies.append(
                {
                    "raw_reply": str(resp.get("raw_reply", "")),
                    "parsed_value": parsed.value,
                    "miss": parsed.miss,
                    "finish_reason": str(resp.get("finish_reason", "") or ""),
                    "cost_usd": resp.get("cost_usd"),
                    "model_slug": str(resp.get("model_slug", "")),
                    "provider": str(resp.get("provider", "")),
                }
            )
            if not parsed.miss and parsed.log10_value is not None:
                answers.append(parsed.log10_value)
        errors = [abs(a - log_truth) for a in answers]
        rows.append(
            {
                "id": task.id,
                "numeric_truth": task.numeric_truth,
                "unit": task.unit,
                "source_note": task.source_note,
                "n_answers": len(answers),
                "n_misses": repeats - len(answers),
                "mean_abs_log10_error": round(sum(errors) / len(errors), 6) if errors else None,
                "log10_spread": round(statistics.pstdev(answers), 6) if len(answers) > 1 else None,
                "replies": replies,
            }
        )

    ranked = sorted(
        (r for r in rows if r["mean_abs_log10_error"] is not None),
        key=lambda r: (-r["mean_abs_log10_error"], r["id"]),
    )
    selected_ids = [r["id"] for r in ranked[:keep]]
    return {
        "rule": SELECTION_RULE.format(repeats=repeats, keep=keep),
        "repeats": repeats,
        "keep": keep,
        "n_candidates": len(candidates),
        "n_excluded_no_answer": sum(1 for r in rows if r["mean_abs_log10_error"] is None),
        "selected_ids": selected_ids,
        "candidates": rows,
    }
