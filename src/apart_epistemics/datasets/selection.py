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

MISS_RULE = (
    " Candidates with more than {allowed} of {repeats} unusable answers (no positive number, "
    "including a stated zero) were dropped."
)

SELECTION_RULE = (
    "Each candidate was asked {repeats} times with no user estimate, crowd statistic or history. "
    "Error is the mean absolute log10 distance of the parsed answers from the documented truth. "
    "The {keep} candidates with the largest error were kept; ties broke by question id. "
    "Candidates with no parseable answer were excluded."
)


STRATIFIED_RULE = (
    "Each candidate was asked {repeats} times with no user estimate, crowd statistic or history. "
    "Error is the mean absolute log10 distance of the parsed answers from the documented truth; "
    "spread is the population standard deviation of the parsed log10 answers. Candidates were "
    "required to have error >= {min_error} log10 units (about {pct:.0f}% off) and at least two "
    "parseable answers. Candidates with error >= {review_error} were held back until a person "
    "confirmed the recorded truth (a very large error often means the recorded answer is wrong). "
    "From the rest, the {half} with the highest spread and the {half} with the lowest spread "
    "were kept; ties broke by error then id."
)


class SelectionError(ValueError):
    """Selection could not be completed; `partial` holds everything measured so far.

    The cold answers were already paid for, so callers should save `partial` for diagnosis.
    """

    def __init__(self, message: str, partial: dict[str, Any]) -> None:
        super().__init__(message)
        self.partial = partial


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


def allowed_misses(repeats: int, max_miss_rate: float) -> int:
    """Most unusable answers a candidate may have: `more than 1 of 8` -> 1 at 8 repeats."""
    return int(max_miss_rate * repeats + 1e-9)


def cold_prompts(task: TaskItem, current_date: str = "") -> tuple[str, str]:
    """Same system prompt as the experiment's C1; the user turn has no current estimate."""
    # Imported here: engine.numeric_core imports from datasets, so a top-level import would loop.
    from apart_epistemics.engine.numeric_core import shared_system_prompt

    system = shared_system_prompt(current_date)
    user = (
        f"Question: {task.question}\n\n"
        f"What is your best estimate of the true value? End with "
        f"'Final estimate: <plain number in {task.unit}>' and nothing after the number."
    )
    return system, user


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    errs = [r["mean_abs_log10_error"] for r in rows]
    spreads = [r["log10_spread"] for r in rows]
    families: dict[str, int] = {}
    for r in rows:
        families[r["domain"]] = families.get(r["domain"], 0) + 1
    return {
        "n": len(rows),
        "by_family": dict(sorted(families.items())),
        "mean_error": round(sum(errs) / len(errs), 6),
        "mean_spread": round(sum(spreads) / len(spreads), 6),
        "min_spread": round(min(spreads), 6),
        "max_spread": round(max(spreads), 6),
    }


def _stratify(
    rows: list[dict[str, Any]],
    keep: int,
    min_error: float,
    review_error: float,
    reviewed_ids: frozenset[str],
    max_misses: int,
) -> dict[str, Any]:
    status: dict[str, str] = {}
    for r in rows:
        err, spread = r["mean_abs_log10_error"], r["log10_spread"]
        if err is None:
            status[r["id"]] = "no_answer"
        elif r["n_misses"] > max_misses:
            status[r["id"]] = "too_many_misses"
        elif spread is None:
            status[r["id"]] = "too_few_answers"
        elif err < min_error:
            status[r["id"]] = "below_error_floor"
        elif err >= review_error and r["id"] not in reviewed_ids:
            status[r["id"]] = "held_for_review"
        else:
            status[r["id"]] = "eligible"

    eligible = sorted(
        (r for r in rows if status[r["id"]] == "eligible"),
        key=lambda r: (-r["log10_spread"], -r["mean_abs_log10_error"], r["id"]),
    )
    held = sorted(i for i, st in status.items() if st == "held_for_review")
    if len(eligible) < keep:
        counts: dict[str, int] = {}
        for st in status.values():
            counts[st] = counts.get(st, 0) + 1
        hint = (
            f" {len(held)} held for hand review ({', '.join(held)}): check the recorded truth "
            "and pass the confirmed ids as reviewed."
            if held
            else ""
        )
        raise SelectionError(
            f"Only {len(eligible)} eligible candidates for keep={keep}; outcomes {counts}.{hint}",
            {"status_by_id": status, "held_for_review": held, "outcome_counts": counts},
        )

    half = keep // 2
    high = eligible[:half]
    low = sorted(eligible[half:], key=lambda r: (r["log10_spread"], r["id"]))[:half]
    return {
        "status_by_id": status,
        "held_for_review": held,
        "high_spread_ids": [r["id"] for r in high],
        "low_spread_ids": [r["id"] for r in low],
        "boundary_tied": high[-1]["log10_spread"] <= low[-1]["log10_spread"],
        "summary": {
            "overall": _summarize(high + low),
            "high_spread": _summarize(high),
            "low_spread": _summarize(low),
        },
    }


def run_cold_selection(
    candidates: list[TaskItem],
    backend: Any,
    repeats: int = 8,
    keep: int = 10,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
    rule: str = "stratified",
    min_error: float = 0.1,
    review_error: float = 0.5,
    reviewed_ids: frozenset[str] = frozenset(),
    max_miss_rate: float = 0.125,
    current_date: str = "",
) -> dict[str, Any]:
    """Query every candidate cold `repeats` times, then select `keep` by the chosen rule.

    `rule="stratified"` (default): require error >= `min_error`, hold back error >=
    `review_error` until listed in `reviewed_ids`, then keep the `keep // 2` highest-spread
    and `keep // 2` lowest-spread of what remains. `rule="error"`: largest error only.
    """
    if repeats < 1 or keep < 1:
        raise ValueError("repeats and keep must be at least 1.")
    if rule not in {"stratified", "error"}:
        raise ValueError("rule must be 'stratified' or 'error'.")
    validate_candidates(candidates)
    if keep > len(candidates):
        raise ValueError(f"Cannot keep {keep} of only {len(candidates)} candidates.")
    if rule == "stratified":
        # Check everything that can be checked before any paid call is made.
        if keep % 2:
            raise ValueError("keep must be even for a high/low spread split.")
        unknown = reviewed_ids - {t.id for t in candidates}
        if unknown:
            raise ValueError(f"reviewed ids not among the candidates: {sorted(unknown)}")

    max_misses = allowed_misses(repeats, max_miss_rate)
    rows = _ask_cold(candidates, backend, repeats, max_tokens, current_date)
    result: dict[str, Any] = {
        "rule_name": rule,
        "current_date": current_date,
        "max_misses": max_misses,
        "repeats": repeats,
        "keep": keep,
        "n_candidates": len(candidates),
        "n_excluded_no_answer": sum(1 for r in rows if r["mean_abs_log10_error"] is None),
        "candidates": rows,
    }
    if rule == "error":
        ranked = sorted(
            (
                r
                for r in rows
                if r["mean_abs_log10_error"] is not None and r["n_misses"] <= max_misses
            ),
            key=lambda r: (-r["mean_abs_log10_error"], r["id"]),
        )
        result.update(
            rule=SELECTION_RULE.format(repeats=repeats, keep=keep)
            + MISS_RULE.format(allowed=max_misses, repeats=repeats),
            selected_ids=[r["id"] for r in ranked[:keep]],
            strata={},
        )
        return result

    rule_text = STRATIFIED_RULE.format(
        repeats=repeats,
        min_error=min_error,
        pct=(10.0**min_error - 1.0) * 100.0,
        review_error=review_error,
        half=keep // 2,
    ) + MISS_RULE.format(allowed=max_misses, repeats=repeats)
    try:
        strat = _stratify(rows, keep, min_error, review_error, reviewed_ids, max_misses)
    except SelectionError as exc:
        exc.partial.update(
            {**result, "rule": rule_text, "min_error": min_error, "review_error": review_error}
        )
        raise
    result.update(
        rule=rule_text,
        min_error=min_error,
        review_error=review_error,
        reviewed_ids=sorted(reviewed_ids),
        selected_ids=strat["high_spread_ids"] + strat["low_spread_ids"],
        strata={
            **dict.fromkeys(strat["high_spread_ids"], "high_spread"),
            **dict.fromkeys(strat["low_spread_ids"], "low_spread"),
        },
        **{k: v for k, v in strat.items() if k not in {"high_spread_ids", "low_spread_ids"}},
        high_spread_ids=strat["high_spread_ids"],
        low_spread_ids=strat["low_spread_ids"],
    )
    return result


def _ask_cold(
    candidates: list[TaskItem],
    backend: Any,
    repeats: int,
    max_tokens: int,
    current_date: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in candidates:
        system, user = cold_prompts(task, current_date)
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
                "domain": task.domain,
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
    return rows
