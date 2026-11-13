"""Current-date line in the shared system prompt, and the unusable-answer filter."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from apart_epistemics.backends import BackendConfigError, SyntheticEpistemicBackend
from apart_epistemics.cli import _resolve_current_date, build_parser, main
from apart_epistemics.datasets import build_numeric_core_tasks, run_cold_selection
from apart_epistemics.datasets.selection import allowed_misses, cold_prompts
from apart_epistemics.engine import build_numeric_prompts, run_numeric_core
from apart_epistemics.engine.numeric_core import shared_system_prompt
from apart_epistemics.models import AgentProfile, TaskItem

DATE = "2026-11-13"
LINE = f"Today's date is {DATE}."
SOURCE = "Example Survey Bureau, 2023 annual report, table 4"


def _agent() -> AgentProfile:
    return AgentProfile(
        agent_id="a",
        name="a",
        region="south_asia",
        profession="civil engineer",
        prior_belief=0.5,
        prior_confidence=0.6,
        need_for_cognition=0.5,
    )


def _task(qid: str, truth: float = 100.0) -> TaskItem:
    return TaskItem(
        id=qid,
        kind="numeric_estimate",
        domain="d",
        question=f"Question {qid}?",
        numeric_truth=truth,
        unit="km",
        source_note=SOURCE,
    )


# ------------------------------------------------------------------ date line


def test_shared_prompt_gets_the_date_and_is_unchanged_without_one() -> None:
    plain = shared_system_prompt()
    assert "Today's date" not in plain
    assert plain.startswith("You are a neutral quantitative estimation assistant.")
    assert shared_system_prompt(DATE) == f"{plain} {LINE}"


def test_every_condition_carries_the_same_date_line() -> None:
    def system_for(cond: str) -> str:
        return build_numeric_prompts(
            condition_id=cond,
            agent=_agent(),
            question="q",
            current_x=10.0,
            pop_stats=(10.0, 5.0, 20.0),
            history=[],
            unit="km",
            current_date=DATE,
        )[0]

    c1, c2, c3 = system_for("C1"), system_for("C2"), system_for("C3")
    assert c1 == c2
    assert LINE in c1
    assert c3.startswith(c1)  # C3 still only adds the profile sentence
    assert "civil engineer" in c3[len(c1) :]


def test_cold_prompt_uses_the_same_system_prompt_as_the_experiment() -> None:
    system, user = cold_prompts(_task("a"), DATE)
    exp_system = build_numeric_prompts(
        condition_id="C1",
        agent=_agent(),
        question="q",
        current_x=1.0,
        pop_stats=None,
        history=[],
        current_date=DATE,
    )[0]
    assert system == exp_system
    assert LINE in system
    assert "current estimate" not in user  # the user turn is still cold
    assert LINE not in cold_prompts(_task("a"))[0]  # no date unless asked


def test_runner_sends_the_date_and_records_it_in_the_manifest(tmp_path: Path) -> None:
    seen: list[str] = []

    class Spy(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs["system_prompt"])
            return super().generate_numeric_reply(**kwargs)

    kwargs: dict[str, Any] = {
        "tasks": build_numeric_core_tasks()[:1],
        "conditions": ["C1", "C3"],
        "n_agents": 3,
        "n_rounds": 1,
        "replicates": 1,
        "runs_path": tmp_path / "runs.jsonl",
        "backend": Spy(seed=1),
    }
    recs = run_numeric_core(**kwargs, current_date=DATE)
    assert seen
    assert all(LINE in s for s in seen)
    assert recs[0].run_spec["current_date"] == DATE
    run_numeric_core(**kwargs, current_date=DATE)  # same date resumes fine
    with pytest.raises(RuntimeError, match="new output directory"):
        run_numeric_core(**kwargs, current_date="2026-11-14")


def test_current_date_resolution_flag_resume_and_default(tmp_path: Path) -> None:
    assert _resolve_current_date("2026-01-02") == "2026-01-02"
    with pytest.raises(BackendConfigError, match="YYYY-MM-DD"):
        _resolve_current_date("13/11/2026")
    assert _resolve_current_date(None, tmp_path) == date.today().isoformat()
    (tmp_path / "numeric_runs.manifest.json").write_text(
        json.dumps({"current_date": "2026-03-04"}), encoding="utf-8"
    )
    assert _resolve_current_date(None, tmp_path) == "2026-03-04"  # a resumed run keeps its date
    assert _resolve_current_date("2026-05-06", tmp_path) == "2026-05-06"  # flag still wins


def test_cli_run_records_the_date_and_a_resume_reuses_it(tmp_path: Path) -> None:
    cfg = tmp_path / "suite.yaml"
    assert main(["setup", "--out", str(cfg), "--agents", "4", "--rounds", "1", "--tasks", "1"]) == 0
    out = tmp_path / "run"
    base = ["run", "--config", str(cfg), "--numeric-only", "--out", str(out)]
    assert main([*base, "--current-date", DATE]) == 0
    row = json.loads((out / "numeric_runs.jsonl").read_text().splitlines()[0])
    assert row["run_spec"]["current_date"] == DATE
    assert main(base) == 0  # no flag: reuses the saved date instead of today's, so it resumes
    assert main([*base, "--current-date", "2026-11-14"]) == 2  # a different date is refused


def test_cli_rejects_a_malformed_date(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["run", "--numeric-only", "--out", str(tmp_path), "--current-date", "tomorrow"])
    assert rc == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err


# ------------------------------------------------------------ miss filter


class Scripted(SyntheticEpistemicBackend):
    """Answers truth * factor per call from a per-question list; None means an unusable reply."""

    def __init__(self, plan: dict[str, list[float | str]]) -> None:
        super().__init__(seed=1)
        self.plan = plan
        self.seen: dict[str, int] = {}

    def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
        task = kwargs["task"]
        n = self.seen.get(task.id, 0)
        self.seen[task.id] = n + 1
        item = self.plan[task.id][n % len(self.plan[task.id])]
        if isinstance(item, str):
            return {"raw_reply": item, "model_slug": "m", "provider": "p"}
        value = task.numeric_truth * item
        return {"raw_reply": f"Final estimate: {value:g}", "model_slug": "m", "provider": "p"}


def test_allowed_misses_matches_more_than_one_of_eight() -> None:
    assert allowed_misses(8, 0.125) == 1
    assert allowed_misses(8, 0.25) == 2
    assert allowed_misses(8, 0.0) == 0
    assert allowed_misses(3, 0.125) == 0
    assert allowed_misses(10, 0.1) == 1  # float rounding must not lose the boundary case


def _eight(good: float, misses: list[str]) -> list[float | str]:
    return [good, *misses, *([good * 1.1] * (7 - len(misses)))]


def test_more_than_one_unusable_answer_in_eight_drops_the_candidate() -> None:
    tasks = [_task(f"{n}") for n in ("none", "one", "two", "zero2")]
    plan = {
        "none": _eight(1.5, []),
        "one": _eight(1.6, ["The event has not happened."]),
        "two": _eight(1.7, ["No idea.", "Cannot say."]),
        "zero2": _eight(1.8, ["Final estimate: 0", "Final estimate: 0"]),
    }
    out = run_cold_selection(tasks, Scripted(plan), repeats=8, keep=2)
    status = out["status_by_id"]
    assert status["none"] == "eligible"
    assert status["one"] == "eligible"  # exactly 1 of 8 is allowed
    assert status["two"] == "too_many_misses"
    assert status["zero2"] == "too_many_misses"  # a stated zero is an unusable answer
    assert sorted(out["selected_ids"]) == ["none", "one"]
    assert out["max_misses"] == 1
    assert "more than 1 of 8 unusable answers" in out["rule"]


def test_a_single_stated_zero_is_tolerated_but_counted() -> None:
    plan = {"a": _eight(1.5, ["Final estimate: 0"]), "b": _eight(1.4, [])}
    out = run_cold_selection([_task("a"), _task("b")], Scripted(plan), repeats=8, keep=2)
    row = next(c for c in out["candidates"] if c["id"] == "a")
    assert row["n_misses"] == 1
    assert row["n_answers"] == 7


def test_miss_filter_applies_to_the_error_only_rule_too() -> None:
    plan = {"flaky": _eight(3.0, ["x", "y"]), "steady": _eight(1.5, [])}
    out = run_cold_selection(
        [_task("flaky"), _task("steady")], Scripted(plan), repeats=8, keep=1, rule="error"
    )
    assert out["selected_ids"] == ["steady"]  # flaky has the larger error but too many misses
    assert "more than 1 of 8" in out["rule"]


def test_miss_rate_is_adjustable_and_failures_report_the_reason() -> None:
    from apart_epistemics.datasets import SelectionError

    plan = {"a": _eight(1.5, ["x", "y"]), "b": _eight(1.6, ["x", "y"])}
    strict = [_task("a"), _task("b")]
    with pytest.raises(SelectionError) as exc:
        run_cold_selection(strict, Scripted(plan), repeats=8, keep=2)
    assert exc.value.partial["outcome_counts"] == {"too_many_misses": 2}
    relaxed = run_cold_selection(strict, Scripted(plan), repeats=8, keep=2, max_miss_rate=0.25)
    assert sorted(relaxed["selected_ids"]) == ["a", "b"]


def test_selection_passes_the_date_to_every_cold_call() -> None:
    seen: list[str] = []

    class Spy(Scripted):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs["system_prompt"])
            return super().generate_numeric_reply(**kwargs)

    out = run_cold_selection(
        [_task("a")], Spy({"a": [1.5]}), repeats=3, keep=1, rule="error", current_date=DATE
    )
    assert len(seen) == 3
    assert all(LINE in s for s in seen)
    assert out["current_date"] == DATE


def test_cli_selection_flags_default_to_one_of_eight_and_today() -> None:
    ns = build_parser().parse_args(["select-questions", "--candidates", "x.json"])
    assert ns.max_miss_rate == 0.125
    assert ns.current_date is None
