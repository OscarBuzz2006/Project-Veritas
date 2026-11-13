"""Question selection (#16) and live-smoke-test reporting (#17), tested fully offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apart_epistemics.backends import SyntheticEpistemicBackend
from apart_epistemics.cli import main
from apart_epistemics.datasets import (
    build_numeric_core_tasks,
    run_cold_selection,
    validate_candidates,
)
from apart_epistemics.engine import run_numeric_core
from apart_epistemics.models import TaskItem
from apart_epistemics.smoke_report import build_smoke_report

SOURCE = "Example Survey Bureau, 2023 annual report, table 4"


def _task(qid: str, truth: float, unit: str = "km", source: str = SOURCE) -> TaskItem:
    return TaskItem(
        id=qid,
        kind="numeric_estimate",
        domain="d",
        question=f"Question {qid}?",
        numeric_truth=truth,
        unit=unit,
        source_note=source,
    )


class Scripted(SyntheticEpistemicBackend):
    """Answers each question with truth * factor[question id]; records the prompts it saw."""

    def __init__(self, factors: dict[str, float]) -> None:
        super().__init__(seed=1)
        self.factors = factors
        self.prompts: list[str] = []

    def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
        task = kwargs["task"]
        self.prompts.append(kwargs["user_prompt"])
        value = task.numeric_truth * self.factors[task.id]
        return {"raw_reply": f"Final estimate: {value:g}", "model_slug": "m", "provider": "p"}


def test_selection_keeps_the_questions_the_model_gets_most_wrong() -> None:
    tasks = [
        _task("known", 1000.0),
        _task("off10x", 500.0),
        _task("off100x", 80.0),
        _task("close", 9.0),
    ]
    backend = Scripted({"known": 1.0, "off10x": 10.0, "off100x": 0.01, "close": 1.2})
    out = run_cold_selection(tasks, backend, repeats=3, keep=2, rule="error")
    assert out["selected_ids"] == ["off100x", "off10x"]  # 2.0 and 1.0 log10 units off
    by_id = {c["id"]: c for c in out["candidates"]}
    assert by_id["known"]["mean_abs_log10_error"] == pytest.approx(0.0)
    assert by_id["off10x"]["mean_abs_log10_error"] == pytest.approx(1.0)
    assert by_id["off10x"]["log10_spread"] == pytest.approx(0.0)
    assert len(by_id["off10x"]["replies"]) == 3
    assert "3 times" in out["rule"]
    assert "The 2 candidates" in out["rule"]


def test_selection_asks_cold_with_no_estimate_crowd_or_history() -> None:
    backend = Scripted({"a": 1.0})
    run_cold_selection([_task("a", 10.0, unit="meters")], backend, repeats=2, keep=1, rule="error")
    assert len(backend.prompts) == 2
    for prompt in backend.prompts:
        assert "current estimate" not in prompt
        assert "Other people" not in prompt
        assert "history" not in prompt
        assert "Final estimate: <plain number in meters>" in prompt


def test_selection_spread_reflects_disagreement_between_repeats() -> None:
    class Wobbly(Scripted):
        n = 0

        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            Wobbly.n += 1
            task = kwargs["task"]
            value = task.numeric_truth * (10.0 if Wobbly.n % 2 else 1.0)
            return {"raw_reply": f"Final estimate: {value:g}", "model_slug": "m", "provider": "p"}

    out = run_cold_selection([_task("a", 100.0)], Wobbly({}), repeats=4, keep=1, rule="error")
    assert out["candidates"][0]["log10_spread"] == pytest.approx(0.5)


def test_selection_is_deterministic_and_breaks_ties_by_id() -> None:
    tasks = [_task("b", 100.0), _task("a", 100.0), _task("c", 100.0)]
    backend = Scripted({"a": 10.0, "b": 10.0, "c": 10.0})
    first = run_cold_selection(tasks, backend, keep=2, rule="error")
    assert first["selected_ids"] == ["a", "b"]
    assert run_cold_selection(tasks, backend, keep=2, rule="error") == first


def test_unparseable_candidates_are_excluded_not_selected() -> None:
    class Silent(Scripted):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["task"].id == "mute":
                return {"raw_reply": "I do not know.", "model_slug": "m", "provider": "p"}
            return super().generate_numeric_reply(**kwargs)

    out = run_cold_selection(
        [_task("mute", 5.0), _task("ok", 5.0)],
        Silent({"ok": 100.0}),
        repeats=2,
        keep=1,
        rule="error",
    )
    assert out["selected_ids"] == ["ok"]
    assert out["n_excluded_no_answer"] == 1


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        ([_task("a", 1.0), _task("a", 2.0)], "unique"),
        ([_task("a", 1.0, source="survey")], "specific, citable source"),
        ([_task("a", 1.0, source="")], "specific, citable source"),
        ([_task("a", 1.0, unit="")], "unit is required"),
    ],
)
def test_candidates_need_unique_ids_units_and_a_specific_source(
    tasks: list[TaskItem], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_candidates(tasks)


def test_cannot_keep_more_than_the_candidates() -> None:
    with pytest.raises(ValueError, match="Cannot keep 5"):
        run_cold_selection([_task("a", 1.0)], Scripted({"a": 1.0}), keep=5)


def test_select_questions_cli_writes_the_rule_and_the_chosen_tasks(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"q{i}",
            "question": f"Q{i}?",
            "numeric_truth": 10.0 ** (i + 1),
            "unit": "km",
            "source_note": SOURCE,
        }
        for i in range(5)
    ]
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "sel"
    rc = main(
        [
            "select-questions",
            "--candidates",
            str(cand),
            "--keep",
            "2",
            "--repeats",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    result = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    chosen = json.loads((out / "selected_tasks.json").read_text(encoding="utf-8"))
    assert len(chosen) == 2
    assert {c["id"] for c in chosen} == set(result["selected_ids"])
    assert result["backend"]["class"] == "SyntheticEpistemicBackend"
    assert "highest spread" in result["rule"]
    assert {c["selection_stratum"] for c in chosen} == {"high_spread", "low_spread"}


def test_select_questions_cli_rejects_uncitable_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cand = tmp_path / "cand.json"
    cand.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "question": "Q?",
                    "numeric_truth": 5,
                    "unit": "km",
                    "source_note": "vague",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "select-questions",
                "--candidates",
                str(cand),
                "--keep",
                "1",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 2
    )
    assert "citable source" in capsys.readouterr().err


# ------------------------------------------------------------ smoke report


def _smoke_run(tmp_path: Path, mutate: Any = None) -> Path:
    run_numeric_core(
        tasks=build_numeric_core_tasks()[:1],
        conditions=["C1", "C3"],
        n_agents=4,
        n_rounds=2,
        replicates=1,
        runs_path=tmp_path / "numeric_runs.jsonl",
        backend=SyntheticEpistemicBackend(seed=1),
    )
    if mutate:
        path = tmp_path / "numeric_runs.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        for i, row in enumerate(rows):
            mutate(i, row)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def _good(_i: int, row: dict[str, Any]) -> None:
    row.update(
        finish_reason="stop",
        reasoning_tokens=12,
        provider="OpenAI",
        model_slug="openai/gpt-6-luna",
        input_tokens=900,
        output_tokens=40,
    )
    row["cost_usd"] = 0.0001


def _good_without_cost(i: int, row: dict[str, Any]) -> None:
    """A row as written before cost_usd existed."""
    _good(i, row)
    row["cost_usd"] = None


def test_smoke_report_passes_a_clean_run(tmp_path: Path) -> None:
    run = _smoke_run(tmp_path, _good)
    rep = build_smoke_report(
        run,
        expected_model="openai/gpt-6-luna",
        expected_provider="OpenAI",
        price_in=0.1,
        price_out=0.5,
    )
    assert rep["all_automatic_checks_pass"] is True
    assert rep["checks"]["cost_known_per_call"]["pass"] is True  # every row carries a real cost
    assert 0 < len(rep["sample_rows"]) <= 10
    assert "C1" in rep["checks"]["c3_prompt_size"]["detail"]


def test_smoke_report_flags_truncation_zero_reasoning_misses_and_wrong_provider(
    tmp_path: Path,
) -> None:
    def bad(i: int, row: dict[str, Any]) -> None:
        _good(i, row)
        row["reasoning_tokens"] = 0
        if i == 0:
            row["finish_reason"] = "length"
        if i < 4:
            row.update(miss=True, parsed_value=None, assistant_log10=None)
        if i == 5:
            row["provider"] = "Amazon Bedrock"

    rep = build_smoke_report(
        _smoke_run(tmp_path, bad), expected_model="openai/gpt-6-luna", expected_provider="OpenAI"
    )
    assert rep["all_automatic_checks_pass"] is False
    checks = rep["checks"]
    assert checks["no_truncated_replies"]["pass"] is False
    assert checks["reasoning_tokens_recorded"]["pass"] is False
    assert checks["miss_rate_near_zero"]["pass"] is False
    assert checks["model_and_provider_as_expected"]["pass"] is False


def test_smoke_report_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _smoke_run(tmp_path / "good", _good)
    assert main(["smoke-report", "--input", str(run), "--expected-provider", "OpenAI"]) == 0
    assert json.loads(capsys.readouterr().out)["all_automatic_checks_pass"] is True
    assert main(["smoke-report", "--input", str(run), "--expected-provider", "Other"]) == 1
    assert main(["smoke-report", "--input", str(tmp_path / "missing")]) == 2


def test_smoke_report_cost_check_is_automatic_when_every_row_has_a_cost(tmp_path: Path) -> None:
    def priced(i: int, row: dict[str, Any]) -> None:
        _good(i, row)
        row["cost_usd"] = 0.0001 * (i + 1)

    rep = build_smoke_report(_smoke_run(tmp_path, priced), price_in=0.1, price_out=0.5)
    cost = rep["checks"]["cost_known_per_call"]
    assert cost["pass"] is True
    assert "actual" in cost["detail"]
    assert "token estimate" in cost["detail"]


def test_smoke_report_falls_back_to_the_request_journal_for_older_runs(tmp_path: Path) -> None:
    run = _smoke_run(tmp_path, _good_without_cost)
    journal = [{"response": {"usage": {"cost": 0.0002}}} for _ in range(5)]
    (run / "provider_requests.jsonl").write_text(
        "\n".join(json.dumps(j) for j in journal) + "\n", encoding="utf-8"
    )
    cost = build_smoke_report(run)["checks"]["cost_known_per_call"]
    assert cost["pass"] is True
    assert "0.0010 USD over 5 requests" in cost["detail"]
    assert "provider_requests.jsonl" in cost["detail"]


def test_smoke_report_fails_the_cost_check_when_no_real_cost_exists(tmp_path: Path) -> None:
    rep = build_smoke_report(_smoke_run(tmp_path, _good_without_cost))
    assert rep["checks"]["cost_known_per_call"]["pass"] is False
    assert rep["all_automatic_checks_pass"] is False

    journal_without_cost = {"response": {"usage": {"prompt_tokens": 3}}}
    (tmp_path / "provider_requests.jsonl").write_text(
        json.dumps(journal_without_cost) + "\n", encoding="utf-8"
    )
    assert build_smoke_report(tmp_path)["checks"]["cost_known_per_call"]["pass"] is False


def test_selection_records_the_cost_of_each_cold_reply() -> None:
    class Priced(Scripted):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            return {**super().generate_numeric_reply(**kwargs), "cost_usd": 0.0003}

    out = run_cold_selection(
        [_task("a", 10.0)], Priced({"a": 2.0}), repeats=2, keep=1, rule="error"
    )
    assert [r["cost_usd"] for r in out["candidates"][0]["replies"]] == [0.0003, 0.0003]


# ------------------------------------------------- stratified selection rule


class Spready(SyntheticEpistemicBackend):
    """Per question: answers cycle through truth * factor for each factor in the list."""

    def __init__(self, plan: dict[str, list[float]]) -> None:
        super().__init__(seed=1)
        self.plan = plan
        self.seen: dict[str, int] = {}

    def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
        task = kwargs["task"]
        n = self.seen.get(task.id, 0)
        self.seen[task.id] = n + 1
        factors = self.plan[task.id]
        value = task.numeric_truth * factors[n % len(factors)]
        return {"raw_reply": f"Final estimate: {value:g}", "model_slug": "m", "provider": "p"}


def _plan(n_each: int = 3) -> tuple[list[TaskItem], dict[str, list[float]]]:
    """Every id is 'hN' (wrong by ~30-60%, varied) or 'lN' (wrong by ~30%, constant)."""
    tasks, plan = [], {}
    for i in range(n_each):
        tasks.append(_task(f"h{i}", 100.0))
        plan[f"h{i}"] = [1.3, 2.0 + i * 0.5, 0.6]  # spread grows with i
        tasks.append(_task(f"l{i}", 100.0))
        plan[f"l{i}"] = [1.4 + i * 0.01]  # constant answers
    return tasks, plan


def test_stratified_keeps_equal_numbers_of_high_and_low_spread_questions() -> None:
    tasks, plan = _plan()
    out = run_cold_selection(tasks, Spready(plan), repeats=6, keep=4)
    assert out["rule_name"] == "stratified"
    assert sorted(out["high_spread_ids"]) == ["h1", "h2"]
    assert sorted(out["low_spread_ids"]) == ["l0", "l1"] or sorted(out["low_spread_ids"]) == [
        "l0",
        "l2",
    ]
    assert len(out["selected_ids"]) == 4
    assert out["strata"]["h2"] == "high_spread"
    assert out["strata"][out["low_spread_ids"][0]] == "low_spread"
    s = out["summary"]
    assert s["high_spread"]["n"] == s["low_spread"]["n"] == 2
    assert s["overall"]["n"] == 4
    assert s["high_spread"]["mean_spread"] > s["low_spread"]["mean_spread"]


def test_stratified_requires_the_error_floor_about_25_percent() -> None:
    tasks = [_task("close", 100.0), _task("far_a", 100.0), _task("far_b", 100.0)]
    plan = {"close": [1.10, 1.12], "far_a": [1.5, 1.4], "far_b": [1.6, 1.3]}  # 1.1 = 0.041 log
    out = run_cold_selection(tasks, Spready(plan), repeats=4, keep=2)
    assert out["status_by_id"]["close"] == "below_error_floor"
    assert sorted(out["selected_ids"]) == ["far_a", "far_b"]
    assert "about 26% off" in out["rule"]


def test_very_large_errors_are_held_for_hand_review_until_confirmed() -> None:
    tasks = [_task(f"ok{i}", 100.0) for i in range(3)] + [_task("suspect", 100.0)]
    plan = {
        "ok0": [1.4, 1.5],
        "ok1": [1.5, 1.7],
        "ok2": [1.3, 1.9],
        "suspect": [10.0, 40.0],  # 1-1.6 log units off, and the widest spread
    }

    with pytest.raises(ValueError, match="held for hand review") as exc:
        run_cold_selection(tasks, Spready(plan), repeats=4, keep=4)
    assert "suspect" in str(exc.value)

    # With keep=2 the suspect is simply held back, not selected.
    held = run_cold_selection(tasks, Spready(plan), repeats=4, keep=2)
    assert held["held_for_review"] == ["suspect"]
    assert "suspect" not in held["selected_ids"]

    # Once a person confirms the recorded truth, it is eligible.
    confirmed = run_cold_selection(
        tasks, Spready(plan), repeats=4, keep=2, reviewed_ids=frozenset({"suspect"})
    )
    assert "suspect" in confirmed["selected_ids"]
    assert confirmed["held_for_review"] == []


def test_reviewing_an_unknown_id_is_an_error_not_silently_ignored() -> None:
    tasks, plan = _plan()
    with pytest.raises(ValueError, match="not among the candidates"):
        run_cold_selection(
            tasks, Spready(plan), repeats=4, keep=2, reviewed_ids=frozenset({"typo"})
        )


def test_stratified_needs_an_even_keep_and_enough_eligible_candidates() -> None:
    tasks, plan = _plan()
    with pytest.raises(ValueError, match="even"):
        run_cold_selection(tasks, Spready(plan), repeats=4, keep=3)
    near = [_task(f"n{i}", 100.0) for i in range(4)]
    with pytest.raises(ValueError, match="Only 0 eligible"):
        run_cold_selection(
            near, Spready({f"n{i}": [1.01, 1.02] for i in range(4)}), repeats=4, keep=2
        )


def test_single_answer_candidates_cannot_be_stratified() -> None:
    tasks, plan = _plan()
    out = run_cold_selection(tasks, Spready(plan), repeats=1, keep=2, rule="error")
    assert out["rule_name"] == "error"  # the old rule still works with one answer
    with pytest.raises(ValueError, match="too_few_answers"):
        run_cold_selection(tasks, Spready(plan), repeats=1, keep=2)


def test_a_tied_spread_boundary_is_flagged() -> None:
    tasks = [_task(f"c{i}", 100.0) for i in range(4)]
    plan = {f"c{i}": [1.5] for i in range(4)}  # all constant: spread 0 everywhere
    out = run_cold_selection(tasks, Spready(plan), repeats=3, keep=4)
    assert out["boundary_tied"] is True
    tasks2, plan2 = _plan()
    assert run_cold_selection(tasks2, Spready(plan2), repeats=6, keep=4)["boundary_tied"] is False


def test_stratified_selection_is_deterministic() -> None:
    tasks, plan = _plan()
    first = run_cold_selection(tasks, Spready(plan), repeats=6, keep=4)
    assert run_cold_selection(tasks, Spready(plan), repeats=6, keep=4) == first


def test_cli_default_repeats_and_thresholds_and_review_flow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from apart_epistemics.cli import build_parser

    ns = build_parser().parse_args(["select-questions", "--candidates", "x.json"])
    assert (ns.rule, ns.repeats, ns.keep, ns.min_error, ns.review_error) == (
        "stratified",
        8,
        10,
        0.1,
        0.5,
    )

    rows = [
        {
            "id": f"q{i}",
            "question": f"Q{i}?",
            "numeric_truth": 10.0 ** (i + 1),
            "unit": "km",
            "source_note": SOURCE,
        }
        for i in range(6)
    ]
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps(rows), encoding="utf-8")
    rc = main(
        ["select-questions", "--candidates", str(cand), "--keep", "5", "--out", str(tmp_path / "o")]
    )
    assert rc == 2  # odd keep is refused with a message, not a traceback
    assert "even" in capsys.readouterr().err


def test_selected_tasks_file_runs_and_the_analysis_reports_each_half(tmp_path: Path) -> None:
    from apart_epistemics.cli import _default_conditions, _execute_numeric_track
    from apart_epistemics.engine import analyze_numeric_core
    from apart_epistemics.models import ExperimentSuiteConfig

    base = build_numeric_core_tasks()[:4]
    strata = ["high_spread", "high_spread", "low_spread", "low_spread"]
    tasks = [
        t.model_copy(update={"selection_stratum": s}) for t, s in zip(base, strata, strict=True)
    ]
    tasks_file = tmp_path / "selected_tasks.json"
    tasks_file.write_text(json.dumps([t.model_dump(mode="json") for t in tasks]), encoding="utf-8")

    cfg = ExperimentSuiteConfig(
        num_agents=4,
        num_rounds=2,
        numeric_replicates=1,
        numeric_tasks_file=str(tasks_file),
        conditions=_default_conditions(),
    )
    analysis, recs = _execute_numeric_track(
        cfg, SyntheticEpistemicBackend(seed=2), tmp_path / "run", n_agents=4, budget_usd=50.0
    )
    assert {r.question_id for r in recs} == {t.id for t in tasks}  # the file replaced the bank
    by = analysis["by_stratum"]
    assert set(by) == {"high_spread", "low_spread"}
    assert by["high_spread"]["questions"] == sorted(t.id for t in tasks[:2])
    assert set(by["high_spread"]["by_condition_final"]) == {"C1", "C2", "C3"}
    assert by["low_spread"]["by_condition_final"]["C1"]["n_cells"] == 2
    assert "slope_user" in by["high_spread"]["h4_slopes"]["by_condition"]["C1"]

    # analyze rebuilds the same per-half report from the saved manifest alone.
    reloaded = analyze_numeric_core(recs, tasks)
    assert reloaded["by_stratum"] == by

    # Without strata nothing extra is reported.
    plain = [t.model_copy(update={"selection_stratum": ""}) for t in tasks]
    assert analyze_numeric_core(recs, plain)["by_stratum"] == {}


# ------------------------------------- failed selections keep their paid answers


class Counting(Spready):
    def __init__(self, plan: dict[str, list[float]]) -> None:
        super().__init__(plan)
        self.calls = 0

    def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return super().generate_numeric_reply(**kwargs)


def test_bad_keep_or_reviewed_ids_are_rejected_before_any_paid_call() -> None:
    tasks, plan = _plan()
    backend = Counting(plan)
    with pytest.raises(ValueError, match="even"):
        run_cold_selection(tasks, backend, repeats=4, keep=3)
    with pytest.raises(ValueError, match="not among the candidates"):
        run_cold_selection(tasks, backend, repeats=4, keep=2, reviewed_ids=frozenset({"typo"}))
    assert backend.calls == 0  # nothing was spent on a run that could never succeed


def test_rows_and_summaries_carry_the_question_family() -> None:
    tasks = [
        _task("a", 100.0).model_copy(update={"domain": "census"}),
        _task("b", 100.0).model_copy(update={"domain": "census"}),
        _task("c", 100.0).model_copy(update={"domain": "events"}),
        _task("d", 100.0).model_copy(update={"domain": "events"}),
    ]
    plan = {"a": [1.5, 1.2], "b": [1.5], "c": [1.4, 3.0], "d": [1.6, 1.5]}
    out = run_cold_selection(tasks, Spready(plan), repeats=4, keep=4)
    assert {c["id"]: c["domain"] for c in out["candidates"]} == {
        "a": "census",
        "b": "census",
        "c": "events",
        "d": "events",
    }
    assert out["summary"]["overall"]["by_family"] == {"census": 2, "events": 2}
    halves = [out["summary"]["high_spread"]["by_family"], out["summary"]["low_spread"]["by_family"]]
    assert sum(sum(h.values()) for h in halves) == 4


def test_selection_error_carries_every_measurement_for_diagnosis() -> None:
    from apart_epistemics.datasets import SelectionError

    tasks = [_task(f"n{i}", 100.0) for i in range(4)]
    plan = {f"n{i}": [1.01, 1.02] for i in range(4)}  # all below the error floor
    with pytest.raises(SelectionError) as exc:
        run_cold_selection(tasks, Spready(plan), repeats=4, keep=2)
    partial = exc.value.partial
    assert len(partial["candidates"]) == 4
    assert all(len(c["replies"]) == 4 for c in partial["candidates"])
    assert set(partial["status_by_id"].values()) == {"below_error_floor"}
    assert partial["outcome_counts"] == {"below_error_floor": 4}
    assert partial["min_error"] == 0.1
    assert "about 26% off" in partial["rule"]


def test_cli_saves_measurements_and_prints_a_table_when_selection_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        {
            "id": f"q{i}",
            "question": f"Q{i}?",
            "domain": "census",
            "numeric_truth": 10.0 ** (i + 1),
            "unit": "km",
            "source_note": SOURCE,
        }
        for i in range(4)
    ]
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "sel"
    # The offline backend answers close to the truth, so every candidate is under the floor.
    rc = main(["select-questions", "--candidates", str(cand), "--keep", "2", "--out", str(out)])
    assert rc == 2
    saved = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert "Only 0 eligible" in saved["failed"]
    assert len(saved["candidates"]) == 4
    assert saved["backend"]["class"] == "SyntheticEpistemicBackend"
    assert not (out / "selected_tasks.json").exists()  # no selection, so no tasks file
    err = capsys.readouterr().err
    assert "Measurements saved to" in err
    assert "below_error_floor" in err
    assert "census" in err  # the table shows the family
