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
    out = run_cold_selection(tasks, backend, repeats=3, keep=2)
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
    run_cold_selection([_task("a", 10.0, unit="meters")], backend, repeats=2, keep=1)
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

    out = run_cold_selection([_task("a", 100.0)], Wobbly({}), repeats=4, keep=1)
    assert out["candidates"][0]["log10_spread"] == pytest.approx(0.5)


def test_selection_is_deterministic_and_breaks_ties_by_id() -> None:
    tasks = [_task("b", 100.0), _task("a", 100.0), _task("c", 100.0)]
    backend = Scripted({"a": 10.0, "b": 10.0, "c": 10.0})
    first = run_cold_selection(tasks, backend, keep=2)
    assert first["selected_ids"] == ["a", "b"]
    assert run_cold_selection(tasks, backend, keep=2) == first


def test_unparseable_candidates_are_excluded_not_selected() -> None:
    class Silent(Scripted):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["task"].id == "mute":
                return {"raw_reply": "I do not know.", "model_slug": "m", "provider": "p"}
            return super().generate_numeric_reply(**kwargs)

    out = run_cold_selection(
        [_task("mute", 5.0), _task("ok", 5.0)], Silent({"ok": 100.0}), repeats=2, keep=1
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
    assert "largest error" in result["rule"]


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
    assert rep["checks"]["cost_known_per_call"]["pass"] == "manual"
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
