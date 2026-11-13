"""Integration tests for Kevin's crash-resumable C1/C2/C3 numeric engine & Page analysis."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from apart_epistemics.backends import BudgetExceededError, SyntheticEpistemicBackend
from apart_epistemics.datasets import build_numeric_core_tasks
from apart_epistemics.engine import analyze_numeric_core, build_numeric_prompts, run_numeric_core
from apart_epistemics.models import AgentProfile


def test_c1_c2_c3_share_one_system_prompt_and_c3_only_appends_profile(
    sample_agent: AgentProfile,
) -> None:
    def system_for(cond: str) -> str:
        return build_numeric_prompts(
            condition_id=cond,
            agent=sample_agent,
            question="Length of river in km?",
            current_x=1500.0,
            pop_stats=(2100.0, 1400.0, 3100.0),
            history=[],
        )[0]

    sys_c1, sys_c2, sys_c3 = system_for("C1"), system_for("C2"), system_for("C3")
    assert sys_c1 == sys_c2
    # C3 keeps the shared prompt verbatim as a prefix and only adds the profile sentence.
    assert sys_c3.startswith(sys_c1)
    added = sys_c3[len(sys_c1) :]
    assert sample_agent.profession in added
    assert sample_agent.region in added
    assert sample_agent.age_band in added
    # No role/persona change: C3 must not swap in a different "advisor" framing.
    assert "advisor" not in sys_c3.lower()


def test_numeric_prompt_asks_for_best_estimate_in_task_unit(sample_agent: AgentProfile) -> None:
    _, usr = build_numeric_prompts(
        condition_id="C1",
        agent=sample_agent,
        question="Maximum depth of Lake Baikal?",
        current_x=1500.0,
        pop_stats=None,
        history=[],
        unit="meters",
    )
    assert "My current estimate is 1500 meters." in usr
    assert "What is your best estimate of the true value?" in usr
    assert "Final estimate: <plain number in meters>" in usr
    assert "nothing after the number" in usr
    assert "updated" not in usr

    _, usr_no_unit = build_numeric_prompts(
        condition_id="C1",
        agent=sample_agent,
        question="q",
        current_x=2.0,
        pop_stats=None,
        history=[],
    )
    assert "Final estimate: <plain number>" in usr_no_unit


def test_runner_passes_task_unit_into_prompt(tmp_path: Path) -> None:
    task = build_numeric_core_tasks()[0]
    seen: list[str] = []

    class Spy(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs["user_prompt"])
            return super().generate_numeric_reply(**kwargs)

    run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=2,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "runs.jsonl",
        backend=Spy(seed=1),
    )
    assert seen
    assert all(f"Final estimate: <plain number in {task.unit}>" in p for p in seen)


def test_runner_records_finish_reason_and_forwards_max_tokens(tmp_path: Path) -> None:
    task = build_numeric_core_tasks()[0]
    seen_max_tokens: list[int] = []

    class Backend(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            seen_max_tokens.append(kwargs["max_tokens"])
            out = super().generate_numeric_reply(**kwargs)
            out["finish_reason"] = "length"
            return out

    default_recs = run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=2,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "a.jsonl",
        backend=Backend(seed=1),
    )
    assert set(seen_max_tokens) == {1024}
    assert all(r.finish_reason == "length" for r in default_recs)
    # Sampling was never sent, so it must not be logged as if it had been.
    assert all(r.temperature is None and r.top_p is None for r in default_recs)

    seen_max_tokens.clear()
    run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=2,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "b.jsonl",
        backend=Backend(seed=1),
        max_tokens=4096,
    )
    assert set(seen_max_tokens) == {4096}


def test_runner_treats_values_over_three_log_units_away_as_misses(tmp_path: Path) -> None:
    task = build_numeric_core_tasks()[0]
    prompts: list[str] = []

    class UnitSlipBackend(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            prompts.append(kwargs["user_prompt"])
            # A metres-read-as-millions style slip: roughly 1e9 times the true size.
            return {
                "raw_reply": f"Final estimate: {task.numeric_truth * 1e9:g}",
                "model_slug": "m",
                "provider": "p",
                "reasoning_tokens": 0,
            }

    records = run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=2,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "runs.jsonl",
        backend=UnitSlipBackend(seed=1),
    )
    assert len(records) == 2
    assert all(r.miss for r in records)
    assert all(r.parsed_value is None and r.assistant_log10 is None for r in records)
    # Each implausible reply is retried once before being recorded as a miss.
    assert len(prompts) == 4
    # A miss leaves the agent's belief unchanged.
    assert all(r.agent_log10_after == r.agent_log10_before for r in records)


def test_numeric_prompt_builder_and_crash_resumable_runner(
    tmp_path: Path, sample_agent: AgentProfile
) -> None:
    _sys_c1, usr_c1 = build_numeric_prompts(
        condition_id="C1",
        agent=sample_agent,
        question="Length of river in km?",
        current_x=1500.0,
        pop_stats=None,
        history=[],
    )
    assert "Other people" not in usr_c1
    assert "civil engineer" not in _sys_c1

    sys_c2, usr_c2 = build_numeric_prompts(
        condition_id="C2",
        agent=sample_agent,
        question="Length of river in km?",
        current_x=1500.0,
        pop_stats=(2100.0, 1400.0, 3100.0),
        history=[],
    )
    assert "Other people asking this recently gave a median of" in usr_c2
    assert "civil engineer" not in sys_c2

    sys_c3, usr_c3 = build_numeric_prompts(
        condition_id="C3",
        agent=sample_agent,
        question="Length of river in km?",
        current_x=1500.0,
        pop_stats=(2100.0, 1400.0, 3100.0),
        history=[("My current estimate is 1200.", "Final estimate: 1800")],
    )
    assert "civil engineer" in sys_c3
    assert "south_asia" in sys_c3
    assert "1500" not in sys_c3
    assert "1800" in usr_c3

    tasks = build_numeric_core_tasks()[:2]
    runs_file = tmp_path / "numeric_runs.jsonl"

    class FlakyBackend(SyntheticEpistemicBackend):
        def __init__(self) -> None:
            super().__init__(seed=7)
            self.calls = 0

        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "raw_reply": "Unclear without data.",
                    "model_slug": "m",
                    "provider": "p",
                    "reasoning_tokens": 0,
                }
            if self.calls in (2, 3):
                return {
                    "raw_reply": "Still unclear.",
                    "model_slug": "m",
                    "provider": "p",
                    "reasoning_tokens": 0,
                }
            return super().generate_numeric_reply(**kwargs)

    flaky = FlakyBackend()
    records = run_numeric_core(
        tasks=tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=8,
        n_rounds=3,
        replicates=1,
        runs_path=runs_file,
        backend=flaky,
        seed=7,
    )
    assert len(records) == 2 * 3 * 8 * 3
    assert any(r.miss for r in records)
    calls_first_run = flaky.calls

    # Resuming on the same runs_path must skip every completed cell (0 extra backend calls).
    records_resumed = run_numeric_core(
        tasks=tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=8,
        n_rounds=3,
        replicates=1,
        runs_path=runs_file,
        backend=flaky,
        seed=7,
    )
    assert len(records_resumed) == len(records)
    assert flaky.calls == calls_first_run

    analysis = analyze_numeric_core(records_resumed, tasks)
    assert analysis["max_identity_residual"] < 1e-11
    assert len(analysis["contrasts"]) == 4

    with pytest.raises(BudgetExceededError):
        run_numeric_core(
            tasks=tasks,
            conditions=["C1"],
            n_agents=8,
            n_rounds=2,
            replicates=1,
            runs_path=tmp_path / "budget_runs.jsonl",
            backend=SyntheticEpistemicBackend(seed=7),
            budget_usd=0.0,
            seed=7,
        )


def test_cross_process_determinism_and_truncated_jsonl_resume(tmp_path: Path) -> None:
    outs = []
    for hseed in ("111", "999"):
        sub_out = tmp_path / f"det_{hseed}.jsonl"
        code = (
            "import json; from pathlib import Path; "
            "from apart_epistemics import ("
            "SyntheticEpistemicBackend, build_numeric_core_tasks, run_numeric_core"
            "); "
            f"recs = run_numeric_core(build_numeric_core_tasks()[:1], ['C3'], 4, 2, 1, "
            f"Path({str(sub_out)!r}), SyntheticEpistemicBackend(seed=42), seed=42); "
            "print(json.dumps([round(r.agent_log10_after, 6) for r in recs]))"
        )
        env = dict(os.environ, PYTHONHASHSEED=hseed)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1]

    trunc_path = tmp_path / "trunc.jsonl"
    tasks = build_numeric_core_tasks()[:1]
    backend = SyntheticEpistemicBackend(seed=42)
    full = run_numeric_core(tasks, ["C3"], 4, 2, 1, trunc_path, backend, seed=42)
    lines = trunc_path.read_text(encoding="utf-8").splitlines()
    trunc_path.write_text(
        "\n".join([*lines[:5], '{"condition_id": "C3", "broken']), encoding="utf-8"
    )
    repaired = run_numeric_core(tasks, ["C3"], 4, 2, 1, trunc_path, backend, seed=42)
    assert len(repaired) == len(full)
    assert [round(r.agent_log10_after, 6) for r in repaired] == [
        round(r.agent_log10_after, 6) for r in full
    ]
