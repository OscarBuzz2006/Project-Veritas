"""Numeric-run settings must come from the config, never be silently overridden."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from apart_epistemics.backends import SyntheticEpistemicBackend
from apart_epistemics.cli import _default_conditions, _execute_numeric_track, build_parser, main
from apart_epistemics.datasets import build_numeric_core_tasks
from apart_epistemics.engine import run_numeric_core
from apart_epistemics.models import ExperimentSuiteConfig


def _cfg(**overrides: Any) -> ExperimentSuiteConfig:
    params: dict[str, Any] = {
        "num_agents": 4,
        "num_rounds": 6,
        "tasks_per_track": 1,
        "numeric_replicates": 1,
        "conditions": _default_conditions(),
    }
    params.update(overrides)
    return ExperimentSuiteConfig(**params)


def _run(tmp_path: Path, cfg: ExperimentSuiteConfig, name: str = "out") -> Any:
    return _execute_numeric_track(
        cfg, SyntheticEpistemicBackend(seed=1), tmp_path / name, n_agents=4, budget_usd=100.0
    )[1]


def test_round_count_is_not_capped_at_four(tmp_path: Path) -> None:
    recs = _run(tmp_path, _cfg(num_rounds=6))
    assert max(r.round_idx for r in recs) == 6
    recs_one = _run(tmp_path, _cfg(num_rounds=1), name="one")
    assert max(r.round_idx for r in recs_one) == 1  # and no hidden minimum of 2 either


def test_numeric_tasks_is_independent_of_tasks_per_track(tmp_path: Path) -> None:
    own = _run(tmp_path, _cfg(tasks_per_track=1, numeric_tasks=3), name="own")
    assert len({r.question_id for r in own}) == 3
    fallback = _run(tmp_path, _cfg(tasks_per_track=2, numeric_tasks=None), name="fallback")
    assert len({r.question_id for r in fallback}) == 2
    everything = _run(tmp_path, _cfg(numeric_tasks=len(build_numeric_core_tasks())), name="all")
    assert len({r.question_id for r in everything}) == len(build_numeric_core_tasks())


def test_effective_settings_are_printed_not_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(tmp_path, _cfg(num_rounds=2, numeric_tasks=2))
    out = capsys.readouterr().out
    assert f"2 of {len(build_numeric_core_tasks())} questions" in out
    assert "4 agents" in out
    assert "2 rounds" in out
    assert "start bias weight 0.22" in out


def test_setup_command_sets_numeric_tasks_so_smoke_runs_stay_small(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    assert (
        main(["setup", "--out", str(path), "--agents", "8", "--rounds", "2", "--tasks", "2"]) == 0
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["numeric_tasks"] == 2
    assert raw["num_rounds"] == 2
    assert build_parser()  # parser still builds


def test_start_bias_weight_zero_centres_the_crowd_on_the_truth(tmp_path: Path) -> None:
    task = max(build_numeric_core_tasks(), key=lambda t: abs(t.popular_misconception_bias))
    assert abs(task.popular_misconception_bias) >= 0.15
    log_truth = math.log10(task.numeric_truth)

    def mean_start_error(weight: float, name: str) -> float:
        recs = run_numeric_core(
            tasks=[task],
            conditions=["C1"],
            n_agents=200,
            n_rounds=1,
            replicates=1,
            runs_path=tmp_path / name,
            backend=SyntheticEpistemicBackend(seed=1),
            start_bias_weight=weight,
        )
        return sum(r.agent_log10_before - log_truth for r in recs) / len(recs)

    # Standard error of the mean is 0.24 / sqrt(200) ~ 0.017.
    assert mean_start_error(0.0, "zero.jsonl") == pytest.approx(0.0, abs=0.06)
    shifted = mean_start_error(0.22, "shifted.jsonl")
    assert shifted == pytest.approx(0.22 * task.popular_misconception_bias, abs=0.06)


def test_start_spread_is_configurable(tmp_path: Path) -> None:
    task = build_numeric_core_tasks()[0]

    def spread(sd: float, name: str) -> float:
        recs = run_numeric_core(
            tasks=[task],
            conditions=["C1"],
            n_agents=200,
            n_rounds=1,
            replicates=1,
            runs_path=tmp_path / name,
            backend=SyntheticEpistemicBackend(seed=1),
            start_sd=sd,
        )
        xs = [r.agent_log10_before for r in recs]
        mean = sum(xs) / len(xs)
        return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))

    assert spread(0.1, "narrow.jsonl") == pytest.approx(0.1, abs=0.02)
    assert spread(0.3, "wide.jsonl") == pytest.approx(0.3, abs=0.04)


def test_start_settings_are_recorded_in_the_manifest(tmp_path: Path) -> None:
    kwargs: dict[str, Any] = {
        "tasks": build_numeric_core_tasks()[:1],
        "conditions": ["C1"],
        "n_agents": 3,
        "n_rounds": 1,
        "replicates": 1,
        "runs_path": tmp_path / "runs.jsonl",
        "backend": SyntheticEpistemicBackend(seed=1),
    }
    recs = run_numeric_core(**kwargs, start_bias_weight=0.0, start_sd=0.3)
    assert recs[0].run_spec["start_bias_weight"] == 0.0
    assert recs[0].run_spec["start_sd"] == 0.3
    with pytest.raises(RuntimeError, match="new output directory"):
        run_numeric_core(**kwargs)  # defaults differ from the saved run
