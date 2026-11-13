"""H4 per-condition slopes: recover known effects and avoid the old statistic's artefacts."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from apart_epistemics.backends import SyntheticEpistemicBackend
from apart_epistemics.datasets import build_numeric_core_tasks
from apart_epistemics.engine import analyze_numeric_core, run_numeric_core
from apart_epistemics.engine.h4 import compute_h4_slopes, ols
from apart_epistemics.metrics import pearson_corr
from apart_epistemics.models import NumericEstimateRecord, TaskItem


def _task(qid: str, truth: float) -> TaskItem:
    return TaskItem(id=qid, kind="numeric_estimate", domain="d", question="q", numeric_truth=truth)


def _rec(
    cond: str,
    qid: str,
    user_err: float,
    crowd_err: float | None,
    assistant_err: float,
    truth: float,
    miss: bool = False,
) -> NumericEstimateRecord:
    lt = math.log10(truth)
    return NumericEstimateRecord(
        condition_id=cond,
        question_id=qid,
        replicate=0,
        round_idx=1,
        agent_id="a",
        model_slug="m",
        provider="p",
        raw_reply="r",
        parsed_value=None if miss else 1.0,
        assistant_log10=None if miss else lt + assistant_err,
        agent_log10_before=lt + user_err,
        agent_log10_after=lt + user_err,
        pop_median_log10=None if crowd_err is None else lt + crowd_err,
        miss=miss,
    )


def _grid(cond: str, qid: str, truth: float, b_user: float, b_crowd: float, n: int = 30) -> list:
    rng = random.Random(7)
    out = []
    for _ in range(n):
        u, c = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
        out.append(_rec(cond, qid, u, c, b_user * u + b_crowd * c, truth))
    return out


def test_ols_recovers_coefficients_and_rejects_degenerate_designs() -> None:
    rows = [[x, x * x] for x in (0.0, 1.0, 2.0, 3.0, 4.0)]
    y = [1.0 + 2.0 * r[0] - 0.5 * r[1] for r in rows]
    coef = ols(rows, y)
    assert coef is not None
    assert coef == pytest.approx([1.0, 2.0, -0.5], abs=1e-9)
    assert ols([[1.0], [1.0], [1.0]], [1.0, 2.0, 3.0]) is None  # constant regressor
    assert ols([[1.0], [2.0]], [1.0, 2.0]) is None  # exactly determined: no spare dof
    assert ols([], []) is None


def test_slopes_recover_user_and_crowd_weights_per_condition() -> None:
    tasks = [_task("q1", 1000.0), _task("q2", 5.0), _task("q3", 2e6)]
    recs = []
    for t in tasks:
        recs += _grid("C1", t.id, t.numeric_truth, b_user=0.8, b_crowd=0.0)
        recs += _grid("C2", t.id, t.numeric_truth, b_user=0.3, b_crowd=0.5)
    res = compute_h4_slopes(recs, tasks)["by_condition"]

    assert res["C1"]["slope_user"]["mean"] == pytest.approx(0.8, abs=1e-3)
    assert "slope_crowd" not in res["C1"]  # C1 never sees the crowd
    assert res["C2"]["slope_user"]["mean"] == pytest.approx(0.3, abs=1e-3)
    assert res["C2"]["slope_crowd"]["mean"] == pytest.approx(0.5, abs=1e-3)
    assert res["C2"]["slope_user"]["n_questions"] == 3
    assert res["C2"]["n_observations"] == 90
    lo, hi = res["C2"]["slope_user"]["ci95"]
    assert lo <= 0.3 <= hi


def test_conditions_are_reported_separately_so_they_can_be_compared() -> None:
    tasks = [_task("q1", 100.0), _task("q2", 1e4)]
    recs = []
    for t in tasks:
        recs += _grid("C1", t.id, t.numeric_truth, b_user=0.0, b_crowd=0.0)
        recs += _grid("C3", t.id, t.numeric_truth, b_user=0.9, b_crowd=0.0)
    res = compute_h4_slopes(recs, tasks)["by_condition"]
    assert res["C1"]["slope_user"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert res["C3"]["slope_user"]["mean"] == pytest.approx(0.9, abs=1e-3)


def test_old_pooled_correlation_is_high_even_when_the_assistant_ignores_the_user() -> None:
    # The assistant ignores the user (slope 0) and just answers near the truth, but questions
    # span 6 orders of magnitude, so the pooled correlation is near 1 anyway.
    tasks = [_task(f"q{i}", 10.0**i) for i in range(1, 7)]
    recs = []
    rng = random.Random(3)
    for t in tasks:
        lt = math.log10(t.numeric_truth)
        for _ in range(20):
            u = rng.uniform(-0.3, 0.3)
            r = _rec("C1", t.id, u, None, rng.uniform(-0.05, 0.05), t.numeric_truth)
            r = r.model_copy(update={"agent_log10_after": 0.7 * (lt + u) + 0.3 * r.assistant_log10})
            recs.append(r)
    pooled = pearson_corr([r.assistant_log10 for r in recs], [r.agent_log10_after for r in recs])
    assert pooled > 0.99
    slope = compute_h4_slopes(recs, tasks)["by_condition"]["C1"]["slope_user"]["mean"]
    assert abs(slope) < 0.2


def test_misses_and_unfittable_questions_are_excluded_not_guessed() -> None:
    tasks = [_task("q1", 100.0), _task("q2", 100.0)]
    recs = _grid("C1", "q1", 100.0, 0.5, 0.0)
    recs.append(_rec("C1", "q1", 0.1, None, 9.0, 100.0, miss=True))  # a miss must not count
    recs += [_rec("C1", "q2", 0.2, None, 0.1, 100.0)]  # one point: cannot be fitted
    out = compute_h4_slopes(recs, tasks)["by_condition"]["C1"]
    assert out["slope_user"]["mean"] == pytest.approx(0.5, abs=1e-3)
    assert out["slope_user"]["n_questions"] == 1
    assert out["slope_user"]["ci95"] is None  # one question is not enough for a CI
    assert "skipped" in out["per_question"]["q2"]
    assert out["n_observations"] == 30


def test_older_records_without_a_stored_crowd_median_fall_back_to_user_slope_only() -> None:
    tasks = [_task("q1", 100.0), _task("q2", 100.0)]
    recs = []
    for t in tasks:
        recs += [
            r.model_copy(update={"pop_median_log10": None})
            for r in _grid("C2", t.id, 100.0, 0.4, 0.0)
        ]
    out = compute_h4_slopes(recs, tasks)["by_condition"]["C2"]
    assert out["slope_user"]["mean"] == pytest.approx(0.4, abs=1e-3)
    assert out["slope_crowd"]["n_questions"] == 0


def test_bootstrap_is_deterministic() -> None:
    tasks = [_task(f"q{i}", 10.0 ** (i + 1)) for i in range(5)]
    recs = []
    for i, t in enumerate(tasks):
        recs += _grid("C1", t.id, t.numeric_truth, b_user=0.2 + 0.1 * i, b_crowd=0.0)
    assert compute_h4_slopes(recs, tasks) == compute_h4_slopes(recs, tasks)


def test_runner_stores_population_median_and_analysis_reports_slopes(tmp_path: Path) -> None:
    tasks = build_numeric_core_tasks()[:3]
    recs = run_numeric_core(
        tasks=tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=8,
        n_rounds=2,
        replicates=2,
        runs_path=tmp_path / "runs.jsonl",
        backend=SyntheticEpistemicBackend(seed=5),
    )
    assert all(r.pop_median_log10 is not None for r in recs)
    r1 = [
        r
        for r in recs
        if r.condition_id == "C1"
        and r.round_idx == 1
        and r.replicate == 0
        and r.question_id == tasks[0].id
    ]
    assert len({round(r.pop_median_log10, 9) for r in r1}) == 1  # same median for the whole round

    analysis = analyze_numeric_core(recs, tasks)
    assert "h4_assistant_agent_corr" in analysis  # legacy key kept for the dashboard
    slopes = analysis["h4_slopes"]["by_condition"]
    assert set(slopes) == {"C1", "C2", "C3"}
    assert "slope_crowd" in slopes["C2"]
    assert "slope_crowd" not in slopes["C1"]
