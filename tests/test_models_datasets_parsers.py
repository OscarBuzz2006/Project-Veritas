"""Tests for domain models, dual-track datasets, and adversarial numeric/JSON parsers."""

import math
from importlib import resources

import pytest
from pydantic import ValidationError

import apart_epistemics
from apart_epistemics.datasets import (
    audit_question_selection_bias,
    build_ground_truth_tasks,
    build_numeric_core_tasks,
    build_open_debate_tasks,
    generate_agent_population,
)
from apart_epistemics.metrics import clamp, quantile_sorted
from apart_epistemics.models import (
    ConditionConfig,
    HeterogeneityType,
    NationalityType,
    NumericEstimateRecord,
    RoundRecord,
    TaskKind,
)
from apart_epistemics.parsers import parse_numeric_estimate, parse_structured_belief


def test_condition_config_validates_bounds() -> None:
    cfg = ConditionConfig(
        condition_id="us_k1",
        label="Monolithic US",
        heterogeneity_type=HeterogeneityType.MONOLITHIC_SHARED,
        nationality_type=NationalityType.US_ONLY,
        num_models=1,
        temperature=0.8,
        top_p=0.95,
    )
    assert cfg.num_models == 1
    assert cfg.temperature == pytest.approx(0.8)

    with pytest.raises(ValidationError):
        ConditionConfig(
            condition_id="bad_temp",
            label="Bad",
            heterogeneity_type=HeterogeneityType.MONOLITHIC_SHARED,
            nationality_type=NationalityType.US_ONLY,
            num_models=1,
            temperature=3.5,
        )

    with pytest.raises(ValidationError):
        RoundRecord(
            run_id="r1",
            condition_id="us_k1",
            seed=42,
            task_id="t1",
            task_kind=TaskKind.GROUND_TRUTH,
            round_idx=0,
            agent_id="a1",
            agent_region="north_america",
            model_assigned="none",
            model_family="none",
            assistant_response_text="",
            agent_response_text="test",
            extracted_topics=[],
            stance_prob=1.5,
            confidence=0.8,
            stated_answer=1,
            goldman_source_overlap=0.0,
            persuasion_bias_exposure=0.0,
        )


def test_datasets_provide_rich_diverse_tasks_and_agents() -> None:
    gt_tasks = build_ground_truth_tasks()
    debate_tasks = build_open_debate_tasks()
    numeric_tasks = build_numeric_core_tasks()

    assert len(gt_tasks) >= 8
    assert len(debate_tasks) >= 8
    assert len(numeric_tasks) == 10
    assert all(t.ground_truth in (0, 1) for t in gt_tasks)
    assert all(t.ground_truth is None and len(t.canonical_topics) >= 6 for t in debate_tasks)
    assert all(t.numeric_truth is not None and t.numeric_truth > 0 for t in numeric_tasks)

    agents = generate_agent_population(num_agents=30, seed=42)
    assert len(agents) == 30
    regions = {a.region for a in agents}
    assert len(regions) >= 5  # Stratified global regions, not monocultural
    assert all(0.05 <= a.prior_belief <= 0.95 for a in agents)
    # Reproducibility across calls with same seed
    agents_repeat = generate_agent_population(num_agents=30, seed=42)
    assert [a.model_dump() for a in agents] == [a.model_dump() for a in agents_repeat]


def test_question_selection_bias_audit_detects_model_a_handicap() -> None:
    # Model A got 0/10 right cold while B and C got 8/10 right -> unfair C4 comparison
    cold_scores = {
        "model_a": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "model_b": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        "model_c": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    }
    report = audit_question_selection_bias(cold_scores, reference_model="model_a")
    assert report["selection_bias_detected"] is True
    assert report["max_accuracy_gap"] == pytest.approx(0.8)

    fair_scores = {
        "model_a": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "model_b": [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        "model_c": [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    }
    fair_report = audit_question_selection_bias(fair_scores, reference_model="model_a")
    assert fair_report["selection_bias_detected"] is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final estimate: 1,200,000", 1_200_000.0),
        (
            "Based on scaling laws, I suggest roughly 500k.\nFinal estimate: 1.2 million",
            1_200_000.0,
        ),
        ("Final estimate: 4.5 billion", 4_500_000_000.0),
        ("Final estimate: ~350 thousand", 350_000.0),
        ("Final estimate: 3.2e5", 320_000.0),
        ("Final estimate: $12.5k", 12_500.0),
        ("My reasoning covers 1990 to 2020. Final estimate: 42,500", 42_500.0),
        ("Estimate without prefix is 87500.", 87_500.0),
    ],
)
def test_numeric_parser_handles_units_commas_and_distractors(text: str, expected: float) -> None:
    result = parse_numeric_estimate(text)
    assert result.miss is False
    assert result.value == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "bad_text",
    [
        "",
        "I cannot provide an exact number without more data.",
        "Final estimate: 0",  # non-positive cannot enter log10
        "Final estimate: -150",  # negative cannot enter log10
        "Final estimate: N/A",
        "Final estimate: 1e9999",  # float overflow must become a clean miss
    ],
)
def test_numeric_parser_marks_misses_safely_without_crashing(bad_text: str) -> None:
    result = parse_numeric_estimate(bad_text, require_positive=True)
    assert result.miss is True
    assert result.value is None


def test_math_helpers_and_public_package_surface() -> None:
    assert clamp(float("nan"), 0.0, 1.0) == 0.0
    assert clamp(float("inf"), 0.0, 1.0) == 0.0
    assert clamp(1.5, 0.0, 1.0) == 1.0
    assert quantile_sorted([], 0.5) == 0.0
    assert quantile_sorted([4.2], 0.75) == pytest.approx(4.2)
    assert quantile_sorted([0.0, 10.0], 0.25) == pytest.approx(2.5)
    for symbol in apart_epistemics.__all__:
        assert hasattr(apart_epistemics, symbol)
    assert (resources.files("apart_epistemics") / "py.typed").is_file()


def test_structured_belief_parser_handles_json_and_fenced_text() -> None:
    fenced = (
        '```json\n{"stance_prob": 0.78, "confidence": 0.85, '
        '"rationale": "Grid storage matters.", "topics": ["grid_storage", "cost"]}\n```'
    )
    parsed = parse_structured_belief(fenced)
    assert parsed["stance_prob"] == pytest.approx(0.78)
    assert parsed["confidence"] == pytest.approx(0.85)
    assert parsed["topics"] == ["grid_storage", "cost"]

    # Out-of-range values clamped safely, malformed falls back gracefully
    fallback = parse_structured_belief("Non-JSON prose with no explicit probability.")
    assert math.isnan(fallback["stance_prob"]) or 0.0 <= fallback["stance_prob"] <= 1.0
    assert isinstance(fallback["rationale"], str)


def test_numeric_estimate_record_log10_consistency() -> None:
    rec = NumericEstimateRecord(
        condition_id="C1",
        question_id="q1",
        replicate=0,
        round_idx=1,
        agent_id="a1",
        model_slug="openai/gpt-6-luna",
        provider="openai",
        raw_reply="Final estimate: 1000",
        parsed_value=1000.0,
        assistant_log10=3.0,
        agent_log10_before=2.0,
        agent_log10_after=2.3,
        miss=False,
        reasoning_tokens=12,
        temperature=0.8,
        top_p=0.95,
    )
    assert rec.agent_log10_after == pytest.approx(0.7 * 2.0 + 0.3 * 3.0)
