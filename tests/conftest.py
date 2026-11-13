"""Shared pytest fixtures for Epistemic Prism unit, integration, and report suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apart_epistemics import (
    AgentProfile,
    ConditionConfig,
    ExperimentSuiteConfig,
    HeterogeneityType,
    NationalityType,
    RoundRecord,
    SyntheticEpistemicBackend,
    TaskKind,
    analyze_numeric_core,
    build_numeric_core_tasks,
    run_numeric_core,
    run_suite,
)


@pytest.fixture
def sample_agent() -> AgentProfile:
    """Return a reproducible sample agent profile."""
    return AgentProfile(
        agent_id="agent_00",
        name="Agent 01",
        region="south_asia",
        profession="civil engineer",
        age_band="25-34",
        values=["empirical rigor", "local autonomy"],
        prior_belief=0.5,
        prior_confidence=0.7,
        need_for_cognition=0.8,
        preferred_topics=["evidence"],
    )


@pytest.fixture
def record_factory():
    """Return a factory building a concise `RoundRecord` fixture with sensible defaults."""

    def _build(
        *,
        condition_id: str,
        round_idx: int,
        agent_id: str,
        model_assigned: str,
        model_family: str,
        topics: list[str],
        stance_prob: float,
        text: str,
        overlap: float,
    ) -> RoundRecord:
        return RoundRecord(
            run_id="run_test",
            condition_id=condition_id,
            seed=42,
            task_id="od_sovereign_ai",
            task_kind=TaskKind.OPEN_DEBATE,
            round_idx=round_idx,
            agent_id=agent_id,
            agent_region="east_asia",
            model_assigned=model_assigned,
            model_family=model_family,
            assistant_response_text="",
            agent_response_text=text,
            extracted_topics=topics,
            stance_prob=stance_prob,
            confidence=max(stance_prob, 1.0 - stance_prob),
            stated_answer=1 if stance_prob >= 0.5 else 0,
            goldman_source_overlap=overlap,
            persuasion_bias_exposure=overlap * 2.0,
        )

    return _build


@pytest.fixture
def core_suite_config() -> ExperimentSuiteConfig:
    """Return a compact Track 4 config spanning decoding, origin, and personalization."""
    return ExperimentSuiteConfig(
        suite_name="Compact Verification Suite",
        seed=42,
        num_agents=12,
        num_rounds=3,
        peer_degree=2,
        tasks_per_track=2,
        conditions=[
            ConditionConfig(
                condition_id="no_ai",
                label="No AI",
                heterogeneity_type=HeterogeneityType.NO_AI,
                nationality_type=NationalityType.NONE,
                num_models=0,
                group="baseline",
            ),
            ConditionConfig(
                condition_id="us_k1_greedy",
                label="US K1 T=0",
                heterogeneity_type=HeterogeneityType.MONOLITHIC_SHARED,
                nationality_type=NationalityType.US_ONLY,
                num_models=1,
                temperature=0.0,
                top_p=0.0,
                group="decoding_ablation",
            ),
            ConditionConfig(
                condition_id="us_k1",
                label="US K1 T=0.8",
                heterogeneity_type=HeterogeneityType.MONOLITHIC_SHARED,
                nationality_type=NationalityType.US_ONLY,
                num_models=1,
                group="k_sweep",
            ),
            ConditionConfig(
                condition_id="cross_national_k4",
                label="Cross-National K4",
                heterogeneity_type=HeterogeneityType.CULTURAL_REGIONAL_MIX,
                nationality_type=NationalityType.CROSS_NATIONAL,
                num_models=4,
                group="nationality",
            ),
            ConditionConfig(
                condition_id="personalized_advisor",
                label="Advisor",
                heterogeneity_type=HeterogeneityType.PERSONALIZED_ADVISOR,
                nationality_type=NationalityType.CROSS_NATIONAL,
                num_models=12,
                group="personalization",
            ),
            ConditionConfig(
                condition_id="personalized_peer",
                label="Peer",
                heterogeneity_type=HeterogeneityType.PERSONALIZED_PEER,
                nationality_type=NationalityType.US_ONLY,
                num_models=12,
                group="personalization",
            ),
        ],
    )


@pytest.fixture
def mini_demo_bundle(
    tmp_path: Path, core_suite_config: ExperimentSuiteConfig
) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    """Run both Track 4 and Numeric Core on compact fixtures for downstream report tests."""
    backend = SyntheticEpistemicBackend(seed=42)
    suite_res = run_suite(core_suite_config, backend=backend, out_dir=tmp_path / "suite")
    num_tasks = build_numeric_core_tasks()[:2]
    num_recs = run_numeric_core(
        tasks=num_tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=6,
        n_rounds=2,
        replicates=1,
        runs_path=tmp_path / "numeric_runs.jsonl",
        backend=backend,
        seed=42,
    )
    num_analysis = analyze_numeric_core(num_recs, num_tasks)
    return suite_res, num_analysis, num_recs
