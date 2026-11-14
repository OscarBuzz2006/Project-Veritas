"""Integration tests for Track 4 multi-round simulation dynamics and replay fidelity."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from apart_epistemics.backends import (
    OllamaBackend,
    ReplayBackend,
    SyntheticEpistemicBackend,
    create_backend,
)
from apart_epistemics.datasets import build_ground_truth_tasks, generate_agent_population
from apart_epistemics.engine import run_suite
from apart_epistemics.engine.track4 import _step_live_llm
from apart_epistemics.models import (
    ConditionConfig,
    ExperimentSuiteConfig,
    HeterogeneityType,
    NationalityType,
)


def test_synthetic_suite_verifies_decoding_dissociation_heterogeneity_and_personalization(
    tmp_path: Path,
) -> None:
    cfg = ExperimentSuiteConfig(
        suite_name="Test Suite",
        seed=42,
        num_agents=18,
        num_rounds=4,
        peer_degree=3,
        tasks_per_track=4,
        conditions=[
            ConditionConfig(
                condition_id="no_ai",
                label="No AI",
                heterogeneity_type=HeterogeneityType.NO_AI,
                nationality_type=NationalityType.NONE,
                num_models=0,
            ),
            ConditionConfig(
                condition_id="us_k1_greedy",
                label="US K1 T=0",
                heterogeneity_type=HeterogeneityType.MONOLITHIC_SHARED,
                nationality_type=NationalityType.US_ONLY,
                num_models=1,
                temperature=0.0,
                top_p=0.0,
            ),
            ConditionConfig(
                condition_id="us_k1",
                label="US K1 T=0.8",
                heterogeneity_type=HeterogeneityType.MONOLITHIC_SHARED,
                nationality_type=NationalityType.US_ONLY,
                num_models=1,
                temperature=0.8,
                top_p=0.95,
            ),
            ConditionConfig(
                condition_id="us_k4",
                label="US K4",
                heterogeneity_type=HeterogeneityType.MULTI_LAB_MIX,
                nationality_type=NationalityType.US_ONLY,
                num_models=4,
            ),
            ConditionConfig(
                condition_id="divergent_sft_pair",
                label="Divergent SFT",
                heterogeneity_type=HeterogeneityType.DIVERGENT_SFT_PAIR,
                nationality_type=NationalityType.DIVERGENT_SFT,
                num_models=2,
            ),
            ConditionConfig(
                condition_id="cross_national_k4",
                label="Cross-National K4",
                heterogeneity_type=HeterogeneityType.CULTURAL_REGIONAL_MIX,
                nationality_type=NationalityType.CROSS_NATIONAL,
                num_models=4,
            ),
            ConditionConfig(
                condition_id="personalized_advisor",
                label="Advisor",
                heterogeneity_type=HeterogeneityType.PERSONALIZED_ADVISOR,
                nationality_type=NationalityType.CROSS_NATIONAL,
                num_models=18,
            ),
            ConditionConfig(
                condition_id="personalized_peer",
                label="Peer",
                heterogeneity_type=HeterogeneityType.PERSONALIZED_PEER,
                nationality_type=NationalityType.US_ONLY,
                num_models=18,
            ),
        ],
    )
    out = run_suite(cfg, backend=SyntheticEpistemicBackend(seed=42), out_dir=tmp_path)
    by_id = {s.condition_id: s for s in out["summaries"]}

    # Decoding double dissociation: T=0.8 restores lexical variance vs T=0, not topic collapse.
    greedy = by_id["us_k1_greedy"]
    default = by_id["us_k1"]
    assert default.pairwise_jaccard_distance - greedy.pairwise_jaccard_distance > 0.08
    assert default.distinct_2 > greedy.distinct_2
    assert default.goldman_effective_sources < 5.0

    # Cross-national and divergent SFT preserve higher topic breadth and Goldman sources.
    assert by_id["cross_national_k4"].effective_topic_count > default.effective_topic_count
    assert (
        by_id["cross_national_k4"].goldman_effective_sources
        > by_id["us_k4"].goldman_effective_sources
    )
    assert by_id["divergent_sft_pair"].effective_topic_count > default.effective_topic_count

    # Personalization calibration split: peer stays diverse, advisor calibrates better.
    peer = by_id["personalized_peer"]
    advisor = by_id["personalized_advisor"]
    assert peer.effective_topic_count > default.effective_topic_count
    assert peer.confident_wrong_rate > advisor.confident_wrong_rate
    assert peer.brier_score > advisor.brier_score

    # Blinded dual-judge diagnostics and artifact exports.
    assert out["topic_audit"]["inter_rater_kappa"] is None
    assert out["topic_audit"]["reliability_status"] == "not_independently_judged"
    assert (tmp_path / "runs.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()

    replay = ReplayBackend.from_jsonl(tmp_path / "runs.jsonl")
    assert replay.record_count > 0
    assert create_backend("synthetic", seed=42) is not None


def test_live_llm_driver_replay_roundtrip_and_nonfinite_fallback(tmp_path: Path) -> None:
    def mock_chat_tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        return {
            "message": {
                "content": (
                    '{"stance_prob": 0.62, "confidence": 0.78, "answer": 1, '
                    '"topics": ["grid_carbon_intensity"]} Evidence highlights grid carbon.'
                )
            }
        }

    cfg = ExperimentSuiteConfig(
        suite_name="Live Mock",
        seed=5,
        num_agents=6,
        num_rounds=1,
        peer_degree=2,
        tasks_per_track=1,
        conditions=[
            ConditionConfig(
                condition_id="us_k1",
                label="US K1",
                heterogeneity_type=HeterogeneityType.MONOLITHIC_SHARED,
                nationality_type=NationalityType.US_ONLY,
                num_models=1,
            )
        ],
    )
    out_live = run_suite(
        cfg,
        backend=OllamaBackend(model="llama3.1:8b", transport=mock_chat_tx),
        out_dir=tmp_path / "live",
    )
    replayed = run_suite(
        cfg,
        backend=create_backend("replay", jsonl_path=tmp_path / "live" / "runs.jsonl"),
        out_dir=tmp_path / "replay",
    )
    assert out_live["summaries"][0].model_dump() == replayed["summaries"][0].model_dump()

    out_parallel = run_suite(
        cfg,
        backend=OllamaBackend(model="llama3.1:8b", transport=mock_chat_tx),
        out_dir=tmp_path / "parallel",
        workers=4,
    )
    assert [r.model_dump() for r in out_live["records"]] == [
        r.model_dump() for r in out_parallel["records"]
    ]

    # Non-JSON prose from live LLM must fall back cleanly to prev_stance (not NaN/0.99).
    def unparseable_tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        return {"message": {"content": "Free-form reasoning with no JSON block at all."}}

    task = build_ground_truth_tasks()[0]
    agent = generate_agent_population(num_agents=1, seed=42)[0]
    step = _step_live_llm(
        backend=OllamaBackend(model="llama3.1:8b", transport=unparseable_tx),
        cond=cfg.conditions[0],
        task=task,
        agent=agent,
        round_idx=1,
        prev_stance=0.41,
        prev_topics=[task.canonical_topics[0]],
        peer_stances=[0.41],
        model_assigned="ollama/llama3.1:8b",
    )
    assert math.isfinite(step["stance_prob"])
    assert step["stance_prob"] == 0.41
