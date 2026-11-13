"""Tests for backends, HTTP payload contracts, crash resume, and behavioral invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apart_epistemics.backends import (
    AnthropicBackend,
    BackendConfigError,
    BudgetExceededError,
    GeminiBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
    OpenRouterBackend,
    ReplayBackend,
    SyntheticEpistemicBackend,
    create_backend,
)
from apart_epistemics.datasets import build_numeric_core_tasks
from apart_epistemics.engine import (
    analyze_numeric_core,
    build_numeric_prompts,
    run_numeric_core,
    run_suite,
)
from apart_epistemics.models import (
    AgentProfile,
    ConditionConfig,
    ExperimentSuiteConfig,
    HeterogeneityType,
    NationalityType,
)


def test_openrouter_backend_enforces_pinning_no_online_and_safe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(BackendConfigError, match="OPENROUTER_API_KEY"):
        OpenRouterBackend(model="openai/gpt-6-luna")

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter-value")
    captured: dict[str, Any] = {}

    def fake_transport(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "choices": [{"message": {"content": "Final estimate: 4200"}}],
            "provider": "OpenAI",
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "reasoning_tokens": 8},
        }

    backend = OpenRouterBackend(
        model="openai/gpt-6-luna:online",
        pin_provider="OpenAI",
        reasoning_effort="low",
        transport=fake_transport,
    )
    resp = backend.generate_chat(
        system_prompt="You are an assistant.",
        user_prompt="Estimate something.",
        temperature=0.8,
        top_p=0.95,
        max_tokens=200,
    )
    assert resp["text"] == "Final estimate: 4200"
    assert resp["provider"] == "OpenAI"
    assert resp["reasoning_tokens"] == 8
    # Verify :online stripped and allow_fallbacks=False pinned
    assert ":online" not in captured["payload"]["model"]
    assert captured["payload"]["provider"] == {"order": ["OpenAI"], "allow_fallbacks": False}
    assert captured["payload"]["reasoning"] == {"effort": "low"}
    assert captured["payload"]["temperature"] == pytest.approx(0.8)
    assert captured["payload"]["top_p"] == pytest.approx(0.95)
    # Credential must never appear in returned metadata or response
    assert "dummy-openrouter-value" not in json.dumps(resp)


def test_ollama_backend_serializes_temperature_and_handles_connection_failure() -> None:
    captured: dict[str, Any] = {}

    def fake_transport(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": '{"stance_prob": 0.65, "confidence": 0.72, "topics": []}'}}

    ollama = OllamaBackend(model="llama3.1:8b", transport=fake_transport)
    out = ollama.generate_chat("sys", "usr", temperature=0.0, top_p=0.0, max_tokens=128)
    assert "0.65" in out["text"]
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["options"]["temperature"] == 0.0
    assert captured["payload"]["options"]["top_p"] == 0.0

    failing = OllamaBackend(
        model="llama3.1:8b",
        base_url="http://127.0.0.1:59999",
    )
    with pytest.raises(RuntimeError, match="Ollama"):
        failing.generate_chat("sys", "usr", temperature=0.7, top_p=0.95, max_tokens=32)


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

    # Decoding double dissociation: T=0.8 restores lexical variance vs T=0, not topic collapse
    greedy = by_id["us_k1_greedy"]
    default = by_id["us_k1"]
    assert default.pairwise_jaccard_distance - greedy.pairwise_jaccard_distance > 0.08
    assert default.distinct_2 > greedy.distinct_2
    assert default.goldman_effective_sources < 5.0

    # Cross-national & divergent SFT preserve higher topic breadth and Goldman sources than us_k1
    assert by_id["cross_national_k4"].effective_topic_count > default.effective_topic_count
    assert (
        by_id["cross_national_k4"].goldman_effective_sources
        > by_id["us_k4"].goldman_effective_sources
    )
    assert by_id["divergent_sft_pair"].effective_topic_count > default.effective_topic_count

    # Personalization calibration split: peer stays diverse, advisor calibrates much better
    peer = by_id["personalized_peer"]
    advisor = by_id["personalized_advisor"]
    assert peer.effective_topic_count > default.effective_topic_count
    assert peer.confident_wrong_rate > advisor.confident_wrong_rate
    assert peer.brier_score > advisor.brier_score

    # Dual-judge blinded summarizer diagnostics attached
    assert out["topic_audit"]["inter_rater_kappa"] >= 0.70
    assert (tmp_path / "runs.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()

    # Replay backend reproduces exact summaries from saved JSONL
    replay = ReplayBackend.from_jsonl(tmp_path / "runs.jsonl")
    assert replay.record_count > 0
    assert create_backend("synthetic", seed=42) is not None


def test_numeric_prompt_builder_and_crash_resumable_runner(tmp_path: Path) -> None:
    agent = AgentProfile(
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
    sys_c1, usr_c1 = build_numeric_prompts(
        condition_id="C1",
        agent=agent,
        question="Length of river in km?",
        current_x=1500.0,
        pop_stats=None,
        history=[],
    )
    assert "Other people" not in usr_c1
    assert "civil engineer" not in sys_c1

    sys_c2, usr_c2 = build_numeric_prompts(
        condition_id="C2",
        agent=agent,
        question="Length of river in km?",
        current_x=1500.0,
        pop_stats=(2100.0, 1400.0, 3100.0),
        history=[],
    )
    assert "Other people asking this recently gave a median of" in usr_c2
    assert "civil engineer" not in sys_c2

    sys_c3, usr_c3 = build_numeric_prompts(
        condition_id="C3",
        agent=agent,
        question="Length of river in km?",
        current_x=1500.0,
        pop_stats=(2100.0, 1400.0, 3100.0),
        history=[("My current estimate is 1200.", "Final estimate: 1800")],
    )
    assert "civil engineer" in sys_c3
    assert "south_asia" in sys_c3
    assert "1500" not in sys_c3  # Profile excludes current estimate
    assert "1800" in usr_c3

    # Test crash resume + retry-once + miss handling + budget guard
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
                # Force double miss once to verify miss=True fallback preserves agent_log10_before
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

    # Second run against same runs_path must skip all already-written rows (0 new calls!)
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
    assert len(analysis["contrasts"]) == 4  # Holm-corrected 4 core tests (C2/C3 vs C1 x div/err)

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


def test_direct_vendor_adapters_and_replay_reproduction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(BackendConfigError):
        OpenAICompatibleBackend(model="gpt-4o-mini")
    with pytest.raises(BackendConfigError):
        AnthropicBackend(model="claude-3-7-sonnet-latest")
    with pytest.raises(BackendConfigError):
        GeminiBackend(model="gemini-2.5-flash")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-ant-value")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-value")

    def openai_tx(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "Final estimate: 55"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def ant_tx(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": [{"text": "Final estimate: 89"}],
            "usage": {"input_tokens": 12, "output_tokens": 6},
        }

    def gem_tx(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidates": [{"content": {"parts": [{"text": "Final estimate: 144"}]}}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4},
        }

    assert (
        create_backend("openai", transport=openai_tx)
        .generate_numeric_reply(system_prompt="s", user_prompt="u")["raw_reply"]
        .endswith("55")
    )
    assert (
        create_backend("anthropic", transport=ant_tx)
        .generate_numeric_reply(system_prompt="s", user_prompt="u")["raw_reply"]
        .endswith("89")
    )
    assert (
        create_backend("gemini", transport=gem_tx)
        .generate_numeric_reply(system_prompt="s", user_prompt="u")["raw_reply"]
        .endswith("144")
    )

    # Live-chat backend driver + ReplayBackend round-trip reproduction
    def mock_chat_tx(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
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


def test_cross_process_determinism_and_truncated_jsonl_resume(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    outs = []
    for hseed in ("111", "999"):
        sub_out = tmp_path / f"det_{hseed}.jsonl"
        code = (
            "import json; from pathlib import Path; "
            "from apart_epistemics.backends import SyntheticEpistemicBackend; "
            "from apart_epistemics.datasets import build_numeric_core_tasks; "
            "from apart_epistemics.engine import run_numeric_core; "
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

    # Verify truncated trailing line recovery in run_numeric_core
    trunc_path = tmp_path / "trunc.jsonl"
    tasks = build_numeric_core_tasks()[:1]
    backend = SyntheticEpistemicBackend(seed=42)
    full = run_numeric_core(tasks, ["C3"], 4, 2, 1, trunc_path, backend, seed=42)
    lines = trunc_path.read_text(encoding="utf-8").splitlines()
    # Keep first 5 valid lines and corrupt the 6th trailing line
    trunc_path.write_text(
        "\n".join(lines[:5] + ['{"condition_id": "C3", "broken']) + "", encoding="utf-8"
    )
    repaired = run_numeric_core(tasks, ["C3"], 4, 2, 1, trunc_path, backend, seed=42)
    assert len(repaired) == len(full)
    assert [round(r.agent_log10_after, 6) for r in repaired] == [
        round(r.agent_log10_after, 6) for r in full
    ]
