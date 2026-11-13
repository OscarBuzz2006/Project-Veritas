"""Regressions for experiment validity, persistence, and provider safety."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import pytest

from apart_epistemics.backends import (
    BackendConfigError,
    BudgetExceededError,
    GeminiBackend,
    OpenRouterBackend,
    ReplayBackend,
    SyntheticEpistemicBackend,
)
from apart_epistemics.cli import _run_both_tracks, main
from apart_epistemics.datasets import build_numeric_core_tasks
from apart_epistemics.engine import analyze_numeric_core, run_numeric_core, run_suite
from apart_epistemics.models import ConditionConfig, ExperimentSuiteConfig
from apart_epistemics.parsers import parse_structured_belief
from apart_epistemics.run_state import configure_budget
from apart_epistemics.summarizer import evaluate_dual_judge_topics


def config() -> ExperimentSuiteConfig:
    return ExperimentSuiteConfig(
        num_agents=4,
        num_rounds=1,
        tasks_per_track=1,
        conditions=[
            ConditionConfig(
                condition_id="mix",
                label="Mix",
                heterogeneity_type="multi_lab_mix",
                num_models=2,
                models=["lab/model-a", "lab/model-b"],
            ),
        ],
    )


def chat_response(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "provider": "test-provider",
        "choices": [
            {
                "message": {
                    "content": '{"stance_prob":0.8,"confidence":0.8,"topics":["evidence"]} '
                    "Evidence matters. Consider evidence."
                }
            }
        ],
    }


def test_live_routing_actual_identity_and_prompt_parse(tmp_path: Path) -> None:
    requests = []

    def transport(_url, _headers, payload):
        requests.append(payload)
        return chat_response(payload["model"])

    backend = OpenRouterBackend(model="agent-simulator", api_key="FAKE", transport=transport)
    result = run_suite(config(), backend, tmp_path)
    assistant_requests = [
        p for p in requests if p["messages"][0]["content"].startswith("You are model")
    ]
    assert {p["model"] for p in assistant_requests} == {"lab/model-a", "lab/model-b"}
    assert all(r.stance_prob == 0.8 for r in result["records"])
    final = [r for r in result["records"] if r.round_idx == 1]
    assert {r.metadata["assistant_model_actual"] for r in final} == {"lab/model-a", "lab/model-b"}
    assert all(r.metadata["agent_model_actual"] == "agent-simulator" for r in final)
    assert all(r.metadata["parse_status"] == "parsed" for r in final)
    assert all(r.metadata["agent_system_prompt"] for r in final)


@pytest.mark.parametrize("suffix", ["\nTwo sentences. With prose.", "", " prose with {braces}"])
def test_prompted_json_and_nonfinite_validation(suffix: str) -> None:
    parsed = parse_structured_belief('{"stance_prob":0.8,"confidence":0.9}' + suffix)
    assert parsed["stance_prob"] == 0.8
    assert parsed["parse_status"] == "parsed"
    assert parse_structured_belief('{"stance_prob":NaN}')["parse_status"] == "invalid"


def numeric(path: Path, **overrides):
    args = dict(
        tasks=build_numeric_core_tasks()[:1],
        conditions=["C1", "C2"],
        n_agents=4,
        n_rounds=2,
        replicates=1,
        runs_path=path,
        backend=SyntheticEpistemicBackend(seed=42),
        seed=42,
        alpha=0.3,
    )
    args.update(overrides)
    return run_numeric_core(**args)


@pytest.mark.parametrize(
    "change",
    [
        dict(seed=99),
        dict(alpha=1),
        dict(n_agents=5),
        dict(backend=SyntheticEpistemicBackend(seed=99)),
    ],
)
def test_resume_rejects_changed_experiment_without_touching_data(tmp_path, change):
    path = tmp_path / "numeric.jsonl"
    numeric(path)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="manifest"):
        numeric(path, **change)
    assert path.read_bytes() == original


def test_partial_results_cannot_masquerade_as_final_or_zero_effect(tmp_path):
    records = numeric(tmp_path / "numeric.jsonl")
    tasks = build_numeric_core_tasks()[:1]
    assert analyze_numeric_core(records, tasks)["completion_status"] == "complete"
    for partial in [
        records[:-1],
        [r for r in records if r.condition_id == "C1" or r.round_idx == 1],
    ]:
        result = analyze_numeric_core(partial, tasks)
        assert result["completion_status"] == "incomplete_or_unverified"
        assert result["by_condition_final"] == {}
        assert result["contrasts"] == []
    with pytest.raises(RuntimeError, match="truth"):
        analyze_numeric_core(records, [tasks[0].model_copy(update={"numeric_truth": 2.0})])


def test_zero_budget_prevents_all_http_calls_and_reservations_persist(tmp_path):
    calls = []

    def transport(_u, _h, p):
        calls.append(p)
        return chat_response(p["model"])

    backend = OpenRouterBackend(model="agent", api_key="FAKE", transport=transport)
    with pytest.raises(BudgetExceededError):
        _run_both_tracks(config(), backend, tmp_path / "zero", budget_usd=0)
    assert calls == []
    configure_budget(backend, tmp_path / "ledger", 2, 1)
    backend.for_model("other").generate_chat("s", "u")
    # Reconstructing the backend/ledger must not grant a fresh allowance.
    fresh = OpenRouterBackend(model="agent", api_key="FAKE", transport=transport)
    configure_budget(fresh, tmp_path / "ledger", 2, 1)
    fresh.generate_chat("s", "u")
    with pytest.raises(BudgetExceededError):
        fresh.generate_chat("s", "u")
    assert len(calls) == 2
    assert json.loads((tmp_path / "ledger/budget.json").read_text())["reserved_usd"] == 2
    assert len((tmp_path / "ledger/provider_requests.jsonl").read_text().splitlines()) == 2


def test_failed_requests_consume_allowance_and_tracebacks_are_redacted(tmp_path):
    def fail(*_args):
        raise RuntimeError("url?key=AUDIT_FAKE_SECRET")

    backend = OpenRouterBackend(model="agent", api_key="AUDIT_FAKE_SECRET", transport=fail)
    configure_budget(backend, tmp_path, 1)
    try:
        backend.generate_chat("s", "u")
    except RuntimeError:
        assert "AUDIT_FAKE_SECRET" not in traceback.format_exc()
    else:
        pytest.fail("Expected transport failure")
    with pytest.raises(BudgetExceededError):
        backend.generate_chat("s", "u")


def test_gemini_uses_header_instead_of_secret_url():
    def transport(url, headers, _payload):
        assert "SECRET" not in url
        assert headers["x-goog-api-key"] == "SECRET"
        return {}

    GeminiBackend(api_key="SECRET", transport=transport).generate_chat("s", "u")


def test_track4_preserves_existing_and_incremental_records(tmp_path):
    calls = 0

    def transport(_u, _h, p):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("offline failure")
        return chat_response(p["model"])

    backend = OpenRouterBackend(model="agent", api_key="FAKE", transport=transport)
    with pytest.raises(RuntimeError, match="offline failure"):
        run_suite(config(), backend, tmp_path)
    original = (tmp_path / "runs.jsonl").read_bytes()
    assert len(original.splitlines()) == 2
    with pytest.raises(RuntimeError, match="already exists"):
        run_suite(config(), backend, tmp_path)
    assert (tmp_path / "runs.jsonl").read_bytes() == original
    assert calls == 3


def test_seed_labels_do_not_verify_themselves(tmp_path):
    result = run_suite(config(), SyntheticEpistemicBackend(), tmp_path)
    for i, record in enumerate(result["records"]):
        record.agent_response_text = "Meaningless unrelated text."
        record.extracted_topics = ["topic_alpha" if i % 2 else "topic_beta"]
    audit = evaluate_dual_judge_topics(result["records"])
    assert audit["inter_rater_kappa"] is None
    assert audit["reliability_warning"]
    # Compare full label sets, not only their first element.
    audit = evaluate_dual_judge_topics(
        result["records"],
        judge_a_override=lambda _t, _v: ["same", "different_a"],
        judge_b_override=lambda _t, _v: ["same", "different_b"],
    )
    assert audit["inter_rater_kappa"] < 0.7


def test_unsupported_replay_is_rejected_before_output(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(BackendConfigError, match="replay"):
        numeric(out / "numeric.jsonl", backend=ReplayBackend([]))
    assert not out.exists()
    source = tmp_path / "empty.jsonl"
    source.write_text("")
    assert (
        main(["run", "--backend", "replay", "--replay-jsonl", str(source), "--out", str(out)]) == 2
    )
    assert not out.exists()


def test_live_placeholder_pools_rejected_before_output(tmp_path):
    cfg = config()
    cfg.conditions[0].models = []
    backend = OpenRouterBackend(model="agent", api_key="FAKE")
    with pytest.raises(BackendConfigError, match="explicit"):
        run_suite(cfg, backend, tmp_path)
    assert not (tmp_path / "runs.jsonl").exists()
