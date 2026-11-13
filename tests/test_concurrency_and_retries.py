"""Concurrency, ordering, budget and retry behaviour for the numeric core and transport."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from apart_epistemics.backends import (
    BudgetExceededError,
    OpenRouterBackend,
    SyntheticEpistemicBackend,
    TransientBackendError,
)
from apart_epistemics.backends.base import post_json
from apart_epistemics.cli import build_parser, main
from apart_epistemics.datasets import build_numeric_core_tasks
from apart_epistemics.engine import run_numeric_core
from apart_epistemics.run_state import BudgetLedger

N_AGENTS = 6


def _run(tmp_path: Path, name: str, backend: Any, **kwargs: Any) -> Any:
    params: dict[str, Any] = {
        "tasks": build_numeric_core_tasks()[:1],
        "conditions": ["C1", "C2", "C3"],
        "n_agents": N_AGENTS,
        "n_rounds": 2,
        "replicates": 1,
        "runs_path": tmp_path / name,
        "backend": backend,
    }
    params.update(kwargs)
    return run_numeric_core(**params)


# ---------------------------------------------------------------- concurrency


def test_parallel_run_matches_sequential_run_exactly(tmp_path: Path) -> None:
    seq = _run(tmp_path, "seq.jsonl", SyntheticEpistemicBackend(seed=3), workers=1)
    par = _run(tmp_path, "par.jsonl", SyntheticEpistemicBackend(seed=3), workers=4)
    assert [r.model_dump() for r in seq] == [r.model_dump() for r in par]
    assert (tmp_path / "seq.jsonl").read_text() == (tmp_path / "par.jsonl").read_text()


def test_calls_really_overlap_but_output_stays_in_agent_order(tmp_path: Path) -> None:
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    class SlowEarlyAgents(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            # Earlier agents finish last, so completion order is the reverse of agent order.
            time.sleep(0.05 * (N_AGENTS - int(kwargs["agent"].agent_id[-2:])))
            try:
                return super().generate_numeric_reply(**kwargs)
            finally:
                with lock:
                    state["active"] -= 1

    backend = SlowEarlyAgents(seed=3)
    recs = _run(tmp_path, "par.jsonl", backend, workers=N_AGENTS, conditions=["C1"], n_rounds=1)
    assert state["peak"] > 1
    ids = [r.agent_id for r in recs]
    assert ids[:N_AGENTS] == [f"agent_{i:02d}" for i in range(N_AGENTS)]
    sequential = _run(
        tmp_path, "seq.jsonl", SyntheticEpistemicBackend(seed=3), conditions=["C1"], n_rounds=1
    )
    assert ids == [r.agent_id for r in sequential]
    lines = (tmp_path / "par.jsonl").read_text().splitlines()
    written = [json.loads(line)["agent_id"] for line in lines]
    assert written == ids


def test_failure_in_one_agent_keeps_finished_calls_and_resume_completes(tmp_path: Path) -> None:
    class FailsOnce(SyntheticEpistemicBackend):
        def __init__(self, target: str) -> None:
            super().__init__(seed=3)
            self.target = target
            self.failed = False

        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["agent"].agent_id == self.target and not self.failed:
                self.failed = True
                time.sleep(0.3)  # let every other agent's call start and finish first
                raise RuntimeError("network down")
            return super().generate_numeric_reply(**kwargs)

    reference = _run(tmp_path, "ref.jsonl", SyntheticEpistemicBackend(seed=3), workers=1)

    with pytest.raises(RuntimeError, match="network down"):
        _run(tmp_path, "run.jsonl", FailsOnce("agent_02"), workers=N_AGENTS)
    saved = [json.loads(x) for x in (tmp_path / "run.jsonl").read_text().splitlines()]
    # Round 1 of C1: every agent except the failing one was paid for and kept.
    assert {r["agent_id"] for r in saved} == {f"agent_0{i}" for i in range(N_AGENTS) if i != 2}

    resumed = _run(tmp_path, "run.jsonl", FailsOnce("none"), workers=4)
    # run_spec embeds the backend class name, which differs for the test double.
    strip = {"run_spec"}
    assert [r.model_dump(exclude=strip) for r in resumed] == [
        r.model_dump(exclude=strip) for r in reference
    ]


def test_budget_cap_holds_under_concurrency(tmp_path: Path) -> None:
    class Costly(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            out = super().generate_numeric_reply(**kwargs)
            out["input_tokens"], out["output_tokens"] = 1_000_000, 0
            return out

    # Each call costs $0.50 at the default price; a $1 cap stops the run long before it ends.
    with pytest.raises(BudgetExceededError):
        _run(tmp_path, "b.jsonl", Costly(seed=3), workers=4, budget_usd=1.0)
    saved = (tmp_path / "b.jsonl").read_text().splitlines()
    # At most the cap plus one in-flight call per worker can have been paid for.
    assert len(saved) <= 2 + 4


def test_workers_must_be_positive(tmp_path: Path) -> None:
    from apart_epistemics.backends import BackendConfigError

    with pytest.raises(BackendConfigError, match="workers"):
        _run(tmp_path, "w.jsonl", SyntheticEpistemicBackend(seed=3), workers=0)


def test_cli_workers_flag_defaults_to_eight_and_rejects_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert build_parser().parse_args(["run"]).workers == 8
    assert main(["run", "--workers", "0", "--out", str(tmp_path)]) == 2
    assert "--workers" in capsys.readouterr().err


# ------------------------------------------------------------------- retries


def _backend(monkeypatch: pytest.MonkeyPatch, transport: Any, **kw: Any) -> OpenRouterBackend:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key-123")
    backend = OpenRouterBackend(model="m", transport=transport, **kw)
    backend.delays = []  # type: ignore[attr-defined]
    backend.retry_sleep = backend.delays.append  # type: ignore[attr-defined,method-assign]
    return backend


def _ok() -> dict[str, Any]:
    return {"choices": [{"message": {"content": "Final estimate: 5"}}]}


def test_transient_errors_are_retried_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] <= 3:
            raise TransientBackendError("HTTP 429")
        return _ok()

    backend = _backend(monkeypatch, tx)
    assert backend.generate_numeric_reply(system_prompt="s", user_prompt="u")["raw_reply"]
    assert calls["n"] == 4  # first attempt + 3 retries
    assert backend.delays == [1.0, 2.0, 4.0]  # type: ignore[attr-defined]


def test_retries_are_capped_and_secrets_stay_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        raise TransientBackendError("HTTP 503 token secret-key-123")

    backend = _backend(monkeypatch, tx)
    with pytest.raises(RuntimeError, match="REDACTED") as exc:
        backend.generate_numeric_reply(system_prompt="s", user_prompt="u")
    assert "secret-key-123" not in str(exc.value)
    assert calls["n"] == 4


def test_non_transient_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        raise RuntimeError("HTTP 400 bad request")

    backend = _backend(monkeypatch, tx)
    with pytest.raises(RuntimeError, match="400"):
        backend.generate_numeric_reply(system_prompt="s", user_prompt="u")
    assert calls["n"] == 1
    assert backend.delays == []  # type: ignore[attr-defined]


def test_every_retry_attempt_reserves_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"n": 0}

    def tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientBackendError("timeout")
        return _ok()

    backend = _backend(monkeypatch, tx)
    backend.budget_ledger = BudgetLedger(tmp_path / "budget.json", cap=10.0, per_request=1.0)
    backend.generate_numeric_reply(system_prompt="s", user_prompt="u")
    assert backend.budget_ledger.requests == 3  # failed attempts may be billed too

    # A retry that would exceed the cap stops with a budget error, not another request.
    calls["n"] = 0
    backend.budget_ledger = BudgetLedger(tmp_path / "tight.json", cap=2.0, per_request=1.0)
    with pytest.raises(BudgetExceededError):
        backend.generate_numeric_reply(system_prompt="s", user_prompt="u")
    assert calls["n"] == 2


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body.encode()


@pytest.mark.parametrize(
    ("raised", "transient"),
    [
        (urllib.error.HTTPError("u", 429, "Too Many", {}, None), True),  # type: ignore[arg-type]
        (urllib.error.HTTPError("u", 500, "Server", {}, None), True),  # type: ignore[arg-type]
        (urllib.error.HTTPError("u", 503, "Unavail", {}, None), True),  # type: ignore[arg-type]
        (urllib.error.HTTPError("u", 400, "Bad", {}, None), False),  # type: ignore[arg-type]
        (urllib.error.HTTPError("u", 401, "Auth", {}, None), False),  # type: ignore[arg-type]
        (TimeoutError("slow"), True),
        (urllib.error.URLError("refused"), True),
    ],
)
def test_post_json_classifies_transient_failures(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, transient: bool
) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise raised

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError) as exc:
        post_json("http://x", {}, {})
    assert isinstance(exc.value, TransientBackendError) is transient


def test_post_json_treats_malformed_body_as_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: _FakeResponse("not json"))
    with pytest.raises(RuntimeError) as exc:
        post_json("http://x", {}, {})
    assert not isinstance(exc.value, TransientBackendError)


def test_budget_ledger_is_exact_under_concurrent_reservations(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json", cap=40.0, per_request=1.0)
    granted: list[int] = []
    refused: list[int] = []

    def reserve() -> None:
        try:
            ledger.reserve()
            granted.append(1)
        except BudgetExceededError:
            refused.append(1)

    threads = [threading.Thread(target=reserve) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 40
    assert len(refused) == 60
    assert ledger.requests == 40
    assert json.loads((tmp_path / "budget.json").read_text())["requests"] == 40
