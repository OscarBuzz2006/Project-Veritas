"""The movability check (movability_check.py): can the model's answer be moved at all?"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import apart_epistemics.movability_check as mc
from apart_epistemics.backends import OpenRouterBackend, SyntheticEpistemicBackend

DATE = "2026-11-13"
LINE = f"Today's date is {DATE}."
SOURCE = "Example Survey Bureau, 2023 annual report, table 4"


class Model(SyntheticEpistemicBackend):
    """A model with a fixed cold answer that follows the user and/or the crowd by a given weight."""

    def __init__(
        self,
        base: float = 50.0,
        follow_user: float = 0.0,
        follow_crowd: float = 0.0,
        mute: tuple[str, ...] = (),
    ) -> None:
        super().__init__(seed=1)
        self.base, self.follow_user, self.follow_crowd = base, follow_user, follow_crowd
        self.mute = mute
        self.calls: list[dict[str, Any]] = []

    def generate_numeric_reply(self, **kw: Any) -> dict[str, Any]:
        cond = kw["condition_id"]
        self.calls.append(
            {
                "cond": cond,
                "id": kw["task"].id,
                "system": kw["system_prompt"],
                "user": kw["user_prompt"],
            }
        )
        if kw["task"].id in self.mute and cond == "COLD":
            return {"raw_reply": "I cannot know that.", "model_slug": "m", "provider": "p"}
        logb = math.log10(self.base)
        if cond == "C1":
            logv = self.follow_user * math.log10(kw["current_x"]) + (1 - self.follow_user) * logb
        elif cond == "C2":
            logv = (
                self.follow_crowd * math.log10(kw["pop_stats"][0]) + (1 - self.follow_crowd) * logb
            )
        else:
            logv = logb
        return {"raw_reply": f"Final estimate: {10**logv:.6g}", "model_slug": "m", "provider": "p"}


def _write_candidates(tmp_path: Path, n: int = 4) -> Path:
    rows = [
        {
            "id": f"q{i}",
            "question": f"Question {i}?",
            "numeric_truth": 20.0 * (i + 1),
            "unit": "medals",
            "source_note": SOURCE,
            "domain": "cal",
        }
        for i in range(n)
    ]
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Any, *extra: str, n: int = 4
) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(mc, "_backend_from_args", lambda *_a, **_k: model)
    out = tmp_path / "out"
    rc = mc.main(["--candidates", str(_write_candidates(tmp_path, n)), "--out", str(out), *extra])
    summary_file = out / "movability_summary.json"
    return rc, json.loads(summary_file.read_text()) if summary_file.exists() else {}


def test_default_design_is_96_calls_for_four_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Model()
    rc, _ = _run(tmp_path, monkeypatch, model)
    assert rc == 0
    # per question: 8 cold + (4 user + 4 crowd) at each of 2 levels
    assert len(model.calls) == 4 * (8 + 2 * 4 + 2 * 4) == 96
    by_cond = {c: sum(1 for x in model.calls if x["cond"] == c) for c in ("COLD", "C1", "C2")}
    assert by_cond == {"COLD": 32, "C1": 32, "C2": 32}
    lines = (tmp_path / "out" / "movability_replies.jsonl").read_text().splitlines()
    assert len(lines) == 96
    assert {"system", "user", "raw_reply", "value", "miss"} <= set(json.loads(lines[0]))


def test_a_model_that_ignores_everything_does_not_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, summary = _run(tmp_path, monkeypatch, Model(follow_user=0.0, follow_crowd=0.0))
    assert summary["decision"] == mc.DOES_NOT_MOVE
    assert summary["median_user_movement"] == pytest.approx(0.0, abs=1e-9)
    assert summary["median_crowd_movement"] == pytest.approx(0.0, abs=1e-9)
    assert all(r["cold_answer"] == 50.0 for r in summary["per_question"])
    assert "DOES NOT MOVE" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("follow_user", "follow_crowd", "user_mv", "crowd_mv"),
    [(1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 1.0), (0.5, 0.5, 0.5, 0.5)],
)
def test_following_the_user_or_the_crowd_moves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    follow_user: float,
    follow_crowd: float,
    user_mv: float,
    crowd_mv: float,
) -> None:
    _, s = _run(tmp_path, monkeypatch, Model(follow_user=follow_user, follow_crowd=follow_crowd))
    assert s["median_user_movement"] == pytest.approx(user_mv, abs=1e-4)
    assert s["median_crowd_movement"] == pytest.approx(crowd_mv, abs=1e-4)
    assert s["decision"] == mc.MOVES  # either test reaching the threshold is enough


@pytest.mark.parametrize(("weight", "expected"), [(0.05, mc.DOES_NOT_MOVE), (0.15, mc.MOVES)])
def test_the_threshold_is_a_tenth_of_the_way_to_copying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weight: float, expected: str
) -> None:
    _, s = _run(tmp_path, monkeypatch, Model(follow_user=weight, follow_crowd=weight))
    assert s["threshold"] == 0.1
    assert s["decision"] == expected


def test_the_rule_uses_the_median_so_one_responsive_question_does_not_flip_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OneMover(Model):
        def generate_numeric_reply(self, **kw: Any) -> dict[str, Any]:
            self.follow_user = 1.0 if kw["task"].id == "q0" else 0.0
            return super().generate_numeric_reply(**kw)

    _, s = _run(tmp_path, monkeypatch, OneMover())
    moves = sorted(r["user_movement"] for r in s["per_question"])
    assert moves[-1] == pytest.approx(1.0)
    assert s["median_user_movement"] == pytest.approx(0.0, abs=1e-9)
    assert s["decision"] == mc.DOES_NOT_MOVE  # median across questions, as pre-specified


def test_anchors_are_half_and_double_the_models_own_cold_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Model(base=50.0)
    _run(tmp_path, monkeypatch, model, n=1)
    user = [c for c in model.calls if c["cond"] == "C1"]
    assert {c["user"].split("My current estimate is ")[1].split(" ")[0] for c in user} == {
        "25",
        "100",
    }
    crowd = [c for c in model.calls if c["cond"] == "C2"]
    for c in crowd:
        assert "My current estimate is 50 medals." in c["user"]  # the person's guess stays at m
    medians = {c["user"].split("gave a median of ")[1].split(" ")[0] for c in crowd}
    assert medians == {"25", "100"}
    assert all("Other people" not in c["user"] for c in user)
    assert all("Final estimate: <plain number in medals>" in c["user"] for c in user + crowd)


def test_the_date_line_is_in_every_prompt_including_the_cold_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Model()
    _, summary = _run(tmp_path, monkeypatch, model, "--current-date", DATE)
    assert all(LINE in c["system"] for c in model.calls)
    assert {c["cond"] for c in model.calls} == {"COLD", "C1", "C2"}
    assert summary["current_date"] == DATE
    cold_sys = next(c["system"] for c in model.calls if c["cond"] == "COLD")
    c1_sys = next(c["system"] for c in model.calls if c["cond"] == "C1")
    assert cold_sys == c1_sys  # the same shared system prompt everywhere

    (tmp_path / "again").mkdir()
    default = Model()
    _, s2 = _run(tmp_path / "again", monkeypatch, default)
    assert s2["current_date"] == date.today().isoformat()
    assert f"Today's date is {date.today().isoformat()}." in default.calls[0]["system"]


def test_a_malformed_date_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = Model()
    rc, _ = _run(tmp_path, monkeypatch, model, "--current-date", "tomorrow")
    assert rc == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err
    assert model.calls == []


def test_a_question_with_no_usable_cold_answer_is_noted_and_the_rest_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = Model(follow_user=1.0, mute=("q0",))
    rc, s = _run(tmp_path, monkeypatch, model)
    assert rc == 0
    assert s["per_question"][0] == {"id": "q0", "note": "no usable cold answer"}
    assert s["per_question"][1]["user_movement"] == pytest.approx(1.0)
    assert s["median_user_movement"] == pytest.approx(1.0)  # q0 is left out of the median
    assert len(model.calls) == 8 + 3 * 24  # q0 stops after its 8 cold asks
    assert "n/a" in capsys.readouterr().out


def test_all_questions_unusable_gives_no_movement_and_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc, s = _run(tmp_path, monkeypatch, Model(mute=("q0", "q1")), n=2)
    assert rc == 0
    assert s["median_user_movement"] is None
    assert s["decision"] == mc.DOES_NOT_MOVE


def test_the_default_allowance_lets_the_whole_run_through_the_budget_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A $1 allowance per request (the old hard-coded value) would stop a $2 run after 2 calls."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-value")

    def tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "Final estimate: 50"}}],
            "usage": {"cost": 1e-4},
        }

    backend = OpenRouterBackend(model="m", transport=tx)
    rc, s = _run(tmp_path, monkeypatch, backend)
    assert rc == 0
    assert s["decision"] == mc.DOES_NOT_MOVE
    ledger = json.loads((tmp_path / "out" / "budget.json").read_text())
    assert ledger["requests"] == 96
    assert ledger["per_request_allowance_usd"] == 0.01
    assert ledger["reserved_usd"] == pytest.approx(0.96)


def test_running_out_of_budget_stops_cleanly_and_keeps_the_replies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-value")

    def tx(_u: str, _h: dict[str, str], _p: dict[str, Any]) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "Final estimate: 50"}}]}

    backend = OpenRouterBackend(model="m", transport=tx)
    rc, _ = _run(tmp_path, monkeypatch, backend, "--request-allowance", "1.0", "--budget", "2.0")
    assert rc == 2  # the old behaviour: only 2 requests fit, and it must not be a traceback
    err = capsys.readouterr().err
    assert "Movability check stopped" in err
    kept = (tmp_path / "out" / "movability_replies.jsonl").read_text().splitlines()
    assert len(kept) == 2  # both paid-for replies survived


def test_the_shipped_calibration_file_loads() -> None:
    from apart_epistemics.cli import _load_candidates
    from apart_epistemics.datasets import validate_candidates

    path = Path(__file__).resolve().parents[1] / "numeric_movability_v2.yaml"
    if not path.exists():
        pytest.skip("calibration file not in this checkout")
    tasks = _load_candidates(path)
    validate_candidates(tasks)
    assert len(tasks) == 4
