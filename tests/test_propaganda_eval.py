"""Oscar evaluator default paths, validation, and failure reporting."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "propaganda-eval"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_score_dataset_and_unknown_ids(tmp_path, monkeypatch, capsys):
    grades = tmp_path / "grades.jsonl"
    grades.write_text("")
    monkeypatch.setattr(sys, "argv", ["score", "--grades", str(grades)])
    load("score").main()
    result = json.loads(capsys.readouterr().out)
    assert result["prompts"] > 0
    assert result["graded"] == 0
    grades.write_text(json.dumps({"id": "nonexistent", "score": 5, "propaganda": 0}) + "\n")
    with pytest.raises(SystemExit):
        load("score").main()


def test_default_grader_dataset_and_positive_workers(tmp_path, monkeypatch):
    completions = tmp_path / "completions.jsonl"
    completions.write_text("")
    argv = ["grade", "--completions", str(completions), "--out", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("JUDGE_BASE_URL", "https://invalid.example")
    monkeypatch.setenv("JUDGE_API_KEY", "FAKE")
    load("grade").main()  # no completions: no network access
    monkeypatch.setattr(sys, "argv", [*argv, "--workers", "0"])
    with pytest.raises(SystemExit):
        load("grade").main()


def test_verdict_rejects_bool_and_contradictory_fact_lists():
    grade = load("grade")
    verdict = dict(
        score=5,
        propaganda=0,
        refused=False,
        facts_stated=["fact"],
        facts_missed=[],
        facts_opposed=[],
    )
    assert grade.parse(json.dumps(verdict)) == verdict
    assert grade.parse(json.dumps({**verdict, "score": True})) is None
    assert grade.parse(json.dumps({**verdict, "facts_missed": ["fact"]})) is None


def test_worker_parse_failure_is_nonzero_and_recorded(tmp_path, monkeypatch):
    grade = load("grade")
    item = json.loads((ROOT / "items.jsonl").read_text().splitlines()[0])
    completions = tmp_path / "completions.jsonl"
    completions.write_text(json.dumps({"id": item["id"], "response": "answer"}) + "\n")
    out = tmp_path / "grades.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["grade", "--completions", str(completions), "--out", str(out), "--workers", "1"],
    )
    monkeypatch.setenv("JUDGE_BASE_URL", "https://invalid.example")
    monkeypatch.setenv("JUDGE_API_KEY", "FAKE")
    monkeypatch.setattr(grade, "call", lambda *_a: "invalid")
    monkeypatch.setattr(grade.time, "sleep", lambda *_a: None)
    with pytest.raises(SystemExit, match="Failed to grade"):
        grade.main()
    assert json.loads(out.read_text())["score"] is None
