"""End-to-end CLI subcommand tests for setup, demo, run, analyze, report, and audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from apart_epistemics.cli import main


def test_cli_setup_demo_analyze_report_and_audit(tmp_path: Path) -> None:
    cfg_path = tmp_path / "suite.yaml"
    rc = main(["setup", "--out", str(cfg_path), "--agents", "8", "--rounds", "2", "--tasks", "2"])
    assert rc == 0
    assert cfg_path.exists()

    out_dir = tmp_path / "results"
    site_dir = tmp_path / "site"
    rc = main(["demo", "--config", str(cfg_path), "--out", str(out_dir), "--site", str(site_dir)])
    assert rc == 0
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "numeric_analysis.json").exists()
    assert (out_dir / "figures" / "pages_headline.svg").exists()
    assert (site_dir / "index.html").exists()

    num_data = json.loads((out_dir / "numeric_analysis.json").read_text(encoding="utf-8"))
    assert num_data["max_identity_residual"] < 1e-11
    assert len(num_data["contrasts"]) == 4

    # Re-analyze and re-build report purely from persisted JSONL outputs
    rc = main(["analyze", "--input", str(out_dir)])
    assert rc == 0

    site2 = tmp_path / "site2"
    rc = main(["report", "--input", str(out_dir), "--site", str(site2)])
    assert rc == 0
    assert (site2 / "index.html").exists()

    audit_path = tmp_path / "selection_audit.json"
    rc = main(["audit-selection", "--out", str(audit_path)])
    assert rc == 0
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert "max_accuracy_gap" in audit
    assert "selection_bias_detected" in audit


def test_cli_run_missing_secret_returns_clean_error_and_module_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out_dir = tmp_path / "run_out"
    rc = main(
        [
            "run",
            "--config",
            "configs/default_suite.yaml",
            "--backend",
            "openrouter",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 2

    proc = subprocess.run(
        [sys.executable, "-m", "apart_epistemics", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "demo" in proc.stdout
    assert "audit-selection" in proc.stdout


def test_cli_run_pin_provider_requires_openrouter_and_max_tokens_is_parsed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from apart_epistemics.cli import build_parser

    ns = build_parser().parse_args(["run", "--pin-provider", "OpenAI", "--max-tokens", "2048"])
    assert ns.pin_provider == "OpenAI"
    assert ns.max_tokens == 2048
    assert build_parser().parse_args(["run"]).max_tokens == 1024

    rc = main(["run", "--backend", "synthetic", "--pin-provider", "OpenAI", "--out", str(tmp_path)])
    assert rc == 2
    assert "only supported with --backend openrouter" in capsys.readouterr().err


def test_cli_run_price_flags_default_and_parse() -> None:
    from apart_epistemics.cli import build_parser

    default = build_parser().parse_args(["run"])
    assert (default.price_in, default.price_out) == (0.5, 1.5)
    ns = build_parser().parse_args(["run", "--price-in", "0.1", "--price-out", "0.5"])
    assert (ns.price_in, ns.price_out) == (0.1, 0.5)
