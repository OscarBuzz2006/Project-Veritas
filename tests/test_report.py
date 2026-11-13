"""Tests for SVG figure generation and self-contained editorial static dashboard builder."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from apart_epistemics.backends import SyntheticEpistemicBackend
from apart_epistemics.datasets import build_numeric_core_tasks
from apart_epistemics.engine import analyze_numeric_core, run_numeric_core, run_suite
from apart_epistemics.models import (
    ConditionConfig,
    ExperimentSuiteConfig,
    HeterogeneityType,
    NationalityType,
)
from apart_epistemics.report import build_static_site, generate_figures


def _mini_suite_and_numeric(tmp_path: Path):
    cfg = ExperimentSuiteConfig(
        suite_name="Report Test Suite",
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
    backend = SyntheticEpistemicBackend(seed=42)
    suite_res = run_suite(cfg, backend=backend, out_dir=tmp_path / "suite")
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


def test_generate_figures_produces_valid_svgs(tmp_path: Path) -> None:
    suite_res, num_analysis, _ = _mini_suite_and_numeric(tmp_path)
    fig_dir = tmp_path / "figures"
    paths = generate_figures(suite_res, num_analysis, fig_dir)

    expected = {
        "dissociation_decoding.svg",
        "epistemic_quadrant.svg",
        "round_trajectories.svg",
        "pages_headline.svg",
    }
    assert {p.name for p in paths} == expected
    for svg_path in paths:
        raw = svg_path.read_text(encoding="utf-8")
        assert "NaN" not in raw
        assert "undefined" not in raw
        root = ET.fromstring(raw)
        assert root.tag.endswith("svg")
        assert root.attrib.get("viewBox")


def test_build_static_site_self_contained_and_safe(tmp_path: Path) -> None:
    suite_res, num_analysis, num_recs = _mini_suite_and_numeric(tmp_path)
    # Inject XSS-like sequence into one record to ensure safe JSON embedding.
    suite_res["records"][0].agent_response_text += " </script><script>window.__xss=1</script>"

    fig_dir = tmp_path / "figures"
    generate_figures(suite_res, num_analysis, fig_dir)
    site_dir = tmp_path / "site"
    index_path = build_static_site(
        suite_result=suite_res,
        numeric_analysis=num_analysis,
        site_dir=site_dir,
        figures_dir=fig_dir,
        numeric_records=num_recs,
    )

    assert index_path.exists()
    html = index_path.read_text(encoding="utf-8")

    # Zero external CDNs
    assert "https://cdn." not in html
    assert "fonts.googleapis" not in html
    assert '<script src="http' not in html
    assert "window.__xss=1" not in html or "\\u003c/script\\u003e" in html

    for section in (
        "masthead-banner",
        "headline-findings",
        "epistemic-quadrant",
        "decoding-dissociation",
        "round-trajectory-explorer",
        "condition-score-table",
        "trace-specimen-drawer",
    ):
        assert f'data-section="{section}"' in html

    m = re.search(
        r'<script id="epistemic-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert m is not None
    embedded = json.loads(m.group(1))
    assert len(embedded["summaries"]) == 6
    assert embedded["numeric_analysis"]["max_identity_residual"] < 1e-11
    assert len(embedded["trace_specimens"]) > 0
    assert (site_dir / "figures" / "pages_headline.svg").exists()
    assert (site_dir / "data" / "benchmark_summary.json").exists()
