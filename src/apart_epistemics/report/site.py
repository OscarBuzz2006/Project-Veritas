"""Self-contained interactive Editorial Quantitative Lab Notebook site builder."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any

from apart_epistemics.models import ConditionSummary, NumericEstimateRecord, RoundRecord
from apart_epistemics.report.figures import generate_figures


def _safe_json_for_html(payload: Any) -> str:
    """Serialize JSON safely for inline `<script type='application/json'>` embedding."""
    raw = json.dumps(payload, separators=(",", ":"))
    return raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _sample_trace_specimens(
    records: list[RoundRecord],
    per_condition: int = 3,
) -> list[dict[str, Any]]:
    """Select final-round specimen traces per condition for qualitative inspection."""
    by_cond: dict[str, list[RoundRecord]] = defaultdict(list)
    for rec in records:
        by_cond[rec.condition_id].append(rec)

    specimens: list[dict[str, Any]] = []
    for recs in by_cond.values():
        max_r = max((r.round_idx for r in recs), default=0)
        final_recs = [r for r in recs if r.round_idx == max_r]
        seen_models: set[str] = set()
        chosen: list[RoundRecord] = []
        for r in final_recs:
            if r.model_assigned not in seen_models or len(chosen) < per_condition:
                chosen.append(r)
                seen_models.add(r.model_assigned)
            if len(chosen) >= per_condition:
                break
        for r in chosen:
            specimens.append(
                {
                    "condition_id": r.condition_id,
                    "task_id": r.task_id,
                    "task_kind": r.task_kind.value,
                    "round_idx": r.round_idx,
                    "agent_id": r.agent_id,
                    "agent_region": r.agent_region,
                    "model_assigned": r.model_assigned,
                    "model_family": r.model_family,
                    "extracted_topics": r.extracted_topics,
                    "stance_prob": r.stance_prob,
                    "confidence": r.confidence,
                    "assistant_response_text": r.assistant_response_text,
                    "agent_response_text": r.agent_response_text,
                }
            )
    return specimens


def build_static_site(
    suite_result: dict[str, Any],
    numeric_analysis: dict[str, Any],
    site_dir: Path,
    figures_dir: Path | None = None,
    numeric_records: list[NumericEstimateRecord] | None = None,
) -> Path:
    """Assemble the self-contained interactive static site in `site_dir`."""
    site_dir.mkdir(parents=True, exist_ok=True)
    site_figs = site_dir / "figures"
    site_figs.mkdir(parents=True, exist_ok=True)

    if figures_dir and figures_dir.exists() and figures_dir.resolve() != site_figs.resolve():
        for svg in figures_dir.glob("*.svg"):
            shutil.copy2(svg, site_figs / svg.name)
    else:
        generate_figures(suite_result, numeric_analysis, site_figs)

    summaries: list[ConditionSummary] = suite_result["summaries"]
    comparisons = suite_result.get("comparisons", [])
    topic_audit = suite_result.get("topic_audit", {})
    records: list[RoundRecord] = suite_result.get("records", [])

    clean_audit = {
        "canonical_topics": topic_audit.get("canonical_topics", []),
        "inter_rater_kappa": topic_audit.get("inter_rater_kappa", 0.0),
        "reliability_warning": topic_audit.get("reliability_warning", False),
        "n_evaluated": topic_audit.get("n_evaluated", len(records)),
    }

    payload = {
        "metadata": {
            "title": "Epistemic Prism",
            "subtitle": "Multi-Agent Model Diversity, Decoding Illusions, and Crowd Wisdom",
            "n_conditions": len(summaries),
            "n_track4_records": len(records),
            "n_numeric_records": len(numeric_records) if numeric_records else 0,
        },
        "summaries": [s.model_dump() for s in summaries],
        "comparisons": [c.model_dump() if hasattr(c, "model_dump") else c for c in comparisons],
        "topic_audit": clean_audit,
        "numeric_analysis": {
            "max_identity_residual": numeric_analysis.get("max_identity_residual", 0.0),
            "h4_assistant_agent_corr": numeric_analysis.get("h4_assistant_agent_corr", 0.0),
            "by_condition_final": numeric_analysis.get("by_condition_final", {}),
            "condition_round_metrics": numeric_analysis.get("condition_round_metrics", []),
            "contrasts": [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in numeric_analysis.get("contrasts", [])
            ],
        },
        "trace_specimens": _sample_trace_specimens(records, per_condition=3),
    }

    data_dir = site_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    tpl_ref = resources.files("apart_epistemics") / "templates" / "dashboard.html"
    template_html = tpl_ref.read_text(encoding="utf-8")
    rendered = template_html.replace("__EPISTEMIC_DATA_JSON__", _safe_json_for_html(payload))

    index_path = site_dir / "index.html"
    index_path.write_text(rendered, encoding="utf-8")
    return index_path
