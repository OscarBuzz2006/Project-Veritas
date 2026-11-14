"""Turn saved run records into the numbers the paper reports (Page's theorem, H4, contrasts)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from apart_epistemics.datasets import generate_agent_population
from apart_epistemics.engine.contrasts import compute_paired_holm_contrasts
from apart_epistemics.engine.h4 import compute_h4_slopes
from apart_epistemics.metrics import compute_page_decomposition, pearson_corr
from apart_epistemics.models import NumericEstimateRecord, TaskItem

# A "cell" is one (condition, question, replicate, round): the group of agents measured together.
CellKey = tuple[str, str, int, int]
CellMetrics = dict[CellKey, dict[str, float]]

# The per-cell measurements that get averaged for reporting.
REPORTED_METRICS = ("crowd_error", "avg_individual_error", "diversity", "iqr", "miss_rate")

# The four comparisons the paper tests: (name, metric, baseline condition, other condition).
CORE_CONTRASTS = [
    ("c2_vs_c1_diversity", "diversity", "C1", "C2"),
    ("c2_vs_c1_crowd_error", "crowd_error", "C1", "C2"),
    ("c3_vs_c1_diversity", "diversity", "C1", "C3"),
    ("c3_vs_c1_crowd_error", "crowd_error", "C1", "C3"),
]

H4_INTERPRETATION = (
    "Legacy pooled correlation, kept for the dashboard: mechanically coupled by the "
    "update rule and pooled across questions. Use h4_slopes instead."
)


def analyze_numeric_core(
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
) -> dict[str, Any]:
    """Compute Page decomposition per cell, H4 correlation, and 4 Holm-corrected contrasts."""
    truth_by_qid = {t.id: math.log10(t.numeric_truth) for t in tasks if t.numeric_truth is not None}
    by_cell = _group_by_cell(records)
    spec = _single_run_spec(records)

    expected_cells, complete_cells = _expected_and_complete_cells(spec, by_cell)
    complete = (
        bool(expected_cells) and set(by_cell) == expected_cells and complete_cells == expected_cells
    )
    _check_ground_truth(spec, records, truth_by_qid)

    cell_metrics, max_resid = _score_cells(by_cell, truth_by_qid)
    max_round = spec.get("n_rounds", max((r.round_idx for r in records), default=1))
    condition_round_metrics = _condition_round_metrics(cell_metrics)

    return {
        "completion_status": "complete" if complete else "incomplete_or_unverified",
        "missing_or_incomplete_cells": len(expected_cells - complete_cells),
        "h4_interpretation": H4_INTERPRETATION,
        "h4_slopes": compute_h4_slopes(records, tasks),
        "by_stratum": _stratum_report(records, tasks, cell_metrics, max_round, complete),
        "max_identity_residual": max_resid,
        "h4_assistant_agent_corr": round(_assistant_agent_corr(records), 4),
        "by_condition_final": {
            row["condition_id"]: row
            for row in condition_round_metrics
            if complete and row["round_idx"] == max_round
        },
        "condition_round_metrics": condition_round_metrics,
        "contrasts": _core_contrasts(cell_metrics, max_round, complete),
    }


def _group_by_cell(
    records: list[NumericEstimateRecord],
) -> dict[CellKey, list[NumericEstimateRecord]]:
    """Group the rows so that each cell holds one row per agent."""
    by_cell: dict[CellKey, list[NumericEstimateRecord]] = defaultdict(list)
    for rec in records:
        by_cell[(rec.condition_id, rec.question_id, rec.replicate, rec.round_idx)].append(rec)
    return by_cell


def _single_run_spec(records: list[NumericEstimateRecord]) -> dict[str, Any]:
    """The run settings, if every row agrees on them; otherwise empty (cannot be verified)."""
    specs = {json.dumps(r.run_spec, sort_keys=True) for r in records}
    return records[0].run_spec if records and len(specs) == 1 else {}


def _expected_and_complete_cells(
    spec: dict[str, Any], by_cell: dict[CellKey, list[NumericEstimateRecord]]
) -> tuple[set[CellKey], set[CellKey]]:
    """Which cells the run should contain, and which of those have every agent present."""
    expected_agents = (
        {a.agent_id for a in generate_agent_population(spec["n_agents"], seed=spec["seed"])}
        if spec
        else set()
    )
    expected_cells = {
        (c, t["id"], rep, rnd)
        for c in spec.get("conditions", [])
        for t in spec.get("tasks", [])
        for rep in range(spec.get("replicates", 0))
        for rnd in range(1, spec.get("n_rounds", 0) + 1)
    }
    # A cell counts as complete only if exactly the expected agents are in it, no more or less.
    complete_cells = {
        key
        for key, rows in by_cell.items()
        if expected_agents
        and len(rows) == len(expected_agents)
        and {r.agent_id for r in rows} == expected_agents
    }
    return expected_cells, complete_cells


def _check_ground_truth(
    spec: dict[str, Any], records: list[NumericEstimateRecord], truth_by_qid: dict[str, float]
) -> None:
    """Stop if the answers being scored against are not the ones the run was made with."""
    if spec and any(
        truth_by_qid.get(t["id"]) != math.log10(t["numeric_truth"]) for t in spec["tasks"]
    ):
        raise RuntimeError("Analysis ground truth differs from the saved run specification.")
    if any(r.question_id not in truth_by_qid for r in records):
        raise RuntimeError("Cannot analyze records without their question ground truth.")


def _score_cells(
    by_cell: dict[CellKey, list[NumericEstimateRecord]], truth_by_qid: dict[str, float]
) -> tuple[CellMetrics, float]:
    """Page's decomposition of each cell, plus the largest rounding error seen in its identity."""
    cell_metrics: CellMetrics = {}
    max_resid = 0.0
    for key, cell_recs in by_cell.items():
        _, qid, _, _ = key
        decomp = compute_page_decomposition(
            [r.agent_log10_after for r in cell_recs],
            log_truth=truth_by_qid[qid],
        )
        decomp["miss_rate"] = sum(1 for r in cell_recs if r.miss) / len(cell_recs)
        max_resid = max(max_resid, decomp["identity_residual"])
        cell_metrics[key] = decomp
    return cell_metrics, max_resid


def _assistant_agent_corr(records: list[NumericEstimateRecord]) -> float:
    """Correlation between the model's answers and the agents' updated beliefs (legacy H4)."""
    valid = [r for r in records if not r.miss and r.assistant_log10 is not None]
    return pearson_corr(
        [float(r.assistant_log10) for r in valid if r.assistant_log10 is not None],
        [r.agent_log10_after for r in valid],
    )


def _mean_metrics(cells: list[dict[str, float]]) -> dict[str, float]:
    """Average each reported metric over the given cells."""
    n = len(cells)
    return {key: round(sum(m[key] for m in cells) / n, 6) for key in REPORTED_METRICS}


def _condition_round_metrics(cell_metrics: CellMetrics) -> list[dict[str, Any]]:
    """Per condition and round: each metric averaged over all questions and replicates."""
    by_cond_round: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for (cond_id, _qid, _rep, r_idx), m in cell_metrics.items():
        by_cond_round[(cond_id, r_idx)].append(m)
    return [
        {"condition_id": cond_id, "round_idx": r_idx, **_mean_metrics(ms)}
        for (cond_id, r_idx), ms in sorted(by_cond_round.items())
    ]


def _core_contrasts(cell_metrics: CellMetrics, max_round: int, complete: bool) -> Any:
    """The four planned comparisons: final-round values, Holm multiple-test correction."""

    def final_round_metric(cond_id: str, metric: str) -> dict[tuple[str, int], float]:
        # One value per (question, replicate), so C1 and C2/C3 can be paired up.
        return {
            (qid, rep): m[metric]
            for (c, qid, rep, r_idx), m in cell_metrics.items()
            if complete and c == cond_id and r_idx == max_round
        }

    return compute_paired_holm_contrasts(
        CORE_CONTRASTS,
        final_round_metric,
        seed=42,
        effect_decimals=6,
        include_empty=False,
    )


def _stratum_report(
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
    cell_metrics: CellMetrics,
    max_round: int,
    complete: bool,
) -> dict[str, Any]:
    """Slopes and final-round metrics for each question stratum (e.g. high/low spread)."""
    by_stratum: dict[str, list[TaskItem]] = defaultdict(list)
    for t in tasks:
        if t.selection_stratum:
            by_stratum[t.selection_stratum].append(t)

    report: dict[str, Any] = {}
    for name, stratum_tasks in sorted(by_stratum.items()):
        qids = {t.id for t in stratum_tasks}
        report[name] = {
            "questions": sorted(qids),
            "by_condition_final": (
                _stratum_final(cell_metrics, qids, max_round) if complete else {}
            ),
            "h4_slopes": compute_h4_slopes(
                [r for r in records if r.question_id in qids], stratum_tasks
            ),
        }
    return report


def _stratum_final(
    cell_metrics: CellMetrics, qids: set[str], max_round: int
) -> dict[str, dict[str, Any]]:
    """Final-round averages per condition, using only the questions in one stratum."""
    final: dict[str, dict[str, Any]] = {}
    for cond in sorted({c for c, _, _, _ in cell_metrics}):
        cells = [
            m
            for (c, q, _rep, r), m in cell_metrics.items()
            if c == cond and q in qids and r == max_round
        ]
        if cells:
            final[cond] = {"n_cells": len(cells), **_mean_metrics(cells)}
    return final
