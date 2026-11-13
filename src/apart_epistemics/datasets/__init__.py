"""Dual-track benchmark datasets, stratified global agents, and selection-bias audit."""

from __future__ import annotations

from apart_epistemics.datasets.population import (
    MODEL_POOLS,
    assign_models_for_condition,
    audit_question_selection_bias,
    generate_agent_population,
)
from apart_epistemics.datasets.selection import (
    SelectionError,
    run_cold_selection,
    validate_candidates,
)
from apart_epistemics.datasets.task_bank import (
    build_ground_truth_tasks,
    build_numeric_core_tasks,
    build_open_debate_tasks,
)

__all__ = [
    "MODEL_POOLS",
    "SelectionError",
    "assign_models_for_condition",
    "audit_question_selection_bias",
    "build_ground_truth_tasks",
    "build_numeric_core_tasks",
    "build_open_debate_tasks",
    "generate_agent_population",
    "run_cold_selection",
    "validate_candidates",
]
