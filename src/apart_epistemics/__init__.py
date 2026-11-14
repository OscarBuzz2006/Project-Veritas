"""Epistemic Prism: simulation and testing suite for AI-mediated epistemic diversity."""

from __future__ import annotations

from apart_epistemics.backends import (
    BackendConfigError,
    BudgetExceededError,
    ReplayBackend,
    SyntheticEpistemicBackend,
    create_backend,
)
from apart_epistemics.datasets import (
    audit_question_selection_bias,
    build_ground_truth_tasks,
    build_numeric_core_tasks,
    build_open_debate_tasks,
    generate_agent_population,
)
from apart_epistemics.engine import (
    analyze_numeric_core,
    build_numeric_prompts,
    format_prompt_number,
    run_numeric_core,
    run_suite,
)
from apart_epistemics.models import (
    AgentProfile,
    ConditionConfig,
    ConditionSummary,
    ExperimentSuiteConfig,
    HeterogeneityType,
    NationalityType,
    NumericEstimateRecord,
    RoundRecord,
    StatisticalComparison,
    TaskItem,
    TaskKind,
)
from apart_epistemics.report import build_static_site, generate_figures

__version__ = "0.1.0"

__all__ = [
    "AgentProfile",
    "BackendConfigError",
    "BudgetExceededError",
    "ConditionConfig",
    "ConditionSummary",
    "ExperimentSuiteConfig",
    "HeterogeneityType",
    "NationalityType",
    "NumericEstimateRecord",
    "ReplayBackend",
    "RoundRecord",
    "StatisticalComparison",
    "SyntheticEpistemicBackend",
    "TaskItem",
    "TaskKind",
    "__version__",
    "analyze_numeric_core",
    "audit_question_selection_bias",
    "build_ground_truth_tasks",
    "build_numeric_core_tasks",
    "build_numeric_prompts",
    "build_open_debate_tasks",
    "build_static_site",
    "create_backend",
    "format_prompt_number",
    "generate_agent_population",
    "generate_figures",
    "run_numeric_core",
    "run_suite",
]
