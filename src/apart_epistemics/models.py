"""Core Pydantic domain models for Epistemic Prism simulations and evaluations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskKind(StrEnum):
    """Supported evaluation task tracks."""

    GROUND_TRUTH = "ground_truth"
    OPEN_DEBATE = "open_debate"
    NUMERIC_ESTIMATE = "numeric_estimate"


class HeterogeneityType(StrEnum):
    """Model deployment and personalization regimes."""

    NO_AI = "no_ai"
    MONOLITHIC_SHARED = "monolithic_shared"
    SHARED_FEEDBACK = "shared_feedback"
    MULTI_LAB_MIX = "multi_lab_mix"
    DIVERGENT_SFT_PAIR = "divergent_sft_pair"
    CULTURAL_REGIONAL_MIX = "cultural_regional_mix"
    PERSONALIZED_ADVISOR = "personalized_advisor"
    PERSONALIZED_PEER = "personalized_peer"
    PERSONALIZED_PROFILE = "personalized_profile"


class NationalityType(StrEnum):
    """Origin/provenance grouping of deployed assistant models."""

    NONE = "none"
    US_ONLY = "us_only"
    CN_ONLY = "cn_only"
    EU_ONLY = "eu_only"
    DIVERGENT_SFT = "divergent_sft"
    CROSS_NATIONAL = "cross_national"


class TaskItem(BaseModel):
    """Single prompt/question in ground-truth, open-debate, or numeric core tracks."""

    id: str
    kind: TaskKind
    domain: str
    question: str
    hypothesis_a: str = ""
    hypothesis_b: str = ""
    ground_truth: int | None = Field(default=None, ge=0, le=1)
    numeric_truth: float | None = Field(default=None, gt=0.0)
    unit: str = ""
    source_note: str = ""
    popular_misconception_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    global_north_prior: float = Field(default=0.5, ge=0.0, le=1.0)
    canonical_topics: list[str] = Field(default_factory=list)


class AgentProfile(BaseModel):
    """Persona-grounded agent profile with explicit initial epistemic priors."""

    agent_id: str
    name: str
    region: str
    profession: str = "analyst"
    age_band: str = "25-34"
    values: list[str] = Field(default_factory=list)
    prior_belief: float = Field(ge=0.0, le=1.0)
    prior_confidence: float = Field(ge=0.5, le=1.0)
    need_for_cognition: float = Field(ge=0.0, le=1.0)
    preferred_topics: list[str] = Field(default_factory=list)


class ConditionConfig(BaseModel):
    """Configuration for one experimental arm in the suite."""

    condition_id: str
    label: str
    heterogeneity_type: HeterogeneityType
    nationality_type: NationalityType = NationalityType.US_ONLY
    num_models: int = Field(ge=0, le=100)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    peer_exchange: bool = True
    alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    group: str = "main"
    models: list[str] = Field(default_factory=list)


class ExperimentSuiteConfig(BaseModel):
    """Top-level configuration parsed from YAML or CLI."""

    suite_name: str = "Epistemic Prism Benchmark"
    seed: int = 42
    num_agents: int = Field(default=24, ge=4, le=200)
    num_rounds: int = Field(default=5, ge=1, le=20)
    peer_degree: int = Field(default=3, ge=0, le=20)
    tasks_per_track: int = Field(default=6, ge=1, le=50)
    numeric_replicates: int = Field(default=2, ge=1, le=10)
    alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    conditions: list[ConditionConfig]


class RoundRecord(BaseModel):
    """One agent-round observation in Track 4 open-debate or ground-truth simulation."""

    run_id: str
    condition_id: str
    seed: int
    task_id: str
    task_kind: TaskKind
    round_idx: int = Field(ge=0)
    agent_id: str
    agent_region: str
    model_assigned: str
    model_family: str
    assistant_response_text: str
    agent_response_text: str
    extracted_topics: list[str]
    stance_prob: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.5, le=1.0)
    stated_answer: int = Field(ge=0, le=1)
    goldman_source_overlap: float = Field(ge=0.0, le=1.0)
    persuasion_bias_exposure: float = Field(ge=0.0, le=10.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NumericEstimateRecord(BaseModel):
    """One call/update observation in Kevin's log10 Page's Theorem numeric core."""

    condition_id: str
    question_id: str
    replicate: int = Field(ge=0)
    round_idx: int = Field(ge=0)
    agent_id: str
    model_slug: str
    provider: str
    raw_reply: str
    run_spec: dict[str, Any] = Field(default_factory=dict)
    parsed_value: float | None = None
    assistant_log10: float | None = None
    agent_log10_before: float
    agent_log10_after: float
    # log10 of the population median shown (C2/C3) or computed (C1) at the start of the round.
    pop_median_log10: float | None = None
    miss: bool = False
    reasoning_tokens: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    finish_reason: str = ""
    # None means the setting was not sent to the model (provider default applied).
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class ConditionSummary(BaseModel):
    """Aggregated final-round and trajectory metrics for one condition."""

    condition_id: str
    label: str
    heterogeneity_type: str
    nationality_type: str
    num_models: int
    temperature: float
    top_p: float
    group: str
    ground_truth_accuracy: float
    brier_score: float
    ece: float
    confident_wrong_rate: float
    topic_shannon_entropy: float
    effective_topic_count: float
    minority_topic_survival: float
    stance_std: float
    bimodality_coefficient: float
    global_north_skew_shift: float
    goldman_effective_sources: float
    persuasion_bias_drift: float
    distinct_2: float
    pairwise_jaccard_distance: float
    round_trajectory: list[dict[str, float]] = Field(default_factory=list)


class StatisticalComparison(BaseModel):
    """Paired statistical contrast with Wilcoxon p-value, bootstrap 95% CI, and Holm adjustment."""

    contrast_id: str
    metric: str
    condition_a: str
    condition_b: str
    mean_diff: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    holm_adjusted_p: float
    n_cells: int
