"""Stratified global agent populations, model pools, and selection-bias audit."""

from __future__ import annotations

import random

from apart_epistemics.models import AgentProfile, NationalityType

MODEL_POOLS: dict[NationalityType, list[tuple[str, str]]] = {
    NationalityType.NONE: [("none", "independent")],
    NationalityType.US_ONLY: [
        ("openai/gpt-frontier", "us_openai"),
        ("anthropic/claude-frontier", "us_anthropic"),
        ("google/gemini-frontier", "us_google"),
        ("xai/grok-frontier", "us_xai"),
    ],
    NationalityType.CN_ONLY: [
        ("qwen/qwen3.8-27b", "cn_alibaba"),
        ("deepseek/deepseek-v3", "cn_deepseek"),
        ("zhipu/glm-4", "cn_zhipu"),
        ("moonshot/kimi-k2", "cn_moonshot"),
    ],
    NationalityType.EU_ONLY: [
        ("mistral/mistral-large", "eu_mistral"),
        ("aleph/pharia-1", "eu_aleph"),
    ],
    NationalityType.DIVERGENT_SFT: [
        ("qwen/qwen3.8-27b", "cn_qwen_base"),
        ("nous/arete-27b", "us_arete_sft"),
    ],
    NationalityType.CROSS_NATIONAL: [
        ("openai/gpt-frontier", "us_openai"),
        ("qwen/qwen3.8-27b", "cn_alibaba"),
        ("mistral/mistral-large", "eu_mistral"),
        ("nous/arete-27b", "us_arete_sft"),
    ],
}

_REGIONS = (
    "north_america",
    "western_europe",
    "east_asia",
    "south_asia",
    "latin_america",
    "sub_saharan_africa",
)

_PROFESSIONS = (
    "civil engineer",
    "public health researcher",
    "high school teacher",
    "economist",
    "software developer",
    "agricultural planner",
    "journalist",
    "policy analyst",
)

_AGE_BANDS = ("18-24", "25-34", "35-49", "50-64")

_VALUE_PAIRS = (
    ["empirical rigor", "local autonomy"],
    ["collective stability", "long-term equity"],
    ["open inquiry", "harm reduction"],
    ["institutional trust", "cultural pluralism"],
)

_TOPIC_POOL = (
    "empirical_evidence",
    "equity_distribution",
    "institutional_trust",
    "tail_risk",
    "local_context",
    "cost_efficiency",
)


def generate_agent_population(num_agents: int = 24, seed: int = 42) -> list[AgentProfile]:
    """Generate a deterministic, regionally stratified population of persona-grounded agents."""
    rng = random.Random(seed)
    agents: list[AgentProfile] = []
    for i in range(num_agents):
        region = _REGIONS[i % len(_REGIONS)]
        profession = _PROFESSIONS[i % len(_PROFESSIONS)]
        age_band = _AGE_BANDS[(i // len(_REGIONS)) % len(_AGE_BANDS)]
        values = list(_VALUE_PAIRS[i % len(_VALUE_PAIRS)])
        base_prior = 0.14 + 0.72 * ((i + 0.5) / num_agents)
        prior_belief = max(0.05, min(0.95, base_prior + rng.uniform(-0.08, 0.08)))
        prior_confidence = round(0.55 + 0.35 * rng.random(), 4)
        nfc = round(0.25 + 0.70 * rng.random(), 4)
        preferred = rng.sample(_TOPIC_POOL, k=2)
        agents.append(
            AgentProfile(
                agent_id=f"agent_{i:02d}",
                name=f"Agent {i + 1:02d}",
                region=region,
                profession=profession,
                age_band=age_band,
                values=values,
                prior_belief=round(prior_belief, 4),
                prior_confidence=prior_confidence,
                need_for_cognition=nfc,
                preferred_topics=preferred,
            )
        )
    return agents


def assign_models_for_condition(
    nationality_type: NationalityType,
    num_models: int,
    explicit_models: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Resolve the exact list of (model_slug, family_id) pairs for an experimental condition."""
    if num_models == 0 or nationality_type == NationalityType.NONE:
        return [("none", "independent")]
    if explicit_models:
        return [(m, m.split("/")[0]) for m in explicit_models[: max(1, num_models)]]
    pool = MODEL_POOLS[nationality_type]
    if num_models <= len(pool):
        return list(pool[:num_models])
    return [pool[i % len(pool)] for i in range(num_models)]


def audit_question_selection_bias(
    cold_correctness_by_model: dict[str, list[int]],
    reference_model: str,
    max_allowed_gap: float = 0.25,
) -> dict[str, float | bool | dict[str, float]]:
    """Audit if questions filtered on `reference_model` failures bias multi-model comparisons."""
    if reference_model not in cold_correctness_by_model:
        raise ValueError(f"Reference model '{reference_model}' not found in cold scores.")
    acc_by_model = {
        model: (sum(scores) / len(scores) if scores else 0.0)
        for model, scores in cold_correctness_by_model.items()
    }
    ref_acc = acc_by_model[reference_model]
    gaps = {m: abs(acc - ref_acc) for m, acc in acc_by_model.items() if m != reference_model}
    max_gap = max(gaps.values(), default=0.0)
    return {
        "reference_model": ref_acc,
        "accuracy_by_model": acc_by_model,
        "max_accuracy_gap": round(max_gap, 4),
        "selection_bias_detected": max_gap > max_allowed_gap,
    }
