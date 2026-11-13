"""Dual-track benchmark datasets, stratified global agents, and selection-bias audit."""

from __future__ import annotations

import random

from apart_epistemics.models import AgentProfile, NationalityType, TaskItem, TaskKind

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

_REGIONS = [
    "north_america",
    "western_europe",
    "east_asia",
    "south_asia",
    "latin_america",
    "sub_saharan_africa",
]

_PROFESSIONS = [
    "civil engineer",
    "public health researcher",
    "high school teacher",
    "economist",
    "software developer",
    "agricultural planner",
    "journalist",
    "policy analyst",
]

_AGE_BANDS = ["18-24", "25-34", "35-49", "50-64"]

_VALUE_PAIRS = [
    ["empirical rigor", "local autonomy"],
    ["collective stability", "long-term equity"],
    ["open inquiry", "harm reduction"],
    ["institutional trust", "cultural pluralism"],
]


def build_ground_truth_tasks() -> list[TaskItem]:
    """Return 8 verifiable tasks with known ground truth and misconception lures."""
    return [
        TaskItem(
            id="gt_ev_grid_net",
            kind=TaskKind.GROUND_TRUTH,
            domain="energy",
            question=(
                "Do electric vehicles over a 150,000 km lifecycle emit less CO2 than internal "
                "combustion vehicles on a coal-heavy grid (~65% coal, e.g., Poland/India averages)?"
            ),
            hypothesis_a=(
                "Net lifecycle emissions are higher or equal due to battery manufacturing."
            ),
            hypothesis_b="Net lifecycle emissions remain lower even on coal-heavy grids.",
            ground_truth=1,
            popular_misconception_bias=-0.22,
            global_north_prior=0.72,
            canonical_topics=[
                "battery_manufacturing_debt",
                "grid_carbon_intensity",
                "drivetrain_efficiency",
                "mineral_refining",
                "lifetime_mileage_crossover",
                "recycling_credits",
            ],
        ),
        TaskItem(
            id="gt_base_rate_screening",
            kind=TaskKind.GROUND_TRUTH,
            domain="biostatistics",
            question=(
                "With 1% prevalence, 95% sensitivity, and 5% false-positive rate, is a person who "
                "tests positive more likely than not (>50%) to actually have the condition?"
            ),
            hypothesis_a="No, PPV is roughly 16%, well below 50%.",
            hypothesis_b="Yes, 95% test accuracy implies >50% probability.",
            ground_truth=0,
            popular_misconception_bias=0.30,
            global_north_prior=0.58,
            canonical_topics=[
                "base_rate_neglect",
                "false_positive_dominance",
                "bayes_theorem_ppv",
                "confirmatory_retesting",
                "screening_cost",
                "clinical_communication",
            ],
        ),
        TaskItem(
            id="gt_antibiotic_course",
            kind=TaskKind.GROUND_TRUTH,
            domain="medicine",
            question=(
                "For common community-acquired bacterial infections, do randomized trials show "
                "shorter 3-5 day courses are generally non-inferior to 10-14 day courses?"
            ),
            hypothesis_a="Longer fixed courses are always needed to prevent resistance.",
            hypothesis_b=(
                "Shorter symptom-guided courses achieve non-inferior cure with less resistance."
            ),
            ground_truth=1,
            popular_misconception_bias=-0.28,
            global_north_prior=0.45,
            canonical_topics=[
                "antimicrobial_stewardship",
                "collateral_microbiome_selection",
                "randomized_trial_evidence",
                "patient_adherence",
                "clinical_guidelines_lag",
                "exception_pathogens",
            ],
        ),
        TaskItem(
            id="gt_flood_levee_paradox",
            kind=TaskKind.GROUND_TRUTH,
            domain="civil_infrastructure",
            question=(
                "Does raising river levees without floodplain zoning reliably reduce long-run "
                "aggregate flood damages across decades?"
            ),
            hypothesis_a=(
                "No, levee effect induces denser development and larger catastrophic losses."
            ),
            hypothesis_b="Yes, taller engineered levees strictly lower cumulative flood losses.",
            ground_truth=0,
            popular_misconception_bias=0.25,
            global_north_prior=0.52,
            canonical_topics=[
                "levee_effect_moral_hazard",
                "extreme_tail_events",
                "land_use_zoning",
                "wetland_retention",
                "insurance_subsidies",
                "upstream_downstream_tradeoff",
            ],
        ),
        TaskItem(
            id="gt_braess_road_capacity",
            kind=TaskKind.GROUND_TRUTH,
            domain="urban_systems",
            question=(
                "Can adding a high-speed shortcut link to a congested road network increase "
                "average equilibrium travel time for all drivers?"
            ),
            hypothesis_a="No, adding capacity strictly reduces or preserves travel time.",
            hypothesis_b=(
                "Yes, selfish routing equilibrium can worsen total delay (Braess paradox)."
            ),
            ground_truth=1,
            popular_misconception_bias=-0.18,
            global_north_prior=0.68,
            canonical_topics=[
                "nash_equilibrium_routing",
                "induced_demand",
                "congestion_pricing",
                "marginal_vs_average_cost",
                "network_topology",
                "public_transit_substitution",
            ],
        ),
        TaskItem(
            id="gt_simpson_admissions",
            kind=TaskKind.GROUND_TRUTH,
            domain="statistics",
            question=(
                "If aggregate university admission rates favor Group A over Group B, must at least "
                "several individual departments also favor Group A?"
            ),
            hypothesis_a="No, every single department can favor Group B due to application mix.",
            hypothesis_b="Yes, aggregate advantage requires advantage in at least one department.",
            ground_truth=0,
            popular_misconception_bias=0.24,
            global_north_prior=0.49,
            canonical_topics=[
                "simpsons_paradox",
                "confounding_application_rate",
                "stratified_analysis",
                "causal_dag_direction",
                "fairness_metrics",
                "department_selectivity",
            ],
        ),
        TaskItem(
            id="gt_cold_chain_spoilage",
            kind=TaskKind.GROUND_TRUTH,
            domain="global_health",
            question=(
                "In last-mile rural clinics, is solar direct-drive refrigeration typically more "
                "reliable than kerosene/grid backup fridges for preserving vaccine potency?"
            ),
            hypothesis_a="Grid/kerosene backup remains more reliable due to battery storage.",
            hypothesis_b=(
                "Solar direct-drive with phase-change ice banks cuts freeze/heat excursions."
            ),
            ground_truth=1,
            popular_misconception_bias=-0.15,
            global_north_prior=0.60,
            canonical_topics=[
                "phase_change_thermal_mass",
                "freeze_damage_risk",
                "fuel_supply_interruption",
                "maintenance_capacity",
                "capital_vs_opex",
                "last_mile_logistics",
            ],
        ),
        TaskItem(
            id="gt_ensemble_correlated_errors",
            kind=TaskKind.GROUND_TRUTH,
            domain="ai_epistemics",
            question=(
                "Does majority-voting across 10 responses from models trained on overlapping web "
                "corpora guarantee Condorcet-style asymptotic error elimination?"
            ),
            hypothesis_a=(
                "No, shared training and alignment induce correlated errors violating independence."
            ),
            hypothesis_b=(
                "Yes, majority vote across many model samples cancels random errors to zero."
            ),
            ground_truth=0,
            popular_misconception_bias=0.26,
            global_north_prior=0.55,
            canonical_topics=[
                "condorcet_independence",
                "correlated_error_floor",
                "shared_pretraining_corpus",
                "goldman_guru_followers",
                "calibration_vs_consensus",
                "deliberation_diversity",
            ],
        ),
    ]


def build_open_debate_tasks() -> list[TaskItem]:
    """Return 8 normative/policy open-ended debate tasks spanning global perspectives."""
    return [
        TaskItem(
            id="od_sovereign_ai",
            kind=TaskKind.OPEN_DEBATE,
            domain="ai_governance",
            question=(
                "Should countries invest public R&D in sovereign national foundation models rather "
                "than relying primarily on frontier models hosted by US or Chinese labs?"
            ),
            hypothesis_a="Concentrate on global frontier APIs with local fine-tuning.",
            hypothesis_b="Fund sovereign models for linguistic, cultural, and strategic autonomy.",
            ground_truth=None,
            global_north_prior=0.38,
            canonical_topics=[
                "linguistic_preservation",
                "compute_subsidy_opportunity_cost",
                "geopolitical_dependency",
                "regulatory_auditability",
                "cultural_value_alignment",
                "open_weights_commons",
                "energy_water_footprint",
                "procurement_diversification",
            ],
        ),
        TaskItem(
            id="od_open_weights",
            kind=TaskKind.OPEN_DEBATE,
            domain="ai_safety",
            question=(
                "Should frontier AI weights above compute thresholds be kept closed by default "
                "or released open-weight for independent academic audit?"
            ),
            hypothesis_a="Restrict frontier weights to prevent irreversible misuse.",
            hypothesis_b="Release weights to avoid concentrated corporate/state gatekeeping.",
            ground_truth=None,
            global_north_prior=0.64,
            canonical_topics=[
                "bio_cyber_offense_asymmetry",
                "independent_academic_audit",
                "monopoly_concentration",
                "irreversible_proliferation",
                "post_training_safety_stripping",
                "global_south_research_access",
                "defensive_patching_speed",
                "compute_governance",
            ],
        ),
        TaskItem(
            id="od_congestion_pricing",
            kind=TaskKind.OPEN_DEBATE,
            domain="urban_policy",
            question=(
                "Is cordon congestion pricing in dense megacities an equitable climate policy "
                "or a regressive tax on outer-suburb commuters?"
            ),
            hypothesis_a="Regressive burden unless transit alternatives exist first.",
            hypothesis_b="Progressive when revenues directly fund frequent bus/rail corridors.",
            ground_truth=None,
            global_north_prior=0.68,
            canonical_topics=[
                "transit_revenue_hypothecation",
                "outer_ring_commuter_equity",
                "informal_freight_impact",
                "air_quality_health_gains",
                "emergency_vehicle_speed",
                "political_backlash_design",
                "exemption_loopholes",
                "spatial_housing_mismatch",
            ],
        ),
        TaskItem(
            id="od_algorithmic_hiring",
            kind=TaskKind.OPEN_DEBATE,
            domain="labor_markets",
            question=(
                "Should employers adopt standardized AI resume/interview scoring over human review "
                "when single-vendor tools promise higher predictive validity?"
            ),
            hypothesis_a="Standardized scoring reduces idiosyncratic human bias.",
            hypothesis_b="Algorithmic monoculture locks out non-standard talent permanently.",
            ground_truth=None,
            global_north_prior=0.54,
            canonical_topics=[
                "kleinberg_monoculture_lockout",
                "human_bias_inconsistency",
                "homogenized_applicant_writing",
                "audit_transparency_rights",
                "disability_language_penalty",
                "goodhart_resume_gaming",
                "appeal_human_in_loop",
                "labor_bargaining_power",
            ],
        ),
        TaskItem(
            id="od_solar_geoengineering",
            kind=TaskKind.OPEN_DEBATE,
            domain="climate_policy",
            question=(
                "Should international scientific bodies expand small-scale outdoor stratospheric "
                "aerosol research as a climate risk hedge?"
            ),
            hypothesis_a="Moratorium needed to prevent termination shock and governance conflict.",
            hypothesis_b="Targeted research needed to evaluate tail risks for vulnerable nations.",
            ground_truth=None,
            global_north_prior=0.44,
            canonical_topics=[
                "monsoon_precipitation_shift",
                "termination_shock_hazard",
                "mitigation_moral_hazard",
                "global_south_heat_vulnerability",
                "unilateral_deployment_risk",
                "governance_veto_legitimacy",
                "indoor_modeling_limits",
                "ozone_chemistry_side_effects",
            ],
        ),
        TaskItem(
            id="od_personalized_ai_tutors",
            kind=TaskKind.OPEN_DEBATE,
            domain="education",
            question=(
                "Do hyper-personalized AI companions in secondary education deepen mastery or "
                "erode shared civic discourse and intellectual friction?"
            ),
            hypothesis_a="Tailored scaffolding closes achievement gaps rapidly.",
            hypothesis_b=(
                "Sycophantic frictionless answers weaken metacognition and civic common ground."
            ),
            ground_truth=None,
            global_north_prior=0.61,
            canonical_topics=[
                "cognitive_forcing_friction",
                "metacognitive_calibration",
                "epistemic_echo_chambers",
                "teacher_ratio_constraints",
                "multilingual_access",
                "sycophantic_overconfidence",
                "shared_classroom_canon",
                "assessment_integrity",
            ],
        ),
        TaskItem(
            id="od_digital_public_infra",
            kind=TaskKind.OPEN_DEBATE,
            domain="public_infrastructure",
            question=(
                "Should national digital identity and AI public services be mandatory defaults "
                "for welfare distribution to cut leakage, or strictly opt-in?"
            ),
            hypothesis_a="Default digital ID dramatically lowers exclusion and graft.",
            hypothesis_b="Mandatory biometrics create single points of exclusion and surveillance.",
            ground_truth=None,
            global_north_prior=0.42,
            canonical_topics=[
                "last_mile_biometric_failure",
                "welfare_leakage_reduction",
                "offline_fallback_rights",
                "state_surveillance_creep",
                "interoperable_open_protocols",
                "marginalized_documentation_gaps",
                "vendor_lock_in",
                "grievance_redress_speed",
            ],
        ),
        TaskItem(
            id="od_ai_assisted_peer_review",
            kind=TaskKind.OPEN_DEBATE,
            domain="science_policy",
            question=(
                "Should scientific journals permit authors and reviewers to rely heavily on "
                "frontier LLMs for framing literature reviews and referee reports?"
            ),
            hypothesis_a="Speeds cross-disciplinary synthesis and lowers language barriers.",
            hypothesis_b="Compresses scientific paradigms and reinforces citation monoculture.",
            ground_truth=None,
            global_north_prior=0.57,
            canonical_topics=[
                "non_native_english_equity",
                "knowledge_collapse_long_tail",
                "citation_matthew_effect",
                "reviewer_outsourcing_epistemia",
                "novel_hypothesis_suppression",
                "verification_burden_asymmetry",
                "methodology_auditability",
                "disclosure_norms",
            ],
        ),
    ]


def build_numeric_core_tasks() -> list[TaskItem]:
    """Return 10 verifiable numeric ground-truth tasks matching Kevin's Page-theorem core."""
    return [
        TaskItem(
            id="num_yarlung_length_km",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="geography",
            question=(
                "What is the total length of the Yarlung Tsangpo / Brahmaputra River "
                "in kilometers?"
            ),
            numeric_truth=2900.0,
            unit="km",
            source_note="Pre-2026 hydrological survey (~2,840-2,900 km)",
            popular_misconception_bias=-0.12,
        ),
        TaskItem(
            id="num_timbuktu_pop",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="demographics",
            question=(
                "What was the municipal census population of Timbuktu, Mali in the 2009 census?"
            ),
            numeric_truth=54453.0,
            unit="people",
            source_note="2009 Mali census (54,453)",
            popular_misconception_bias=0.18,
        ),
        TaskItem(
            id="num_lake_baikal_depth_m",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="geography",
            question="What is the maximum measured depth of Lake Baikal in meters?",
            numeric_truth=1642.0,
            unit="meters",
            source_note="Standard bathymetric depth (1,642 m)",
            popular_misconception_bias=-0.06,
        ),
        TaskItem(
            id="num_kerguelen_area_km2",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="geography",
            question=(
                "What is the total land area of the Kerguelen Islands archipelago "
                "in square kilometers?"
            ),
            numeric_truth=7215.0,
            unit="km2",
            source_note="French Southern and Antarctic Lands survey (7,215 km2)",
            popular_misconception_bias=-0.16,
        ),
        TaskItem(
            id="num_challenger_deep_m",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="oceanography",
            question="What is the approximate depth of Challenger Deep in meters?",
            numeric_truth=10935.0,
            unit="meters",
            source_note="NOAA / Caladan Oceanic bathymetry (~10,935 m)",
            popular_misconception_bias=-0.04,
        ),
        TaskItem(
            id="num_bhutan_pop_2017",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="demographics",
            question=(
                "What was Bhutan's resident population in the 2017 Population and Housing Census?"
            ),
            numeric_truth=727145.0,
            unit="people",
            source_note="National Statistics Bureau of Bhutan (727,145)",
            popular_misconception_bias=0.14,
        ),
        TaskItem(
            id="num_trans_siberian_km",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="infrastructure",
            question=(
                "What is the length of the main Trans-Siberian Railway line from Moscow "
                "to Vladivostok in km?"
            ),
            numeric_truth=9289.0,
            unit="km",
            source_note="Official route marker (9,289 km)",
            popular_misconception_bias=-0.07,
        ),
        TaskItem(
            id="num_atacama_annual_mm",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="climatology",
            question=(
                "What is the long-term mean annual rainfall at Arica in Chile's Atacama Desert "
                "in millimeters?"
            ),
            numeric_truth=1.8,
            unit="mm",
            source_note="Chilean meteorological record (~1.8 mm/year)",
            popular_misconception_bias=0.22,
        ),
        TaskItem(
            id="num_iceland_pop_2024",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="demographics",
            question="What was Iceland's total resident population on January 1, 2024?",
            numeric_truth=383726.0,
            unit="people",
            source_note="Statistics Iceland (383,726)",
            popular_misconception_bias=0.09,
        ),
        TaskItem(
            id="num_speed_of_sound_water",
            kind=TaskKind.NUMERIC_ESTIMATE,
            domain="physics",
            question=(
                "What is the speed of sound in seawater at 20 C and standard salinity "
                "in meters per second?"
            ),
            numeric_truth=1522.0,
            unit="m/s",
            source_note="UNESCO acoustic tables (~1,522 m/s)",
            popular_misconception_bias=-0.19,
        ),
    ]


def generate_agent_population(num_agents: int = 24, seed: int = 42) -> list[AgentProfile]:
    """Generate a deterministic, regionally stratified population of persona-grounded agents."""
    rng = random.Random(seed)
    agents: list[AgentProfile] = []
    topic_pool = [
        "empirical_evidence",
        "equity_distribution",
        "institutional_trust",
        "tail_risk",
        "local_context",
        "cost_efficiency",
    ]
    for i in range(num_agents):
        region = _REGIONS[i % len(_REGIONS)]
        profession = _PROFESSIONS[i % len(_PROFESSIONS)]
        age_band = _AGE_BANDS[(i // len(_REGIONS)) % len(_AGE_BANDS)]
        values = _VALUE_PAIRS[i % len(_VALUE_PAIRS)]
        # Stratified symmetric priors spanning [0.12, 0.88] with small deterministic jitter
        base_prior = 0.14 + 0.72 * ((i + 0.5) / num_agents)
        prior_belief = max(0.05, min(0.95, base_prior + rng.uniform(-0.08, 0.08)))
        prior_confidence = round(0.55 + 0.35 * rng.random(), 4)
        nfc = round(0.25 + 0.70 * rng.random(), 4)
        preferred = rng.sample(topic_pool, k=2)
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
        return pool[:num_models]
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
