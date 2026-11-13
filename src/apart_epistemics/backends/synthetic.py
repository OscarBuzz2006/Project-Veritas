"""Deterministic stochastic epistemic simulator modeling decoding, origin, and personalization."""

from __future__ import annotations

import math
import random
import zlib
from typing import Any

from apart_epistemics.metrics import clamp
from apart_epistemics.models import (
    AgentProfile,
    ConditionConfig,
    HeterogeneityType,
    NationalityType,
    TaskItem,
    TaskKind,
)

_STOCHASTIC_OPENERS = (
    "Looking closely at",
    "From an empirical vantage point on",
    "In practical operational contexts involving",
    "Weighing longitudinal evidence around",
    "Setting aside textbook assumptions about",
    "Considering cross-regional constraints in",
    "Focusing on structural trade-offs within",
    "Examining baseline mechanisms behind",
    "Reassessing conventional priors regarding",
    "Evaluating heterogeneous evidence tied to",
)

_STOCHASTIC_CONNECTORS = (
    "directly shifts probability toward",
    "provides decisive signal favoring",
    "tempers sweeping claims in favor of",
    "highlights localized sensitivity around",
    "exposes hidden correlation across",
    "sharpens conditional likelihood toward",
    "demonstrates persistent divergence over",
    "anchors the causal assessment near",
)

_STOCHASTIC_CLOSINGS = (
    "pending out-of-sample replication across additional cohorts.",
    "even after accounting for institutional measurement noise.",
    "whereas narrow monoculture priors overlook tail regimes.",
    "especially under shifting regional resource baselines.",
    "which resists premature convergence onto a single framing.",
    "when calibrated against historical base rates and variance.",
    "contingent on transparent empirical monitoring.",
    "leaving substantive room for competing mechanistic hypotheses.",
)

_STOCHASTIC_REGISTERS = (
    "systemic vulnerability",
    "institutional inertia",
    "counterfactual baseline",
    "distributional asymmetry",
    "measurement elasticity",
    "feedback amplification",
    "marginal constraint",
    "structural heterogeneity",
)

_PERSONALIZED_REGIMES = frozenset(
    {
        HeterogeneityType.PERSONALIZED_ADVISOR,
        HeterogeneityType.PERSONALIZED_PEER,
        HeterogeneityType.PERSONALIZED_PROFILE,
    }
)


def _humanize_topic(slug: str) -> str:
    return slug.replace("_", " ")


class SyntheticEpistemicBackend:
    """Grounded stochastic mechanism engine modeling decoding, origin, and personalization."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _task_truth_target(self, task: TaskItem) -> float:
        if task.kind == TaskKind.GROUND_TRUTH and task.ground_truth is not None:
            return 0.88 if task.ground_truth == 1 else 0.12
        return task.global_north_prior

    def _model_epistemic_profile(
        self,
        cond: ConditionConfig,
        task: TaskItem,
        model_slug: str,
        model_family: str,
    ) -> tuple[float, list[int], float]:
        """Return `(assistant_target_prob, preferred_topic_indices, assistant_confidence)`."""
        topics = task.canonical_topics or ["general_analysis"]
        m_len = len(topics)
        truth_target = self._task_truth_target(task)
        lure_shift = task.popular_misconception_bias * 0.85
        regime = cond.heterogeneity_type

        if regime == HeterogeneityType.MONOLITHIC_SHARED:
            if cond.nationality_type == NationalityType.CN_ONLY:
                target = clamp(
                    0.5 * truth_target + 0.5 * (1.0 - task.global_north_prior) + 0.55 * lure_shift,
                    0.08,
                    0.92,
                )
                return (target, [2 % m_len, 3 % m_len], 0.84)
            target = clamp(
                0.55 * truth_target + 0.45 * task.global_north_prior + 0.72 * lure_shift,
                0.08,
                0.92,
            )
            return (target, [0, 1 % m_len], 0.87)

        if regime == HeterogeneityType.SHARED_FEEDBACK:
            target = clamp(
                0.48 * truth_target + 0.52 * task.global_north_prior + 0.82 * lure_shift,
                0.07,
                0.93,
            )
            return (target, [0, 1 % m_len], 0.90)

        if regime == HeterogeneityType.MULTI_LAB_MIX:
            lab_hash = sum(ord(ch) for ch in model_slug) % max(m_len, 1)
            if cond.nationality_type == NationalityType.CN_ONLY:
                target = clamp(
                    0.62 * truth_target
                    + 0.38 * (1.0 - task.global_north_prior)
                    + 0.42 * lure_shift,
                    0.10,
                    0.90,
                )
            else:
                target = clamp(
                    0.64 * truth_target + 0.36 * task.global_north_prior + 0.48 * lure_shift,
                    0.10,
                    0.90,
                )
            return (target, [lab_hash, (lab_hash + 1) % m_len], 0.82)

        if regime == HeterogeneityType.DIVERGENT_SFT_PAIR:
            if "arete" in model_family or "arete" in model_slug:
                target = clamp(
                    0.66 * truth_target
                    + 0.34 * max(0.65, task.global_north_prior)
                    + 0.3 * lure_shift,
                    0.10,
                    0.90,
                )
                return (target, [1 % m_len, 4 % m_len, 5 % m_len], 0.84)
            target = clamp(
                0.62 * truth_target
                + 0.38 * min(0.38, 1.0 - task.global_north_prior)
                + 0.3 * lure_shift,
                0.10,
                0.90,
            )
            return (target, [0, 2 % m_len, 3 % m_len], 0.82)

        if regime == HeterogeneityType.CULTURAL_REGIONAL_MIX:
            return self._regional_mix_profile(task, truth_target, lure_shift, m_len, model_family)

        return (truth_target, [0, 1 % m_len], 0.75)

    def _regional_mix_profile(
        self,
        task: TaskItem,
        truth_target: float,
        lure_shift: float,
        m_len: int,
        model_family: str,
    ) -> tuple[float, list[int], float]:
        slot = sum(ord(ch) for ch in model_family) % max(m_len, 1)
        if model_family.startswith("cn_"):
            target = clamp(
                0.70 * truth_target + 0.30 * (1.0 - task.global_north_prior),
                0.12,
                0.88,
            )
        elif model_family.startswith("eu_"):
            target = clamp(0.74 * truth_target + 0.26 * 0.50, 0.12, 0.88)
        elif model_family.startswith("us_arete"):
            target = clamp(0.72 * truth_target + 0.28 * task.global_north_prior, 0.12, 0.88)
        else:
            target = clamp(
                0.70 * truth_target + 0.30 * task.global_north_prior + 0.22 * lure_shift,
                0.12,
                0.88,
            )
        return (target, [slot, (slot + 2) % m_len, (slot + 4) % m_len], 0.78)

    def _compose_prose(
        self,
        rng: random.Random,
        cond: ConditionConfig,
        agent: AgentProfile,
        task: TaskItem,
        topics: list[str],
        stance_prob: float,
        confidence: float,
        round_idx: int,
    ) -> str:
        t_phrases = [_humanize_topic(t) for t in topics]
        focus = ", ".join(t_phrases[:2]) if len(t_phrases) >= 2 else t_phrases[0]
        direction = "supports Hypothesis B" if stance_prob >= 0.5 else "favors Hypothesis A"
        if cond.temperature <= 0.05:
            # Near-deterministic greedy decoding collapses phrasing variety across agents.
            return (
                f"Round {round_idx} assessment for {task.id}: Focus on {focus}. "
                f"Evidence {direction} with probability {stance_prob:.2f} "
                f"and confidence {confidence:.2f}."
            )

        scale = clamp(cond.temperature, 0.1, 1.5)
        opener = rng.choice(_STOCHASTIC_OPENERS)
        conn = rng.choice(_STOCHASTIC_CONNECTORS)
        closing = rng.choice(_STOCHASTIC_CLOSINGS)
        reg = rng.choice(_STOCHASTIC_REGISTERS)
        persona_cue = (
            f"As a {agent.profession} in {agent.region.replace('_', ' ')} "
            f"prioritizing {' and '.join(agent.values)}, "
            if scale > 0.6
            else ""
        )
        return (
            f"{persona_cue}{opener} {focus}, the balance of {reg} {conn} "
            f"{direction} (p={stance_prob:.2f}, conf={confidence:.2f}), {closing}"
        )

    def _simulate_round_zero(
        self,
        rng: random.Random,
        cond: ConditionConfig,
        task: TaskItem,
        agent: AgentProfile,
        agent_idx: int,
        canonical: list[str],
    ) -> dict[str, Any]:
        m_len = len(canonical)
        region_offset = (zlib.crc32(agent.region.encode("utf-8")) % 5 - 2) * 0.04
        task_pull = (
            (0.28 * (self._task_truth_target(task) - 0.5) + 0.35 * task.popular_misconception_bias)
            if task.kind == TaskKind.GROUND_TRUTH
            else 0.22 * (task.global_north_prior - 0.5)
        )
        stance = clamp(
            agent.prior_belief + task_pull + region_offset + rng.uniform(-0.07, 0.07),
            0.05,
            0.95,
        )
        conf = clamp(agent.prior_confidence + rng.uniform(-0.05, 0.05), 0.51, 0.96)
        idx1 = (agent_idx * 3 + len(agent.region)) % m_len
        idx2 = (idx1 + 1 + (agent_idx % 2)) % m_len
        topics = [canonical[idx1], canonical[idx2]] if idx2 != idx1 else [canonical[idx1]]
        prose = self._compose_prose(rng, cond, agent, task, topics, stance, conf, 0)
        return {
            "assistant_response_text": (
                "Independent baseline prior (no assistant prompt in Round 0)."
            ),
            "agent_response_text": prose,
            "extracted_topics": topics,
            "stance_prob": round(stance, 4),
            "confidence": round(conf, 4),
            "stated_answer": 1 if stance >= 0.5 else 0,
            "goldman_source_overlap": 0.0,
            "persuasion_bias_exposure": 0.0,
        }

    def _simulate_no_ai(
        self,
        rng: random.Random,
        task: TaskItem,
        agent_idx: int,
        round_idx: int,
        prev_stance: float,
        peer_mean: float,
        canonical: list[str],
    ) -> tuple[float, float, list[str], str, float, float]:
        m_len = len(canonical)
        truth_target = self._task_truth_target(task)
        noisy_prior = 0.5 + 0.72 * (truth_target - 0.5) + 0.42 * task.popular_misconception_bias
        indep_signal = clamp(noisy_prior + rng.gauss(0.0, 0.14), 0.05, 0.95)
        next_stance = clamp(
            0.56 * prev_stance + 0.24 * peer_mean + 0.20 * indep_signal,
            0.04,
            0.96,
        )
        next_conf = clamp(0.55 + 0.55 * abs(next_stance - 0.5), 0.52, 0.88)
        slot_a = (agent_idx + round_idx) % m_len
        slot_b = (agent_idx * 2 + 1) % m_len
        topics = [canonical[slot_a], canonical[slot_b]] if slot_b != slot_a else [canonical[slot_a]]
        assistant_text = "No AI advisor in loop; pure agent reasoning and peer discussion."
        return next_stance, next_conf, topics, assistant_text, 0.05, 0.10

    def _simulate_personalized(
        self,
        rng: random.Random,
        cond: ConditionConfig,
        task: TaskItem,
        agent: AgentProfile,
        agent_idx: int,
        prev_stance: float,
        peer_mean: float,
        canonical: list[str],
    ) -> tuple[float, float, list[str], str, float, float]:
        m_len = len(canonical)
        truth_target = self._task_truth_target(task)
        region_shift = (agent_idx % 3 - 1) * 0.06
        primary_slot = (agent_idx * 5 + sum(ord(c) for c in agent.profession)) % m_len
        secondary_slot = (primary_slot + 2) % m_len
        topics = (
            [canonical[primary_slot], canonical[secondary_slot]]
            if secondary_slot != primary_slot
            else [canonical[primary_slot]]
        )

        if cond.heterogeneity_type == HeterogeneityType.PERSONALIZED_ADVISOR:
            advisor_target = clamp(
                0.86 * truth_target + 0.14 * (0.5 + region_shift) + rng.gauss(0.0, 0.05),
                0.06,
                0.94,
            )
            next_stance = clamp(
                0.28 * prev_stance + 0.60 * advisor_target + 0.12 * peer_mean,
                0.05,
                0.95,
            )
            next_conf = clamp(0.52 + 0.72 * abs(next_stance - 0.5), 0.52, 0.87)
            assistant_text = (
                f"Critical advisor tailored for {agent.profession}: stress-testing assumptions "
                f"on {_humanize_topic(topics[0])} against calibrated base rates."
            )
            return next_stance, next_conf, topics, assistant_text, 0.15, 0.18

        if cond.heterogeneity_type == HeterogeneityType.PERSONALIZED_PEER:
            lure_pull = task.popular_misconception_bias * 1.15
            syc_target = clamp(
                prev_stance + lure_pull + region_shift + rng.gauss(0.0, 0.06),
                0.04,
                0.96,
            )
            if task.kind == TaskKind.GROUND_TRUTH and task.ground_truth is not None:
                wrong_target = 0.14 if task.ground_truth == 1 else 0.86
                anchor_hash = (agent_idx + sum(ord(c) for c in task.id)) % 10
                syc_target = (
                    wrong_target
                    if anchor_hash < 5
                    else clamp(
                        0.62 * truth_target + 0.38 * prev_stance + rng.gauss(0.0, 0.06),
                        0.06,
                        0.94,
                    )
                )
            next_stance = clamp(
                0.35 * prev_stance + 0.55 * syc_target + 0.10 * peer_mean,
                0.04,
                0.96,
            )
            next_conf = clamp(0.84 + 0.12 * rng.random(), 0.78, 0.98)
            assistant_text = (
                f"Warm peer resonance for {agent.profession}: your intuition regarding "
                f"{_humanize_topic(topics[0])} makes total sense here."
            )
            return next_stance, next_conf, topics, assistant_text, 0.22, 1.65

        prof_target = clamp(
            0.68 * truth_target + 0.32 * prev_stance + 0.3 * task.popular_misconception_bias,
            0.06,
            0.94,
        )
        next_stance = clamp(0.42 * prev_stance + 0.45 * prof_target + 0.13 * peer_mean, 0.05, 0.95)
        next_conf = clamp(0.58 + 0.60 * abs(next_stance - 0.5), 0.54, 0.91)
        assistant_text = (
            f"Profile-adapted note ({agent.region}): emphasizing {_humanize_topic(topics[0])}."
        )
        return next_stance, next_conf, topics, assistant_text, 0.18, 0.45

    def _simulate_model_mediated(
        self,
        rng: random.Random,
        cond: ConditionConfig,
        task: TaskItem,
        agent_idx: int,
        round_idx: int,
        prev_stance: float,
        prev_confidence: float,
        prev_topics: list[str],
        peer_mean: float,
        canonical: list[str],
        model_assigned: str,
        model_family: str,
    ) -> tuple[float, float, list[str], str, float, float]:
        assistant_target, pref_slots, assistant_conf = self._model_epistemic_profile(
            cond, task, model_assigned, model_family
        )
        monolithic = cond.heterogeneity_type in (
            HeterogeneityType.MONOLITHIC_SHARED,
            HeterogeneityType.SHARED_FEEDBACK,
        )
        assistant_weight = 0.76 if monolithic else 0.62
        noise_sd = 0.03 if cond.temperature <= 0.05 else 0.055 * (0.6 + 0.4 * cond.temperature)
        next_stance = clamp(
            (1.0 - assistant_weight) * (0.6 * prev_stance + 0.4 * peer_mean)
            + assistant_weight * assistant_target
            + rng.gauss(0.0, noise_sd),
            0.04,
            0.96,
        )
        next_conf = clamp(
            0.4 * prev_confidence + 0.6 * assistant_conf + rng.uniform(-0.04, 0.04),
            0.52,
            0.96,
        )

        if monolithic:
            top_idx = pref_slots[0] if (round_idx >= 2 or rng.random() < 0.78) else pref_slots[-1]
            topics = [canonical[top_idx]]
            if round_idx == 1 and len(pref_slots) > 1 and rng.random() < 0.35:
                topics = [canonical[pref_slots[0]], canonical[pref_slots[1]]]
            overlap = 0.85 if cond.heterogeneity_type == HeterogeneityType.SHARED_FEEDBACK else 0.72
            bias_exp = 1.25 + 0.2 * round_idx
        elif cond.heterogeneity_type == HeterogeneityType.MULTI_LAB_MIX:
            primary = pref_slots[0]
            sec = pref_slots[1] if len(pref_slots) > 1 else pref_slots[0]
            topics = (
                [canonical[primary]]
                if round_idx >= 3 and rng.random() < 0.45
                else [canonical[primary], canonical[sec]]
            )
            overlap = 0.52
            bias_exp = 0.85
        else:
            a_slot = pref_slots[(agent_idx + round_idx) % len(pref_slots)]
            b_slot = pref_slots[(agent_idx + 1) % len(pref_slots)]
            extra = prev_topics[0] if prev_topics and rng.random() < 0.5 else canonical[b_slot]
            topics = list(dict.fromkeys([canonical[a_slot], extra]))
            overlap = (
                0.22 if cond.heterogeneity_type == HeterogeneityType.CULTURAL_REGIONAL_MIX else 0.30
            )
            bias_exp = 0.42

        assistant_text = (
            f"[{model_assigned}] Highlighting {_humanize_topic(topics[0])} "
            f"toward p={assistant_target:.2f}."
        )
        return next_stance, next_conf, topics, assistant_text, overlap, bias_exp

    def generate_agent_step(
        self,
        cond: ConditionConfig,
        task: TaskItem,
        agent: AgentProfile,
        agent_idx: int,
        round_idx: int,
        prev_stance: float,
        prev_confidence: float,
        prev_topics: list[str],
        peer_stances: list[float],
        peer_topics: list[list[str]],
        model_assigned: str,
        model_family: str,
    ) -> dict[str, Any]:
        """Simulate one agent-assistant-peer round with mechanistic epistemic dynamics."""
        rng_seed = (
            self.seed * 100003
            + sum(ord(c) for c in cond.condition_id) * 977
            + sum(ord(c) for c in task.id) * 131
            + agent_idx * 43
            + round_idx * 17
        )
        rng = random.Random(rng_seed)
        canonical = task.canonical_topics or ["general_analysis"]

        if round_idx == 0:
            return self._simulate_round_zero(rng, cond, task, agent, agent_idx, canonical)

        peer_mean = sum(peer_stances) / len(peer_stances) if peer_stances else prev_stance
        if cond.heterogeneity_type == HeterogeneityType.NO_AI:
            next_stance, next_conf, topics, assistant_text, overlap, bias_exp = (
                self._simulate_no_ai(
                    rng, task, agent_idx, round_idx, prev_stance, peer_mean, canonical
                )
            )
        elif cond.heterogeneity_type in _PERSONALIZED_REGIMES:
            next_stance, next_conf, topics, assistant_text, overlap, bias_exp = (
                self._simulate_personalized(
                    rng, cond, task, agent, agent_idx, prev_stance, peer_mean, canonical
                )
            )
        else:
            next_stance, next_conf, topics, assistant_text, overlap, bias_exp = (
                self._simulate_model_mediated(
                    rng,
                    cond,
                    task,
                    agent_idx,
                    round_idx,
                    prev_stance,
                    prev_confidence,
                    prev_topics,
                    peer_mean,
                    canonical,
                    model_assigned,
                    model_family,
                )
            )

        flat_peer_topics = [t for sublist in peer_topics for t in sublist]
        if flat_peer_topics and cond.heterogeneity_type == HeterogeneityType.SHARED_FEEDBACK:
            most_common = max(set(flat_peer_topics), key=flat_peer_topics.count)
            topics = [most_common]

        prose = self._compose_prose(
            rng, cond, agent, task, topics, next_stance, next_conf, round_idx
        )
        return {
            "assistant_response_text": assistant_text,
            "agent_response_text": prose,
            "extracted_topics": topics,
            "stance_prob": round(next_stance, 4),
            "confidence": round(next_conf, 4),
            "stated_answer": 1 if next_stance >= 0.5 else 0,
            "goldman_source_overlap": round(overlap, 4),
            "persuasion_bias_exposure": round(bias_exp, 4),
        }

    def generate_numeric_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        condition_id: str = "C1",
        task: TaskItem | None = None,
        round_idx: int = 1,
        current_x: float = 100.0,
        pop_stats: tuple[float, float, float] | None = None,
        agent: AgentProfile | None = None,
        model_slug: str = "openai/gpt-frontier",
        temperature: float = 0.8,
        **_: Any,
    ) -> dict[str, Any]:
        """Simulate Kevin's C1/C2/C3 numeric epistemic responses around log10 ground truth."""
        del system_prompt, user_prompt
        truth = task.numeric_truth if (task and task.numeric_truth) else max(current_x, 1.0)
        log_truth = math.log10(truth)
        log_cur = math.log10(max(current_x, 1e-6))

        seed_val = (
            self.seed * 7919
            + sum(ord(c) for c in condition_id) * 101
            + round_idx * 37
            + sum(ord(c) for c in (agent.agent_id if agent else "a0")) * 19
            + int(abs(log_truth) * 100)
        )
        rng = random.Random(seed_val)
        shared_bias = 0.16 if (task and abs(task.popular_misconception_bias) > 0.15) else 0.10

        if condition_id == "C2" and pop_stats is not None:
            pop_med = math.log10(max(pop_stats[0], 1e-6))
            target_log = 0.65 * pop_med + 0.25 * (log_truth + shared_bias) + 0.10 * log_cur
            noise = rng.gauss(0.0, 0.03 * max(0.3, temperature))
        elif condition_id == "C3" and agent is not None:
            persona_shift = ((zlib.crc32(agent.profession.encode("utf-8")) % 7) - 3) * 0.06
            target_log = 0.55 * (log_truth + persona_shift) + 0.45 * log_cur
            noise = rng.gauss(0.0, 0.12 * max(0.4, temperature))
        else:
            target_log = 0.50 * (log_truth + shared_bias * 0.7) + 0.50 * log_cur
            noise = rng.gauss(0.0, 0.11 * max(0.4, temperature))

        pred_val = 10.0 ** (target_log + noise)
        formatted = f"{round(pred_val, 1):.1f}" if pred_val >= 100.0 else f"{pred_val:.3g}"
        return {
            "raw_reply": (
                f"Considering the reference scaling and plausible mechanisms, "
                f"my reasoned Final estimate: {formatted}"
            ),
            "model_slug": model_slug,
            "provider": "synthetic_sim",
            "reasoning_tokens": rng.randint(12, 64),
            "input_tokens": rng.randint(180, 320),
            "output_tokens": rng.randint(24, 58),
        }
