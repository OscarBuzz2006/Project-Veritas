"""Track 4 multi-round agent-assistant-peer simulation engine and round aggregator."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from apart_epistemics.backends import ReplayBackend
from apart_epistemics.datasets import (
    assign_models_for_condition,
    build_ground_truth_tasks,
    build_open_debate_tasks,
    generate_agent_population,
)
from apart_epistemics.engine.contrasts import compute_paired_holm_contrasts
from apart_epistemics.metrics import (
    clamp,
    compute_accuracy,
    compute_brier_score,
    compute_confident_wrong_rate,
    compute_ece,
    compute_goldman_effective_sources,
    compute_lexical_diversity,
    compute_minority_topic_survival,
    compute_stance_stats,
    compute_topic_entropy_and_hill,
)
from apart_epistemics.models import (
    AgentProfile,
    ConditionConfig,
    ConditionSummary,
    ExperimentSuiteConfig,
    HeterogeneityType,
    RoundRecord,
    StatisticalComparison,
    TaskItem,
    TaskKind,
)
from apart_epistemics.parsers import parse_structured_belief
from apart_epistemics.summarizer import evaluate_dual_judge_topics, export_audit_sample

_SUMMARY_CSV_FIELDS = (
    "condition_id",
    "label",
    "heterogeneity_type",
    "nationality_type",
    "num_models",
    "temperature",
    "top_p",
    "group",
    "ground_truth_accuracy",
    "brier_score",
    "ece",
    "confident_wrong_rate",
    "topic_shannon_entropy",
    "effective_topic_count",
    "minority_topic_survival",
    "stance_std",
    "bimodality_coefficient",
    "global_north_skew_shift",
    "goldman_effective_sources",
    "persuasion_bias_drift",
    "distinct_2",
    "pairwise_jaccard_distance",
)


def _build_peer_graph(num_agents: int, degree: int, seed: int) -> dict[int, list[int]]:
    """Build a lightweight regular-plus-chord peer communication graph."""
    if num_agents <= 1 or degree <= 0:
        return {i: [] for i in range(num_agents)}
    rng = random.Random(seed)
    eff_deg = min(degree, num_agents - 1)
    neighbors: dict[int, list[int]] = {}
    for i in range(num_agents):
        ring_nbrs = [(i + offset) % num_agents for offset in range(1, eff_deg + 1)]
        if num_agents > eff_deg + 2 and eff_deg >= 2:
            chord = rng.randrange(num_agents)
            if chord != i and chord not in ring_nbrs:
                ring_nbrs[-1] = chord
        neighbors[i] = ring_nbrs
    return neighbors


def _step_live_llm(
    backend: Any,
    cond: ConditionConfig,
    task: TaskItem,
    agent: AgentProfile,
    round_idx: int,
    prev_stance: float,
    prev_topics: list[str],
    peer_stances: list[float],
    model_assigned: str,
) -> dict[str, Any]:
    """Execute one assistant + agent step via a live chat-completions backend."""
    canonical = task.canonical_topics or ["general_analysis"]
    assistant_text = "No AI assistant in loop."
    if cond.heterogeneity_type != HeterogeneityType.NO_AI and round_idx > 0:
        style = (
            "Offer balanced, calibrated Socratic counter-arguments noting base rates."
            if cond.heterogeneity_type == HeterogeneityType.PERSONALIZED_ADVISOR
            else (
                "Respond warmly as an empathetic peer validating the user's intuition."
                if cond.heterogeneity_type == HeterogeneityType.PERSONALIZED_PEER
                else "Provide a concise, authoritative analytical perspective."
            )
        )
        asst_sys = (
            f"You are model '{model_assigned}'. {style} "
            f"Discuss relevant aspects from: {', '.join(canonical)}."
        )
        asst_usr = (
            f"Question: {task.question}\n"
            f"Hypothesis A: {task.hypothesis_a}\nHypothesis B: {task.hypothesis_b}\n"
            f"User context: region={agent.region}, current probability p(B)={prev_stance:.2f}."
        )
        asst_resp = backend.generate_chat(
            system_prompt=asst_sys,
            user_prompt=asst_usr,
            temperature=cond.temperature,
            top_p=cond.top_p,
            max_tokens=180,
        )
        assistant_text = asst_resp["text"]

    peer_avg = sum(peer_stances) / len(peer_stances) if peer_stances else prev_stance
    agent_sys = (
        f"Roleplay a {agent.profession} in {agent.region} valuing {', '.join(agent.values)}. "
        f"Return a JSON object with keys stance_prob (0..1 float), confidence (0.5..1 float), "
        f"answer (0 for A, 1 for B), and topics (1-3 items from {canonical}), followed by "
        f"2 sentences mentioning those topic phrases explicitly."
    )
    agent_usr = (
        f"Task: {task.question}\n"
        f"Hypothesis A: {task.hypothesis_a}\nHypothesis B: {task.hypothesis_b}\n"
        f"Your prior stance p(B)={prev_stance:.2f}; Peer mean p(B)={peer_avg:.2f}.\n"
        f"Assistant reply: {assistant_text[:400]}"
    )
    agent_resp = backend.generate_chat(
        system_prompt=agent_sys,
        user_prompt=agent_usr,
        temperature=cond.temperature,
        top_p=cond.top_p,
        max_tokens=220,
    )
    parsed = parse_structured_belief(agent_resp["text"])
    topics = [t for t in parsed.get("topics", []) if t in canonical] or (
        prev_topics or [canonical[0]]
    )
    raw_stance = float(parsed.get("stance_prob", prev_stance))
    stance = raw_stance if math.isfinite(raw_stance) else prev_stance
    raw_conf = float(parsed.get("confidence", 0.65))
    conf = raw_conf if math.isfinite(raw_conf) else 0.65
    return {
        "assistant_response_text": assistant_text,
        "agent_response_text": agent_resp["text"] or f"Focusing on {topics[0].replace('_', ' ')}.",
        "extracted_topics": topics,
        "stance_prob": round(clamp(stance, 0.01, 0.99), 4),
        "confidence": round(clamp(conf, 0.5, 0.99), 4),
        "stated_answer": 1 if stance >= 0.5 else 0,
        "goldman_source_overlap": 0.6 if cond.num_models == 1 else 0.25,
        "persuasion_bias_exposure": 0.5,
    }


def _aggregate_by_task(
    by_task: dict[str, list[tuple[int, RoundRecord]]],
    task_map: dict[str, TaskItem],
    topic_assignments: dict[int, list[str]],
    r0_by_task: dict[str, list[tuple[int, RoundRecord]]],
) -> dict[str, float]:
    """Compute per-task-averaged Track 4 metrics for one round."""
    if not by_task:
        return {}

    entropies: list[float] = []
    eff_topics: list[float] = []
    survivals: list[float] = []
    stance_stds: list[float] = []
    bimodalities: list[float] = []
    gn_skews: list[float] = []
    goldmans: list[float] = []
    bias_drifts: list[float] = []
    distinct2_list: list[float] = []
    jaccard_list: list[float] = []
    gt_probs: list[float] = []
    gt_truths: list[int] = []
    gt_conf_probs: list[float] = []

    for task_id, indexed_recs in by_task.items():
        task = task_map[task_id]
        recs = [rec for _, rec in indexed_recs]
        topic_lists = [
            topic_assignments.get(idx, rec.extracted_topics) for idx, rec in indexed_recs
        ]
        vocab = task.canonical_topics or sorted({t for tl in topic_lists for t in tl})

        t_stats = compute_topic_entropy_and_hill(topic_lists, vocab)
        entropies.append(t_stats["normalized_entropy"])
        eff_topics.append(t_stats["effective_topics"])

        r0_pairs = r0_by_task.get(task_id, [])
        r0_topics = [topic_assignments.get(idx, r0.extracted_topics) for idx, r0 in r0_pairs]
        survivals.append(compute_minority_topic_survival(r0_topics, topic_lists))

        probs = [r.stance_prob for r in recs]
        s_stats = compute_stance_stats(probs)
        stance_stds.append(s_stats["std"])
        bimodalities.append(s_stats["bimodality_coefficient"])

        r0_mean = sum(r0.stance_prob for _, r0 in r0_pairs) / len(r0_pairs) if r0_pairs else 0.5
        cur_mean = sum(probs) / len(probs)
        gn_sign = 1.0 if task.global_north_prior >= 0.5 else -1.0
        gn_skews.append((cur_mean - r0_mean) * gn_sign)

        goldmans.append(compute_goldman_effective_sources(recs))

        r0_by_agent = {r0.agent_id: r0.stance_prob for _, r0 in r0_pairs}
        drift_vals = [
            r.persuasion_bias_exposure * abs(r.stance_prob - r0_by_agent.get(r.agent_id, 0.5))
            for r in recs
        ]
        bias_drifts.append(sum(drift_vals) / len(drift_vals))

        lex = compute_lexical_diversity([r.agent_response_text for r in recs])
        distinct2_list.append(lex["distinct_2"])
        jaccard_list.append(lex["pairwise_jaccard_distance"])

        if task.kind == TaskKind.GROUND_TRUTH and task.ground_truth is not None:
            for r in recs:
                p_y1 = r.confidence if r.stated_answer == 1 else (1.0 - r.confidence)
                gt_probs.append(r.stance_prob)
                gt_conf_probs.append(p_y1)
                gt_truths.append(task.ground_truth)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "ground_truth_accuracy": compute_accuracy(gt_probs, gt_truths) if gt_truths else 0.0,
        "brier_score": compute_brier_score(gt_conf_probs, gt_truths) if gt_truths else 0.0,
        "ece": compute_ece(gt_conf_probs, gt_truths) if gt_truths else 0.0,
        "confident_wrong_rate": (
            compute_confident_wrong_rate(gt_conf_probs, gt_truths, threshold=0.75)
            if gt_truths
            else 0.0
        ),
        "topic_shannon_entropy": _mean(entropies),
        "effective_topic_count": _mean(eff_topics),
        "minority_topic_survival": _mean(survivals),
        "stance_std": _mean(stance_stds),
        "bimodality_coefficient": _mean(bimodalities),
        "global_north_skew_shift": _mean(gn_skews),
        "goldman_effective_sources": _mean(goldmans),
        "persuasion_bias_drift": _mean(bias_drifts),
        "distinct_2": _mean(distinct2_list),
        "pairwise_jaccard_distance": _mean(jaccard_list),
    }


def _compute_suite_contrasts(
    summaries: list[ConditionSummary],
    by_cond_round_task: dict[str, dict[int, dict[str, list[tuple[int, RoundRecord]]]]],
    task_map: dict[str, TaskItem],
    topic_assignments: dict[int, list[str]],
    final_round: int,
) -> list[StatisticalComparison]:
    """Compute paired Wilcoxon, bootstrap 95% CI, and Holm-Bonferroni Track 4 contrasts."""
    by_id = {s.condition_id: s for s in summaries}
    specs: list[tuple[str, str, str, str]] = []
    if "us_k1_greedy" in by_id and "us_k1" in by_id:
        specs.append(
            ("decoding_lexical_shift", "pairwise_jaccard_distance", "us_k1_greedy", "us_k1")
        )
    if "us_k1" in by_id and "us_k4" in by_id:
        specs.append(("us_k1_vs_k4_topics", "effective_topic_count", "us_k1", "us_k4"))
    if "us_k1" in by_id and "cross_national_k4" in by_id:
        specs.append(
            ("us_k1_vs_cross_nat_topics", "effective_topic_count", "us_k1", "cross_national_k4")
        )
        specs.append(
            (
                "us_k1_vs_cross_nat_goldman",
                "goldman_effective_sources",
                "us_k1",
                "cross_national_k4",
            )
        )
    if "us_k4" in by_id and "cross_national_k4" in by_id:
        specs.append(
            (
                "us_k4_vs_cross_nat_goldman",
                "goldman_effective_sources",
                "us_k4",
                "cross_national_k4",
            )
        )
    if "personalized_advisor" in by_id and "personalized_peer" in by_id:
        specs.append(
            (
                "advisor_vs_peer_conf_wrong",
                "confident_wrong_rate",
                "personalized_advisor",
                "personalized_peer",
            )
        )
        specs.append(
            ("advisor_vs_peer_brier", "brier_score", "personalized_advisor", "personalized_peer")
        )

    def _per_task_metric(cond_id: str, metric: str) -> dict[str, float]:
        out: dict[str, float] = {}
        grouped = by_cond_round_task[cond_id].get(final_round, {})
        for tid, indexed_recs in grouped.items():
            task = task_map[tid]
            trecs = [r for _, r in indexed_recs]
            if metric == "pairwise_jaccard_distance":
                out[tid] = compute_lexical_diversity([r.agent_response_text for r in trecs])[
                    "pairwise_jaccard_distance"
                ]
            elif metric == "effective_topic_count":
                t_lists = [
                    topic_assignments.get(idx, r.extracted_topics) for idx, r in indexed_recs
                ]
                vocab = task.canonical_topics or ["general"]
                out[tid] = compute_topic_entropy_and_hill(t_lists, vocab)["effective_topics"]
            elif metric == "goldman_effective_sources":
                out[tid] = compute_goldman_effective_sources(trecs)
            elif metric in ("confident_wrong_rate", "brier_score"):
                if task.kind != TaskKind.GROUND_TRUTH or task.ground_truth is None:
                    continue
                probs = [
                    r.confidence if r.stated_answer == 1 else (1.0 - r.confidence) for r in trecs
                ]
                truths = [task.ground_truth] * len(trecs)
                out[tid] = (
                    compute_confident_wrong_rate(probs, truths, threshold=0.75)
                    if metric == "confident_wrong_rate"
                    else compute_brier_score(probs, truths)
                )
        return out

    return compute_paired_holm_contrasts(
        specs,
        _per_task_metric,
        seed=42,
        effect_decimals=4,
        include_empty=False,
    )


def _run_condition_rounds(
    cfg: ExperimentSuiteConfig,
    cond: ConditionConfig,
    tasks: list[TaskItem],
    agents: list[AgentProfile],
    peer_graph: dict[int, list[int]],
    backend: Any,
) -> list[RoundRecord]:
    """Execute rounds `0..cfg.num_rounds` for a single Track 4 condition."""
    model_pool = assign_models_for_condition(
        nationality_type=cond.nationality_type,
        num_models=cond.num_models,
        explicit_models=cond.models or None,
    )
    is_personalized = cond.heterogeneity_type in (
        HeterogeneityType.PERSONALIZED_ADVISOR,
        HeterogeneityType.PERSONALIZED_PEER,
        HeterogeneityType.PERSONALIZED_PROFILE,
    )
    cond_records: list[RoundRecord] = []

    for task in tasks:
        agent_stance = [a.prior_belief for a in agents]
        agent_conf = [a.prior_confidence for a in agents]
        agent_topics: list[list[str]] = [
            [task.canonical_topics[0]] if task.canonical_topics else ["general"] for _ in agents
        ]

        for round_idx in range(cfg.num_rounds + 1):
            round_stances = list(agent_stance)
            round_topics = [list(tl) for tl in agent_topics]

            for a_idx, agent in enumerate(agents):
                base_model, base_family = model_pool[a_idx % len(model_pool)]
                model_assigned = (
                    f"{base_model}@persona-{agent.agent_id}"
                    if is_personalized and base_model != "none"
                    else base_model
                )
                nbrs = peer_graph[a_idx] if (cond.peer_exchange and round_idx > 0) else []
                peer_st = [round_stances[n] for n in nbrs]
                peer_tp = [round_topics[n] for n in nbrs]

                if isinstance(backend, ReplayBackend):
                    replayed = backend.get_round_record(
                        cond.condition_id, task.id, round_idx, agent.agent_id
                    )
                    if replayed is None:
                        miss_key = (cond.condition_id, task.id, round_idx, agent.agent_id)
                        raise RuntimeError(f"Replay miss for {miss_key}")
                    rec = replayed
                else:
                    if hasattr(backend, "generate_agent_step"):
                        step = backend.generate_agent_step(
                            cond=cond,
                            task=task,
                            agent=agent,
                            agent_idx=a_idx,
                            round_idx=round_idx,
                            prev_stance=round_stances[a_idx],
                            prev_confidence=agent_conf[a_idx],
                            prev_topics=round_topics[a_idx],
                            peer_stances=peer_st,
                            peer_topics=peer_tp,
                            model_assigned=model_assigned,
                            model_family=base_family,
                        )
                    else:
                        step = _step_live_llm(
                            backend=backend,
                            cond=cond,
                            task=task,
                            agent=agent,
                            round_idx=round_idx,
                            prev_stance=round_stances[a_idx],
                            prev_topics=round_topics[a_idx],
                            peer_stances=peer_st,
                            model_assigned=model_assigned,
                        )
                    rec = RoundRecord(
                        run_id=f"{cond.condition_id}_{task.id}_r{round_idx}",
                        condition_id=cond.condition_id,
                        seed=cfg.seed,
                        task_id=task.id,
                        task_kind=task.kind,
                        round_idx=round_idx,
                        agent_id=agent.agent_id,
                        agent_region=agent.region,
                        model_assigned=model_assigned,
                        model_family=base_family,
                        assistant_response_text=step["assistant_response_text"],
                        agent_response_text=step["agent_response_text"],
                        extracted_topics=step["extracted_topics"],
                        stance_prob=step["stance_prob"],
                        confidence=step["confidence"],
                        stated_answer=step["stated_answer"],
                        goldman_source_overlap=step["goldman_source_overlap"],
                        persuasion_bias_exposure=step["persuasion_bias_exposure"],
                        metadata={
                            "temperature": cond.temperature,
                            "top_p": cond.top_p,
                            "heterogeneity_type": cond.heterogeneity_type.value,
                            "nationality_type": cond.nationality_type.value,
                        },
                    )

                agent_stance[a_idx] = rec.stance_prob
                agent_conf[a_idx] = rec.confidence
                agent_topics[a_idx] = list(rec.extracted_topics)
                cond_records.append(rec)

    return cond_records


def run_suite(
    cfg: ExperimentSuiteConfig,
    backend: Any,
    out_dir: Path,
) -> dict[str, Any]:
    """Run all Track 4 experimental conditions end-to-end and export summary artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = (
        build_ground_truth_tasks()[: cfg.tasks_per_track]
        + (build_open_debate_tasks()[: cfg.tasks_per_track])
    )
    task_map = {t.id: t for t in tasks}

    agents = generate_agent_population(num_agents=cfg.num_agents, seed=cfg.seed)
    peer_graph = _build_peer_graph(cfg.num_agents, cfg.peer_degree, seed=cfg.seed)

    all_records: list[RoundRecord] = []
    by_cond_round_task: dict[str, dict[int, dict[str, list[tuple[int, RoundRecord]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    with (out_dir / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for cond in cfg.conditions:
            cond_recs = _run_condition_rounds(cfg, cond, tasks, agents, peer_graph, backend)
            for rec in cond_recs:
                global_idx = len(all_records)
                all_records.append(rec)
                by_cond_round_task[cond.condition_id][rec.round_idx][rec.task_id].append(
                    (global_idx, rec)
                )
                fh.write(rec.model_dump_json() + "\n")

    topic_audit = evaluate_dual_judge_topics(all_records, seed=cfg.seed)
    export_audit_sample(
        all_records,
        n=min(40, len(all_records)),
        out_path=out_dir / "audit_human_check.csv",
        seed=cfg.seed,
    )

    assignments: dict[int, list[str]] = topic_audit["assignments"]
    summaries: list[ConditionSummary] = []

    for cond in cfg.conditions:
        cond_rounds = by_cond_round_task[cond.condition_id]
        r0_by_task = cond_rounds.get(0, {})
        trajectory: list[dict[str, float]] = []
        for r_idx in range(cfg.num_rounds + 1):
            rm = _aggregate_by_task(
                by_task=cond_rounds.get(r_idx, {}),
                task_map=task_map,
                topic_assignments=assignments,
                r0_by_task=r0_by_task,
            )
            rm["round_idx"] = float(r_idx)
            trajectory.append({k: round(v, 4) for k, v in rm.items()})

        final_metrics = trajectory[-1]
        summaries.append(
            ConditionSummary(
                condition_id=cond.condition_id,
                label=cond.label,
                heterogeneity_type=cond.heterogeneity_type.value,
                nationality_type=cond.nationality_type.value,
                num_models=cond.num_models,
                temperature=cond.temperature,
                top_p=cond.top_p,
                group=cond.group,
                ground_truth_accuracy=final_metrics["ground_truth_accuracy"],
                brier_score=final_metrics["brier_score"],
                ece=final_metrics["ece"],
                confident_wrong_rate=final_metrics["confident_wrong_rate"],
                topic_shannon_entropy=final_metrics["topic_shannon_entropy"],
                effective_topic_count=final_metrics["effective_topic_count"],
                minority_topic_survival=final_metrics["minority_topic_survival"],
                stance_std=final_metrics["stance_std"],
                bimodality_coefficient=final_metrics["bimodality_coefficient"],
                global_north_skew_shift=final_metrics["global_north_skew_shift"],
                goldman_effective_sources=final_metrics["goldman_effective_sources"],
                persuasion_bias_drift=final_metrics["persuasion_bias_drift"],
                distinct_2=final_metrics["distinct_2"],
                pairwise_jaccard_distance=final_metrics["pairwise_jaccard_distance"],
                round_trajectory=trajectory,
            )
        )

    comparisons = _compute_suite_contrasts(
        summaries=summaries,
        by_cond_round_task=by_cond_round_task,
        task_map=task_map,
        topic_assignments=assignments,
        final_round=cfg.num_rounds,
    )

    summary_payload = {
        "suite_name": cfg.suite_name,
        "seed": cfg.seed,
        "num_agents": cfg.num_agents,
        "num_rounds": cfg.num_rounds,
        "summaries": [s.model_dump() for s in summaries],
        "comparisons": [c.model_dump() for c in comparisons],
        "topic_audit": {
            "canonical_topics": topic_audit["canonical_topics"],
            "inter_rater_kappa": topic_audit["inter_rater_kappa"],
            "reliability_warning": topic_audit["reliability_warning"],
            "n_evaluated": topic_audit["n_evaluated"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for s in summaries:
            data = s.model_dump()
            writer.writerow({k: data[k] for k in _SUMMARY_CSV_FIELDS})

    return {
        "summaries": summaries,
        "records": all_records,
        "topic_audit": topic_audit,
        "comparisons": comparisons,
    }
