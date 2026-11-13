"""Unified multi-round simulation engine, prompt builders, and statistical analyzers."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from apart_epistemics.backends import BudgetExceededError, ReplayBackend
from apart_epistemics.datasets import (
    assign_models_for_condition,
    build_ground_truth_tasks,
    build_open_debate_tasks,
    generate_agent_population,
)
from apart_epistemics.metrics import (
    bootstrap_ci,
    compute_accuracy,
    compute_brier_score,
    compute_confident_wrong_rate,
    compute_ece,
    compute_goldman_effective_sources,
    compute_lexical_diversity,
    compute_minority_topic_survival,
    compute_page_decomposition,
    compute_stance_stats,
    compute_topic_entropy_and_hill,
    holm_bonferroni,
    paired_wilcoxon_p,
    pearson_corr,
)
from apart_epistemics.models import (
    AgentProfile,
    ConditionConfig,
    ConditionSummary,
    ExperimentSuiteConfig,
    HeterogeneityType,
    NumericEstimateRecord,
    RoundRecord,
    StatisticalComparison,
    TaskItem,
    TaskKind,
)
from apart_epistemics.parsers import parse_numeric_estimate, parse_structured_belief
from apart_epistemics.summarizer import evaluate_dual_judge_topics, export_audit_sample


def _build_peer_graph(num_agents: int, degree: int, seed: int) -> dict[int, list[int]]:
    """Build an undirected regular-plus-chord peer communication graph."""
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
    peer_topics: list[list[str]],
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
    stance = float(parsed.get("stance_prob", prev_stance))
    conf = float(parsed.get("confidence", 0.65))
    return {
        "assistant_response_text": assistant_text,
        "agent_response_text": agent_resp["text"] or f"Focusing on {topics[0].replace('_', ' ')}.",
        "extracted_topics": topics,
        "stance_prob": round(max(0.01, min(0.99, stance)), 4),
        "confidence": round(max(0.5, min(0.99, conf)), 4),
        "stated_answer": 1 if stance >= 0.5 else 0,
        "goldman_source_overlap": 0.6 if cond.num_models == 1 else 0.25,
        "persuasion_bias_exposure": 0.5,
    }


def _aggregate_records_for_round(
    records: list[RoundRecord],
    task_map: dict[str, TaskItem],
    topic_assignments: dict[int, list[str]],
    record_indices: dict[int, int],
    r0_by_task: dict[str, list[RoundRecord]],
    target_round: int,
) -> dict[str, float]:
    """Compute per-task-averaged Track 4 metrics at a single simulation round."""
    by_task: dict[str, list[tuple[int, RoundRecord]]] = defaultdict(list)
    for rec in records:
        if rec.round_idx == target_round:
            orig_idx = record_indices[id(rec)]
            by_task[rec.task_id].append((orig_idx, rec))

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

        r0_recs = r0_by_task.get(task_id, [])
        r0_topics = [
            topic_assignments.get(record_indices[id(r0)], r0.extracted_topics) for r0 in r0_recs
        ]
        survivals.append(compute_minority_topic_survival(r0_topics, topic_lists))

        probs = [r.stance_prob for r in recs]
        s_stats = compute_stance_stats(probs)
        stance_stds.append(s_stats["std"])
        bimodalities.append(s_stats["bimodality_coefficient"])

        r0_mean = sum(r.stance_prob for r in r0_recs) / len(r0_recs) if r0_recs else 0.5
        cur_mean = sum(probs) / len(probs)
        gn_sign = 1.0 if task.global_north_prior >= 0.5 else -1.0
        gn_skews.append((cur_mean - r0_mean) * gn_sign)

        goldmans.append(compute_goldman_effective_sources(recs))

        r0_by_agent = {r.agent_id: r.stance_prob for r in r0_recs}
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
                # Reconstruct signed P(y=1) weighted by stated confidence vs prior stance.
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
    records_by_cond: dict[str, list[RoundRecord]],
    task_map: dict[str, TaskItem],
    topic_assignments: dict[int, list[str]],
    record_indices: dict[int, int],
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

    if not specs:
        return []

    def _per_task_metric(cond_id: str, metric: str) -> dict[str, float]:
        out: dict[str, float] = {}
        recs = [r for r in records_by_cond[cond_id] if r.round_idx == final_round]
        grouped: dict[str, list[RoundRecord]] = defaultdict(list)
        for r in recs:
            grouped[r.task_id].append(r)
        for tid, trecs in grouped.items():
            task = task_map[tid]
            if metric == "pairwise_jaccard_distance":
                out[tid] = compute_lexical_diversity([r.agent_response_text for r in trecs])[
                    "pairwise_jaccard_distance"
                ]
            elif metric == "effective_topic_count":
                t_lists = [
                    topic_assignments.get(record_indices[id(r)], r.extracted_topics) for r in trecs
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

    raw_results: list[dict[str, Any]] = []
    for cid, metric, cond_a, cond_b in specs:
        ma = _per_task_metric(cond_a, metric)
        mb = _per_task_metric(cond_b, metric)
        common = sorted(set(ma) & set(mb))
        if not common:
            continue
        xs = [ma[t] for t in common]
        ys = [mb[t] for t in common]
        diffs = [y - x for x, y in zip(xs, ys, strict=True)]
        ci_lo, ci_hi = bootstrap_ci(diffs, seed=42)
        p_val = paired_wilcoxon_p(xs, ys)
        raw_results.append(
            {
                "contrast_id": cid,
                "metric": metric,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "mean_diff": sum(diffs) / len(diffs),
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "wilcoxon_p": p_val,
                "n_cells": len(common),
            }
        )

    holm_ps = holm_bonferroni([item["wilcoxon_p"] for item in raw_results])
    comparisons: list[StatisticalComparison] = []
    for item, adj_p in zip(raw_results, holm_ps, strict=True):
        comparisons.append(
            StatisticalComparison(
                contrast_id=item["contrast_id"],
                metric=item["metric"],
                condition_a=item["condition_a"],
                condition_b=item["condition_b"],
                mean_diff=round(item["mean_diff"], 4),
                ci_low=round(item["ci_low"], 4),
                ci_high=round(item["ci_high"], 4),
                wilcoxon_p=round(item["wilcoxon_p"], 6),
                holm_adjusted_p=round(adj_p, 6),
                n_cells=item["n_cells"],
            )
        )
    return comparisons


def run_suite(
    cfg: ExperimentSuiteConfig,
    backend: Any,
    out_dir: Path,
) -> dict[str, Any]:
    """Run all Track 4 experimental conditions end-to-end and export summary artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_tasks = build_ground_truth_tasks()[: cfg.tasks_per_track]
    debate_tasks = build_open_debate_tasks()[: cfg.tasks_per_track]
    tasks = gt_tasks + debate_tasks
    task_map = {t.id: t for t in tasks}

    agents = generate_agent_population(num_agents=cfg.num_agents, seed=cfg.seed)
    peer_graph = _build_peer_graph(cfg.num_agents, cfg.peer_degree, seed=cfg.seed)

    all_records: list[RoundRecord] = []
    records_by_cond: dict[str, list[RoundRecord]] = defaultdict(list)

    for cond in cfg.conditions:
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

        for task in tasks:
            # Per-agent rolling state across rounds 0..num_rounds.
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
                    if is_personalized and base_model != "none":
                        model_assigned = f"{base_model}@persona-{agent.agent_id}"
                    else:
                        model_assigned = base_model

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
                                peer_topics=peer_tp,
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
                    all_records.append(rec)
                    records_by_cond[cond.condition_id].append(rec)

    # Persist JSONL interaction logs.
    runs_path = out_dir / "runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(rec.model_dump_json() + "\n")

    # Run blinded dual-judge topic audit across all records.
    topic_audit = evaluate_dual_judge_topics(all_records, seed=cfg.seed)
    export_audit_sample(
        all_records,
        n=min(40, len(all_records)),
        out_path=out_dir / "audit_human_check.csv",
        seed=cfg.seed,
    )

    record_indices = {id(rec): idx for idx, rec in enumerate(all_records)}
    assignments: dict[int, list[str]] = topic_audit["assignments"]

    summaries: list[ConditionSummary] = []
    for cond in cfg.conditions:
        cond_recs = records_by_cond[cond.condition_id]
        r0_by_task: dict[str, list[RoundRecord]] = defaultdict(list)
        for r in cond_recs:
            if r.round_idx == 0:
                r0_by_task[r.task_id].append(r)

        trajectory: list[dict[str, float]] = []
        for r_idx in range(cfg.num_rounds + 1):
            rm = _aggregate_records_for_round(
                records=cond_recs,
                task_map=task_map,
                topic_assignments=assignments,
                record_indices=record_indices,
                r0_by_task=r0_by_task,
                target_round=r_idx,
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
        records_by_cond=records_by_cond,
        task_map=task_map,
        topic_assignments=assignments,
        record_indices=record_indices,
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

    csv_fields = [
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
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for s in summaries:
            data = s.model_dump()
            writer.writerow({k: data[k] for k in csv_fields})

    return {
        "summaries": summaries,
        "records": all_records,
        "topic_audit": topic_audit,
        "comparisons": comparisons,
    }


def build_numeric_prompts(
    condition_id: str,
    agent: AgentProfile,
    question: str,
    current_x: float,
    pop_stats: tuple[float, float, float] | None,
    history: list[tuple[str, str]],
) -> tuple[str, str]:
    """Construct C1 (baseline), C2 (social summary), or C3 (personalized+history) prompts."""
    cond = condition_id.upper()
    est_str = f"{current_x:g}"

    if cond == "C3":
        system_prompt = (
            f"You are an analytical estimation advisor speaking with a {agent.profession} "
            f"in {agent.region} (age band {agent.age_band}) who values "
            f"{', '.join(agent.values) or 'evidence'}. "
            f"Help them reason accurately and end cleanly with 'Final estimate: <number>'."
        )
    else:
        system_prompt = (
            "You are a neutral quantitative estimation assistant. "
            "Reason step by step without web lookup and end with 'Final estimate: <number>'."
        )

    parts = [f"Question: {question}"]
    if cond in ("C2", "C3") and pop_stats is not None:
        med, q25, q75 = pop_stats
        parts.append(
            f"Other people asking this recently gave a median of {med:g} (IQR {q25:g} to {q75:g})."
        )
    if cond == "C3" and history:
        recent = history[-2:]
        hist_lines = [
            f"Turn {idx + 1} User: {u} | Assistant: {a}" for idx, (u, a) in enumerate(recent)
        ]
        parts.append("Recent conversation history:\n" + "\n".join(hist_lines))

    parts.append(f"My current estimate is {est_str}. What should my updated numerical estimate be?")
    return (system_prompt, "\n\n".join(parts))


def _load_existing_numeric_records(
    runs_path: Path,
) -> dict[tuple[str, str, int, int, str], NumericEstimateRecord]:
    existing: dict[tuple[str, str, int, int, str], NumericEstimateRecord] = {}
    if not runs_path.exists():
        return existing
    raw_lines = runs_path.read_text(encoding="utf-8").splitlines()
    had_truncated_tail = False
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = NumericEstimateRecord.model_validate_json(stripped)
        except ValidationError:
            # Tolerate a single truncated final line if a previous run died mid-write.
            if idx == len(raw_lines) - 1:
                had_truncated_tail = True
                continue
            raise
        key = (rec.condition_id, rec.question_id, rec.replicate, rec.round_idx, rec.agent_id)
        existing[key] = rec
    if had_truncated_tail:
        clean_lines = [r.model_dump_json() for r in existing.values()]
        runs_path.write_text(
            "\n".join(clean_lines) + ("\n" if clean_lines else ""), encoding="utf-8"
        )
    return existing


def _quantile_sorted(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _estimate_call_cost_usd(resp: dict[str, Any]) -> float:
    in_tok = int(resp.get("input_tokens", 0) or 0)
    out_tok = int(resp.get("output_tokens", 0) or 0) + int(resp.get("reasoning_tokens", 0) or 0)
    return max(1e-5, (in_tok * 0.5 + out_tok * 1.5) / 1_000_000.0)


def run_numeric_core(
    tasks: list[TaskItem],
    conditions: list[str],
    n_agents: int,
    n_rounds: int,
    replicates: int,
    runs_path: Path,
    backend: Any,
    alpha: float = 0.3,
    budget_usd: float = 50.0,
    seed: int = 42,
) -> list[NumericEstimateRecord]:
    """Run crash-resumable C1/C2/C3 numeric estimation sweeps with retry-once and budget cap."""
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    agents = generate_agent_population(num_agents=n_agents, seed=seed)
    existing = _load_existing_numeric_records(runs_path)

    spent_usd = 0.0
    ordered_records: list[NumericEstimateRecord] = []

    with runs_path.open("a", encoding="utf-8") as fh:
        for cond_id in conditions:
            for task in tasks:
                truth = task.numeric_truth or 100.0
                log_truth = math.log10(truth)

                for rep in range(replicates):
                    init_rng = random.Random(
                        seed * 313 + sum(ord(c) for c in task.id) * 19 + rep * 7
                    )
                    # Same Round 0 crowd priors across C1/C2/C3 per (question, replicate).
                    current_logs = [
                        log_truth
                        + 0.22 * task.popular_misconception_bias
                        + init_rng.gauss(0.0, 0.24)
                        for _ in range(n_agents)
                    ]
                    histories: list[list[tuple[str, str]]] = [[] for _ in range(n_agents)]

                    for r_idx in range(1, n_rounds + 1):
                        sorted_x = sorted(10.0**lg for lg in current_logs)
                        pop_stats = (
                            statistics.median(sorted_x),
                            _quantile_sorted(sorted_x, 0.25),
                            _quantile_sorted(sorted_x, 0.75),
                        )

                        for a_idx, agent in enumerate(agents):
                            key = (cond_id, task.id, rep, r_idx, agent.agent_id)
                            if key in existing:
                                rec = existing[key]
                                current_logs[a_idx] = rec.agent_log10_after
                                histories[a_idx].append(
                                    (
                                        f"My current estimate is {10.0**rec.agent_log10_before:g}.",
                                        rec.raw_reply,
                                    )
                                )
                                ordered_records.append(rec)
                                continue

                            if budget_usd <= 0.0 or spent_usd >= budget_usd:
                                raise BudgetExceededError(
                                    f"Budget cap exhausted (spent={spent_usd:.4f}, "
                                    f"cap={budget_usd:.4f})."
                                )

                            before_log = current_logs[a_idx]
                            cur_x = 10.0**before_log
                            sys_p, usr_p = build_numeric_prompts(
                                condition_id=cond_id,
                                agent=agent,
                                question=task.question,
                                current_x=cur_x,
                                pop_stats=pop_stats if cond_id in ("C2", "C3") else None,
                                history=histories[a_idx],
                            )
                            resp = backend.generate_numeric_reply(
                                system_prompt=sys_p,
                                user_prompt=usr_p,
                                condition_id=cond_id,
                                task=task,
                                round_idx=r_idx,
                                current_x=cur_x,
                                pop_stats=pop_stats,
                                agent=agent,
                            )
                            spent_usd += _estimate_call_cost_usd(resp)
                            raw_reply = str(resp.get("raw_reply", ""))
                            parsed = parse_numeric_estimate(raw_reply, require_positive=True)

                            # Retry once with explicit formatting instruction on parse failure.
                            if parsed.miss:
                                if spent_usd >= budget_usd:
                                    raise BudgetExceededError("Budget cap exhausted during retry.")
                                retry_usr = (
                                    usr_p
                                    + "\n\nRespond with a single positive number prefixed by "
                                    + "'Final estimate: '."
                                )
                                resp = backend.generate_numeric_reply(
                                    system_prompt=sys_p,
                                    user_prompt=retry_usr,
                                    condition_id=cond_id,
                                    task=task,
                                    round_idx=r_idx,
                                    current_x=cur_x,
                                    pop_stats=pop_stats,
                                    agent=agent,
                                )
                                spent_usd += _estimate_call_cost_usd(resp)
                                raw_reply = str(resp.get("raw_reply", ""))
                                parsed = parse_numeric_estimate(raw_reply, require_positive=True)

                            if parsed.miss or parsed.log10_value is None:
                                after_log = before_log
                                asst_log = None
                                parsed_val = None
                                is_miss = True
                            else:
                                asst_log = parsed.log10_value
                                parsed_val = parsed.value
                                after_log = (1.0 - alpha) * before_log + alpha * asst_log
                                is_miss = False

                            current_logs[a_idx] = after_log
                            histories[a_idx].append(
                                (f"My current estimate is {cur_x:g}.", raw_reply)
                            )

                            rec = NumericEstimateRecord(
                                condition_id=cond_id,
                                question_id=task.id,
                                replicate=rep,
                                round_idx=r_idx,
                                agent_id=agent.agent_id,
                                model_slug=str(resp.get("model_slug", "synthetic")),
                                provider=str(resp.get("provider", "synthetic_sim")),
                                raw_reply=raw_reply,
                                parsed_value=parsed_val,
                                assistant_log10=asst_log,
                                agent_log10_before=before_log,
                                agent_log10_after=after_log,
                                miss=is_miss,
                                reasoning_tokens=int(resp.get("reasoning_tokens", 0) or 0),
                                input_tokens=int(resp.get("input_tokens", 0) or 0),
                                output_tokens=int(resp.get("output_tokens", 0) or 0),
                            )
                            existing[key] = rec
                            ordered_records.append(rec)
                            fh.write(rec.model_dump_json() + "\n")
                            fh.flush()

    return ordered_records


def analyze_numeric_core(
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
) -> dict[str, Any]:
    """Compute Page decomposition per cell, H4 correlation, and 4 Holm-corrected contrasts."""
    truth_by_qid = {t.id: math.log10(t.numeric_truth) for t in tasks if t.numeric_truth is not None}
    by_cell: dict[tuple[str, str, int, int], list[NumericEstimateRecord]] = defaultdict(list)
    for rec in records:
        by_cell[(rec.condition_id, rec.question_id, rec.replicate, rec.round_idx)].append(rec)

    cell_metrics: dict[tuple[str, str, int, int], dict[str, float]] = {}
    max_resid = 0.0
    for key, cell_recs in by_cell.items():
        _, qid, _, _ = key
        log_truth = truth_by_qid.get(qid, 0.0)
        decomp = compute_page_decomposition(
            [r.agent_log10_after for r in cell_recs],
            log_truth=log_truth,
        )
        decomp["miss_rate"] = sum(1 for r in cell_recs if r.miss) / len(cell_recs)
        max_resid = max(max_resid, decomp["identity_residual"])
        cell_metrics[key] = decomp

    # H4 correlation between assistant log10 suggestion and agent's posterior log10.
    valid_recs = [r for r in records if not r.miss and r.assistant_log10 is not None]
    h4_corr = pearson_corr(
        [float(r.assistant_log10) for r in valid_recs],
        [r.agent_log10_after for r in valid_recs],
    )

    max_round = max((r.round_idx for r in records), default=1)
    by_cond_round: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for (cond_id, _qid, _rep, r_idx), m in cell_metrics.items():
        by_cond_round[(cond_id, r_idx)].append(m)

    condition_round_metrics: list[dict[str, Any]] = []
    for (cond_id, r_idx), ms in sorted(by_cond_round.items()):
        n = len(ms)
        condition_round_metrics.append(
            {
                "condition_id": cond_id,
                "round_idx": r_idx,
                "crowd_error": round(sum(m["crowd_error"] for m in ms) / n, 6),
                "avg_individual_error": round(sum(m["avg_individual_error"] for m in ms) / n, 6),
                "diversity": round(sum(m["diversity"] for m in ms) / n, 6),
                "iqr": round(sum(m["iqr"] for m in ms) / n, 6),
                "miss_rate": round(sum(m["miss_rate"] for m in ms) / n, 6),
            }
        )

    by_condition_final = {
        row["condition_id"]: row for row in condition_round_metrics if row["round_idx"] == max_round
    }

    # Confirmatory family of 4 core Holm-corrected contrasts on final-round Page terms.
    core_pairs = [
        ("c2_vs_c1_diversity", "diversity", "C1", "C2"),
        ("c2_vs_c1_crowd_error", "crowd_error", "C1", "C2"),
        ("c3_vs_c1_diversity", "diversity", "C1", "C3"),
        ("c3_vs_c1_crowd_error", "crowd_error", "C1", "C3"),
    ]
    raw_tests: list[dict[str, Any]] = []
    for cid, metric, ca, cb in core_pairs:
        cells_a = {
            (qid, rep): m[metric]
            for (c, qid, rep, r_idx), m in cell_metrics.items()
            if c == ca and r_idx == max_round
        }
        cells_b = {
            (qid, rep): m[metric]
            for (c, qid, rep, r_idx), m in cell_metrics.items()
            if c == cb and r_idx == max_round
        }
        common = sorted(set(cells_a) & set(cells_b))
        xs = [cells_a[k] for k in common] if common else [0.0]
        ys = [cells_b[k] for k in common] if common else [0.0]
        diffs = [y - x for x, y in zip(xs, ys, strict=True)]
        ci_lo, ci_hi = bootstrap_ci(diffs, seed=42)
        p_val = paired_wilcoxon_p(xs, ys) if common else 1.0
        raw_tests.append(
            {
                "contrast_id": cid,
                "metric": metric,
                "condition_a": ca,
                "condition_b": cb,
                "mean_diff": sum(diffs) / len(diffs),
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "wilcoxon_p": p_val,
                "n_cells": len(common),
            }
        )

    adj_ps = holm_bonferroni([t["wilcoxon_p"] for t in raw_tests])
    contrasts = [
        StatisticalComparison(
            contrast_id=t["contrast_id"],
            metric=t["metric"],
            condition_a=t["condition_a"],
            condition_b=t["condition_b"],
            mean_diff=round(t["mean_diff"], 6),
            ci_low=round(t["ci_low"], 6),
            ci_high=round(t["ci_high"], 6),
            wilcoxon_p=round(t["wilcoxon_p"], 6),
            holm_adjusted_p=round(adj_p, 6),
            n_cells=t["n_cells"],
        )
        for t, adj_p in zip(raw_tests, adj_ps, strict=True)
    ]

    return {
        "max_identity_residual": max_resid,
        "h4_assistant_agent_corr": round(h4_corr, 4),
        "by_condition_final": by_condition_final,
        "condition_round_metrics": condition_round_metrics,
        "contrasts": contrasts,
    }
