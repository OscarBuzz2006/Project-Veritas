"""Blinded, two-pass dual-judge topic summarization and human-audit exporter."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from apart_epistemics.metrics import cohens_kappa
from apart_epistemics.models import RoundRecord

JudgeFn = Callable[[str, list[str]], list[str]]


def blind_and_shuffle_records(records: list[RoundRecord], seed: int = 42) -> list[dict[str, Any]]:
    """Strip all condition/model/nationality/round metadata and shuffle deterministically."""
    blinded: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        digest = hashlib.sha256(f"{seed}:{idx}:{rec.agent_response_text}".encode()).hexdigest()[:12]
        blinded.append(
            {
                "blind_id": f"item_{digest}",
                "orig_index": idx,
                "task_id": rec.task_id,
                "blind_text": rec.agent_response_text,
                "judge_payload": {
                    "blind_id": f"item_{digest}",
                    "text": rec.agent_response_text,
                },
                "seed_topics": list(rec.extracted_topics),
            }
        )
    rng = random.Random(seed)
    rng.shuffle(blinded)
    return blinded


def _default_keyword_judge(text: str, vocabulary: list[str]) -> list[str]:
    low = text.lower()
    matched: list[str] = []
    for topic in vocabulary:
        parts = [p for p in topic.lower().split("_") if len(p) > 3]
        if parts and any(p in low for p in parts):
            matched.append(topic)
    return matched if matched else ([vocabulary[0]] if vocabulary else ["general"])


def _secondary_keyword_judge(text: str, vocabulary: list[str]) -> list[str]:
    low = text.lower()
    matched: list[str] = []
    for topic in vocabulary:
        parts = [p for p in topic.lower().split("_") if len(p) > 3]
        hits = sum(1 for p in parts if p in low)
        if hits >= 1:
            matched.append(topic)
    return matched if matched else ([vocabulary[0]] if vocabulary else ["general"])


def consolidate_global_codebook(records: list[RoundRecord]) -> list[str]:
    """Consolidate unique conceptual topics across all conditions and rounds into one codebook."""
    topics = sorted({t for rec in records for t in rec.extracted_topics if t})
    return topics if topics else ["general_discourse"]


def evaluate_dual_judge_topics(
    records: list[RoundRecord],
    seed: int = 42,
    kappa_threshold: float = 0.70,
    judge_a_override: JudgeFn | None = None,
    judge_b_override: JudgeFn | None = None,
) -> dict[str, Any]:
    """Run blinded two-pass topic evaluation with dual judges and kappa reliability gate."""
    canonical = consolidate_global_codebook(records)
    blinded = blind_and_shuffle_records(records, seed=seed)

    judge_a = judge_a_override or _default_keyword_judge
    judge_b = judge_b_override or _secondary_keyword_judge

    labels_a: list[str] = []
    labels_b: list[str] = []
    assignments: dict[int, list[str]] = {}

    for item in blinded:
        text = item["judge_payload"]["text"]
        pred_a = judge_a(text, canonical)
        pred_b = judge_b(text, canonical)
        # Seed labels may describe content, but cannot verify their own reliability.
        if judge_a_override is None:
            pred_a = item["seed_topics"] or pred_a
        labels_a.append("|".join(sorted(set(pred_a))))
        labels_b.append("|".join(sorted(set(pred_b))))
        assignments[item["orig_index"]] = pred_a

    independent = (
        judge_a_override is not None
        and judge_b_override is not None
        and judge_a_override is not judge_b_override
    )
    kappa = cohens_kappa(labels_a, labels_b) if independent and records else None
    return {
        "canonical_topics": canonical,
        "inter_rater_kappa": round(kappa, 4) if kappa is not None else None,
        "reliability_status": "independently_judged"
        if kappa is not None
        else "not_independently_judged",
        "label_basis": "independent judge"
        if judge_a_override
        else "self-reported or synthetic seed topics",
        "reliability_warning": kappa is None or kappa < kappa_threshold,
        "assignments": assignments,
        "n_evaluated": len(blinded),
    }


def export_audit_sample(
    records: list[RoundRecord],
    n: int,
    out_path: Path,
    seed: int = 42,
) -> Path:
    """Export condition-stratified blinded responses for human verification of LLM topic labels."""
    blinded = blind_and_shuffle_records(records, seed=seed)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in blinded:
        rec = records[item["orig_index"]]
        strata[rec.condition_id].append(item)
    keys = sorted(strata)
    random.Random(seed).shuffle(keys)
    sample: list[dict[str, Any]] = []
    while len(sample) < min(max(0, n), len(blinded)):
        for key in keys:
            if strata[key] and len(sample) < n:
                sample.append(strata[key].pop())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            lineterminator="\n",
            fieldnames=["blind_id", "task_id", "response_text", "llm_topics", "human_topics"],
        )
        writer.writeheader()
        for item in sample:
            writer.writerow(
                {
                    "blind_id": item["blind_id"],
                    "task_id": item["task_id"],
                    "response_text": item["blind_text"],
                    "llm_topics": "|".join(item["seed_topics"]),
                    "human_topics": "",
                }
            )
    return out_path
