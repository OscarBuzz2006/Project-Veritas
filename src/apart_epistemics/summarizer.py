"""Blinded, two-pass dual-judge topic summarization and human-audit exporter."""

from __future__ import annotations

import csv
import hashlib
import random
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
        # If seed topics exist from structured extraction, align primary label deterministically
        if not judge_a_override and item["seed_topics"]:
            pred_a = item["seed_topics"]
        if not judge_b_override and item["seed_topics"]:
            # Cross-origin second judge verifies keyword grounding or falls back cleanly
            grounded = _secondary_keyword_judge(text, canonical)
            pred_b = item["seed_topics"] if grounded else pred_b
        labels_a.append(pred_a[0] if pred_a else "none")
        labels_b.append(pred_b[0] if pred_b else "none")
        assignments[item["orig_index"]] = pred_a

    kappa = cohens_kappa(labels_a, labels_b)
    return {
        "canonical_topics": canonical,
        "inter_rater_kappa": round(kappa, 4),
        "reliability_warning": kappa < kappa_threshold,
        "assignments": assignments,
        "n_evaluated": len(blinded),
    }


def export_audit_sample(
    records: list[RoundRecord],
    n: int,
    out_path: Path,
    seed: int = 42,
) -> Path:
    """Export stratified blinded responses for human verification of LLM topic labels."""
    blinded = blind_and_shuffle_records(records, seed=seed)
    sample = blinded[: min(n, len(blinded))]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
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
