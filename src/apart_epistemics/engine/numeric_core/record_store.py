"""Reading saved run records back from disk so an interrupted run can resume."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from apart_epistemics.models import NumericEstimateRecord

# One saved row is identified by: condition, question, replicate, round, agent.
RecordKey = tuple[str, str, int, int, str]


def load_existing_records(runs_path: Path) -> dict[RecordKey, NumericEstimateRecord]:
    """Load every saved row, repairing the file if the last line was cut off mid-write."""
    existing: dict[RecordKey, NumericEstimateRecord] = {}
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
            # Only the very last line is allowed to be broken: that is what a crash while
            # writing looks like. A broken line anywhere else means the file is corrupt, so
            # we stop rather than silently skip data.
            if idx == len(raw_lines) - 1:
                try:
                    json.loads(stripped)
                except json.JSONDecodeError:
                    had_truncated_tail = True
                    continue
            raise
        existing[record_key(rec)] = rec

    if had_truncated_tail:
        # Rewrite the file without the broken last line so new rows append cleanly.
        clean_lines = [r.model_dump_json() for r in existing.values()]
        runs_path.write_text(
            "\n".join(clean_lines) + ("\n" if clean_lines else ""), encoding="utf-8"
        )
    return existing


def record_key(rec: NumericEstimateRecord) -> RecordKey:
    """The identity of a saved row."""
    return (rec.condition_id, rec.question_id, rec.replicate, rec.round_idx, rec.agent_id)
