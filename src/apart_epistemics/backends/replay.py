"""Deterministic offline replay backend backed by archived JSONL logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apart_epistemics.models import RoundRecord


class ReplayBackend:
    """Deterministic offline replay backend backed by an archived JSONL run log."""

    def __init__(self, raw_rows: list[dict[str, Any]]) -> None:
        self._raw_rows = raw_rows
        self._round_index: dict[tuple[str, str, int, str], RoundRecord] = {}
        for row in raw_rows:
            if "task_id" in row and "agent_response_text" in row:
                rec = RoundRecord.model_validate(row)
                key = (rec.condition_id, rec.task_id, rec.round_idx, rec.agent_id)
                self._round_index[key] = rec

    @classmethod
    def from_jsonl(cls, path: Path) -> ReplayBackend:
        """Load replay records line by line from a persisted JSONL trace log."""
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return cls(rows)

    @property
    def record_count(self) -> int:
        """Return total number of rows loaded from archive."""
        return len(self._raw_rows)

    def get_round_record(
        self, condition_id: str, task_id: str, round_idx: int, agent_id: str
    ) -> RoundRecord | None:
        """Look up one deterministic `RoundRecord` by coordinate key."""
        return self._round_index.get((condition_id, task_id, round_idx, agent_id))
