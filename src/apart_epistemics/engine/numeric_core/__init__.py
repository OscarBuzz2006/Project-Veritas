"""Crash-resumable C1/C2/C3 numeric estimation engine and Page's Theorem analyzer.

One module per job:
  prompts.py        the text sent to the model
  reply_parsing.py  reading a number out of a reply
  pricing.py        cost estimates and the budget check
  retry.py          one backend call, plus one retry
  record_store.py   loading saved rows to resume a run
  round_runner.py   one round: ask every agent, update their beliefs
  runner.py         the whole sweep over conditions, questions and replicates
  analysis.py       the statistics reported from saved rows
"""

from __future__ import annotations

from apart_epistemics.engine.numeric_core.analysis import analyze_numeric_core
from apart_epistemics.engine.numeric_core.pricing import TokenPricing, estimate_call_cost_usd
from apart_epistemics.engine.numeric_core.prompts import build_numeric_prompts, shared_system_prompt
from apart_epistemics.engine.numeric_core.runner import run_numeric_core

__all__ = [
    "TokenPricing",
    "analyze_numeric_core",
    "build_numeric_prompts",
    "estimate_call_cost_usd",
    "run_numeric_core",
    "shared_system_prompt",
]
