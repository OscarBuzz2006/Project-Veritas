"""Unified multi-round simulation engine, prompt builders, and statistical analyzers."""

from __future__ import annotations

from apart_epistemics.engine.contrasts import compute_paired_holm_contrasts
from apart_epistemics.engine.numeric_core import (
    analyze_numeric_core,
    build_numeric_prompts,
    run_numeric_core,
)
from apart_epistemics.engine.track4 import run_suite

__all__ = [
    "analyze_numeric_core",
    "build_numeric_prompts",
    "compute_paired_holm_contrasts",
    "run_numeric_core",
    "run_suite",
]
