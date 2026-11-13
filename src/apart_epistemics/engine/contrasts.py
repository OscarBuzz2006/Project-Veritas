"""Shared paired statistical contrast builder (Wilcoxon + bootstrap CI + Holm-Bonferroni)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from apart_epistemics.metrics import bootstrap_ci, holm_bonferroni, paired_wilcoxon_p
from apart_epistemics.models import StatisticalComparison


def compute_paired_holm_contrasts(
    specs: list[tuple[str, str, str, str]],
    cell_metric_fn: Callable[[str, str], dict[Any, float]],
    *,
    seed: int = 42,
    effect_decimals: int = 4,
    include_empty: bool = False,
) -> list[StatisticalComparison]:
    """Compute paired contrasts with Wilcoxon signed-rank, 95% bootstrap CI, and Holm adjustment."""
    if not specs:
        return []

    raw_tests: list[dict[str, Any]] = []
    for contrast_id, metric, cond_a, cond_b in specs:
        cells_a = cell_metric_fn(cond_a, metric)
        cells_b = cell_metric_fn(cond_b, metric)
        common = sorted(set(cells_a) & set(cells_b))
        if not common:
            # Missing observations are not evidence of a zero effect, even when
            # older callers request include_empty.
            continue
        xs = [cells_a[k] for k in common]
        ys = [cells_b[k] for k in common]
        p_val = paired_wilcoxon_p(xs, ys)

        diffs = [y - x for x, y in zip(xs, ys, strict=True)]
        ci_lo, ci_hi = bootstrap_ci(diffs, seed=seed)
        raw_tests.append(
            {
                "contrast_id": contrast_id,
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

    if not raw_tests:
        return []

    adjusted_ps = holm_bonferroni([item["wilcoxon_p"] for item in raw_tests])
    return [
        StatisticalComparison(
            contrast_id=item["contrast_id"],
            metric=item["metric"],
            condition_a=item["condition_a"],
            condition_b=item["condition_b"],
            mean_diff=round(item["mean_diff"], effect_decimals),
            ci_low=round(item["ci_low"], effect_decimals),
            ci_high=round(item["ci_high"], effect_decimals),
            wilcoxon_p=round(item["wilcoxon_p"], 6),
            holm_adjusted_p=round(adj_p, 6),
            n_cells=item["n_cells"],
        )
        for item, adj_p in zip(raw_tests, adjusted_ps, strict=True)
    ]
