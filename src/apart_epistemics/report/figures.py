"""Publication-grade SVG chart generators for Epistemic Prism analytics."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from apart_epistemics.models import ConditionSummary

_PALETTE: dict[str, str] = {
    "paper": "#FAF8F4",
    "surface": "#FFFFFF",
    "ink": "#171614",
    "muted": "#5E5A52",
    "grid": "#E5E0D5",
    "vermilion": "#C84B31",
    "cerulean": "#245A7A",
    "moss": "#2F6F4E",
    "ochre": "#A96A0B",
    "plum": "#6B4C7A",
}

_GROUP_COLORS: dict[str, str] = {
    "baseline": _PALETTE["moss"],
    "decoding_ablation": "#8C2D19",
    "k_sweep": _PALETTE["ochre"],
    "nationality": _PALETTE["cerulean"],
    "personalization": _PALETTE["plum"],
    "main": _PALETTE["vermilion"],
}

_ALIASES: dict[str, str] = {
    "no_ai": "No AI",
    "us_k1_greedy": "US K1 (T=0)",
    "us_k1": "US K1 (T=0.8)",
    "us_k2": "US K2",
    "us_k4": "US K4",
    "cn_k1": "CN K1",
    "cn_k4": "CN K4",
    "divergent_sft_pair": "Divergent SFT",
    "cross_national_k4": "Cross-Nat K4",
    "personalized_advisor": "Advisor (Calib)",
    "personalized_peer": "Peer (Mirror)",
}


def _color_for_group(group: str, condition_id: str) -> str:
    if condition_id.startswith(("cross_national", "divergent")):
        return _PALETTE["cerulean"]
    if condition_id.startswith("us_k1"):
        return _PALETTE["vermilion"]
    return _GROUP_COLORS.get(group, _PALETTE["ink"])


def _short_label(cond_id: str) -> str:
    return _ALIASES.get(cond_id, cond_id)


def _svg_wrap(width: int, height: int, title: str, body: str) -> str:
    esc_title = html.escape(title)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{esc_title}">\n'
        f"  <title>{esc_title}</title>\n"
        f'  <rect width="{width}" height="{height}" fill="{_PALETTE["surface"]}" rx="6"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def _svg_header_lines(title: str, subtitle: str) -> list[str]:
    return [
        f'  <text x="24" y="34" font-family="Georgia,serif" font-size="18" font-weight="700" '
        f'fill="{_PALETTE["ink"]}">{title}</text>',
        f'  <text x="24" y="54" font-family="sans-serif" font-size="12" '
        f'fill="{_PALETTE["muted"]}">{subtitle}</text>',
    ]


def _append_horizontal_legend(
    lines: list[str], lx: int, ly: int, entries: list[tuple[str, str]], step_x: int
) -> None:
    for label, col in entries:
        lines.append(f'  <rect x="{lx}" y="{ly - 9}" width="11" height="11" rx="2" fill="{col}"/>')
        lines.append(
            f'  <text x="{lx + 16}" y="{ly}" font-family="sans-serif" font-size="11" '
            f'fill="{_PALETTE["ink"]}">{html.escape(label)}</text>'
        )
        lx += step_x


def _render_dissociation_svg(summaries: list[ConditionSummary]) -> str:
    """Compare greedy vs stochastic decoding across lexical vs epistemic dimensions."""
    by_id = {s.condition_id: s for s in summaries}
    greedy = by_id.get("us_k1_greedy")
    stoch = by_id.get("us_k1") or by_id.get("cn_k1")
    diverse = by_id.get("cross_national_k4") or by_id.get("us_k4")

    bars = (
        ("Lexical Jaccard Dist", lambda s: s.pairwise_jaccard_distance),
        ("Distinct-2 Bigrams", lambda s: s.distinct_2),
        ("Topic Entropy (Norm)", lambda s: s.topic_shannon_entropy),
        ("Effective Topics / 6", lambda s: min(1.0, s.effective_topic_count / 6.0)),
        ("Goldman N_eff / 6", lambda s: min(1.0, s.goldman_effective_sources / 6.0)),
    )

    w, h = 760, 400
    left, right, top, row_h = 178, 56, 78, 54
    chart_w = w - left - right
    lines = _svg_header_lines(
        "Decoding Dissociation: Lexical Veil vs Monoculture",
        "Recorded lexical and topic metrics; effective-source values are assumption-based proxies",
    )

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * chart_w
        lines.append(
            f'  <line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{top + len(bars) * row_h - 6}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{x:.1f}" y="{top + len(bars) * row_h + 14}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{tick:.2f}</text>'
        )

    series = [
        ("US K1 T=0", greedy, "#8C2D19"),
        ("US K1 T=0.8", stoch, _PALETTE["vermilion"]),
        ("Cross-Nat K4", diverse, _PALETTE["cerulean"]),
    ]

    for r_idx, (metric_name, getter) in enumerate(bars):
        cy = top + r_idx * row_h
        lines.append(
            f'  <text x="{left - 12}" y="{cy + 22}" text-anchor="end" font-family="sans-serif" '
            f'font-size="12" fill="{_PALETTE["ink"]}">{html.escape(metric_name)}</text>'
        )
        for s_idx, (_lbl, cond, col) in enumerate(series):
            val = max(0.0, min(1.0, getter(cond))) if cond else 0.0
            bw = max(2.0, val * chart_w)
            by = cy + s_idx * 13
            lines.append(
                f'  <rect x="{left}" y="{by}" width="{bw:.1f}" height="10" rx="2" fill="{col}"/>'
            )
            lines.append(
                f'  <text x="{left + bw + 5:.1f}" y="{by + 8.5}" font-family="monospace" '
                f'font-size="10" fill="{_PALETTE["ink"]}">{val:.2f}</text>'
            )

    _append_horizontal_legend(
        lines, 24, h - 22, [(lbl, col) for lbl, cond, col in series if cond is not None], 135
    )
    return _svg_wrap(w, h, "Decoding Double Dissociation", "\n".join(lines))


def _render_epistemic_quadrant_svg(summaries: list[ConditionSummary]) -> str:
    """Scatter plot: Effective Topic Count (X) vs Ground-Truth Calibration 1-Brier (Y)."""
    w, h = 760, 430
    left, right, top, bottom = 74, 58, 72, 68
    pw, ph = w - left - right, h - top - bottom

    xs = [s.effective_topic_count for s in summaries] or [1.0]
    ys = [max(0.0, 1.0 - s.brier_score) for s in summaries] or [0.5]
    x_min = max(0.5, min(xs) - 0.5)
    x_max = max(x_min + 1.0, max(xs) + 0.85)
    y_min = max(0.35, min(ys) - 0.06)
    y_max = min(1.02, max(y_min + 0.1, max(ys) + 0.05))

    def px(val: float) -> float:
        return left + ((val - x_min) / (x_max - x_min)) * pw

    def py(val: float) -> float:
        return top + ph - ((val - y_min) / (y_max - y_min)) * ph

    lines = _svg_header_lines(
        "Epistemic Quadrant: Topic Breadth vs Calibration Quality",
        "Bubble radius shows assumption-based source proxy (N_eff), not measured independence",
    )

    for i in range(5):
        tx = x_min + i * (x_max - x_min) / 4.0
        gx = px(tx)
        lines.append(
            f'  <line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + ph}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{gx:.1f}" y="{top + ph + 18}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{tx:.1f}</text>'
        )
        ty = y_min + i * (y_max - y_min) / 4.0
        gy = py(ty)
        lines.append(
            f'  <line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{left - 10}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{ty:.2f}</text>'
        )

    lines.append(
        f'  <rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" '
        f'stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="{left + pw / 2:.1f}" y="{h - 18}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12" fill="{_PALETTE["ink"]}">'
        f"Effective Topic Count (Hill N1)</text>"
    )
    lines.append(
        f'  <text x="20" y="{top + ph / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 20 {top + ph / 2:.1f})" font-family="sans-serif" '
        f'font-size="12" fill="{_PALETTE["ink"]}">Calibration Score (1 - Brier)</text>'
    )

    label_offsets = {
        "us_k1_greedy": (0.0, -1.0, "start"),
        "us_k1": (12.0, 3.0, "start"),
        "cn_k1": (0.0, 14.0, "start"),
        "cn_k4": (0.0, -1.0, "middle"),
        "divergent_sft_pair": (-4.0, 16.0, "end"),
        "cross_national_k4": (0.0, -12.0, "middle"),
        "personalized_advisor": (-10.0, 15.0, "end"),
        "personalized_peer": (-12.0, 3.0, "end"),
        "no_ai": (0.0, 16.0, "middle"),
    }
    for s in summaries:
        cx = px(s.effective_topic_count)
        cy = py(max(0.0, 1.0 - s.brier_score))
        radius = max(6.0, min(18.0, 4.0 + s.goldman_effective_sources * 1.4))
        col = _color_for_group(s.group, s.condition_id)
        lines.append(
            f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{col}" '
            f'fill-opacity="0.78" stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
        )
        dx, dy_raw, anchor = label_offsets.get(
            s.condition_id,
            (0.0, -radius - 5 if cy > top + 28 else radius + 12, "middle"),
        )
        if s.condition_id in ("us_k1_greedy", "cn_k4", "cross_national_k4"):
            dy_raw = -radius - 5
        elif s.condition_id in ("cn_k1", "personalized_advisor", "divergent_sft_pair"):
            dy_raw = radius + 12
        elif s.condition_id == "no_ai":
            dy_raw = radius + 14
        lines.append(
            f'  <text x="{cx + dx:.1f}" y="{cy + dy_raw:.1f}" text-anchor="{anchor}" '
            f'font-family="sans-serif" font-size="10.5" fill="{_PALETTE["ink"]}">'
            f"{html.escape(_short_label(s.condition_id))}</text>"
        )

    return _svg_wrap(w, h, "Epistemic Quadrant", "\n".join(lines))


def _render_round_trajectories_svg(summaries: list[ConditionSummary]) -> str:
    """Plot round-by-round Effective Topic Count trajectories for focal regimes."""
    focal_ids = {
        "no_ai",
        "us_k1",
        "us_k4",
        "divergent_sft_pair",
        "cross_national_k4",
        "personalized_peer",
    }
    selected = [s for s in summaries if s.condition_id in focal_ids] or summaries[:6]

    w, h = 760, 400
    left, right, top, bottom = 68, 178, 72, 54
    pw, ph = w - left - right, h - top - bottom

    max_r = max(
        1, max((len(s.round_trajectory) - 1 for s in selected if s.round_trajectory), default=1)
    )
    all_vals = [pt.get("effective_topic_count", 1.0) for s in selected for pt in s.round_trajectory]
    y_min = max(0.8, min(all_vals, default=1.0) - 0.3)
    y_max = max(y_min + 1.0, max(all_vals, default=6.0) + 0.4)

    def px(r: float) -> float:
        return left + (r / max_r) * pw

    def py(v: float) -> float:
        return top + ph - ((v - y_min) / (y_max - y_min)) * ph

    lines = _svg_header_lines(
        "Multi-Round Topic Decay vs Structural Preservation",
        "Recorded topic trajectories; synthetic runs reflect simulator assumptions",
    )

    for r in range(max_r + 1):
        gx = px(r)
        lines.append(
            f'  <line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + ph}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{gx:.1f}" y="{top + ph + 18}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">R{r}</text>'
        )

    for i in range(5):
        tv = y_min + i * (y_max - y_min) / 4.0
        gy = py(tv)
        lines.append(
            f'  <line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{left - 10}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{tv:.1f}</text>'
        )

    lines.append(
        f'  <rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" '
        f'stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
    )

    series_palette = (
        _PALETTE["moss"],
        _PALETTE["vermilion"],
        _PALETTE["ochre"],
        "#1E4D6B",
        _PALETTE["cerulean"],
        _PALETTE["plum"],
    )

    for idx, s in enumerate(selected):
        if not s.round_trajectory:
            continue
        col = series_palette[idx % len(series_palette)]
        pts = [
            f"{px(pt['round_idx']):.1f},{py(pt['effective_topic_count']):.1f}"
            for pt in s.round_trajectory
        ]
        lines.append(
            f'  <polyline fill="none" stroke="{col}" stroke-width="2.4" points="{" ".join(pts)}"/>'
        )
        for pt in s.round_trajectory:
            cx, cy = px(pt["round_idx"]), py(pt["effective_topic_count"])
            lines.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="{col}"/>')

        ly = top + 18 + idx * 24
        lx = left + pw + 16
        lines.append(
            f'  <line x1="{lx}" y1="{ly - 4}" x2="{lx + 18}" y2="{ly - 4}" '
            f'stroke="{col}" stroke-width="2.6"/>'
        )
        lines.append(
            f'  <text x="{lx + 24}" y="{ly}" font-family="sans-serif" font-size="11" '
            f'fill="{_PALETTE["ink"]}">{html.escape(_short_label(s.condition_id))}</text>'
        )

    return _svg_wrap(w, h, "Round Trajectories", "\n".join(lines))


def _render_pages_headline_svg(numeric_analysis: dict[str, Any]) -> str:
    """Render Kevin's C1/C2/C3 Page's Diversity Prediction Theorem bar comparison."""
    final_map: dict[str, dict[str, Any]] = numeric_analysis.get("by_condition_final", {})
    max_resid = float(numeric_analysis.get("max_identity_residual", 0.0))
    order = [c for c in ("C1", "C2", "C3") if c in final_map] or sorted(final_map.keys())

    w, h = 760, 390
    left, right, top, bottom = 74, 42, 78, 72
    pw, ph = w - left - right, h - top - bottom

    metrics = (
        ("crowd_error", "Crowd Error", _PALETTE["vermilion"]),
        ("avg_individual_error", "Avg Individual Err", _PALETTE["ochre"]),
        ("diversity", "Crowd Diversity", _PALETTE["cerulean"]),
    )
    max_val = max([float(final_map[c].get(k, 0.0)) for c in order for k, _, _ in metrics] or [0.1])
    max_val = max(max_val * 1.18, 0.05)

    def py(v: float) -> float:
        return top + ph - (v / max_val) * ph

    lines = _svg_header_lines(
        "Page&apos;s Diversity Prediction Theorem (Log10 Core: C1 / C2 / C3)",
        (
            "Identity: Crowd Error = Avg Individual Error - Diversity "
            f"(Max verified residual: {max_resid:.2e})"
        ),
    )

    for i in range(5):
        tv = i * max_val / 4.0
        gy = py(tv)
        lines.append(
            f'  <line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
            f'stroke="{_PALETTE["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{left - 10}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="11" fill="{_PALETTE["muted"]}">{tv:.3f}</text>'
        )

    lines.append(
        f'  <line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" '
        f'stroke="{_PALETTE["ink"]}" stroke-width="1"/>'
    )

    group_w = pw / max(len(order), 1)
    bar_w = min(44.0, group_w * 0.22)
    cond_names = {"C1": "C1 Baseline", "C2": "C2 Social Median", "C3": "C3 Persona+Hist"}

    for g_idx, cid in enumerate(order):
        gx_center = left + (g_idx + 0.5) * group_w
        lines.append(
            f'  <text x="{gx_center:.1f}" y="{top + ph + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12.5" font-weight="600" '
            f'fill="{_PALETTE["ink"]}">{html.escape(cond_names.get(cid, cid))}</text>'
        )
        start_x = gx_center - (len(metrics) * bar_w + (len(metrics) - 1) * 8) / 2.0
        for m_idx, (key, _lbl, col) in enumerate(metrics):
            val = float(final_map[cid].get(key, 0.0))
            bx = start_x + m_idx * (bar_w + 8)
            by = py(val)
            bh = max(2.0, (top + ph) - by)
            lines.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                f'rx="2" fill="{col}"/>'
            )
            lines.append(
                f'  <text x="{bx + bar_w / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
                f'font-family="monospace" font-size="10" fill="{_PALETTE["ink"]}">{val:.4f}</text>'
            )

    _append_horizontal_legend(lines, 24, h - 18, [(lbl, col) for _k, lbl, col in metrics], 165)
    return _svg_wrap(w, h, "Pages Diversity Prediction Theorem", "\n".join(lines))


def generate_figures(
    suite_result: dict[str, Any],
    numeric_analysis: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    """Render all four analytical SVG charts to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[ConditionSummary] = suite_result["summaries"]
    outputs = {
        "dissociation_decoding.svg": _render_dissociation_svg(summaries),
        "epistemic_quadrant.svg": _render_epistemic_quadrant_svg(summaries),
        "round_trajectories.svg": _render_round_trajectories_svg(summaries),
        "pages_headline.svg": _render_pages_headline_svg(numeric_analysis),
    }
    written: list[Path] = []
    for filename, content in outputs.items():
        path = out_dir / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
