"""CLI entrypoint for Epistemic Prism (setup, demo, run, analyze, report, audit-selection)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from apart_epistemics.backends import (
    BackendConfigError,
    BudgetExceededError,
    SyntheticEpistemicBackend,
    create_backend,
)
from apart_epistemics.backends.base import DEFAULT_NUMERIC_MAX_TOKENS
from apart_epistemics.datasets import audit_question_selection_bias, build_numeric_core_tasks
from apart_epistemics.engine import (
    TokenPricing,
    analyze_numeric_core,
    run_numeric_core,
    run_suite,
)
from apart_epistemics.models import (
    ConditionConfig,
    ConditionSummary,
    ExperimentSuiteConfig,
    HeterogeneityType,
    NationalityType,
    NumericEstimateRecord,
    RoundRecord,
)
from apart_epistemics.report import build_static_site, generate_figures
from apart_epistemics.run_state import configure_budget
from apart_epistemics.summarizer import evaluate_dual_judge_topics

# (condition_id, label, heterogeneity, nationality, num_models, group, temperature, top_p)
_DEFAULT_CONDITION_SPECS: tuple[
    tuple[str, str, HeterogeneityType, NationalityType, int, str, float, float], ...
] = (
    (
        "no_ai",
        "No AI Baseline (Independent + Peers)",
        HeterogeneityType.NO_AI,
        NationalityType.NONE,
        0,
        "baseline",
        0.8,
        0.95,
    ),
    (
        "us_k1_greedy",
        "Monolithic US (K=1, T=0.0, top_p=0.0)",
        HeterogeneityType.MONOLITHIC_SHARED,
        NationalityType.US_ONLY,
        1,
        "decoding_ablation",
        0.0,
        0.0,
    ),
    (
        "us_k1",
        "Monolithic US (K=1, App Default T=0.8)",
        HeterogeneityType.MONOLITHIC_SHARED,
        NationalityType.US_ONLY,
        1,
        "k_sweep",
        0.8,
        0.95,
    ),
    (
        "us_k2",
        "Multi-Lab US Mix (K=2)",
        HeterogeneityType.MULTI_LAB_MIX,
        NationalityType.US_ONLY,
        2,
        "k_sweep",
        0.8,
        0.95,
    ),
    (
        "us_k4",
        "Multi-Lab US Mix (K=4)",
        HeterogeneityType.MULTI_LAB_MIX,
        NationalityType.US_ONLY,
        4,
        "k_sweep",
        0.8,
        0.95,
    ),
    (
        "cn_k1",
        "Monolithic CN Open Model (K=1)",
        HeterogeneityType.MONOLITHIC_SHARED,
        NationalityType.CN_ONLY,
        1,
        "nationality",
        0.8,
        0.95,
    ),
    (
        "cn_k4",
        "Multi-Lab CN Mix (K=4)",
        HeterogeneityType.MULTI_LAB_MIX,
        NationalityType.CN_ONLY,
        4,
        "nationality",
        0.8,
        0.95,
    ),
    (
        "divergent_sft_pair",
        "Same-Base Divergent SFT (K=2, Qwen3.8 + Arete-27B)",
        HeterogeneityType.DIVERGENT_SFT_PAIR,
        NationalityType.DIVERGENT_SFT,
        2,
        "nationality",
        0.8,
        0.95,
    ),
    (
        "cross_national_k4",
        "Cross-National Mix (K=4, US + CN + EU)",
        HeterogeneityType.CULTURAL_REGIONAL_MIX,
        NationalityType.CROSS_NATIONAL,
        4,
        "nationality",
        0.8,
        0.95,
    ),
    (
        "personalized_advisor",
        "Personalized Advisor (Role-Framed)",
        HeterogeneityType.PERSONALIZED_ADVISOR,
        NationalityType.CROSS_NATIONAL,
        24,
        "personalization",
        0.8,
        0.95,
    ),
    (
        "personalized_peer",
        "Personalized Peer (Sycophantic)",
        HeterogeneityType.PERSONALIZED_PEER,
        NationalityType.US_ONLY,
        24,
        "personalization",
        0.8,
        0.95,
    ),
)


def _default_conditions() -> list[ConditionConfig]:
    return [
        ConditionConfig(
            condition_id=cid,
            label=label,
            heterogeneity_type=het,
            nationality_type=nat,
            num_models=k,
            temperature=temp,
            top_p=top_p,
            group=group,
        )
        for cid, label, het, nat, k, group, temp, top_p in _DEFAULT_CONDITION_SPECS
    ]


def _load_config(config_path: Path | None) -> ExperimentSuiteConfig:
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return ExperimentSuiteConfig.model_validate(raw)
    default_yaml = Path("configs/default_suite.yaml")
    if default_yaml.exists():
        raw = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))
        return ExperimentSuiteConfig.model_validate(raw)
    return ExperimentSuiteConfig(conditions=_default_conditions())


def _serialize_numeric_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_identity_residual": analysis["max_identity_residual"],
        "h4_assistant_agent_corr": analysis["h4_assistant_agent_corr"],
        "by_condition_final": analysis["by_condition_final"],
        "condition_round_metrics": analysis["condition_round_metrics"],
        "contrasts": [
            c.model_dump() if hasattr(c, "model_dump") else c for c in analysis["contrasts"]
        ],
    }


def _resolve_fig_dir(args_figures: str | None, out_dir: Path) -> Path:
    if args_figures is not None:
        return Path(args_figures)
    if out_dir == Path("reports/results"):
        return Path("reports/figures")
    return out_dir / "figures"


def _execute_numeric_track(
    cfg: ExperimentSuiteConfig,
    backend: Any,
    out_dir: Path,
    *,
    n_agents: int,
    budget_usd: float,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
    pricing: TokenPricing | None = None,
) -> tuple[dict[str, Any], list[NumericEstimateRecord]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    num_tasks = build_numeric_core_tasks()[: cfg.tasks_per_track]
    num_recs = run_numeric_core(
        tasks=num_tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=n_agents,
        n_rounds=max(2, min(cfg.num_rounds, 4)),
        replicates=cfg.numeric_replicates,
        runs_path=out_dir / "numeric_runs.jsonl",
        backend=backend,
        alpha=cfg.alpha,
        budget_usd=budget_usd,
        seed=cfg.seed,
        max_tokens=max_tokens,
        pricing=pricing,
    )
    num_analysis = analyze_numeric_core(num_recs, num_tasks)
    (out_dir / "numeric_analysis.json").write_text(
        json.dumps(_serialize_numeric_analysis(num_analysis), indent=2),
        encoding="utf-8",
    )
    return num_analysis, num_recs


def _run_both_tracks(
    cfg: ExperimentSuiteConfig,
    backend: Any,
    out_dir: Path,
    budget_usd: float = 50.0,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
    pricing: TokenPricing | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[NumericEstimateRecord]]:
    if not hasattr(backend, "generate_numeric_reply"):
        raise BackendConfigError(
            "Replay supports Track 4 only; numeric and combined replay are unsupported."
        )
    configure_budget(backend, out_dir, budget_usd, getattr(backend, "request_allowance", 1.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_res = run_suite(cfg, backend=backend, out_dir=out_dir)
    num_analysis, num_recs = _execute_numeric_track(
        cfg,
        backend,
        out_dir,
        n_agents=cfg.num_agents,
        budget_usd=budget_usd,
        max_tokens=max_tokens,
        pricing=pricing,
    )
    return suite_res, num_analysis, num_recs


def _cmd_setup(args: argparse.Namespace) -> int:
    cfg = ExperimentSuiteConfig(
        suite_name="Epistemic Prism Custom Suite",
        seed=args.seed,
        num_agents=args.agents,
        num_rounds=args.rounds,
        tasks_per_track=args.tasks,
        conditions=_default_conditions(),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote validated config to {out_path}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = _load_config(Path(args.config) if args.config else None)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})
    out_dir = Path(args.out)
    site_dir = Path(args.site)
    fig_dir = _resolve_fig_dir(args.figures, out_dir)

    backend = SyntheticEpistemicBackend(seed=cfg.seed)
    suite_res, num_analysis, num_recs = _run_both_tracks(cfg, backend, out_dir)
    generate_figures(suite_res, num_analysis, fig_dir)
    build_static_site(
        suite_result=suite_res,
        numeric_analysis=num_analysis,
        site_dir=site_dir,
        figures_dir=fig_dir,
        numeric_records=num_recs,
    )
    print(
        f"Demo complete: {len(suite_res['summaries'])} conditions, "
        f"kappa={suite_res['topic_audit']['inter_rater_kappa']}, "
        f"residual={num_analysis['max_identity_residual']:.2e} -> {site_dir / 'index.html'}"
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(Path(args.config) if args.config else None)
        backend_kwargs: dict[str, Any] = {"seed": cfg.seed}
        if args.model:
            backend_kwargs["model"] = args.model
        if args.replay_jsonl:
            backend_kwargs["jsonl_path"] = args.replay_jsonl
        if args.pin_provider:
            if args.backend != "openrouter":
                raise BackendConfigError(
                    "--pin-provider is only supported with --backend openrouter."
                )
            backend_kwargs["pin_provider"] = args.pin_provider
        backend = create_backend(args.backend, **backend_kwargs)
        if args.backend == "replay":
            raise BackendConfigError(
                "The run command includes numeric execution; replay is Track "
                "4-only. Use analyze/report for archived runs."
            )
        backend.request_allowance = args.request_allowance
    except (BackendConfigError, FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    pricing = TokenPricing(input_per_m=args.price_in, output_per_m=args.price_out)
    try:
        if args.numeric_only:
            configure_budget(backend, out_dir, args.budget, args.request_allowance)
            _execute_numeric_track(
                cfg,
                backend,
                out_dir,
                n_agents=cfg.num_agents,
                budget_usd=args.budget,
                max_tokens=args.max_tokens,
                pricing=pricing,
            )
        else:
            _run_both_tracks(
                cfg,
                backend,
                out_dir,
                budget_usd=args.budget,
                max_tokens=args.max_tokens,
                pricing=pricing,
            )
    except (BudgetExceededError, RuntimeError, BackendConfigError) as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        return 2

    print(f"Run completed -> {out_dir}")
    return 0


def _load_saved_artifacts(
    in_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[NumericEstimateRecord]]:
    summary_file = in_dir / "summary.json"
    runs_file = in_dir / "runs.jsonl"
    num_runs_file = in_dir / "numeric_runs.jsonl"

    summary_json = (
        json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    )
    records = [
        RoundRecord.model_validate_json(line)
        for line in (
            runs_file.read_text(encoding="utf-8").splitlines() if runs_file.exists() else []
        )
        if line.strip()
    ]
    num_records = [
        NumericEstimateRecord.model_validate_json(line)
        for line in (
            num_runs_file.read_text(encoding="utf-8").splitlines() if num_runs_file.exists() else []
        )
        if line.strip()
    ]

    manifest_path = num_runs_file.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    from apart_epistemics.models import TaskItem

    numeric_tasks = (
        [TaskItem.model_validate(t) for t in manifest["tasks"]]
        if manifest
        else build_numeric_core_tasks()
    )
    num_analysis = analyze_numeric_core(num_records, numeric_tasks) if num_records else {}
    topic_audit = (
        evaluate_dual_judge_topics(records) if records else summary_json.get("topic_audit", {})
    )
    summaries = [ConditionSummary.model_validate(s) for s in summary_json.get("summaries", [])]
    suite_result = {
        "backend": summary_json.get("backend", {"class": "unrecorded"}),
        "summaries": summaries,
        "records": records,
        "comparisons": summary_json.get("comparisons", []),
        "topic_audit": topic_audit,
    }
    return suite_result, num_analysis, num_records


def _cmd_analyze(args: argparse.Namespace) -> int:
    in_dir = Path(args.input)
    suite_res, num_analysis, num_recs = _load_saved_artifacts(in_dir)
    if num_analysis:
        (in_dir / "numeric_analysis.json").write_text(
            json.dumps(_serialize_numeric_analysis(num_analysis), indent=2),
            encoding="utf-8",
        )
    out = {
        "conditions": len(suite_res["summaries"]),
        "track4_records": len(suite_res["records"]),
        "inter_rater_kappa": suite_res["topic_audit"].get("inter_rater_kappa"),
        "numeric_records": len(num_recs),
        "max_identity_residual": num_analysis.get("max_identity_residual"),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    in_dir = Path(args.input)
    suite_res, num_analysis, num_recs = _load_saved_artifacts(in_dir)
    fig_dir = _resolve_fig_dir(args.figures, in_dir)
    site_dir = Path(args.site)
    generate_figures(suite_res, num_analysis, fig_dir)
    build_static_site(
        suite_result=suite_res,
        numeric_analysis=num_analysis,
        site_dir=site_dir,
        figures_dir=fig_dir,
        numeric_records=num_recs,
    )
    print(f"Rebuilt figures in {fig_dir} and site in {site_dir}")
    return 0


def _cmd_audit_selection(args: argparse.Namespace) -> int:
    cold_scores = {
        "openai/gpt-frontier": [0, 0, 0, 1, 0, 0, 1, 0],
        "qwen/qwen3.8-27b": [1, 1, 0, 1, 1, 0, 1, 1],
        "mistral/mistral-large": [1, 0, 1, 1, 1, 1, 0, 1],
    }
    verdict = audit_question_selection_bias(
        cold_correctness_by_model=cold_scores,
        reference_model="openai/gpt-frontier",
        max_allowed_gap=args.max_gap,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    else:
        print(json.dumps(verdict, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level `apart-epistemics` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="apart-epistemics",
        description="Epistemic Prism CLI: Track 4 + Page's Theorem multi-agent testing suite.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Generate a validated experiment YAML config.")
    p_setup.add_argument("--out", default="configs/custom_suite.yaml")
    p_setup.add_argument("--agents", type=int, default=24)
    p_setup.add_argument("--rounds", type=int, default=5)
    p_setup.add_argument("--tasks", type=int, default=6)
    p_setup.add_argument("--seed", type=int, default=42)

    p_demo = sub.add_parser(
        "demo", help="Run zero-API synthetic benchmark and build figures + static site."
    )
    p_demo.add_argument("--config", default=None)
    p_demo.add_argument("--out", default="reports/results")
    p_demo.add_argument("--site", default="docs")
    p_demo.add_argument("--figures", default=None)
    p_demo.add_argument("--seed", type=int, default=None)

    p_run = sub.add_parser("run", help="Run configured suite on a chosen backend adapter.")
    p_run.add_argument("--config", default="configs/default_suite.yaml")
    p_run.add_argument(
        "--backend",
        default="synthetic",
        choices=["synthetic", "replay", "ollama", "openrouter", "openai", "anthropic", "gemini"],
    )
    p_run.add_argument("--model", default=None)
    p_run.add_argument(
        "--pin-provider",
        default=None,
        help="OpenRouter only: route every call to this provider with no fallbacks.",
    )
    p_run.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_NUMERIC_MAX_TOKENS,
        help="Output token limit per numeric call (reasoning tokens may count against it).",
    )
    p_run.add_argument("--replay-jsonl", default=None)
    p_run.add_argument("--out", default="reports/results")
    p_run.add_argument("--budget", type=float, default=15.0)
    p_run.add_argument(
        "--price-in",
        type=float,
        default=TokenPricing().input_per_m,
        help="USD per 1M input tokens, for the budget guard. Set this for your model.",
    )
    p_run.add_argument(
        "--price-out",
        type=float,
        default=TokenPricing().output_per_m,
        help="USD per 1M output tokens (reasoning tokens included), for the budget guard.",
    )
    p_run.add_argument(
        "--request-allowance",
        type=float,
        default=1.0,
        help=(
            "USD reserved before every HTTP attempt (default 1). Set at least "
            "the maximum possible request charge; not actual billing."
        ),
    )
    p_run.add_argument("--numeric-only", action="store_true")

    p_analyze = sub.add_parser("analyze", help="Analyze persisted JSONL logs and verify metrics.")
    p_analyze.add_argument("--input", default="reports/results")

    p_report = sub.add_parser("report", help="Rebuild 4 SVG figures and interactive static site.")
    p_report.add_argument("--input", default="reports/results")
    p_report.add_argument("--site", default="docs")
    p_report.add_argument("--figures", default=None)

    p_audit = sub.add_parser(
        "audit-selection", help="Run cold-start question-selection bias audit."
    )
    p_audit.add_argument("--out", default=None)
    p_audit.add_argument("--max-gap", type=float, default=0.25)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    dispatch = {
        "setup": _cmd_setup,
        "demo": _cmd_demo,
        "run": _cmd_run,
        "analyze": _cmd_analyze,
        "report": _cmd_report,
        "audit-selection": _cmd_audit_selection,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
