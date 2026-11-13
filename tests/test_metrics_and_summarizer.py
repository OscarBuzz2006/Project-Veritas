"""Property, mathematical invariant, and robustness tests for metrics and summarizer."""

import csv
import random
from pathlib import Path

import pytest

from apart_epistemics.metrics import (
    bootstrap_ci,
    cohens_kappa,
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
)
from apart_epistemics.summarizer import (
    blind_and_shuffle_records,
    evaluate_dual_judge_topics,
    export_audit_sample,
)


def test_pages_diversity_prediction_theorem_holds_universally() -> None:
    rng = random.Random(123)
    for _ in range(50):
        truth = rng.uniform(-2.0, 6.0)
        estimates = [truth + rng.gauss(0.2, 0.8) for _ in range(30)]
        decomp = compute_page_decomposition(estimates, truth)
        # Core identity: crowd_error == avg_individual_error - diversity
        assert decomp["identity_residual"] < 1e-12
        assert decomp["crowd_error"] == pytest.approx(
            decomp["avg_individual_error"] - decomp["diversity"], abs=1e-12
        )
        assert decomp["iqr"] >= 0.0


def test_calibration_metrics_exact_and_bounded() -> None:
    # Perfect confident predictions
    assert compute_brier_score([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert compute_ece([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert compute_confident_wrong_rate([0.9, 0.1], [1, 0]) == pytest.approx(0.0)

    # Maximally wrong at 0.90 confidence -> Brier = 0.81, confident_wrong = 1.0
    assert compute_brier_score([0.9, 0.1], [0, 1]) == pytest.approx(0.81)
    assert compute_confident_wrong_rate([0.9, 0.1], [0, 1], threshold=0.75) == pytest.approx(1.0)
    assert 0.0 <= compute_ece([0.9, 0.1, 0.8, 0.2], [0, 1, 0, 1]) <= 1.0


def test_topic_entropy_hill_and_minority_survival() -> None:
    vocab = ["a", "b", "c", "d"]
    uniform = compute_topic_entropy_and_hill([["a"], ["b"], ["c"], ["d"]], vocab)
    assert uniform["normalized_entropy"] == pytest.approx(1.0)
    assert uniform["effective_topics"] == pytest.approx(4.0)

    collapsed = compute_topic_entropy_and_hill([["a"], ["a"], ["a"], ["a"]], vocab)
    assert collapsed["normalized_entropy"] == pytest.approx(0.0)
    assert collapsed["effective_topics"] == pytest.approx(1.0)

    survival = compute_minority_topic_survival(
        round0_topic_lists=[["a"], ["b"], ["c"], ["d"]],
        final_topic_lists=[["a"], ["a"], ["b"], ["a"]],
    )
    assert survival == pytest.approx(0.5)


def test_bimodality_separates_polarized_echo_chambers_from_unimodal() -> None:
    polarized = [0.05] * 12 + [0.95] * 12
    concentrated = [0.48, 0.50, 0.52, 0.49, 0.51] * 4
    pol_stats = compute_stance_stats(polarized)
    con_stats = compute_stance_stats(concentrated)
    assert pol_stats["std"] > con_stats["std"]
    assert pol_stats["bimodality_coefficient"] > 0.555
    assert con_stats["bimodality_coefficient"] < pol_stats["bimodality_coefficient"]


def test_goldman_effective_sources_respects_independence_bounds(record_factory: object) -> None:
    shared_recs = [
        record_factory(
            condition_id="us_k1",
            round_idx=5,
            agent_id=f"a{i}",
            model_assigned="openai/gpt-frontier",
            model_family="us_openai",
            topics=["a"],
            stance_prob=0.5,
            text="shared reasoning",
            overlap=0.95,
        )
        for i in range(20)
    ]
    indep_recs = [
        record_factory(
            condition_id="no_ai",
            round_idx=5,
            agent_id=f"a{i}",
            model_assigned="none",
            model_family=f"indep_{i}",
            topics=["a"],
            stance_prob=0.5,
            text="independent reasoning",
            overlap=0.05,
        )
        for i in range(20)
    ]
    n_eff_shared = compute_goldman_effective_sources(shared_recs)
    n_eff_indep = compute_goldman_effective_sources(indep_recs)
    assert 1.0 <= n_eff_shared < 4.0
    assert 12.0 < n_eff_indep <= 20.0


def test_lexical_diversity_separates_identical_vs_distinct_texts() -> None:
    same = ["The policy should balance equity and stability in cities."] * 8
    varied = [
        "Rural transit networks require dedicated bus lanes before pricing.",
        "Digital vouchers protect low-income outer-suburb shift workers.",
        "Emergency response speeds accelerate dramatically inside priced zones.",
        "Informal freight operators face acute liquidity bottlenecks without rebates.",
    ]
    lex_same = compute_lexical_diversity(same)
    lex_varied = compute_lexical_diversity(varied)
    assert lex_same["pairwise_jaccard_distance"] == pytest.approx(0.0)
    assert lex_varied["pairwise_jaccard_distance"] > 0.75
    assert lex_varied["distinct_2"] > lex_same["distinct_2"]


def test_statistical_tests_wilcoxon_bootstrap_holm_and_kappa() -> None:
    assert cohens_kappa(["a", "b", "c", "a"], ["a", "b", "c", "a"]) == pytest.approx(1.0)
    assert cohens_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) < 0.0

    # Clear positive difference across 20 cells
    x = [0.20 + 0.01 * i for i in range(20)]
    y = [0.55 + 0.01 * i for i in range(20)]
    p_val = paired_wilcoxon_p(x, y)
    assert p_val < 0.001
    assert paired_wilcoxon_p(x, x) == pytest.approx(1.0)

    lo, hi = bootstrap_ci([yi - xi for xi, yi in zip(x, y, strict=True)], seed=42)
    assert lo > 0.30
    assert hi < 0.40

    adj = holm_bonferroni([0.005, 0.02, 0.04, 0.50])
    assert adj[0] == pytest.approx(0.02)
    assert adj[1] == pytest.approx(0.06)
    assert adj[2] == pytest.approx(0.08)
    assert adj[3] == pytest.approx(0.50)


def test_blinded_dual_judge_summarizer_strips_metadata_and_checks_kappa(
    tmp_path: Path, record_factory: object
) -> None:
    records = [
        record_factory(
            condition_id="secret_us_k1",
            round_idx=5,
            agent_id="a1",
            model_assigned="openai/gpt-frontier",
            model_family="us_openai",
            topics=["monopoly_concentration", "auditability"],
            stance_prob=0.6,
            text="We need independent academic audit to prevent corporate monopoly concentration.",
            overlap=0.9,
        ),
        record_factory(
            condition_id="secret_cross_national",
            round_idx=5,
            agent_id="a2",
            model_assigned="qwen/qwen3.8-27b",
            model_family="cn_alibaba",
            topics=["global_south_access", "cultural_alignment"],
            stance_prob=0.4,
            text="Open weights expand global south research access and local cultural alignment.",
            overlap=0.2,
        ),
    ]

    blinded = blind_and_shuffle_records(records, seed=42)
    assert len(blinded) == 2
    for item in blinded:
        assert "secret_us_k1" not in item["blind_text"]
        assert "secret_cross_national" not in item["blind_text"]
        assert "condition_id" not in item["judge_payload"]
        assert "model_assigned" not in item["judge_payload"]

    # Seed labels do not establish independent reliability
    result = evaluate_dual_judge_topics(records, seed=42)
    assert len(result["canonical_topics"]) >= 3
    assert result["inter_rater_kappa"] is None
    assert result["reliability_warning"] is True

    # Disagreement flag when judge_b returns mismatched labels
    noisy = evaluate_dual_judge_topics(
        records,
        seed=42,
        judge_a_override=lambda _txt, vocab: [vocab[0]],
        judge_b_override=lambda _txt, _vocab: ["unrelated_noise"],
    )
    assert noisy["inter_rater_kappa"] < 0.70
    assert noisy["reliability_warning"] is True

    audit_csv = tmp_path / "audit_human_check.csv"
    export_audit_sample(records, n=2, out_path=audit_csv, seed=42)
    assert audit_csv.exists()
    with audit_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert "blind_id" in rows[0]
    assert "condition_id" not in rows[0]
