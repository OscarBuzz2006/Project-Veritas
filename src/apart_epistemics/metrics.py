"""Epistemic, calibration, Page-identity, Goldman independence, and statistical metrics."""

from __future__ import annotations

import itertools
import math
import random
import re
from collections import Counter
from collections.abc import Iterable

from apart_epistemics.models import RoundRecord

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def clamp(value: float, low: float, high: float) -> float:
    """Clamp `value` safely into `[low, high]`, treating non-finite inputs as `low`."""
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def quantile_sorted(sorted_vals: list[float], q: float) -> float:
    """Compute linear-interpolated quantile `q` in [0, 1] from a pre-sorted sequence."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def compute_page_decomposition(log_estimates: list[float], log_truth: float) -> dict[str, float]:
    """Compute Page's Diversity Prediction Theorem terms on log10 estimates."""
    n = len(log_estimates)
    if n == 0:
        return {
            "crowd_error": 0.0,
            "avg_individual_error": 0.0,
            "diversity": 0.0,
            "identity_residual": 0.0,
            "iqr": 0.0,
            "crowd_mean": 0.0,
        }
    mean_g = sum(log_estimates) / n
    crowd_err = (mean_g - log_truth) ** 2
    avg_ind_err = sum((g - log_truth) ** 2 for g in log_estimates) / n
    diversity = sum((g - mean_g) ** 2 for g in log_estimates) / n
    residual = abs(crowd_err - (avg_ind_err - diversity))
    sorted_vals = sorted(log_estimates)
    iqr = quantile_sorted(sorted_vals, 0.75) - quantile_sorted(sorted_vals, 0.25)
    return {
        "crowd_error": crowd_err,
        "avg_individual_error": avg_ind_err,
        "diversity": diversity,
        "identity_residual": residual,
        "iqr": iqr,
        "crowd_mean": mean_g,
    }


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation safely, returning 0.0 on constant inputs."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return clamp(num / (dx * dy), -1.0, 1.0)


def compute_accuracy(probs: list[float], truths: list[int]) -> float:
    """Compute binary classification accuracy at threshold 0.5."""
    if not probs:
        return 0.0
    correct = sum(1 for p, y in zip(probs, truths, strict=True) if int(p >= 0.5) == y)
    return correct / len(probs)


def compute_brier_score(probs: list[float], truths: list[int]) -> float:
    """Compute mean squared probability error."""
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, truths, strict=True)) / len(probs)


def compute_ece(probs: list[float], truths: list[int], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error on stated answer confidence."""
    if not probs:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, truths, strict=True):
        conf = max(p, 1.0 - p)
        correct = 1 if int(p >= 0.5) == y else 0
        idx = min(n_bins - 1, max(0, int((conf - 0.5) / 0.5 * n_bins)))
        bins[idx].append((conf, correct))
    total = len(probs)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(acc for _, acc in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_acc - avg_conf)
    return ece


def compute_confident_wrong_rate(
    probs: list[float], truths: list[int], threshold: float = 0.75
) -> float:
    """Fraction of items answered incorrectly with confidence >= threshold."""
    if not probs:
        return 0.0
    bad = sum(
        1
        for p, y in zip(probs, truths, strict=True)
        if int(p >= 0.5) != y and max(p, 1.0 - p) >= threshold
    )
    return bad / len(probs)


def compute_topic_entropy_and_hill(
    topic_lists: Iterable[list[str]], vocabulary: list[str]
) -> dict[str, float]:
    """Compute normalized Shannon entropy and Hill N1 effective topic count."""
    counts: Counter[str] = Counter()
    for items in topic_lists:
        counts.update(items)
    total = sum(counts.values())
    vocab_size = max(len(set(vocabulary)), len(counts), 1)
    if total == 0 or vocab_size <= 1:
        return {"normalized_entropy": 0.0, "effective_topics": 1.0 if total > 0 else 0.0}

    h_nat = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            h_nat -= p * math.log(p)
    max_h = math.log(vocab_size)
    norm_h = h_nat / max_h if max_h > 0 else 0.0
    return {
        "normalized_entropy": clamp(norm_h, 0.0, 1.0),
        "effective_topics": math.exp(h_nat),
    }


def compute_minority_topic_survival(
    round0_topic_lists: Iterable[list[str]],
    final_topic_lists: Iterable[list[str]],
) -> float:
    """Fraction of distinct topics present at Round 0 that still survive in final round."""
    r0 = {t for items in round0_topic_lists for t in items}
    if not r0:
        return 0.0
    rf = {t for items in final_topic_lists for t in items}
    return len(r0 & rf) / len(r0)


def compute_stance_stats(probs: list[float]) -> dict[str, float]:
    """Compute std, variance, and SAS bimodality coefficient."""
    n = len(probs)
    if n < 2:
        return {"std": 0.0, "variance": 0.0, "bimodality_coefficient": 0.0}
    mean_p = sum(probs) / n
    diffs = [p - mean_p for p in probs]
    m2 = sum(d * d for d in diffs) / n
    if n < 4 or m2 < 1e-12:
        var = m2 if n >= 2 else 0.0
        return {
            "std": math.sqrt(var) if var >= 1e-12 else 0.0,
            "variance": var if var >= 1e-12 else 0.0,
            "bimodality_coefficient": 0.0,
        }
    m3 = sum(d**3 for d in diffs) / n
    m4 = sum(d**4 for d in diffs) / n
    skew = m3 / (m2**1.5)
    excess_kurt = (m4 / (m2**2)) - 3.0
    denom = excess_kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    bc = (skew**2 + 1.0) / denom if denom > 1e-9 else 0.0
    return {
        "std": math.sqrt(m2),
        "variance": m2,
        "bimodality_coefficient": clamp(bc, 0.0, 1.0),
    }


def _pair_correlation_tuple(
    a_model: str,
    a_family: str,
    a_overlap: float,
    b_model: str,
    b_family: str,
    b_overlap: float,
) -> float:
    if a_model == "none" and b_model == "none":
        base = 0.02
    elif a_model == b_model and a_model != "none":
        base = 0.84
    elif a_family == b_family:
        base = 0.70
    elif a_family.startswith("us_") and b_family.startswith("us_"):
        base = 0.48
    elif a_family.startswith("cn_") and b_family.startswith("cn_"):
        base = 0.44
    elif a_model != "none" and b_model != "none":
        base = 0.18
    else:
        base = 0.05
    peer_bump = 0.06 * (a_overlap + b_overlap)
    return min(0.96, base + peer_bump)


def compute_goldman_effective_sources(records: list[RoundRecord]) -> float:
    """Compute Kish / covariance-adjusted effective independent sources N_eff in [1, N]."""
    n = len(records)
    if n <= 1:
        return float(n)
    # Extract primitive fields once to avoid repeated Pydantic attribute lookups inside O(N^2).
    sig = [(r.model_assigned, r.model_family, r.goldman_source_overlap) for r in records]
    corr_sum = float(n)
    for i in range(n):
        am, af, ao = sig[i]
        for j in range(i + 1, n):
            bm, bf, bo = sig[j]
            corr_sum += 2.0 * _pair_correlation_tuple(am, af, ao, bm, bf, bo)
    n_eff = (n * n) / max(corr_sum, 1.0)
    return clamp(n_eff, 1.0, float(n))


def compute_lexical_diversity(texts: list[str]) -> dict[str, float]:
    """Compute distinct-2 bigram ratio and mean pairwise Jaccard token distance."""
    if not texts:
        return {"distinct_2": 0.0, "pairwise_jaccard_distance": 0.0}

    counts_by_text = Counter(texts)
    unique_bigrams: set[tuple[str, str]] = set()
    total_bigrams = 0
    unique_sets: list[tuple[frozenset[str], int]] = []

    for text, count in counts_by_text.items():
        toks = _TOKEN_RE.findall(text.lower())
        unique_sets.append((frozenset(toks), count))
        if len(toks) >= 2:
            total_bigrams += (len(toks) - 1) * count
            unique_bigrams.update(itertools.pairwise(toks))

    distinct_2 = len(unique_bigrams) / total_bigrams if total_bigrams else 0.0
    n_texts = len(texts)
    if n_texts < 2:
        return {"distinct_2": distinct_2, "pairwise_jaccard_distance": 0.0}

    # Identical texts have Jaccard distance 0.0; only evaluate pairs of distinct unique texts.
    dist_sum = 0.0
    u_len = len(unique_sets)
    for i in range(u_len):
        si, ci = unique_sets[i]
        for j in range(i + 1, u_len):
            sj, cj = unique_sets[j]
            union_sz = len(si | sj)
            pair_dist = 1.0 - (len(si & sj) / union_sz if union_sz > 0 else 0.0)
            dist_sum += pair_dist * ci * cj

    total_pairs = (n_texts * (n_texts - 1)) // 2
    return {
        "distinct_2": distinct_2,
        "pairwise_jaccard_distance": dist_sum / total_pairs if total_pairs else 0.0,
    }


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa inter-rater agreement for categorical assignments."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    po = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    cats = set(ca) | set(cb)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def paired_wilcoxon_p(xs: list[float], ys: list[float]) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value (exact when n_nz <= 14, else normal)."""
    if len(xs) != len(ys):
        raise ValueError("Paired inputs must have identical length.")
    diffs = [y - x for x, y in zip(xs, ys, strict=True) if abs(y - x) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 1.0

    indexed = sorted(enumerate(diffs), key=lambda item: abs(item[1]))
    ranks = [0.0] * n
    tie_adj = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(abs(indexed[j][1]) - abs(indexed[i][1])) <= 1e-12:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        t = j - i
        if t > 1:
            tie_adj += t * (t * t - 1)
        i = j

    w_plus = sum(r for d, r in zip(diffs, ranks, strict=True) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, ranks, strict=True) if d < 0)
    w_stat = min(w_plus, w_minus)

    if n <= 14 and tie_adj == 0.0:
        total = 1 << n
        extreme = 0
        for mask in range(total):
            s = sum((idx + 1) for idx in range(n) if (mask >> idx) & 1)
            if s <= w_stat + 1e-12 or (n * (n + 1) / 2.0 - s) <= w_stat + 1e-12:
                extreme += 1
        return min(1.0, extreme / total)

    mean_w = n * (n + 1) / 4.0
    var_w = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_adj) / 24.0
    if var_w <= 1e-12:
        return 1.0
    z = max(0.0, (abs(w_stat - mean_w) - 0.5) / math.sqrt(var_w))
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def bootstrap_ci(
    diffs: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Compute percentile bootstrap confidence interval for mean difference."""
    if not diffs:
        return (0.0, 0.0)
    if len(diffs) == 1:
        return (diffs[0], diffs[0])
    rng = random.Random(seed)
    n = len(diffs)
    randrange = rng.randrange
    inv_n = 1.0 / n
    means = [sum(diffs[randrange(n)] for _ in range(n)) * inv_n for _ in range(n_boot)]
    means.sort()
    return (quantile_sorted(means, alpha / 2.0), quantile_sorted(means, 1.0 - alpha / 2.0))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down family-wise correction preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, orig_idx in enumerate(order):
        scaled = min(1.0, p_values[orig_idx] * (m - rank))
        running_max = max(running_max, scaled)
        adjusted[orig_idx] = running_max
    return adjusted
