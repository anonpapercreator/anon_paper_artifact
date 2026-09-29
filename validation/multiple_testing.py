"""
validation/multiple_testing.py — Family-Wise Error Rate Control
================================================================

Implements the Holm (1979) step-down Bonferroni procedure for multiple
hypothesis testing.

When the simulator is validated by running several Kolmogorov–Smirnov tests
(e.g., one per marginal: recall latency, migrate latency, file size, IAT),
applying the per-test α=0.05 threshold inflates the probability of rejecting
*some* null by chance to roughly 1 − (1 − 0.05)^k. Holm's step-down procedure
controls the family-wise error rate (FWER) at α, with strictly greater power
than the classical Bonferroni correction.

Reference:
    Holm, S. (1979). "A simple sequentially rejective multiple test procedure."
    Scandinavian Journal of Statistics, 6(2), 65–70.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class HolmResult:
    """Per-test outcome of the Holm step-down procedure."""
    name: str
    p_value: float
    adjusted_p: float
    rank: int            # 1-based position in the sorted (ascending) family
    rejected: bool


def holm_bonferroni(
    named_p_values: Sequence[Tuple[str, float]],
    alpha: float = 0.05,
) -> Tuple[List[HolmResult], bool]:
    """
    Apply Holm's step-down Bonferroni procedure to a family of p-values.

    Algorithm:
        1. Sort the m p-values ascending: p_(1) ≤ p_(2) ≤ … ≤ p_(m).
        2. For i = 1, 2, …, m, if p_(i) ≤ α / (m − i + 1), reject the
           corresponding null. Otherwise stop and accept the remaining nulls.
        3. Adjusted p-value for p_(i) is
               p̃_(i) = max_{j ≤ i} min(1, (m − j + 1) · p_(j)).

    Args:
        named_p_values: Sequence of ``(test_name, p_value)`` tuples.
        alpha:          Family-wise significance level (default 0.05).

    Returns:
        (per_test_results, family_rejected):
            per_test_results — list of HolmResult in the input order,
            family_rejected  — True iff *any* null was rejected at level α.
    """
    m = len(named_p_values)
    if m == 0:
        return [], False

    indexed = list(enumerate(named_p_values))
    # Sort by p-value ascending; break ties by original index for stability.
    indexed.sort(key=lambda ix: (ix[1][1], ix[0]))

    adjusted_sorted: List[float] = [0.0] * m
    running_max = 0.0
    stop_rejecting = False
    reject_flags_sorted = [False] * m

    for rank_i, (orig_idx, (_, p)) in enumerate(indexed, start=1):
        factor = m - rank_i + 1
        threshold = alpha / factor
        if not stop_rejecting and p <= threshold:
            reject_flags_sorted[rank_i - 1] = True
        else:
            stop_rejecting = True

        adj = min(1.0, factor * p)
        running_max = max(running_max, adj)
        adjusted_sorted[rank_i - 1] = running_max

    results: List[HolmResult] = [None] * m  # type: ignore[list-item]
    for rank_i, (orig_idx, (name, p)) in enumerate(indexed, start=1):
        results[orig_idx] = HolmResult(
            name=name,
            p_value=float(p),
            adjusted_p=float(adjusted_sorted[rank_i - 1]),
            rank=rank_i,
            rejected=bool(reject_flags_sorted[rank_i - 1]),
        )

    family_rejected = any(r.rejected for r in results)
    return results, family_rejected


def summarise_holm(
    named_p_values: Sequence[Tuple[str, float]],
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Convenience wrapper: run Holm and return a JSON-friendly dict suitable
    for inclusion in a validation report.
    """
    results, family_rejected = holm_bonferroni(named_p_values, alpha=alpha)
    return {
        "procedure": "Holm-Bonferroni (step-down)",
        "alpha": alpha,
        "n_tests": len(named_p_values),
        "family_rejected": family_rejected,
        "family_passed": not family_rejected,
        "per_test": [
            {
                "name": r.name,
                "p_value": round(r.p_value, 6),
                "adjusted_p": round(r.adjusted_p, 6),
                "rank": r.rank,
                "rejected": r.rejected,
            }
            for r in results
        ],
    }
