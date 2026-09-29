"""
tests/test_holm_bonferroni.py — Multiple-testing correction for KS validation.

Phase 3.2 adds the Holm (1979) step-down Bonferroni procedure so that when
several KS tests are run in one validation pass (one per marginal: recall
latency, migrate latency, file size, IAT, …), the family-wise error rate is
controlled at α=0.05 instead of the per-test α accumulating to ~1 − (1−α)^k.

These tests assert the correctness properties of Holm's procedure and its
wiring through the validators module.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.multiple_testing import holm_bonferroni, summarise_holm
from validation.validators import MultiMarginalKSValidator


def test_holm_single_test_reduces_to_identity():
    """With one test, the Holm-adjusted p-value equals the raw p-value
    and rejection matches the raw test."""
    res, family = holm_bonferroni([("only", 0.03)], alpha=0.05)
    assert len(res) == 1
    assert res[0].adjusted_p == pytest.approx(0.03, abs=1e-12)
    assert res[0].rejected is True
    assert family is True

    res, family = holm_bonferroni([("only", 0.07)], alpha=0.05)
    assert res[0].rejected is False
    assert family is False


def test_holm_rejects_all_when_all_significant():
    """Classic sanity check: p-values well below α/m should all reject."""
    raw = [("a", 0.001), ("b", 0.002), ("c", 0.004)]
    res, family = holm_bonferroni(raw, alpha=0.05)
    assert family is True
    assert all(r.rejected for r in res)


def test_holm_retains_order_in_output():
    """Per-test results are returned in the INPUT order (not sorted)."""
    raw = [("c", 0.04), ("a", 0.001), ("b", 0.03)]
    res, _ = holm_bonferroni(raw, alpha=0.05)
    assert [r.name for r in res] == ["c", "a", "b"]


def test_holm_step_down_stops_at_first_non_significant():
    """Holm's step-down rule: once a null is *not* rejected, all remaining
    larger p-values are also not rejected. Here α=0.05, m=4 ⇒ thresholds
    are α/4, α/3, α/2, α. The sequence 0.01, 0.02, 0.03, 0.045 hits the
    second threshold (α/3 ≈ 0.0167) at rank 2 with p=0.02 — so rank 2
    is NOT rejected, and rank 3, 4 must also not be rejected."""
    raw = [("p1", 0.01), ("p2", 0.02), ("p3", 0.03), ("p4", 0.045)]
    res, family = holm_bonferroni(raw, alpha=0.05)
    rejected_names = {r.name for r in res if r.rejected}
    assert rejected_names == {"p1"}, (
        f"Expected only p1 rejected, got {rejected_names}"
    )
    assert family is True  # one null rejected


def test_holm_stricter_than_per_test_but_looser_than_bonferroni():
    """Holm must be stricter than raw per-test α and at least as powerful
    as classical Bonferroni. With α=0.05 and m=5, raw p=0.02 would reject
    per-test; Bonferroni (α/m=0.01) would not. Holm at rank 1 uses
    α/m = 0.01 — so also no rejection at rank 1 ⇒ step-down halts."""
    raw = [("x1", 0.02), ("x2", 0.03), ("x3", 0.04), ("x4", 0.05), ("x5", 0.06)]
    res, family = holm_bonferroni(raw, alpha=0.05)
    # Raw test would reject x1..x4 (p≤0.05); Holm rejects nothing.
    assert family is False
    assert not any(r.rejected for r in res)


def test_holm_adjusted_p_is_monotone_nondecreasing():
    """In the sorted order, the adjusted p-values must be monotone
    non-decreasing (the `running_max` construction guarantees this)."""
    raw = [("a", 0.05), ("b", 0.001), ("c", 0.03), ("d", 0.01)]
    res, _ = holm_bonferroni(raw, alpha=0.05)
    by_rank = sorted(res, key=lambda r: r.rank)
    adj = [r.adjusted_p for r in by_rank]
    for i in range(1, len(adj)):
        assert adj[i] >= adj[i - 1], (
            f"Adjusted p-values not monotone: {adj}"
        )


def test_summarise_holm_empty_family():
    """An empty family is a no-op: passes, with n_tests=0."""
    s = summarise_holm([], alpha=0.05)
    assert s["n_tests"] == 0
    assert s["family_rejected"] is False
    assert s["family_passed"] is True


def test_multi_marginal_ks_validator_accepts_identical_distributions():
    """When all marginals are drawn from the same distribution as their
    reference, the family-wise Holm test should pass (no null rejected)."""
    rng = np.random.default_rng(42)
    marginals = {
        f"marg_{i}": (
            rng.normal(size=1000).tolist(),
            rng.normal(size=1000).tolist(),
        )
        for i in range(4)
    }
    mmks = MultiMarginalKSValidator(alpha=0.05)
    out = mmks.validate({
        k: ([abs(x) + 1e-6 for x in sim], [abs(x) + 1e-6 for x in obs])
        for k, (sim, obs) in marginals.items()
    })
    assert out["family_passed"] is True, f"Family rejected identical distros: {out}"


def test_multi_marginal_ks_validator_rejects_when_one_marginal_differs():
    """If one marginal is systematically shifted, the family-wise test
    should reject *at least* that marginal after Holm correction."""
    rng = np.random.default_rng(7)
    sim_normal = [abs(x) + 1e-6 for x in rng.normal(size=1000)]
    obs_normal = [abs(x) + 1e-6 for x in rng.normal(size=1000)]
    sim_shifted = [abs(x) + 1.5 for x in rng.normal(size=1000)]
    obs_original = [abs(x) + 1e-6 for x in rng.normal(size=1000)]
    marginals = {
        "match_1": (sim_normal, obs_normal),
        "match_2": ([abs(x) + 1e-6 for x in rng.normal(size=1000)],
                    [abs(x) + 1e-6 for x in rng.normal(size=1000)]),
        "different": (sim_shifted, obs_original),
    }
    mmks = MultiMarginalKSValidator(alpha=0.05)
    out = mmks.validate(marginals)
    assert out["family_passed"] is False, f"Holm missed the shifted marginal: {out}"
    # Per-marginal: the 'different' entry must be flagged
    different = out["per_marginal"]["different"]
    assert different["rejected_after_holm"] is True


def test_multi_marginal_ks_validator_skips_tiny_samples():
    """Marginals below min_samples should be reported as SKIP and excluded
    from the Holm family so the correction is not diluted by empty tests."""
    mmks = MultiMarginalKSValidator(alpha=0.05)
    out = mmks.validate(
        {"good": ([1.0] * 20, [1.1] * 20), "tiny": ([1.0], [2.0])},
        min_samples=5,
    )
    assert out["per_marginal"]["tiny"]["status"] == "SKIP"
    # The 'good' marginal went into the family with n_tests = 1
    assert out["holm"]["n_tests"] == 1
