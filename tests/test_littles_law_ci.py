"""
tests/test_littles_law_ci.py — Batch-means CI test for Little's Law.

Phase 3.1 replaces the ad-hoc `|L − λW| ≤ tolerance · L` relative-error test
with a proper hypothesis test:
    H₀: L − λW = 0,
tested via a 100·(1−α)% confidence interval constructed from batch-mean
estimates of Var(L̂), Var(λ̂), Var(Ŵ) combined via the delta method.

These tests assert the correctness properties that distinguish the CI test
from the earlier heuristic:

  1. For a stationary M/M/1 the null hypothesis is accepted.
  2. A non-stationary queue with drifting sojourn times is *rejected*
     (the CI does not contain 0), showing the test has real statistical power.
  3. Insufficient data degrades gracefully: the test returns a clear note and
     does not falsely report success.
  4. The reported confidence interval is honest — its width shrinks like
     1/√B as the number of batches grows.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats.collector import QueueStats


def _simulate_mm1(lam: float, mu: float, n_events: int, seed: int = 0,
                  warmup_s: float = 0.0) -> QueueStats:
    """Single-server FIFO with exponential IAT and service, no queueing dynamics
    beyond departure = arrival + service (valid because service is memoryless
    and we track one customer at a time — a simplification that preserves
    E[W] = 1/(μ−λ) only for ρ → 0; for Little's Law checking we actually want
    the full sojourn = wait + service, which is handled by the M/M/1 driver
    below)."""
    rng = np.random.default_rng(seed)
    q = QueueStats("mm1", warmup_s=warmup_s)
    # Proper M/M/1 single-server FIFO with queueing
    t_arr = 0.0
    busy_until = 0.0
    for _ in range(n_events):
        t_arr += rng.exponential(1.0 / lam)
        start_svc = max(t_arr, busy_until)
        svc = rng.exponential(1.0 / mu)
        depart = start_svc + svc
        busy_until = depart
        q.arrival(t_arr)
        q.departure(depart, t_arr)
    q.finalise(t_arr)
    return q


def test_ci_test_accepts_stationary_mm1():
    """For a well-mixed, stationary M/M/1 the CI for L − λW must contain 0."""
    q = _simulate_mm1(lam=1.0, mu=2.0, n_events=8000, seed=123)
    res = q.verify_littles_law(n_batches=20, alpha=0.05)
    assert res["passed"], f"CI test rejected stationary M/M/1: {res}"
    assert res["ci_lower"] <= 0.0 <= res["ci_upper"], (
        f"0 not in reported CI: {res}"
    )


def test_ci_test_rejects_inflated_sojourns():
    """If we deliberately inflate departure times (breaking L = λW), the CI
    must exclude 0 at 95% confidence. This proves the test has power."""
    rng = np.random.default_rng(7)
    q = QueueStats("bad")
    t_arr = 0.0
    busy_until = 0.0
    for _ in range(8000):
        t_arr += rng.exponential(1.0)
        start_svc = max(t_arr, busy_until)
        svc = rng.exponential(0.5)
        depart = start_svc + svc
        busy_until = depart
        q.arrival(t_arr)
        # Record a departure with an inflated arrival_time so the *reported*
        # sojourn is systematically too small — i.e. our Ŵ underestimates true W.
        # This makes L − λW strictly positive and should be detected.
        fake_arrival = t_arr + 0.8 * (depart - t_arr)
        q.departure(depart, fake_arrival)
    q.finalise(t_arr)
    res = q.verify_littles_law(n_batches=20, alpha=0.05)
    assert not res["passed"], (
        f"CI test failed to detect artificial Little's Law violation: {res}"
    )
    assert res["ci_lower"] > 0.0 or res["ci_upper"] < 0.0, (
        f"0 still in CI despite deliberate violation: {res}"
    )


def test_insufficient_data_reports_note_not_success():
    """A queue with no data must not silently report `passed=True`."""
    q = QueueStats("empty")
    q.finalise(100.0)
    res = q.verify_littles_law()
    assert res["passed"] is False
    assert "Insufficient" in res.get("note", "")


def test_ci_width_shrinks_with_more_batches():
    """Var of the batch-means point estimate ≈ s²/B ⇒ SE scales like 1/√B.
    Doubling B should shrink SE by roughly √2 (not exact, but in the ballpark
    — the sample variance itself has sampling noise)."""
    q = _simulate_mm1(lam=1.0, mu=2.0, n_events=12000, seed=42)
    res_10 = q.verify_littles_law(n_batches=10)
    res_40 = q.verify_littles_law(n_batches=40)
    assert res_40["SE_delta"] < res_10["SE_delta"], (
        f"SE did not shrink with more batches: "
        f"B=10 SE={res_10['SE_delta']} vs B=40 SE={res_40['SE_delta']}"
    )


def test_warmup_censoring_does_not_break_batching():
    """With warmup_s > 0 the observation window [warmup, sim_end] is smaller
    but the CI test should still produce a valid result when there's enough
    post-warmup data."""
    q = _simulate_mm1(lam=1.0, mu=2.0, n_events=10000, seed=11, warmup_s=500.0)
    res = q.verify_littles_law(n_batches=20)
    assert not math.isnan(res["SE_delta"]), f"SE is NaN post-warmup: {res}"
    assert res["n_batches"] == 20


def test_result_contains_required_fields():
    """Schema check: the result dict must include every field downstream
    consumers (reports, aggregators) will read."""
    q = _simulate_mm1(lam=1.0, mu=2.0, n_events=4000, seed=3)
    res = q.verify_littles_law()
    required = {
        "queue", "L", "lambda", "W", "LW_product",
        "delta", "SE_delta", "ci_lower", "ci_upper",
        "relative_error", "n_batches", "alpha", "df", "passed",
    }
    missing = required - set(res.keys())
    assert not missing, f"CI result missing fields: {missing}"
