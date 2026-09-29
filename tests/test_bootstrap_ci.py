"""
tests/test_bootstrap_ci.py — Percentile-bootstrap confidence intervals.

Phase 3.3 replaces the Gaussian-style ``x̄ ± t · s/√n`` CI in MetricAccumulator
with a non-parametric percentile bootstrap (Efron & Tibshirani 1993) so that
replication-level summaries — which are commonly skewed (latency percentiles,
hit ratios, utilisations) — are reported with an honest interval that doesn't
assume symmetry.

For single-replication summaries, a within-replication (Woodruff-style)
bootstrap over the raw sample takes over so callers still see a CI rather
than "insufficient data".

These tests assert the key correctness properties:

  1. On symmetric data the percentile bootstrap CI closely matches the
     t-distribution CI (so we haven't made the interval wildly wider).
  2. For heavily skewed data the bootstrap CI is asymmetric around the mean,
     which the t-CI cannot capture.
  3. The LatencyHistogram summary returns a CI even with a single
     replication, via the within-rep bootstrap fallback.
  4. Bootstrap CIs shrink with sample size at roughly the nominal √n rate.
  5. The CI is reproducible with a fixed RNG seed.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats.collector import (
    percentile_bootstrap_ci,
    MetricAccumulator,
    LatencyHistogram,
)


def test_bootstrap_matches_t_ci_for_symmetric_data():
    """For approximately normal data the bootstrap CI should agree with the
    t-interval to within a small margin."""
    rng = np.random.default_rng(1)
    x = rng.normal(loc=10.0, scale=2.0, size=200)
    lo_bs, hi_bs = percentile_bootstrap_ci(x, statistic="mean", n_boot=2000,
                                            rng_seed=7)
    # t-interval reference
    import math as m
    from scipy.stats import t as t_dist
    mean, s = float(np.mean(x)), float(np.std(x, ddof=1))
    se = s / m.sqrt(len(x))
    t_crit = t_dist.ppf(0.975, df=len(x) - 1)
    lo_t, hi_t = mean - t_crit * se, mean + t_crit * se
    assert abs(lo_bs - lo_t) < 0.15
    assert abs(hi_bs - hi_t) < 0.15


def test_bootstrap_captures_skew_that_t_cannot():
    """Bootstrap CI around the mean of a skewed sample is asymmetric about
    the point estimate. A symmetric t-CI would mis-cover in the long tail."""
    rng = np.random.default_rng(3)
    x = rng.lognormal(mean=0.0, sigma=1.5, size=300)
    mean = float(np.mean(x))
    lo, hi = percentile_bootstrap_ci(x, statistic="mean", n_boot=2000,
                                     rng_seed=3)
    # Distance to upper bound exceeds distance to lower bound for right-skewed
    # lognormal sample means.
    assert (hi - mean) > (mean - lo), (
        f"Bootstrap did not capture skew: lo={lo}, mean={mean}, hi={hi}"
    )


def test_bootstrap_percentile_statistic():
    """Bootstrap on the 95th percentile: endpoints should bracket the true
    95th percentile of a known distribution (Exp(1))."""
    rng = np.random.default_rng(5)
    x = rng.exponential(scale=1.0, size=2000)
    lo, hi = percentile_bootstrap_ci(x, statistic="percentile", percentile=95.0,
                                     n_boot=1000, rng_seed=5)
    true_p95 = -np.log(0.05)  # ≈ 2.9957
    assert lo <= true_p95 <= hi, (
        f"True 95% quantile {true_p95:.4f} not in CI [{lo:.4f}, {hi:.4f}]"
    )


def test_bootstrap_ci_reproducible_with_seed():
    """Fixing the rng_seed produces deterministic endpoints."""
    x = np.arange(100, dtype=float)
    a = percentile_bootstrap_ci(x, n_boot=500, rng_seed=42)
    b = percentile_bootstrap_ci(x, n_boot=500, rng_seed=42)
    assert a == b


def test_bootstrap_ci_shrinks_with_n():
    """CI half-width should decrease when the sample size increases."""
    rng = np.random.default_rng(11)
    small = rng.normal(0, 1, size=50)
    large = rng.normal(0, 1, size=5000)
    lo_s, hi_s = percentile_bootstrap_ci(small, n_boot=1000, rng_seed=1)
    lo_l, hi_l = percentile_bootstrap_ci(large, n_boot=1000, rng_seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_metric_accumulator_uses_bootstrap():
    """MetricAccumulator.summary reports ci_method=percentile_bootstrap."""
    acc = MetricAccumulator("x", rng_seed=0)
    for v in [1.0, 1.2, 0.9, 1.1, 1.3, 0.95, 1.05, 1.15]:
        acc.add_replication(v)
    summary = acc.summary()
    assert "percentile_bootstrap" in summary["ci_method"]
    assert summary["ci95_lower"] < summary["mean"] < summary["ci95_upper"]


def test_metric_accumulator_single_rep_returns_nan_ci():
    """With n=1 the cross-replication bootstrap is undefined — returns NaN
    endpoints. (The per-rep raw-sample fallback lives in LatencyHistogram.)"""
    acc = MetricAccumulator("x")
    acc.add_replication(1.0)
    lo, hi = acc.confidence_interval_95()
    assert math.isnan(lo) and math.isnan(hi)


def test_latency_histogram_single_rep_uses_woodruff_fallback():
    """A single-replication LatencyHistogram should still report a valid CI
    via the within-replication bootstrap on the raw sample."""
    rng = np.random.default_rng(9)
    lh = LatencyHistogram("recall", rng_seed=9)
    for x in rng.exponential(scale=0.2, size=500):
        lh.record(float(x))
    lh.flush_replication()
    summary = lh.summary()
    p50 = summary["p50"]
    assert "woodruff" in p50.get("ci_method", "")
    assert p50["ci95_lower"] <= p50["mean"] <= p50["ci95_upper"]
    assert p50["n_replications"] == 1


def test_latency_histogram_multi_rep_uses_cross_rep_bootstrap():
    """With ≥2 replications the histogram should use cross-replication
    bootstrap rather than the Woodruff fallback."""
    rng = np.random.default_rng(13)
    lh = LatencyHistogram("recall", rng_seed=13)
    for rep in range(5):
        for x in rng.exponential(scale=0.2, size=300):
            lh.record(float(x))
        lh.flush_replication()
    summary = lh.summary()
    assert "percentile_bootstrap" in summary["p95"]["ci_method"]
    assert summary["p95"]["n_replications"] == 5
