"""
tests/test_diurnal_nhpp.py — Diurnal modulation preserves time-averaged λ.

Phase 3.4 replaces the biased ``IAT = base_iat / factor(t)`` modulation with
non-homogeneous Poisson process (NHPP) thinning. The old scheme underestimated
the long-run rate because E[1/φ] > 1/E[φ] = 1 whenever φ is non-constant
(Jensen's inequality).

The contract this module asserts is:

  1. Over a full 24h period, the empirical arrival rate matches λ_base
     within 2% (exponential base: the canonical NHPP case).
  2. The same property holds for the Pareto base (scale-family candidate).
  3. When diurnal modulation is disabled, the empirical rate equals λ_base
     (regression test for the non-modulated path).
  4. The instantaneous rate does vary with time-of-day: arrival counts in
     an on-peak hour exceed arrival counts in an off-peak hour.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload.generator import DiurnalModulator, SyntheticWorkloadGenerator


class _StubRNG:
    """Minimal RNG manager stub that hands the same numpy Generator to every
    stream the workload generator asks for. Enough for testing _sample_iat."""

    def __init__(self, seed: int = 1234):
        self._gen = np.random.default_rng(seed)

    def get(self, name: str):
        return self._gen


class _StubFS:
    pass


def _make_gen(iat_dist: str = "exponential", diurnal: bool = True,
              amplitude: float = 0.6, peak_hour: float = 10.0,
              exp_rate: float = 1.0, pareto_alpha: float = 1.8,
              pareto_xmin: float = 0.1, seed: int = 1):
    cfg = {
        "file_size_lognormal_mu": 1e7,
        "file_size_lognormal_sigma": 2e7,
        "file_size_min_bytes": 1024,
        "file_size_max_bytes": 10**12,
        "iat_distribution": iat_dist,
        "iat_pareto_alpha": pareto_alpha,
        "iat_pareto_xmin_s": pareto_xmin,
        "iat_exponential_rate": exp_rate,
        "read_fraction": 0.7,
        "working_set_size": 1000,
        "zipf_s": 1.0,
        "diurnal_enabled": diurnal,
        "diurnal_amplitude": amplitude,
        "diurnal_peak_hour": peak_hour,
    }
    rng = _StubRNG(seed=seed)
    gen = SyntheticWorkloadGenerator.__new__(SyntheticWorkloadGenerator)
    # Bypass __init__ (which needs simpy & filesystem) by setting just the
    # attributes that _sample_iat and _sample_base_iat touch.
    gen._env = None
    gen._cfg = cfg
    gen._rng = rng
    gen._fs = None
    gen._rng_size = rng.get("file_size")
    gen._rng_iat = rng.get("iat")
    gen._rng_access = rng.get("access_freq")
    gen._rng_rw = rng.get("rw_ratio")
    gen._rng_age = rng.get("file_age_init")
    gen._size_mu = cfg["file_size_lognormal_mu"]
    gen._size_sigma = cfg["file_size_lognormal_sigma"]
    gen._size_min = cfg["file_size_min_bytes"]
    gen._size_max = cfg["file_size_max_bytes"]
    gen._iat_dist = cfg["iat_distribution"]
    gen._pareto_alpha = cfg["iat_pareto_alpha"]
    gen._pareto_xmin = cfg["iat_pareto_xmin_s"]
    gen._exp_rate = cfg["iat_exponential_rate"]
    gen._read_frac = cfg["read_fraction"]
    gen._diurnal_enabled = diurnal
    if diurnal:
        gen._diurnal = DiurnalModulator(amplitude=amplitude, peak_hour=peak_hour)
    return gen


def _simulate_arrivals(gen: SyntheticWorkloadGenerator, t_max: float) -> np.ndarray:
    """Generate arrival timestamps up to t_max."""
    t = 0.0
    times = []
    # Hard cap to prevent runaway generation on bad parameters
    for _ in range(int(t_max * 100) + 100):
        iat = gen._sample_iat(t)
        t += iat
        if t > t_max:
            break
        times.append(t)
    return np.asarray(times)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_diurnal_preserves_time_averaged_rate_exponential(seed):
    """Exponential base + diurnal: empirical λ over 24h within 2% of λ_base."""
    lambda_base = 1.0  # 1 arrival per second
    gen = _make_gen(iat_dist="exponential", exp_rate=lambda_base,
                    amplitude=0.6, seed=seed)
    t_max = 86400.0  # 24 h
    times = _simulate_arrivals(gen, t_max)
    empirical = len(times) / t_max
    rel_err = abs(empirical - lambda_base) / lambda_base
    assert rel_err < 0.02, (
        f"Empirical rate {empirical:.4f} differs from base {lambda_base} "
        f"by {rel_err*100:.2f}% (>2%). seed={seed}"
    )


def test_diurnal_preserves_time_averaged_rate_pareto():
    """Pareto base + diurnal: empirical λ within 3% of 1/E[IAT].
    Slightly looser tolerance because Pareto's heavy tail inflates the
    sampling variance."""
    alpha = 2.5  # mean exists (need α > 1)
    xmin = 0.5
    lambda_base = (alpha - 1.0) / xmin  # Lomax mean = xmin/(α-1)
    gen = _make_gen(iat_dist="pareto", pareto_alpha=alpha, pareto_xmin=xmin,
                    amplitude=0.6, seed=42)
    t_max = 86400.0
    times = _simulate_arrivals(gen, t_max)
    empirical = len(times) / t_max
    rel_err = abs(empirical - lambda_base) / lambda_base
    assert rel_err < 0.03, (
        f"Pareto empirical rate {empirical:.4f} differs from base "
        f"{lambda_base:.4f} by {rel_err*100:.2f}% (>3%)."
    )


def test_non_diurnal_rate_matches_base():
    """Regression: with diurnal disabled the rate just equals the base."""
    gen = _make_gen(iat_dist="exponential", exp_rate=2.0, diurnal=False, seed=11)
    t_max = 3600.0
    times = _simulate_arrivals(gen, t_max)
    empirical = len(times) / t_max
    assert abs(empirical - 2.0) / 2.0 < 0.03


def test_peak_hour_has_higher_rate_than_off_peak():
    """The modulation produces a real time-of-day effect: more arrivals near
    the sine crest than near the trough.

    With λ(t) = λ_base·(1 + A·sin(2π(t − τ_peak)/T)), the sine crest is at
    τ_peak + T/4 and the trough at τ_peak − T/4. For τ_peak=10h and T=24h
    that places the crest at 16:00 and the trough at 04:00."""
    gen = _make_gen(iat_dist="exponential", exp_rate=1.0, amplitude=0.8,
                    peak_hour=10.0, seed=99)
    times = _simulate_arrivals(gen, t_max=86400.0)
    crest_count = int(np.sum((times >= 15 * 3600) & (times <= 17 * 3600)))
    trough_count = int(np.sum((times >= 3 * 3600) & (times <= 5 * 3600)))
    assert crest_count > trough_count, (
        f"Crest window ({crest_count}) did not exceed trough ({trough_count}) "
        f"— modulation is degenerate."
    )


def test_max_factor_consistent_with_amplitude():
    m = DiurnalModulator(amplitude=0.6, peak_hour=10.0)
    assert m.max_factor == pytest.approx(1.6)


def test_time_rescaling_preserves_pareto_tail():
    """The time-rescaling construction must preserve the heavy-tail
    structure of a Pareto base — that's the entire point of using it
    instead of thinning. We check the empirical 95th-percentile IAT
    against the closed-form Lomax 95th percentile.

    For Lomax(α, x_min): F(x) = 1 − (1 + x/x_min)^(−α)
    Inverse: x_p = x_min · ((1 − p)^(−1/α) − 1)

    We pick a long enough sample for the tail estimate to stabilise. The
    diurnal modulation only stretches/compresses time; it should not
    *reshape* the tail of the base IAT distribution.
    """
    alpha = 2.5
    xmin = 0.5
    gen = _make_gen(iat_dist="pareto", pareto_alpha=alpha, pareto_xmin=xmin,
                    amplitude=0.6, seed=7)
    t_max = 5 * 86400.0  # 5 days for a tail estimate
    times = _simulate_arrivals(gen, t_max)
    iats = np.diff(np.concatenate([[0.0], times]))
    # Drop the head IAT (correlates with t=0 origin) and trim very
    # small floor-clamped values (the modulator floors φ at 0.01).
    iats = iats[iats > 1e-5][1:]

    # Theoretical Lomax 95th percentile *un-modulated*. Time-rescaling
    # should leave the IAT distribution invariant (the rescaling
    # preserves the X_i structure; only the *physical* placement of
    # each event in absolute time is changed).
    p = 0.95
    expected = xmin * ((1.0 - p) ** (-1.0 / alpha) - 1.0)
    empirical = float(np.percentile(iats, 95))
    rel_err = abs(empirical - expected) / max(expected, 1e-12)
    assert rel_err < 0.20, (
        f"Empirical 95th-pct IAT {empirical:.4f} differs from Lomax "
        f"closed form {expected:.4f} by {rel_err*100:.1f}% (>20%) — "
        f"time-rescaling reshaped the tail."
    )
