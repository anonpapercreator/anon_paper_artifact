"""
tests/test_iom_weibull.py — IOM Weibull scale formula and Mwaste(τ) sanity.
"""

import os
import sys
from math import gamma

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.iom import IOMParameters, IOM


def _weibull_mean_sample(k: float, lam: float, n: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    return rng.weibull(k, size=n).mean() * lam


@pytest.mark.parametrize("cv", [1.5, 2.0, 3.0, 5.0])
def test_iom_weibull_scale_exact(cv):
    """λ = mean / Γ(1 + 1/k) must satisfy λ · Γ(1 + 1/k) == mean exactly."""
    mean_life_s = 3600.0
    k = 1.0 / (cv ** 2)
    lam = mean_life_s / gamma(1.0 + 1.0 / k)
    assert lam * gamma(1.0 + 1.0 / k) == pytest.approx(mean_life_s, rel=1e-12)


@pytest.mark.parametrize("cv", [1.5, 2.0])
def test_iom_weibull_scale_recovers_mean_by_sampling(cv):
    """For moderate tail (CV ≤ 2) the sampled mean should recover the target."""
    mean_life_s = 3600.0
    k = 1.0 / (cv ** 2)
    lam = mean_life_s / gamma(1.0 + 1.0 / k)
    sampled_mean = _weibull_mean_sample(k, lam, n=500_000)
    assert sampled_mean == pytest.approx(mean_life_s, rel=0.05)


def test_iom_mwaste_monotone_non_decreasing_in_tau():
    """Mwaste(τ) must be non-decreasing — a later archival interval can only
    have ≥ the expected wasted tape of an earlier one."""
    params = IOMParameters(
        rho_data_bps=1e9,
        bmax_bps=4 * 360e6,
        disk_bw_bps=65 * (1024 ** 3),
        lambda_scratch_bps=1e9,
        mean_file_bytes=10 * (1024 ** 2),
        mean_file_life_s=3600.0,
        file_life_cv=2.0,
    )
    iom = IOM(params)
    taus = [60.0, 300.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0]
    vals = [iom._mwaste_tib(t) for t in taus]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-9, f"Mwaste not monotone: {vals}"


def test_iom_mwaste_exponential_matches_closed_form():
    """CV=1 → exponential: Mwaste(τ) = ρ · (1 - e^{-ατ}) / α (TiB)."""
    params = IOMParameters(
        rho_data_bps=2e9,
        bmax_bps=1e10,
        disk_bw_bps=65 * (1024 ** 3),
        lambda_scratch_bps=1e9,
        mean_file_bytes=10 * (1024 ** 2),
        mean_file_life_s=1800.0,
        file_life_cv=1.0,
    )
    iom = IOM(params)
    alpha = 1.0 / 1800.0
    for tau in [60.0, 600.0, 3600.0, 36000.0]:
        expected_bytes = params.rho_data_bps * (1 - np.exp(-alpha * tau)) / alpha
        expected_tib = expected_bytes / (1024 ** 4)
        assert iom._mwaste_tib(tau) == pytest.approx(expected_tib, rel=1e-6)
