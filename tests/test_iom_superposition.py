"""
tests/test_iom_superposition.py — IOM P_loss correctness.

The IOM P_loss is a τ-dependent model that combines two physically
distinct effects:

  - Mount-amortisation back-pressure (∝ 1/τ): the tape side cannot start
    streaming until each batch's mount completes; for small τ this drives
    the effective archival utilisation up, blowing up the cache wait.
  - Compound-Poisson burst SCV (∝ τ): viewed at the file level, batched
    archival has SCV ≈ B(τ), which Whitt-superposes with the scratch SCV.

The two effects make P_loss(τ) U-shaped, so the IOM optimum is interior
rather than pinned at τ_min.

These tests cover the qualitative consequences:
  1. P_loss is *not* τ-invariant.
  2. P_loss → 0 as ρ_data → 0.
  3. P_loss increases with ρ_data at any fixed τ.
  4. P_loss(τ) has a finite interior minimum on (0, ∞).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.iom import IOMParameters, IOM


def _params(lambda_scratch_bps: float = 30e9, rho_data_bps: float = 1e9,
            cv_iat: float = 1.5) -> IOMParameters:
    return IOMParameters(
        rho_data_bps=rho_data_bps,
        bmax_bps=6.4e9,
        disk_bw_bps=65e9,
        lambda_scratch_bps=lambda_scratch_bps,
        mean_file_bytes=64e6,
        cv_iat=cv_iat,
        cv_service=1.0,
        mean_file_life_s=3600.0,
        file_life_cv=1.0,
        mount_time_s=30.0,
        n_tape_drives=8,
        native_rate_bps=360e6,
    )


def test_ploss_is_tau_dependent():
    """P_loss must vary with τ: a constant c_a²_arch would suppress the
    batching physics entirely and make P_loss τ-invariant."""
    iom = IOM(_params())
    pl_short = iom._ploss(60.0)
    pl_long = iom._ploss(86400.0)
    assert pl_short != pytest.approx(pl_long, rel=1e-3), (
        f"Ploss is τ-invariant ({pl_short} == {pl_long}) — burst SCV must "
        f"give Ploss τ-dependence."
    )


def test_ploss_recovers_baseline_when_archival_vanishes():
    """With ρ_data_bps = 0 the archival stream contributes no batches; the
    cache sees only the pre-existing scratch load and P_loss = 0."""
    iom = IOM(_params(rho_data_bps=0.0))
    assert iom._ploss(3600.0) == pytest.approx(0.0, abs=1e-9)


def test_ploss_increases_with_archival_load():
    """At fixed τ, more archival data → more queueing impact on the cache."""
    iom_low = IOM(_params(rho_data_bps=0.5e9))
    iom_high = IOM(_params(rho_data_bps=5.0e9))
    assert iom_high._ploss(3600.0) > iom_low._ploss(3600.0)


def test_ploss_has_interior_minimum():
    """U-shape: P_loss must dip below both small-τ and large-τ values for
    some intermediate τ. This is the structural property that gives the
    IOM an interior τ*."""
    iom = IOM(_params(rho_data_bps=0.5e9))
    pl_small = iom._ploss(1.0)         # mount-overhead heavy
    pl_large = iom._ploss(86400.0)      # burst-SCV heavy
    pl_mid = min(iom._ploss(t) for t in (5.0, 10.0, 30.0, 60.0))
    assert pl_mid < pl_small, \
        f"Ploss(small)={pl_small:.2f} should exceed mid-τ minimum={pl_mid:.2f}"
    assert pl_mid < pl_large, \
        f"Ploss(large)={pl_large:.2f} should exceed mid-τ minimum={pl_mid:.2f}"


def test_ploss_grows_at_large_tau():
    """As τ → ∞, batch sizes B(τ) grow without bound; the burst SCV
    Whitt-blends into the combined process, monotonically inflating
    P_loss in the large-τ regime. This is the model-internal mechanism
    underwriting the J(τ→∞)=∞ family-wise observation."""
    iom = IOM(_params())
    pls = [iom._ploss(t) for t in (3600.0, 7200.0, 14400.0, 86400.0)]
    assert all(pls[i] < pls[i + 1] for i in range(len(pls) - 1)), (
        f"Ploss should be monotone increasing at large τ: {pls}"
    )
