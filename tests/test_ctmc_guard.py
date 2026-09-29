"""
tests/test_ctmc_guard.py — Guards against degenerate CTMC fits.

Phase 3.5 adds two guards that turn previously silent failure modes into
explicit diagnostics:

  1. ``PhaseClassifier.fit_generator_matrix`` emits
     ``AbsorbingPhaseWarning`` (or raises it under ``strict=True``) when a
     fitted phase has no outgoing transitions — either because it was
     never observed, or because it was observed but never left.
     Stationary analysis that assumes an ergodic chain would silently
     return misleading probabilities otherwise.

  2. ``stationary_from_Q`` checks the null-space dimension of Q^T via
     SVD. A dimension greater than one means the chain has more than one
     communicating class, so π is not uniquely defined. The guard raises
     ``DisconnectedChainError`` rather than silently returning a
     degenerate solve that depends on the choice of normalising row.

The tests cover both guards end-to-end.
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ctmc_predictor import stationary_from_Q
from models.phase_classifier import (
    AbsorbingPhaseWarning,
    DisconnectedChainError,
    N_PHASES,
    Phase,
    PhaseClassifier,
    TelemetrySample,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _sample(write_bps: float = 0.0, recall_bps: float = 0.0,
            interval_s: float = 10.0) -> TelemetrySample:
    """Build a telemetry sample that classifies cleanly by I/O rates."""
    return TelemetrySample(
        sim_time_s=0.0,
        cache_bytes=1.0,
        cache_capacity=1e12,
        write_rate_bps=write_bps,
        read_rate_bps=0.0,
        archival_rate_bps=0.0,
        recall_rate_bps=recall_bps,
        files_in_cache=0,
        files_on_tape=0,
        archiver_active=False,
        interval_s=interval_s,
    )


def _make_classifier() -> PhaseClassifier:
    return PhaseClassifier(cache_capacity_bytes=1e12,
                           ess_peak_gbs=65.0, tape_peak_gbs=6.4)


def _feed_checkpoint(clf: PhaseClassifier, n: int = 5) -> None:
    write_bps = 40.0 * (1024 ** 3)  # ≥ 40% of ESS peak → CHECKPOINT
    for _ in range(n):
        clf.observe(_sample(write_bps=write_bps))


def _feed_compute(clf: PhaseClassifier, n: int = 5) -> None:
    # Small amount of write-side I/O: below CHECKPOINT, above IDLE → COMPUTE.
    write_bps = 5.0 * (1024 ** 3)
    for _ in range(n):
        clf.observe(_sample(write_bps=write_bps))


# ---------------------------------------------------------------------------
# fit_generator_matrix: unobserved-phase guard
# ---------------------------------------------------------------------------
class TestUnobservedPhaseGuard:

    def test_unobserved_phase_emits_warning(self):
        """Only CHECKPOINT and COMPUTE observed — IDLE and RECALL never
        visited. Fit should warn."""
        clf = _make_classifier()
        _feed_checkpoint(clf)
        _feed_compute(clf)
        with pytest.warns(AbsorbingPhaseWarning, match="unobserved"):
            Q = clf.fit_generator_matrix()
        # Unobserved rows should be all zero so Q·1 = 0 still holds.
        assert np.allclose(Q[Phase.IDLE, :], 0.0)
        assert np.allclose(Q[Phase.RECALL, :], 0.0)
        assert np.allclose(Q.sum(axis=1), 0.0, atol=1e-9)

    def test_strict_mode_raises_on_unobserved_phase(self):
        clf = _make_classifier()
        _feed_checkpoint(clf)
        _feed_compute(clf)
        with pytest.raises(AbsorbingPhaseWarning):
            clf.fit_generator_matrix(strict=True)


# ---------------------------------------------------------------------------
# fit_generator_matrix: absorbing-phase guard
# ---------------------------------------------------------------------------
class TestAbsorbingPhaseGuard:

    def test_single_phase_entire_trace_is_absorbing(self):
        """If every observation classifies as the same phase, the system
        accrues dwell time there but never transitions out — an absorbing
        row."""
        clf = _make_classifier()
        _feed_checkpoint(clf, n=10)  # stays in CHECKPOINT throughout
        with pytest.warns(AbsorbingPhaseWarning) as record:
            Q = clf.fit_generator_matrix()
        combined_msg = " ".join(str(w.message) for w in record)
        # One absorbing phase (CHECKPOINT) + three unobserved.
        assert "absorbing" in combined_msg
        assert "unobserved" in combined_msg
        # CHECKPOINT row has dwell but no off-diagonal mass.
        assert np.allclose(Q[Phase.CHECKPOINT, :], 0.0)


# ---------------------------------------------------------------------------
# fit_generator_matrix: healthy ergodic case does not warn
# ---------------------------------------------------------------------------
class TestHealthyChain:

    def test_no_warning_when_all_phases_cycle(self):
        """Alternating observations across all four phases — no degeneracy
        flags should fire."""
        clf = _make_classifier()
        # Rotate through all four phases enough times to populate Q fully.
        write_checkpoint = 40.0 * (1024 ** 3)
        recall_high = 5.0 * (1024 ** 3)
        write_compute = 5.0 * (1024 ** 3)
        # Sequence: IDLE → COMPUTE → CHECKPOINT → COMPUTE → RECALL → COMPUTE → IDLE ...
        seq = [
            _sample(),                                   # IDLE
            _sample(write_bps=write_compute),            # COMPUTE
            _sample(write_bps=write_checkpoint),         # CHECKPOINT
            _sample(write_bps=write_compute),            # COMPUTE
            _sample(recall_bps=recall_high),             # RECALL
            _sample(write_bps=write_compute),            # COMPUTE
        ]
        for _ in range(6):
            for s in seq:
                clf.observe(s)
        with warnings.catch_warnings():
            warnings.simplefilter("error", AbsorbingPhaseWarning)
            Q = clf.fit_generator_matrix()
        # Every row should sum to zero and have at least one positive off-diag.
        assert np.allclose(Q.sum(axis=1), 0.0, atol=1e-9)
        for i in range(N_PHASES):
            off_diag = np.delete(Q[i, :], i)
            assert off_diag.sum() > 0, f"phase {i} has no outgoing mass"


# ---------------------------------------------------------------------------
# stationary_from_Q: disconnected-chain guard
# ---------------------------------------------------------------------------
class TestStationaryFromQGuard:

    def test_irreducible_2x2_chain_returns_valid_pi(self):
        """Two-state ergodic chain with rates 1↔2 should give
        π = (2/3, 1/3) modulo numerical noise."""
        Q = np.array([[-1.0,  1.0],
                      [ 2.0, -2.0]])
        pi = stationary_from_Q(Q)
        assert pi.shape == (2,)
        assert np.isclose(pi.sum(), 1.0)
        np.testing.assert_allclose(pi, [2.0 / 3.0, 1.0 / 3.0], atol=1e-9)

    def test_disconnected_chain_raises(self):
        """Block-diagonal Q ⇒ two communicating classes ⇒ π not unique."""
        Q = np.array([
            [-1.0,  1.0,  0.0,  0.0],
            [ 2.0, -2.0,  0.0,  0.0],
            [ 0.0,  0.0, -3.0,  3.0],
            [ 0.0,  0.0,  1.0, -1.0],
        ])
        with pytest.raises(DisconnectedChainError, match=r"singular values"):
            stationary_from_Q(Q)

    def test_absorbing_rows_still_solvable_when_unique(self):
        """A Q with one absorbing phase (all-zero row) but a unique
        stationary distribution is still solvable — all mass accumulates
        on the absorbing state."""
        # 3-state chain: 0→1→2, with state 2 absorbing.
        Q = np.array([
            [-1.0,  1.0,  0.0],
            [ 0.0, -1.0,  1.0],
            [ 0.0,  0.0,  0.0],
        ])
        pi = stationary_from_Q(Q)
        np.testing.assert_allclose(pi, [0.0, 0.0, 1.0], atol=1e-9)

    def test_ergodic_3x3_chain(self):
        """Hand-verified 3-state ergodic chain."""
        # Uniform rate 1 between all pairs ⇒ π = (1/3, 1/3, 1/3).
        Q = np.array([
            [-2.0,  1.0,  1.0],
            [ 1.0, -2.0,  1.0],
            [ 1.0,  1.0, -2.0],
        ])
        pi = stationary_from_Q(Q)
        np.testing.assert_allclose(pi, [1.0 / 3.0] * 3, atol=1e-9)

    def test_fit_then_solve_round_trip_ergodic(self):
        """fit_generator_matrix on a healthy chain should produce a Q that
        stationary_from_Q can solve without raising."""
        clf = _make_classifier()
        write_checkpoint = 40.0 * (1024 ** 3)
        recall_high = 5.0 * (1024 ** 3)
        write_compute = 5.0 * (1024 ** 3)
        seq = [
            _sample(),
            _sample(write_bps=write_compute),
            _sample(write_bps=write_checkpoint),
            _sample(write_bps=write_compute),
            _sample(recall_bps=recall_high),
            _sample(write_bps=write_compute),
        ]
        for _ in range(10):
            for s in seq:
                clf.observe(s)
        Q = clf.fit_generator_matrix()
        pi = stationary_from_Q(Q)
        assert pi.shape == (N_PHASES,)
        assert np.isclose(pi.sum(), 1.0)
        assert np.all(pi >= 0)
