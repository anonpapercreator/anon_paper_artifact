"""
models/phase_classifier.py — Workload Phase Classification via CTMC
Classifies HSM workload into IDLE/COMPUTE/CHECKPOINT/RECALL phases.
Fits the CTMC generator matrix Q from observed dwell times and transitions.

MLE estimator (continuous-time Markov chain):
    q_ij = N_ij / T_i      (i ≠ j)
    q_ii = − Σ_{j≠i} q_ij

where N_ij is the number of observed i → j transitions and T_i the total
dwell time in state i over the observation window.

References:
    Albert, A. (1962). "Estimating the infinitesimal generator of a
        continuous time, finite state Markov process." Annals of
        Mathematical Statistics, 33(2), 727–753.
    Billingsley, P. (1961). "Statistical Inference for Markov Processes."
        University of Chicago Press.

Note: Anderson & Goodman (1957) covers the analogous MLE for *discrete-time*
Markov chains; the formula above is its continuous-time counterpart due to
Albert and Billingsley.
"""
from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum


class AbsorbingPhaseWarning(UserWarning):
    """A fitted phase has no outgoing transitions (absorbing state) or was
    never observed. Stationary analysis that assumes an ergodic chain will
    give misleading results — the caller should either extend the observation
    window or analyse the irreducible sub-chain separately."""


class DisconnectedChainError(ValueError):
    """The estimated generator matrix has more than one communicating class,
    so the linear system for the stationary distribution is rank-deficient and
    π is not uniquely defined."""


class Phase(IntEnum):
    IDLE       = 0
    COMPUTE    = 1
    CHECKPOINT = 2
    RECALL     = 3


PHASE_NAMES = {Phase.IDLE: "IDLE", Phase.COMPUTE: "COMPUTE",
               Phase.CHECKPOINT: "CHECKPOINT", Phase.RECALL: "RECALL"}
N_PHASES = 4


@dataclass
class TelemetrySample:
    sim_time_s:          float
    cache_bytes:         float
    cache_capacity:      float
    write_rate_bps:      float
    read_rate_bps:       float
    archival_rate_bps:   float
    recall_rate_bps:     float
    files_in_cache:      int
    files_on_tape:       int
    archiver_active:     bool
    interval_s:          float

    @property
    def cache_fill_fraction(self) -> float:
        return self.cache_bytes / max(self.cache_capacity, 1)

    @property
    def total_io_bps(self) -> float:
        return self.write_rate_bps + self.read_rate_bps


class PhaseClassifier:
    CHECKPOINT_WRITE_FRACTION = 0.40
    RECALL_READ_FRACTION      = 0.20
    IDLE_TOTAL_FRACTION       = 0.05

    def __init__(self, cache_capacity_bytes: float,
                 ess_peak_gbs: float = 65.0,
                 tape_peak_gbs: float = 6.4) -> None:
        self._capacity  = cache_capacity_bytes
        self._ess_peak  = ess_peak_gbs  * (1024**3)
        self._tape_peak = tape_peak_gbs * (1024**3)
        self._n_transitions  = np.zeros((N_PHASES, N_PHASES), dtype=float)
        self._dwell_time     = np.zeros(N_PHASES, dtype=float)
        self._phase_write_total = np.zeros(N_PHASES, dtype=float)
        self._phase_drain_total = np.zeros(N_PHASES, dtype=float)
        self._phase_count       = np.zeros(N_PHASES, dtype=int)
        self._prev_phase: Optional[Phase] = None
        self._observations: List[dict] = []

    def observe(self, sample: TelemetrySample) -> Phase:
        phase = self._classify(sample)
        self._dwell_time[phase] += sample.interval_s
        self._phase_write_total[phase] += sample.write_rate_bps
        self._phase_drain_total[phase] += sample.archival_rate_bps + sample.recall_rate_bps
        self._phase_count[phase]       += 1
        if self._prev_phase is not None and self._prev_phase != phase:
            self._n_transitions[self._prev_phase, phase] += 1
        self._observations.append({"phase": phase, "sim_time_s": sample.sim_time_s})
        self._prev_phase = phase
        return phase

    def fit_generator_matrix(self, strict: bool = False) -> np.ndarray:
        """
        Estimate the CTMC generator matrix Q by maximum likelihood
        (Albert 1962; Billingsley 1961).

        Guards against degenerate fits:
            - Unobserved phase (no dwell time): warn; leave its row all zero.
            - Absorbing phase (dwell > 0 but no outgoing transitions):
              warn; leave its row all zero (so exp(Qt) keeps mass there).

        Args:
            strict: If True, raise ``AbsorbingPhaseWarning`` as an error
                    when any phase is flagged. Default False (emit a warning
                    and return the Q that best represents the observation).
        """
        Q = np.zeros((N_PHASES, N_PHASES), dtype=float)
        flagged: List[Tuple[int, str]] = []
        for i in range(N_PHASES):
            T_i = self._dwell_time[i]
            if T_i < 1e-9:
                flagged.append((i, "unobserved"))
                continue  # Zero row — preserves Q·1=0
            for j in range(N_PHASES):
                if i != j:
                    Q[i, j] = self._n_transitions[i, j] / T_i
            Q[i, i] = -np.sum(Q[i, :])
            # Absorbing: dwell > 0 but no transitions out.
            if np.all(self._n_transitions[i, :] == 0):
                flagged.append((i, "absorbing"))
        if flagged:
            names = [PHASE_NAMES[Phase(i)] for i, _ in flagged]
            kinds = [k for _, k in flagged]
            msg = (f"Generator matrix fit surfaced degenerate phases: "
                   f"{list(zip(names, kinds))}. Stationary and transient "
                   f"analyses based on Q will treat these rows as zero "
                   f"(no transitions out).")
            if strict:
                raise AbsorbingPhaseWarning(msg)
            warnings.warn(msg, AbsorbingPhaseWarning, stacklevel=2)
        return Q

    def net_drift_rates_bps(self) -> np.ndarray:
        r = np.zeros(N_PHASES, dtype=float)
        for i in range(N_PHASES):
            n = max(self._phase_count[i], 1)
            r[i] = (self._phase_write_total[i] - self._phase_drain_total[i]) / n
        return r

    def stationary_phase_distribution(self) -> np.ndarray:
        Q = self.fit_generator_matrix()
        n = N_PHASES
        A = Q.T.copy()
        A[-1, :] = 1.0
        b = np.zeros(n); b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
            pi = np.maximum(pi, 0)
            pi /= max(pi.sum(), 1e-12)
        except np.linalg.LinAlgError:
            pi = np.ones(n) / n
        return pi

    def _classify(self, s: TelemetrySample) -> Phase:
        write_frac  = s.write_rate_bps  / max(self._ess_peak,  1)
        recall_frac = s.recall_rate_bps / max(self._tape_peak, 1)
        total_frac  = s.total_io_bps    / max(self._ess_peak,  1)
        if write_frac  >= self.CHECKPOINT_WRITE_FRACTION: return Phase.CHECKPOINT
        if recall_frac >= self.RECALL_READ_FRACTION:       return Phase.RECALL
        if total_frac  <= self.IDLE_TOTAL_FRACTION:        return Phase.IDLE
        return Phase.COMPUTE


def build_telemetry_from_simulator_state(
        sim_time_s, cache_bytes, cache_capacity,
        prev_ess_written, curr_ess_written, prev_ess_read, curr_ess_read,
        prev_tape_written, curr_tape_written, prev_tape_read, curr_tape_read,
        files, interval_s, archiver_migrated) -> TelemetrySample:
    dt = max(interval_s, 1e-9)
    files_in_cache = sum(1 for f in files.values() if not f.get("on_tape", True))
    files_on_tape  = sum(1 for f in files.values() if f.get("on_tape", False))
    return TelemetrySample(
        sim_time_s=sim_time_s, cache_bytes=cache_bytes, cache_capacity=cache_capacity,
        write_rate_bps    = max(0.0, (curr_ess_written  - prev_ess_written)  / dt),
        read_rate_bps     = max(0.0, (curr_ess_read     - prev_ess_read)     / dt),
        archival_rate_bps = max(0.0, (curr_tape_written - prev_tape_written) / dt),
        recall_rate_bps   = max(0.0, (curr_tape_read    - prev_tape_read)    / dt),
        files_in_cache=files_in_cache, files_on_tape=files_on_tape,
        archiver_active=archiver_migrated, interval_s=interval_s,
    )
