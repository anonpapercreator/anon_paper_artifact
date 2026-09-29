"""
models/ctmc_predictor.py — CTMC Transient Probability Predictor
P(state=j at t+h | state=i now) = e_i · exp(Q·h)
Enables dynamic τ*(t) updates each archiver cycle.
"""
from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Optional, List, Dict


class CTMCPredictor:
    def __init__(self, Q: np.ndarray, names: Optional[List[str]] = None) -> None:
        self.Q = Q.copy().astype(float)
        self.n = Q.shape[0]
        self.names = names or [str(i) for i in range(self.n)]
        self._belief = np.ones(self.n) / self.n
        self._current_phase: Optional[int] = None
        self._expm_cache: Dict[int, np.ndarray] = {}

    def observe_phase(self, phase_idx: int) -> None:
        self._current_phase = phase_idx
        belief = np.zeros(self.n); belief[phase_idx] = 1.0
        belief = belief @ linalg.expm(self.Q * 1.0)
        belief = np.maximum(belief, 0); s = belief.sum()
        self._belief = belief / s if s > 0 else np.ones(self.n) / self.n

    def predict(self, horizon_s: float) -> np.ndarray:
        eQh = self._matrix_exp(horizon_s)
        p = self._belief @ eQh
        p = np.maximum(p, 0); s = p.sum()
        return p / s if s > 0 else np.ones(self.n) / self.n

    def predicted_drift_rate(self, r: np.ndarray, horizon_s: float) -> float:
        return float(self.predict(horizon_s) @ r)

    def hitting_probability(self, target_phases: List[int], horizon_s: float) -> float:
        p = self.predict(horizon_s)
        return float(sum(p[i] for i in target_phases if i < self.n))

    def phase_summary_at_horizon(self, r: np.ndarray, horizon_s: float) -> dict:
        p = self.predict(horizon_s)
        return {
            "horizon_s": horizon_s, "horizon_min": round(horizon_s / 60, 1),
            "phase_probabilities": {self.names[i]: round(float(p[i]), 4) for i in range(self.n)},
            "most_likely_phase": self.names[int(np.argmax(p))],
            "expected_drift_bps": float(p @ r),
            "expected_drift_gbs": round(float(p @ r) / (1024**3), 4),
            "prob_checkpoint": round(float(p[2]), 4),
            "prob_recall":     round(float(p[3]), 4),
        }

    def _matrix_exp(self, t: float) -> np.ndarray:
        key = round(t)
        if key not in self._expm_cache:
            self._expm_cache[key] = linalg.expm(self.Q * t)
            if len(self._expm_cache) > 500:
                del self._expm_cache[next(iter(self._expm_cache))]
        return self._expm_cache[key]


def transient_trajectory(Q: np.ndarray, initial_phase: int,
                          t_max_s: float = 3600.0, n_points: int = 100) -> dict:
    n = Q.shape[0]; t_grid = np.linspace(0, t_max_s, n_points)
    P_t = np.zeros((n_points, n)); P0 = np.zeros(n); P0[initial_phase] = 1.0
    for k, t in enumerate(t_grid):
        row = P0 @ linalg.expm(Q * t)
        row = np.maximum(row, 0); s = row.sum()
        P_t[k, :] = row / s if s > 0 else np.ones(n) / n
    return {"t_grid_s": t_grid, "t_grid_min": t_grid / 60.0, "P_t": P_t, "n_phases": n}


def stationary_from_Q(Q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Solve π·Q = 0 subject to Σπ_i = 1 for the stationary distribution of a
    continuous-time Markov chain.

    Degeneracy guard:
        The left null-space of Q has dimension equal to the number of
        communicating classes. If more than one class exists (i.e., the
        chain is reducible) the stationary distribution is not unique —
        every convex combination of class-level stationary vectors is
        itself stationary. We detect this by computing rank(Q^T) via SVD
        and raise ``DisconnectedChainError`` rather than silently returning
        a uniform or garbage vector.

    Args:
        Q:   n×n generator matrix (rows sum to zero, off-diagonals ≥ 0).
        tol: Singular-value tolerance for null-space rank estimation.

    Raises:
        DisconnectedChainError: if Q has >1 zero singular values, meaning
            more than one communicating class.
    """
    # Import lazily to avoid circular-ish dependency at module load.
    from models.phase_classifier import DisconnectedChainError

    n = Q.shape[0]
    # Rank of Q^T. The null-space dimension of Q^T is n − rank(Q^T).
    s = np.linalg.svd(Q.T, compute_uv=False)
    s_max = s.max() if s.size else 1.0
    threshold = tol * max(s_max, 1.0) * max(n, 1)
    null_dim = int(np.sum(s < threshold))
    if null_dim > 1:
        raise DisconnectedChainError(
            f"Generator matrix has {null_dim} zero singular values (>1); "
            f"the chain has multiple communicating classes and π is not "
            f"uniquely defined. Singular values: "
            f"{np.round(s, 6).tolist()}. Analyse each irreducible "
            f"sub-chain separately."
        )

    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
        pi = np.maximum(pi, 0)
        total = pi.sum()
        if total <= 0:
            raise np.linalg.LinAlgError("stationary vector has zero mass")
        pi /= total
    except np.linalg.LinAlgError as err:
        raise DisconnectedChainError(
            f"Failed to solve for stationary distribution: {err}. "
            f"The generator matrix may be singular or ill-conditioned."
        ) from err
    return pi
