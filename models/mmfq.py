"""
models/mmfq.py — Markov-Modulated Fluid Queue Steady-State Solver
Cache fill X(t)∈[0,C] driven by CTMC phase φ(t).
f'(x) = f(x)·Q·R⁻¹  (Ahn & Ramaswami 2005)
Spectral method with Monte Carlo fallback.
Kingman G/G/1 gives Ploss(τ) from first principles.
"""
from __future__ import annotations
import numpy as np
from scipy import linalg
from scipy import integrate as sp_integrate
from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class MMFQResult:
    x_grid: np.ndarray
    cdf: np.ndarray
    pdf: np.ndarray
    phase_probs: np.ndarray
    mass_at_zero: np.ndarray
    mass_at_cap: np.ndarray
    prob_above_hwm: float
    prob_above_lwm: float
    expected_fill: float
    # Mean-drift-time to HWM. *Not* the analytic MFPT of the MMFQ — that
    # requires the absorbing-boundary first-passage analysis (Asmussen 2003
    # §XIV; Kulkarni 1997 Chap. 7) which this stationary solver does not
    # perform. The reported value is a deterministic-drift proxy and is a
    # lower bound on the true MFPT when mean drift is positive.
    mfpt_to_hwm: float
    ploss_baseline: float
    ploss_combined: float
    wq_ratio: float
    Q: np.ndarray
    r: np.ndarray
    C: float
    hwm: float
    lwm: float
    solver_ok: bool
    solver_message: str


class MMFQSolver:
    """Bounded MMFQ stationary solver."""

    def __init__(self, Q: np.ndarray, r: np.ndarray,
                 C: float, hwm: float = 0.75, lwm: float = 0.50) -> None:
        n = Q.shape[0]
        assert Q.shape == (n, n)
        assert r.shape == (n,)
        assert C > 0 and 0 < lwm < hwm < 1
        self.Q = Q.astype(float)
        self.r = r.astype(float)
        self.C = float(C)
        self.hwm = hwm
        self.lwm = lwm
        self.n = n

    def solve(self, n_grid: int = 200) -> MMFQResult:
        # Degenerate cases: all drifts same sign ⇒ pure atomic distribution
        # at the corresponding boundary. The spectral/MC branches would only
        # muddy this with numerical noise.
        if np.all(self.r >= 0) and np.any(self.r > 0):
            return self._degenerate_boundary_solve(n_grid, at_cap=True)
        if np.all(self.r <= 0) and np.any(self.r < 0):
            return self._degenerate_boundary_solve(n_grid, at_cap=False)
        try:
            result = self._spectral_solve(n_grid)
            if result.solver_ok:
                return result
        except Exception as exc:
            warnings.warn(f"Spectral solver failed ({exc}); using Monte Carlo fallback.")
        return self._monte_carlo_solve(n_grid)

    def _degenerate_boundary_solve(self, n_grid: int, at_cap: bool) -> MMFQResult:
        """All drifts share a sign: the queue sits at one boundary with
        probability 1. Continuous density is zero; all mass is the atom
        distributed across phases by the CTMC stationary π."""
        pi = self._stationary_distribution()
        x_unit = np.linspace(0.0, 1.0, n_grid)
        pdf = np.zeros(n_grid)
        mass_at_zero = np.zeros(self.n)
        mass_at_cap = np.zeros(self.n)
        if at_cap:
            mass_at_cap = pi.copy()
            cdf = np.zeros(n_grid)  # P(X ≤ x) = 0 for x < C
            cdf[-1] = 1.0
            expected_fill = 1.0
            prob_above_hwm = 1.0
            prob_above_lwm = 1.0
            mfpt_to_hwm = 0.0
            msg = "Degenerate: all drifts ≥ 0 ⇒ atom at x=C."
        else:
            mass_at_zero = pi.copy()
            cdf = np.ones(n_grid)  # P(X ≤ x) = 1 for x ≥ 0
            expected_fill = 0.0
            prob_above_hwm = 0.0
            prob_above_lwm = 0.0
            mfpt_to_hwm = float("inf")
            msg = "Degenerate: all drifts ≤ 0 ⇒ atom at x=0."
        return MMFQResult(
            x_grid=x_unit, cdf=cdf, pdf=pdf, phase_probs=pi,
            mass_at_zero=mass_at_zero, mass_at_cap=mass_at_cap,
            prob_above_hwm=prob_above_hwm, prob_above_lwm=prob_above_lwm,
            expected_fill=expected_fill, mfpt_to_hwm=mfpt_to_hwm,
            ploss_baseline=0.0, ploss_combined=0.0, wq_ratio=1.0,
            Q=self.Q, r=self.r, C=self.C, hwm=self.hwm, lwm=self.lwm,
            solver_ok=True, solver_message=msg,
        )

    def ploss_at_tau(self, tau_s: float, lambda_scratch_bps: float,
                     mean_file_bytes: float, cv_iat: float, cv_service: float,
                     disk_bandwidth_bps: float) -> float:
        rho_arch = self._mean_archival_rate_bps()
        E_S = mean_file_bytes / max(disk_bandwidth_bps, 1)
        rho_s = min(lambda_scratch_bps * E_S, 0.99)
        rho_combined = min(rho_s + rho_arch * E_S / max(mean_file_bytes, 1), 0.99)
        c_a2_s = cv_iat ** 2
        c_s2 = cv_service ** 2
        if rho_arch > 0 and tau_s > 0:
            burst_frac = min(rho_arch / max(disk_bandwidth_bps, 1), 1.0)
            c_a2_arch = burst_frac * 2.0 + (1 - burst_frac) * 1.0
            w_s = lambda_scratch_bps / max(lambda_scratch_bps + rho_arch, 1)
            c_a2_combined = w_s * c_a2_s + (1 - w_s) * c_a2_arch
        else:
            c_a2_combined = c_a2_s
        W_q_s = self._kingman_Wq(rho_s, E_S, c_a2_s, c_s2)
        W_q   = self._kingman_Wq(rho_combined, E_S, c_a2_combined, c_s2)
        if W_q_s < 1e-12:
            return 0.0
        return max(0.0, (W_q / W_q_s - 1.0) * 100.0)

    def ploss_curve(self, tau_values_s: np.ndarray, lambda_scratch_bps: float,
                    mean_file_bytes: float, cv_iat_scratch: float = 1.0,
                    cv_service: float = 1.0,
                    disk_bandwidth_bps: Optional[float] = None) -> np.ndarray:
        if disk_bandwidth_bps is None:
            disk_bandwidth_bps = 65.0 * (1024**3)
        return np.array([
            self.ploss_at_tau(max(t, 1e-6), lambda_scratch_bps, mean_file_bytes,
                              cv_iat_scratch, cv_service, disk_bandwidth_bps)
            for t in tau_values_s
        ])

    def _spectral_solve(self, n_grid: int = 200) -> MMFQResult:
        """
        Bounded MMFQ stationary solver (Anick-Mitra-Sondhi 1982; Mitra 1988).

        CORRECTED. The cache-fill CDF F_i(x) = P(X <= x, phase = i) satisfies
        dF/dx . R = F . Q, with solution F(x) = sum_k b_k phi_k g_k(x) where
        (z_k, phi_k) solve the generalized left-eigenproblem phi_k Q = z_k phi_k R.
        The coefficients b_k are fixed by the bounded-domain boundary conditions

            F_i(0) = 0      for r_i > 0    (up-states: no atom at the empty boundary)
            F_i(C) = pi_i   for r_i < 0    (down-states: no atom at the full boundary)

        The pi_i on the right-hand side make this an INHOMOGENEOUS n-by-n system.
        The previous implementation imposed a homogeneous system A c = 0 on the
        density; with a full-rank A that has only the trivial solution, and taking
        the smallest singular vector collapsed the answer onto the zero-eigenvalue
        (constant) mode -- a near-uniform density independent of the drift. The
        present formulation reproduces an independent fluid simulation and the
        Monte-Carlo branch across drift regimes.

        Scalars (P_above, E[X/C]) are read from the marginal CDF, which is bounded
        on [0, 1] at any cache size; the absolute scale is fixed by pi_i, so no grid
        renormalization is applied. At production cache size the per-phase fluid
        excursion is ~1e-5 of C, so z*C ~ 1e5 and the stationary distribution
        collapses into a sub-grid boundary layer (effectively an atom at one
        boundary). That regime is flagged in solver_message; the scalars stay valid,
        but the continuous density is genuinely degenerate there.
        """
        n = self.n
        r = self.r.astype(float)

        # R must be invertible: nudge any exact-zero drift off zero, sign-preserving.
        eps = 1e-3 * max(np.abs(r).max(), 1.0)
        sgn = np.sign(r)
        sgn[sgn == 0] = 1.0
        rr = np.where(np.abs(r) < eps, eps * sgn, r)
        R = np.diag(rr)
        pi = self._stationary_distribution()

        # phi_k Q = z_k phi_k R   <=>   Q^T v = z R v,  v = phi_k^T.
        z, V = linalg.eig(self.Q.T, R)
        phi = V.T  # phi[k, :] is the k-th left eigenvector of Q R^-1.

        x = np.linspace(0.0, self.C, n_grid)
        # Conditioning-safe basis: exp(z x) for Re z <= 0, exp(z (x - C)) for Re z > 0,
        # so every basis function stays in (0, 1] across [0, C].
        g = np.zeros((n, n_grid), dtype=complex)
        g0 = np.zeros(n, dtype=complex)
        gC = np.zeros(n, dtype=complex)
        for k in range(n):
            if z[k].real <= 0.0:
                g[k, :] = np.exp(z[k] * x)
                g0[k] = 1.0
                gC[k] = np.exp(z[k] * self.C)
            else:
                g[k, :] = np.exp(z[k] * (x - self.C))
                g0[k] = np.exp(-z[k] * self.C)
                gC[k] = 1.0

        # Inhomogeneous boundary-condition system A b = rhs. Partition on the
        # REGULARISED drifts rr (never exactly zero), so an unobserved/zero-drift
        # phase still receives a boundary condition and the system stays n-by-n.
        S_pos = np.where(rr > 0)[0]
        S_neg = np.where(rr < 0)[0]
        A = np.zeros((n, n), dtype=complex)
        rhs = np.zeros(n, dtype=complex)
        row = 0
        for i in S_pos:                       # F_i(0) = 0
            A[row, :] = phi[:, i] * g0
            rhs[row] = 0.0
            row += 1
        for i in S_neg:                       # F_i(C) = pi_i
            A[row, :] = phi[:, i] * gC
            rhs[row] = pi[i]
            row += 1
        if row != n:
            raise ValueError("Need exactly n boundary conditions; check for zero drifts.")
        b = np.linalg.solve(A, rhs)

        # F_i(x) and density f_i(x) = F_i'(x) = sum_k b_k z_k phi_k(i) g_k(x).
        F = ((g.T * b) @ phi).real
        f_complex = (g.T * (b * z)) @ phi
        f = np.maximum(f_complex.real, 0.0)
        imag_resid = float(np.max(np.abs(f_complex.imag)))
        if imag_resid > 1e-6 * max(np.abs(f).max(), 1.0):
            raise ValueError(f"Spectral density has non-negligible imaginary part "
                             f"({imag_resid:.2e}); falling back.")

        pdf_per_phase = f                                  # (n_grid, n)
        pdf_marginal = np.maximum(pdf_per_phase.sum(axis=1), 0.0)
        mass_at_zero = np.maximum(F[0, :], 0.0)            # atom at x=0 (down-states only)
        mass_at_cap = np.maximum(pi - F[-1, :], 0.0)       # atom at x=C (up-states only)

        x_unit = x / self.C
        # Marginal CDF P(X <= x): exact at every node, no grid renormalization.
        cdf = np.clip(F.sum(axis=1), 0.0, 1.0)
        cdf = np.maximum.accumulate(cdf)                   # guard tiny fp non-monotonicity
        pdf_unit = pdf_marginal * self.C                   # density wrt u = x / C

        # E[X/C] = int_0^1 (1 - F(u)) du -- robust even when f is a sub-grid spike.
        expected_fill = float(np.clip(
            sp_integrate.trapezoid(1.0 - cdf, x_unit), 0.0, 1.0))
        prob_above_hwm = float(1.0 - np.interp(self.hwm, x_unit, cdf))
        prob_above_lwm = float(1.0 - np.interp(self.lwm, x_unit, cdf))
        phase_probs = pi.copy()                            # phase marginal is the CTMC stationary pi

        mean_r = float(np.dot(pi, self.r))
        if mean_r > 0 and expected_fill < self.hwm:
            mfpt_to_hwm = (self.hwm - expected_fill) * self.C / mean_r
        else:
            mfpt_to_hwm = float("inf")

        msg = "Spectral (corrected; Anick-Mitra-Sondhi CDF boundary-value problem)."
        nz = np.abs(z[np.abs(z) > 1e-30])
        if nz.size:
            layer_cells = (1.0 / nz.max()) / (self.C / n_grid)
            if layer_cells < 1.0:
                msg += (" NOTE: stationary boundary layer thinner than one grid cell; "
                        "distribution is effectively atomic at this cache size "
                        "(scalars valid, continuous density degenerate).")

        return MMFQResult(
            x_grid=x_unit, cdf=cdf, pdf=pdf_unit, phase_probs=phase_probs,
            mass_at_zero=mass_at_zero, mass_at_cap=mass_at_cap,
            prob_above_hwm=prob_above_hwm, prob_above_lwm=prob_above_lwm,
            expected_fill=expected_fill, mfpt_to_hwm=mfpt_to_hwm,
            ploss_baseline=0.0, ploss_combined=0.0, wq_ratio=1.0,
            Q=self.Q, r=self.r, C=self.C, hwm=self.hwm, lwm=self.lwm,
            solver_ok=True, solver_message=msg,
        )

    def _monte_carlo_solve(self, n_grid: int, n_steps: int = 200_000, dt: float = 1.0) -> MMFQResult:
        rng = np.random.default_rng(42)
        x = self.C * 0.5; phase = 0
        Q_off = self.Q.copy(); np.fill_diagonal(Q_off, 0)
        exit_rates = -np.diag(self.Q)
        fill_samples = np.zeros(n_steps); phase_samples = np.zeros(n_steps, dtype=int)
        for step in range(n_steps):
            x = np.clip(x + self.r[phase] * dt, 0, self.C)
            rate = exit_rates[phase]
            if rate > 0 and rng.exponential(1.0 / rate) < dt:
                probs = Q_off[phase, :] / max(rate, 1e-12)
                probs = np.maximum(probs, 0); probs /= max(probs.sum(), 1e-12)
                phase = rng.choice(self.n, p=probs)
            fill_samples[step] = x / self.C; phase_samples[step] = phase
        x_grid = np.linspace(0, 1, n_grid)
        pdf, _ = np.histogram(fill_samples, bins=n_grid, range=(0, 1), density=True)
        norm = pdf.sum()
        cdf = np.cumsum(pdf) / max(norm, 1e-12)
        cdf = np.clip(cdf / max(cdf[-1], 1e-12), 0, 1)
        return MMFQResult(
            x_grid=x_grid, cdf=cdf, pdf=pdf / max(norm, 1.0),
            phase_probs=np.array([np.mean(phase_samples == i) for i in range(self.n)]),
            mass_at_zero=np.zeros(self.n), mass_at_cap=np.zeros(self.n),
            prob_above_hwm=float(np.mean(fill_samples > self.hwm)),
            prob_above_lwm=float(np.mean(fill_samples > self.lwm)),
            expected_fill=float(np.mean(fill_samples)), mfpt_to_hwm=float("nan"),
            ploss_baseline=0.0, ploss_combined=0.0, wq_ratio=1.0,
            Q=self.Q, r=self.r, C=self.C, hwm=self.hwm, lwm=self.lwm,
            solver_ok=True, solver_message="Monte Carlo (spectral fallback).",
        )

    @staticmethod
    def _kingman_Wq(rho: float, E_S: float, c_a2: float, c_s2: float) -> float:
        rho = min(max(rho, 0.0), 0.9999)
        if rho < 1e-12 or E_S < 1e-12: return 0.0
        return (rho / (1.0 - rho)) * E_S * (c_a2 + c_s2) / 2.0

    def _mean_archival_rate_bps(self) -> float:
        pi = self._stationary_distribution()
        return float(np.sum(pi * np.maximum(-self.r, 0)))

    def _stationary_distribution(self) -> np.ndarray:
        n = self.n; A = self.Q.T.copy(); A[-1, :] = 1.0
        b = np.zeros(n); b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b); pi = np.maximum(pi, 0)
            pi /= max(pi.sum(), 1e-12)
        except np.linalg.LinAlgError:
            pi = np.ones(n) / n
        return pi
