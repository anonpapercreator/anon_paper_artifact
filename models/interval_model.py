"""
models/interval_model.py -- two-control interval model (age threshold a, sweep period p).

Replaces the single-parameter IOM (models/iom.py) for the IPDPS 2027 revision.

Policy semantics (DMF 7 and DESCASSI PolicyEngine): a migration sweep runs every
p seconds and selects every live REG file whose age is at least a. A file
created at a phase uniform relative to the sweep schedule is copied at age
D = a + pU, U ~ Uniform(0, 1), if it is still alive (L > D).

Cost terms, each in its own unit
--------------------------------
  V(a,p)  unprotected volume [bytes]      = rho * E[min(L, D)]
  C(a,p)  tape write rate    [bytes/s]    = rho * Pr(L > D)
  P(a,p)  performance loss   [%]          = relative increase in mean recall
          wait at the drive pool caused by migration write zones
          (M/G/c, Lee-Longton two-moment approximation).

Library-server batching (ls_min_batch_bytes b, ls_max_wait_s w) sets the zone
size Z(a,p); each zone holds one write drive for t_x + Z/R.

Closed forms, with S(t) = Pr(L > t), G(x) = int_0^x S, H(x) = int_0^x G:
  C = rho (G(a+p) - G(a)) / p,   V = rho (H(a+p) - H(a)) / p.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Optional

import numpy as np
from scipy import integrate, optimize


# ---------------------------------------------------------------------------
# File lifetimes
# ---------------------------------------------------------------------------

@dataclass
class Lifetime:
    """Mixture lifetime: fraction s_inf is never deleted (the retained class);
    the remaining 1 - s_inf is exponential or lognormal with the given mean.

    kind="empirical" instead reads a survival curve S(t) from the CSV file `curve`
    (columns age_s, S; uniform grid starting at 0 with S(0) = 1). S is taken as
    linear between grid points and constant at its last value beyond the grid, so
    s_inf is set to that last value. G, H and sampling are exact for this S."""
    s_inf: float = 1.0
    kind: str = "exp"
    mean_s: float = 3600.0
    sigma: float = 1.0          # lognormal shape (ignored for exp)
    curve: str = ""             # CSV path for kind="empirical"

    def __post_init__(self):
        if self.kind not in ("exp", "lognormal", "empirical"):
            raise ValueError("kind must be 'exp', 'lognormal' or 'empirical'")
        if self.kind == "empirical":
            self._load_curve()
        if not 0.0 <= self.s_inf <= 1.0:
            raise ValueError("s_inf must lie in [0, 1]")
        self._mu = np.log(self.mean_s) - self.sigma ** 2 / 2.0

    # empirical curve: S linear on [g_k, g_k + h], constant beyond the grid
    def _load_curve(self):
        import pandas as pd
        d = pd.read_csv(self.curve)
        g = d["age_s"].to_numpy(float); S = d["S"].to_numpy(float)
        h = np.diff(g)
        if g[0] != 0.0 or not np.allclose(h, h[0]) or abs(S[0] - 1.0) > 1e-9:
            raise ValueError("curve must start at age 0 with S = 1 on a uniform grid")
        if np.any(np.diff(S) > 1e-12):
            raise ValueError("curve must be nonincreasing")
        self._g, self._S, self._h = g, S, float(h[0])
        self._m = np.diff(S) / self._h                      # slope on each segment
        self._Gk = np.r_[0.0, np.cumsum(S[:-1] * self._h + self._m * self._h ** 2 / 2)]
        self._Hk = np.r_[0.0, np.cumsum(self._Gk[:-1] * self._h + S[:-1] * self._h ** 2 / 2
                                        + self._m * self._h ** 3 / 6)]
        self.s_inf = float(S[-1])

    def _seg(self, x):
        x = np.asarray(x, dtype=float)
        k = np.clip((x // self._h).astype(int), 0, len(self._g) - 2)
        u = x - self._g[k]
        return k, u

    def _emp_S(self, x):
        x = np.asarray(x, dtype=float)
        k, u = self._seg(np.minimum(x, self._g[-1]))
        return self._S[k] + self._m[k] * u

    # finite-part survival and density
    def _sf(self, t):
        t = np.asarray(t, dtype=float)
        if self.kind == "exp":
            return np.exp(-t / self.mean_s)
        from scipy.stats import lognorm
        return lognorm.sf(t, s=self.sigma, scale=np.exp(self._mu))

    def _pdf(self, t):
        t = np.asarray(t, dtype=float)
        if self.kind == "exp":
            return np.exp(-t / self.mean_s) / self.mean_s
        from scipy.stats import lognorm
        return lognorm.pdf(t, s=self.sigma, scale=np.exp(self._mu))

    def S(self, t):
        if self.kind == "empirical":
            return self._emp_S(t)
        return self.s_inf + (1.0 - self.s_inf) * self._sf(t)

    def f(self, t):
        if self.kind == "empirical":
            t = np.asarray(t, dtype=float)
            k, _ = self._seg(np.minimum(t, self._g[-1]))
            return np.where(t < self._g[-1], -self._m[k], 0.0)
        return (1.0 - self.s_inf) * self._pdf(t)

    def hazard(self, t):
        return self.f(t) / self.S(t)

    def G(self, x: float) -> float:
        x = float(x)
        if self.kind == "empirical":
            if x >= self._g[-1]:
                return float(self._Gk[-1] + self._S[-1] * (x - self._g[-1]))
            k, u = self._seg(x); k = int(k); u = float(u)
            return float(self._Gk[k] + self._S[k] * u + self._m[k] * u * u / 2)
        if self.kind == "exp":
            m = self.mean_s
            return self.s_inf * x + (1 - self.s_inf) * m * (-np.expm1(-x / m))
        # G_fin(x) = E[min(L, x)] = E[L; L <= x] + x P(L > x)   (exact, lognormal partial moment)
        return self.s_inf * x + (1 - self.s_inf) * (self._pm(1, x) + x * float(self._sf(x)))

    def H(self, x: float) -> float:
        x = float(x)
        if self.kind == "empirical":
            if x >= self._g[-1]:
                y = x - self._g[-1]
                return float(self._Hk[-1] + self._Gk[-1] * y + self._S[-1] * y * y / 2)
            k, u = self._seg(x); k = int(k); u = float(u)
            return float(self._Hk[k] + self._Gk[k] * u + self._S[k] * u * u / 2
                         + self._m[k] * u ** 3 / 6)
        if self.kind == "exp":
            m = self.mean_s
            return (self.s_inf * x * x / 2.0
                    + (1 - self.s_inf) * m * (x + m * np.expm1(-x / m)))
        # H_fin(x) = int_0^x E[min(L,u)] du = E[x L - L^2/2; L <= x] + (x^2/2) P(L > x)
        return (self.s_inf * x * x / 2.0
                + (1 - self.s_inf) * (x * self._pm(1, x) - self._pm(2, x) / 2.0
                                      + x * x / 2.0 * float(self._sf(x))))

    def _pm(self, k: int, x: float) -> float:
        """Lognormal partial moment E[L^k; L <= x]."""
        from scipy.stats import norm
        if x <= 0:
            return 0.0
        mu, s = self._mu, self.sigma
        return float(np.exp(k * mu + k * k * s * s / 2.0)
                     * norm.cdf((np.log(x) - mu - k * s * s) / s))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.kind == "empirical":
            # inverse transform: L = S^{-1}(U); U < s_inf means never deleted
            U = rng.random(n)
            L = np.full(n, np.inf)
            fin = U >= self.s_inf
            negS = -self._S
            k = np.searchsorted(negS, -U[fin], side="left") - 1   # S[k] >= U > S[k+1]
            k = np.clip(k, 0, len(self._g) - 2)
            m = self._m[k]
            L[fin] = self._g[k] + np.where(m < 0, (self._S[k] - U[fin]) / np.where(m < 0, -m, 1.0), 0.0)
            return L
        L = np.full(n, np.inf)
        finite = rng.random(n) >= self.s_inf
        k = int(finite.sum())
        if self.kind == "exp":
            L[finite] = rng.exponential(self.mean_s, k)
        else:
            L[finite] = rng.lognormal(self._mu, self.sigma, k)
        return L


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

@dataclass
class TapePlatform:
    rho_data_bps: float            # creation rate of the archive-eligible class
    n_drives: int                  # drives serving recalls
    n_write_drives: int            # drives that also take write zones
    native_rate_bps: float         # per-drive streaming rate R
    zone_overhead_s: float         # t_x: mount (if needed) + locate to EOD per zone
    ls_min_batch_bytes: float      # b
    ls_max_wait_s: float           # w
    recall_rate_per_s: float       # lambda_r
    recall_service_mean_s: float   # E[S_r]
    recall_service_scv: float      # c^2 of S_r


# ---------------------------------------------------------------------------
# V and C
# ---------------------------------------------------------------------------

def tape_write_rate(life: Lifetime, rho: float, a: float, p: float) -> float:
    if p <= 0:
        return rho * float(life.S(a))
    return rho * (life.G(a + p) - life.G(a)) / p


def unprotected_volume(life: Lifetime, rho: float, a: float, p: float) -> float:
    if p <= 0:
        return rho * life.G(a)
    return rho * (life.H(a + p) - life.H(a)) / p


# ---------------------------------------------------------------------------
# Zones and drive-pool queue
# ---------------------------------------------------------------------------

def zone_bytes(C: float, p: float, b: float, w: float) -> float:
    """Mean zone size under library-server accrual (fluid in bytes).

    A sweep delivers A = C p bytes. If A >= b, zones are cut at b. Otherwise
    sweeps accrue until b is reached or w has elapsed since the first accrual.
    As p -> 0 this tends to min(b, C w).
    """
    if C <= 0:
        return 0.0
    if p <= 0:
        return min(b, C * w)
    A = C * p
    if A >= b:
        return b
    # Files reach the accrual buffer one at a time (paced by the movers), so a
    # flush triggered by the byte threshold cuts the zone at b, not at a sweep
    # boundary. The timer instead flushes every sweep that arrived within w.
    k_w = max(np.ceil(w / p), 1.0)   # sweeps in [first, first + w); timer wins ties
    return min(b, A * k_w)


def erlang_c(c: int, offered: float) -> float:
    """Probability of waiting in M/M/c with offered load `offered` (< c)."""
    if offered >= c:
        return 1.0
    s = sum(offered ** k / factorial(k) for k in range(c))
    last = offered ** c / factorial(c) * c / (c - offered)
    return last / (s + last)


def mgc_wait(lam: float, ES: float, ES2: float, c: int) -> float:
    """Lee-Longton approximation: W_q(M/G/c) = W_q(M/M/c) (1 + c_s^2) / 2."""
    if lam <= 0:
        return 0.0
    offered = lam * ES
    if offered >= c:
        return np.inf
    scv = ES2 / ES ** 2 - 1.0
    wmm = erlang_c(c, offered) * ES / (c - offered)
    return wmm * (1.0 + scv) / 2.0


def drive_pool_wait(pl: TapePlatform, C: float, p: float) -> float:
    """Mean wait at the drive pool with recalls and write zones mixed FCFS."""
    ESr = pl.recall_service_mean_s
    ESr2 = (1.0 + pl.recall_service_scv) * ESr ** 2
    lam_r = pl.recall_rate_per_s
    Z = zone_bytes(C, p, pl.ls_min_batch_bytes, pl.ls_max_wait_s)
    if C <= 0 or Z <= 0:
        return mgc_wait(lam_r, ESr, ESr2, pl.n_drives)
    lam_z = C / Z
    Sz = pl.zone_overhead_s + Z / pl.native_rate_bps
    # zones may only use write drives: infeasible if they alone saturate them
    if lam_z * Sz >= pl.n_write_drives:
        return np.inf
    lam = lam_r + lam_z
    ES = (lam_r * ESr + lam_z * Sz) / lam
    ES2 = (lam_r * ESr2 + lam_z * Sz ** 2) / lam
    return mgc_wait(lam, ES, ES2, pl.n_drives)


def performance_loss(pl: TapePlatform, C: float, p: float) -> float:
    """Percent increase in mean recall wait caused by migration zones."""
    w0 = drive_pool_wait(pl, 0.0, p)
    w1 = drive_pool_wait(pl, C, p)
    if not np.isfinite(w1):
        return np.inf
    if w0 <= 0:
        return 0.0 if w1 <= 0 else np.inf
    return max(0.0, (w1 / w0 - 1.0) * 100.0)


# ---------------------------------------------------------------------------
# Objective and optimum
# ---------------------------------------------------------------------------

@dataclass
class Prices:
    """Weights applied to each term after dividing by a site reference.

    J = wp * P / P_ref + wv * V / (rho * T_R) + wc * C / rho
    P_ref: tolerated recall-wait increase [%]; T_R: tolerated exposure window
    [s] (V_ref = rho T_R, the volume written in one RPO window); C_ref = rho.
    """
    wp: float = 1.0
    wv: float = 1.0
    wc: float = 1.0
    P_ref: float = 10.0
    T_R: float = 3600.0


def objective(pl: TapePlatform, life: Lifetime, pr: Prices, a: float, p: float) -> dict:
    rho = pl.rho_data_bps
    C = tape_write_rate(life, rho, a, p)
    V = unprotected_volume(life, rho, a, p)
    P = performance_loss(pl, C, p)
    J = pr.wp * P / pr.P_ref + pr.wv * V / (rho * pr.T_R) + pr.wc * C / rho
    return {"a": a, "p": p, "C": C, "V": V, "P": P, "J": J}


def optimal_age_threshold(pl: TapePlatform, life: Lifetime, pr: Prices,
                          p: float = 0.0, a_max: float = 30 * 86400.0,
                          n_grid: int = 600) -> dict:
    """Minimise J over a >= 0 at fixed period p (grid, then bounded refine)."""
    grid = np.concatenate([[0.0], np.logspace(-1, np.log10(a_max), n_grid)])
    J = np.array([objective(pl, life, pr, a, p)["J"] for a in grid])
    i = int(np.nanargmin(J))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    if hi > lo:
        res = optimize.minimize_scalar(lambda a: objective(pl, life, pr, a, p)["J"],
                                       bounds=(lo, hi), method="bounded",
                                       options={"xatol": 1e-3 * max(hi - lo, 1e-9)})
        if res.fun <= J[i]:
            return objective(pl, life, pr, float(res.x), p)
    return objective(pl, life, pr, float(grid[i]), p)


def hazard_rule_rhs(pl: TapePlatform, life: Lifetime, pr: Prices, a: float,
                    p: float = 0.0, dC_frac: float = 1e-4) -> float:
    """Right-hand side of the generalised hazard rule at a:

        h(a*) = (wv / V_ref) / (wc / C_ref + (wp / P_ref) dP/dC)

    dP/dC by central difference in C. Units 1/s.
    """
    rho = pl.rho_data_bps
    C = tape_write_rate(life, rho, a, p)
    dC = max(C * dC_frac, 1.0)
    dP = (performance_loss(pl, C + dC, p) - performance_loss(pl, max(C - dC, 0.0), p)) \
        / (C + dC - max(C - dC, 0.0))
    num = pr.wv / (rho * pr.T_R)
    den = pr.wc / rho + pr.wp / pr.P_ref * dP
    return num / den
