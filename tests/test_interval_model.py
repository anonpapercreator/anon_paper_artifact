"""Tests for models/interval_model.py.

Each closed form is checked against an independent computation: Monte Carlo
for C and V, an event-level replay of the library-server rule for zone size,
exact M/M/c and Pollaczek-Khinchine results for the queue, and brute-force
minimisation for the hazard rule and the period result.
"""
import numpy as np
import pytest

from models.interval_model import (
    Lifetime, TapePlatform, Prices, tape_write_rate, unprotected_volume,
    zone_bytes, mgc_wait, erlang_c, performance_loss, objective,
    optimal_age_threshold, hazard_rule_rhs)


def _mc_policy(life, lam, a, p, horizon, warm, rng):
    n = rng.poisson(lam * horizon)
    c = np.sort(rng.uniform(0.0, horizon, n))
    L = life.sample(rng, n)
    t_arch = np.ceil((c + a) / p) * p
    D = t_arch - c
    arch = L > D
    e = c + np.minimum(L, D)
    ov = np.clip(np.minimum(e, horizon) - np.maximum(c, warm), 0.0, None)
    win = (t_arch >= warm) & (t_arch < horizon)
    return ov.sum() / (horizon - warm), (arch & win).sum() / (horizon - warm)


@pytest.mark.parametrize("life,a,p", [
    (Lifetime(0.3, "exp", 600.0), 0.0, 300.0),
    (Lifetime(0.3, "exp", 600.0), 900.0, 300.0),
    (Lifetime(0.5, "lognormal", 300.0, 2.0), 120.0, 60.0),
    (Lifetime(1.0), 0.0, 1000.0),
])
def test_C_V_against_monte_carlo(life, a, p):
    rng = np.random.default_rng(2026)
    lam, reps = 2.0, 30
    horizon, warm = np.ceil(1.5e5 / p) * p, np.ceil(3e4 / p) * p
    est = np.array([_mc_policy(life, lam, a, p, horizon, warm, rng) for _ in range(reps)])
    m, se = est.mean(0), est.std(0, ddof=1) / np.sqrt(reps)
    V, C = unprotected_volume(life, lam, a, p), tape_write_rate(life, lam, a, p)
    assert abs(m[0] - V) < 4 * se[0] + 1e-9
    assert abs(m[1] - C) < 4 * se[1] + 1e-9


def test_limits_without_deletion():
    keep = Lifetime(1.0)
    assert unprotected_volume(keep, 3.0, 0.0, 100.0) == pytest.approx(150.0)
    assert unprotected_volume(keep, 3.0, 50.0, 0.0) == pytest.approx(150.0)
    assert tape_write_rate(keep, 3.0, 10.0, 100.0) == pytest.approx(3.0)


def _replay_ls(C, p, b, w, horizon=2e6):
    """Event-level replay of the library-server accrual rule with sweep batches."""
    zones, acc, first = [], 0.0, None
    t = 0.0
    while t < horizon:
        t += p
        if first is not None and t - first >= w:      # timer fired before this sweep
            zones.append(acc); acc, first = 0.0, None
        A = C * p
        while A > 0:
            take = min(A, b - acc)
            if first is None:
                first = t
            acc += take; A -= take
            if acc >= b - 1e-9:
                zones.append(acc); acc, first = 0.0, None
    return np.mean(zones[5:])


@pytest.mark.parametrize("C,p", [(5e8, 1.0), (5e8, 10.0), (5e8, 50.0), (1e7, 60.0), (1e6, 20.0)])
def test_zone_bytes_against_replay(C, p):
    b, w = 1e10, 300.0
    assert zone_bytes(C, p, b, w) == pytest.approx(_replay_ls(C, p, b, w), rel=0.02)


def test_mgc_exact_cases():
    # M/M/c: Lee-Longton factor is 1 when scv = 1
    lam, ES, c = 3.0, 2.0, 8
    rho_c = lam * ES
    exact = erlang_c(c, rho_c) * ES / (c - rho_c)
    assert mgc_wait(lam, ES, 2 * ES ** 2, c) == pytest.approx(exact)
    # M/D/1: Pollaczek-Khinchine, W = rho ES / (2 (1 - rho))
    lam, ES = 0.4, 2.0
    assert mgc_wait(lam, ES, ES ** 2, 1) == pytest.approx(0.8 * 2.0 / (2 * 0.2))


def _platform(lam_r=0.05):
    return TapePlatform(rho_data_bps=0.5 * 1024 ** 3, n_drives=9, n_write_drives=7,
                        native_rate_bps=400 * 1024 ** 2, zone_overhead_s=10.0,
                        ls_min_batch_bytes=1e10, ls_max_wait_s=300.0,
                        recall_rate_per_s=lam_r, recall_service_mean_s=60.0,
                        recall_service_scv=0.5)


def test_P_convex_increasing_in_C():
    pl = _platform()
    # P jumps at C = 0+ (the first zone brings a fixed overhead); test C > 0.
    Cs = np.linspace(1e6, 0.9 * pl.rho_data_bps, 200)
    P = np.array([performance_loss(pl, C, 0.0) for C in Cs])
    assert np.all(np.diff(P) >= -1e-9)
    assert np.all(np.diff(P, 2) >= -1e-6)


@pytest.mark.parametrize("life", [Lifetime(0.7, "exp", 3600.0), Lifetime(0.4, "exp", 600.0),
                                  Lifetime(0.7, "lognormal", 3600.0, 1.5)])
@pytest.mark.parametrize("pr", [Prices(T_R=3600.0), Prices(T_R=86400.0), Prices(wp=0.0, T_R=86400.0)])
def test_hazard_rule_holds_at_optimum(life, pr):
    pl = _platform()
    opt = optimal_age_threshold(pl, life, pr)
    a = opt["a"]
    if a > 1.0:
        assert float(life.hazard(a)) == pytest.approx(hazard_rule_rhs(pl, life, pr, a), rel=0.02)
    else:   # corner solution: copy at once when h(0) <= rhs(0)
        assert float(life.hazard(0.0)) <= hazard_rule_rhs(pl, life, pr, 0.0) * 1.02


def test_period_zero_dominates_for_DFR():
    """For DFR lifetimes and p in the library-server regime, no (a, p>0) beats the
    best deterministic delay (proposition: sweep as often as possible)."""
    pl, pr = _platform(), Prices(T_R=86400.0)
    for life in [Lifetime(0.7, "exp", 3600.0), Lifetime(0.5, "exp", 600.0)]:   # DFR lifetimes
        best0 = optimal_age_threshold(pl, life, pr, p=0.0)["J"]
        for a in [0.0, 1800.0, 7200.0, 20000.0]:
            for p in [1.0, 5.0, 15.0]:   # p << b / C = 18.6 s keeps Z = min(b, C w)
                assert objective(pl, life, pr, a, p)["J"] >= best0 - 1e-9


# ---- Results stated in Section III of the IPDPS 2027 paper ----------------------------

def _J_two_term(life, a, p, cw, ce, rho=1.0):
    return cw * tape_write_rate(life, rho, a, p) + ce * unprotected_volume(life, rho, a, p)


@pytest.mark.parametrize("life", [Lifetime(0.7, "exp", 3600.0), Lifetime(0.3, "exp", 600.0),
                                  Lifetime(0.6, "lognormal", 3600.0, 1.5)])
@pytest.mark.parametrize("p", [0.0, 600.0, 3600.0])
@pytest.mark.parametrize("theta", [1 / 86400.0, 1 / 7200.0])
def test_window_hazard_condition(life, p, theta):
    """Interior minimisers of cw*C + ce*V over a satisfy
    (S(a) - S(a+p)) / (G(a+p) - G(a)) = ce/cw  (h(a) = ce/cw when p = 0)."""
    from scipy.optimize import minimize_scalar
    cw, ce = 1.0, theta
    a_grid = np.linspace(0.0, 20 * 3600.0, 2001)
    J = np.array([_J_two_term(life, a, p, cw, ce) for a in a_grid])
    i = int(np.argmin(J))
    if i == 0 or i == len(a_grid) - 1:
        pytest.skip("corner solution for this case")
    res = minimize_scalar(lambda a: _J_two_term(life, a, p, cw, ce),
                          bounds=(a_grid[i - 1], a_grid[i + 1]), method="bounded",
                          options={"xatol": 1e-3})
    a = res.x
    if p == 0.0:
        lhs = float(life.hazard(a))
    else:
        lhs = (float(life.S(a)) - float(life.S(a + p))) / (life.G(a + p) - life.G(a))
    assert lhs == pytest.approx(theta, rel=1e-3)


@pytest.mark.parametrize("life", [Lifetime(0.7, "exp", 3600.0), Lifetime(0.5, "lognormal", 300.0, 2.0),
                                  Lifetime(1.0)])
@pytest.mark.parametrize("a,p", [(0.0, 3600.0), (1800.0, 21600.0), (7200.0, 60.0)])
def test_exposure_bound_and_monotonicity(life, a, p):
    """V/rho <= a + p/2, with equality when nothing is deleted; for fixed a, V increases
    and C decreases with p."""
    v = unprotected_volume(life, 1.0, a, p)
    assert v <= a + p / 2 + 1e-6
    if life.s_inf == 1.0:
        assert v == pytest.approx(a + p / 2)
    v2 = unprotected_volume(life, 1.0, a, 2 * p)
    c1, c2 = tape_write_rate(life, 1.0, a, p), tape_write_rate(life, 1.0, a, 2 * p)
    assert v2 >= v - 1e-9 and c2 <= c1 + 1e-12


@pytest.mark.parametrize("life", [Lifetime(0.6, "lognormal", 3600.0, 1.5), Lifetime(0.2, "lognormal", 300.0, 2.0)])
@pytest.mark.parametrize("x", [10.0, 900.0, 7200.0, 86400.0])
def test_lognormal_closed_forms_against_quadrature(life, x):
    from scipy import integrate
    G_q = integrate.quad(lambda t: float(life.S(t)), 0.0, x, limit=400, points=[life.mean_s])[0]
    H_q = integrate.quad(life.G, 0.0, x, limit=400)[0]
    assert life.G(x) == pytest.approx(G_q, rel=1e-7)
    assert life.H(x) == pytest.approx(H_q, rel=1e-7)
