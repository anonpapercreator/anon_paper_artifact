"""Tests for Lifetime(kind="empirical") used by e11 (model and simulator share this class)."""
import numpy as np
import pytest
from scipy import integrate
from models.interval_model import Lifetime, tape_write_rate, unprotected_volume


@pytest.fixture
def curve(tmp_path):
    g = np.arange(0, 3601, 60.0)
    S = 0.4 + 0.6 * np.exp(-g / 600.0)
    S[10:15] = S[10]                       # a flat stretch
    S = np.minimum.accumulate(S)
    p = tmp_path / "c.csv"
    import pandas as pd
    pd.DataFrame({"age_s": g.astype(int), "S": S}).to_csv(p, index=False)
    return str(p), g, S


def test_S_G_H_exact(curve):
    path, g, S = curve
    L = Lifetime(kind="empirical", curve=path)
    assert L.s_inf == pytest.approx(S[-1])
    assert np.allclose(L.S(g), S)
    for x in [0, 30, 59.9, 60, 777, 3600, 5000]:
        G = integrate.quad(lambda t: float(L.S(t)), 0, x, limit=400, points=g[g < x][1:] if x > 60 else None)[0]
        assert L.G(x) == pytest.approx(G, rel=1e-7, abs=1e-6)
        H = integrate.quad(lambda t: L.G(t), 0, x, limit=400)[0]
        assert L.H(x) == pytest.approx(H, rel=1e-6, abs=1e-3)


def test_sampling_matches_S_and_G(curve):
    path, g, S = curve
    L = Lifetime(kind="empirical", curve=path)
    rng = np.random.default_rng(1)
    x = L.sample(rng, 400_000)
    for t in [0, 100, 600, 1500, 3600, 10_000]:
        assert np.mean(x > t) == pytest.approx(float(L.S(t)), abs=0.004)
    for t in [300, 1200, 4000]:
        assert np.mean(np.minimum(x, t)) == pytest.approx(L.G(t), rel=0.01)


def test_closed_forms_match_monte_carlo(curve):
    path, _, _ = curve
    L = Lifetime(kind="empirical", curve=path)
    rng = np.random.default_rng(2)
    n = 400_000
    Lx = L.sample(rng, n)
    for a, p in [(0, 600), (300, 1800), (1200, 60)]:
        D = a + p * rng.random(n)
        assert np.mean(Lx > D) == pytest.approx(tape_write_rate(L, 1.0, a, p), abs=0.004)
        assert np.mean(np.minimum(Lx, D)) == pytest.approx(unprotected_volume(L, 1.0, a, p), rel=0.01)


def test_rejects_bad_curve(tmp_path):
    import pandas as pd
    p = tmp_path / "b.csv"
    pd.DataFrame({"age_s": [0, 60, 120], "S": [1.0, 0.5, 0.7]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        Lifetime(kind="empirical", curve=str(p))
