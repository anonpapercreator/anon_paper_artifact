#!/usr/bin/env python3
"""
tests/validate_mmfq.py — correctness check for models/mmfq.py (no monkeypatch).

Exercises the installed MMFQSolver.solve() against an independent bounded-fluid
simulation written from scratch here, plus the solver's own Monte-Carlo branch,
across drift regimes at lab cache size, and checks the production-scale boundary
limit. Self-contained: needs only numpy + scipy and an importable models/mmfq.py.

Run from the repo root:
    DESCASSI_ROOT=$PWD PYTHONPATH=$PWD python3 tests/validate_mmfq.py
Exit 0 iff every lab case matches the simulation within 0.05 and each production
case is correctly boundary-atomic with total mass ~1.
"""
from __future__ import annotations
import numpy as np
from models.mmfq import MMFQSolver


def fluid_sim(Q, r, C, n_sojourns=250_000, seed=3, bins=200):
    """Independent stationary sim: exact CTMC sojourns, fine fluid integration,
    reflecting boundaries; returns time-weighted E[X/C]."""
    rng = np.random.default_rng(seed); n = len(r)
    ex = -np.diag(Q); P = Q.copy(); np.fill_diagonal(P, 0.0)
    P = np.maximum(P, 0.0); P = P / np.where(ex[:, None] > 0, ex[:, None], 1.0)
    x = 0.5 * C; ph = 0; hist = np.zeros(bins); dx = 0.004 * C
    for _ in range(n_sojourns):
        e = ex[ph]
        tau = rng.exponential(1.0 / max(e, 1e-12)) if e > 0 else 50.0
        if abs(r[ph]) < 1e-9:
            hist[min(int(x / C * bins), bins - 1)] += tau
        else:
            h = dx / abs(r[ph]); s = max(1, int(tau / h)); h = tau / s
            xs = np.clip(x + r[ph] * h * np.arange(1, s + 1), 0, C)
            np.add.at(hist, np.minimum((xs / C * bins).astype(int), bins - 1), h)
            x = xs[-1]
        ph = rng.choice(n, p=P[ph]) if e > 0 else ph
    centers = (np.arange(bins) + 0.5) / bins
    return float(np.sum(centers * hist) / hist.sum())


def main() -> int:
    Q = np.array([[-0.10, 0.08, 0.01, 0.01],
                  [0.05, -0.15, 0.08, 0.02],
                  [0.02, 0.10, -0.20, 0.08],
                  [0.05, 0.02, 0.03, -0.10]])
    C_lab = 100 * 1024**3
    C_prod = 977465837092864.0   # 889 TiB
    ok = True

    print("LAB SCALE 100 GiB  (solve() must match the independent fluid sim)")
    print(f"  {'case':12}{'sim':>8}{'MC':>8}{'solve':>8}{'P>hwm':>8}")
    cases = [("drift up",   np.array([-1e8, 5e8, 2e9, -5e8])),
             ("drift down", np.array([-3e9, 2e8, 4e8, -3e9])),
             ("balanced",   np.array([-1e9, 5e8, 1.2e9, -9e8])),
             ("zero-drift",  np.array([-1e8, 5e8, 0.0, -3e8]))]
    for name, r in cases:
        sim = fluid_sim(Q, r, C_lab)
        s = MMFQSolver(Q, r, C_lab, hwm=0.8, lwm=0.6)
        mc = s._monte_carlo_solve(200).expected_fill
        res = s.solve(200)
        good = abs(res.expected_fill - sim) < 0.05; ok &= good
        print(f"  {name:12}{sim:8.3f}{mc:8.3f}{res.expected_fill:8.3f}"
              f"{res.prob_above_hwm:8.3f}   {'PASS' if good else 'FAIL'}")

    print("\nPRODUCTION SCALE 889 TiB  (boundary-atomic, total mass ~1)")
    for name, r in [("net up",   np.array([-5e8, 2e9, 5e9, -5e8])),
                    ("net down", np.array([-2e9, 1e9, 5e9, -3e9]))]:
        A = Q.T.copy(); A[-1, :] = 1.0; b = np.zeros(4); b[-1] = 1.0
        md = float(np.linalg.solve(A, b) @ r)
        res = MMFQSolver(Q, r, C_prod, hwm=0.8, lwm=0.6).solve(200)
        tot = res.cdf[-1] + res.mass_at_cap.sum()
        want = 1.0 if md > 0 else 0.0
        good = abs(res.expected_fill - want) < 0.05 and abs(tot - 1.0) < 0.02; ok &= good
        print(f"  {name:9} pi.r={md:+.2e}  E[X/C]={res.expected_fill:.3f}"
              f"  totMass={tot:.3f}   {'PASS' if good else 'FAIL'}")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
