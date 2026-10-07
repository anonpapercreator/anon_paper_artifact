#!/usr/bin/env python3
"""e11 / analyze.py -- the analysis fixed in PLAN.md, applied to sweep_{E,L1,L2,A}.json.

Per cell (setting x distribution x cost): mean ratio, 95% t interval, 90% t interval,
equivalence (two one-sided t tests, alpha 0.05 each, margin +/-10%), and whether the
95% interval excludes 1. Regression of the per-replication ratio on log(1 + a/60),
log(p/60) and distribution dummies, by cost. Writes summary.json and prints a report.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats as st

HERE = Path(__file__).resolve().parent
DISTS = ["E", "L1", "L2", "A", "E-P", "L1-P", "L2-P", "A-P"]
MARGIN = 0.10


def cells(rows):
    out = []
    for a in sorted({r["a"] for r in rows}):
        for p in sorted({r["p"] for r in rows}):
            sub = [r for r in rows if r["a"] == a and r["p"] == p]
            if not sub:
                continue
            for k in ("C", "V"):
                x = np.array([r[f"{k}_hat"] / r[f"{k}_model"] for r in sub])
                n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
                t95 = st.t.ppf(0.975, n - 1); t90 = st.t.ppf(0.95, n - 1)
                ci95 = [m - t95 * se, m + t95 * se]; ci90 = [m - t90 * se, m + t90 * se]
                out.append(dict(a=a, p=p, cost=k, n=n, mean=m, ci95=ci95, ci90=ci90,
                                equivalent=bool(ci90[0] >= 1 - MARGIN and ci90[1] <= 1 + MARGIN),
                                excludes_1=bool(ci95[0] > 1 or ci95[1] < 1)))
    return out


def regression(all_rows, cost):
    present = list(all_rows)
    X, y = [], []
    for d, rows in all_rows.items():
        for r in rows:
            X.append([1.0, np.log1p(r["a"] / 60), np.log(r["p"] / 60)] + [float(d == q) for q in present[1:]])
            y.append(r[f"{cost}_hat"] / r[f"{cost}_model"])
    X, y = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; n, k = X.shape
    s2 = res @ res / (n - k); cov = s2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
    names = ["const", "log1p(a/60)", "log(p/60)"] + [f"dist={q}" for q in present[1:]]
    return {nm: dict(coef=float(b), se=float(s), p=float(2 * st.t.sf(abs(b / s), n - k)))
            for nm, b, s in zip(names, beta, se)}


def main():
    all_rows, summary = {}, dict(margin=MARGIN, cells={}, totals={}, regression={})
    for d in DISTS:
        f = HERE / f"sweep_{d}.json"
        if not f.exists():
            print(f"missing {f.name}"); continue
        rows = json.loads(f.read_text())["rows"]; all_rows[d] = rows
        c = cells(rows); summary["cells"][d] = c
        for k in ("C", "V"):
            ck = [x for x in c if x["cost"] == k]
            summary["totals"][f"{d}/{k}"] = dict(
                cells=len(ck), equivalent=sum(x["equivalent"] for x in ck),
                excludes_1=sum(x["excludes_1"] for x in ck),
                min_mean=min(x["mean"] for x in ck), max_mean=max(x["mean"] for x in ck),
                max_abs_dev=max(abs(x["mean"] - 1) for x in ck))
    for arr in ("pareto", "poisson"):
        g = {d: r for d, r in all_rows.items() if ("-P" in d) == (arr == "poisson")}
        if g:
            summary["regression"][arr] = {k: regression(g, k) for k in ("C", "V")}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["totals"], indent=1))
    for d, c in summary["cells"].items():
        for x in c:
            if not x["equivalent"]:
                print(f"NOT EQUIVALENT {d} a={x['a']:.0f} p={x['p']:.0f} {x['cost']}: mean {x['mean']:.3f} "
                      f"90% [{x['ci90'][0]:.3f}, {x['ci90'][1]:.3f}]")
    for arr, rr in summary["regression"].items():
        for k, r in rr.items():
            print(arr, k, {n: (round(v['coef'], 4), round(v['se'], 4), round(v['p'], 4)) for n, v in r.items()})


if __name__ == "__main__":
    main()
