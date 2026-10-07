#!/usr/bin/env python3
"""e11 / analyze_cohort.py -- PLAN.md analysis (Addendum 2) applied to cohort_*.json.

Ratios per replication:
  R_C  = c_sel / c_model
  R_V  = v_tap / v_model_delta   (protection at tape write, model at a + delta)
  R_V0 = v_sel / v_model         (protection at selection, A4)
Per cell: mean, 95% and 90% t intervals, TOST equivalence with margin +/-10%, whether the
95% interval excludes 1. Regression of the ratio on log1p(a/60), log(p/60) and distribution
dummies, by ratio and arrival process. Writes summary_cohort.json.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats as st

HERE = Path(__file__).resolve().parent
TAGS = ["E", "L1", "L2", "A", "E-P", "L1-P", "L2-P", "A-P"]
RERUNS = ["L2-P-s1042", "A-s1042"]  # Addendum 3: fresh seed, reported separately
MARGIN = 0.10
RATIOS = {"C": ("c_sel", "c_model"), "V": ("v_tap", "v_model_delta"), "V0": ("v_sel", "v_model")}


def ratio(r, k):
    n, d = RATIOS[k]
    return r[n] / r[d]


def cells(rows):
    out = []
    for a in sorted({r["a"] for r in rows}):
        for p in sorted({r["p"] for r in rows}):
            sub = [r for r in rows if r["a"] == a and r["p"] == p]
            for k in RATIOS:
                x = np.array([ratio(r, k) for r in sub])
                n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
                t95 = st.t.ppf(0.975, n - 1); t90 = st.t.ppf(0.95, n - 1)
                ci95 = [m - t95 * se, m + t95 * se]; ci90 = [m - t90 * se, m + t90 * se]
                out.append(dict(a=a, p=p, cost=k, n=n, mean=m, ci95=ci95, ci90=ci90,
                                equivalent=bool(ci90[0] >= 1 - MARGIN and ci90[1] <= 1 + MARGIN),
                                excludes_1=bool(ci95[0] > 1 or ci95[1] < 1)))
    return out


def regression(groups, k):
    names_d = list(groups)
    X, y, g = [], [], []
    for d, rows in groups.items():
        for r in rows:
            X.append([1.0, np.log1p(r["a"] / 60), np.log(r["p"] / 60)] + [float(d == q) for q in names_d[1:]])
            y.append(ratio(r, k)); g.append(r["rep"])
    X, y, g = np.array(X), np.array(y), np.array(g)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; n, kk = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag((res @ res / (n - kk)) * XtXi))
    # cluster-robust (CR1) by replication id: a replication id shares random streams across settings
    G = np.unique(g); meat = np.zeros((kk, kk))
    for c in G:
        u = X[g == c].T @ res[g == c]; meat += np.outer(u, u)
    fac = len(G) / (len(G) - 1) * (n - 1) / (n - kk)
    se_cl = np.sqrt(np.diag(fac * XtXi @ meat @ XtXi))
    names = ["const", "log1p(a/60)", "log(p/60)"] + [f"dist={q}" for q in names_d[1:]]
    return {nm: dict(coef=float(b), se=float(s), p=float(2 * st.t.sf(abs(b / s), n - kk)),
                     se_cluster=float(sc), p_cluster=float(2 * st.t.sf(abs(b / sc), len(G) - 1)))
            for nm, b, s, sc in zip(names, beta, se, se_cl)}


def main():
    rows_by, summary = {}, dict(margin=MARGIN, cells={}, totals={}, regression={}, unresolved={})
    for t in TAGS + RERUNS:
        f = HERE / f"cohort_{t}.json"
        if not f.exists():
            print(f"missing {f.name}"); continue
        rows = json.loads(f.read_text())["rows"]; rows_by[t] = rows
        summary["unresolved"][t] = dict(files=sum(r["unresolved"] for r in rows),
                                        of=sum(r["n_files"] for r in rows))
        c = cells(rows); summary["cells"][t] = c
        for k in RATIOS:
            ck = [x for x in c if x["cost"] == k]
            summary["totals"][f"{t}/{k}"] = dict(
                cells=len(ck), equivalent=sum(x["equivalent"] for x in ck),
                excludes_1=sum(x["excludes_1"] for x in ck),
                min_mean=min(x["mean"] for x in ck), max_mean=max(x["mean"] for x in ck),
                max_abs_dev=max(abs(x["mean"] - 1) for x in ck),
                max_ci90_halfwidth=max((x["ci90"][1] - x["ci90"][0]) / 2 for x in ck))
    for arr, tags in (("pareto", [t for t in TAGS if "-P" not in t]), ("poisson", [t for t in TAGS if "-P" in t]),
                      ("reruns", [t for t in RERUNS])):
        g = {t: rows_by[t] for t in tags if t in rows_by}
        if len(g) >= 1:
            summary["regression"][arr] = {k: regression(g, k) for k in RATIOS}
    (HERE / "summary_cohort.json").write_text(json.dumps(summary, indent=1))
    for key, v in summary["totals"].items():
        print(f"{key:8s} equiv {v['equivalent']:2d}/{v['cells']}  excl1 {v['excludes_1']:2d}  "
              f"range {v['min_mean']:.3f}-{v['max_mean']:.3f}  max90hw {v['max_ci90_halfwidth']:.3f}")
    print("unresolved", summary["unresolved"])
    for t, c in summary["cells"].items():
        for x in c:
            if not x["equivalent"] and x["cost"] != "V0":
                print(f"NOT EQUIVALENT {t} a={x['a']:.0f} p={x['p']:.0f} {x['cost']}: mean {x['mean']:.3f} "
                      f"90% [{x['ci90'][0]:.3f}, {x['ci90'][1]:.3f}]")
    for arr, rr in summary["regression"].items():
        for k, r in rr.items():
            print(arr, k, {n: (round(v['coef'], 4), round(v['se'], 4), round(v['p'], 4), "cl", round(v['se_cluster'], 4), round(v['p_cluster'], 4))
                           for n, v in r.items() if not n.startswith("dist")})


if __name__ == "__main__":
    main()
