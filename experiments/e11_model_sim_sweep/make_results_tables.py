#!/usr/bin/env python3
"""Print the markdown tables for RESULTS.md from summary.json (flow) and summary_cohort.json (cohort)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
for name, keys in (("summary_cohort.json", ("C", "V", "V0")), ("summary.json", ("C", "V"))):
    s = json.loads((H / name).read_text())
    print(f"\n### {name}\n")
    print("| set | cost | equivalent (of 30) | 95% CI excludes 1 | min mean | max mean |")
    print("|---|---|---|---|---|---|")
    for key, v in s["totals"].items():
        t, k = key.split("/")
        if k in keys:
            print(f"| {t} | {k} | {v['equivalent']} | {v['excludes_1']} | {v['min_mean']:.3f} | {v['max_mean']:.3f} |")
    print("\nRegression of the ratio (coef, se, p):\n")
    for arr, rr in s["regression"].items():
        for k, r in rr.items():
            terms = ", ".join(f"{n} {v['coef']:+.4f} ({v['se']:.4f}, p={v['p']:.3f})"
                              for n, v in r.items() if n in ("log1p(a/60)", "log(p/60)"))
            print(f"- {arr} {k}: {terms}")
