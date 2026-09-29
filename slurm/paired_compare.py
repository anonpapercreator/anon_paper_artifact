#!/usr/bin/env python3
"""
slurm/paired_compare.py -- rigorous static-vs-adaptive comparison.

Both policies are run with the SAME master seed and the SAME replication IDs
(common random numbers), so replication k of each arm sees the same stochastic
workload realisation. That makes the comparison PAIRED, which removes
between-replication variance and is the defensible way to claim "adaptive beats
static". For each matched replication we form the per-rep statistic (default:
recall P95), then report:

  * the paired difference and the ratio (static / adaptive), per replication;
  * a Wilcoxon signed-rank test on the differences (non-parametric, no
    normality assumption -- appropriate for heavy-tailed latency);
  * a 1000-resample bootstrap 95% CI on the MEDIAN ratio (effect size);
  * the matched-pair rank-biserial effect size.

Significance from a large N is necessary but not sufficient: report the CI and
effect size, not just the p-value. With very large N even trivial differences
are "significant", so the CI on the ratio is the number that should carry the
claim.

Usage:
    python slurm/paired_compare.py --static runs/static --adaptive runs/adaptive \
        --metric recall_p95_s
"""
from __future__ import annotations
import sys, os, json, glob, argparse
import numpy as np


def load_metric(run_dir: str, metric: str) -> dict:
    """replication_id -> metric value, from the JSON sidecars."""
    out = {}
    for s in sorted(glob.glob(os.path.join(run_dir, "rep_*.json"))):
        r = json.load(open(s))
        if r.get(metric) is not None:
            out[int(r["replication"])] = float(r[metric])
    return out


def boot_ci_median_ratio(ratio: np.ndarray, B: int = 1000, seed: int = 12345):
    rng = np.random.default_rng(seed)
    n = len(ratio)
    meds = np.empty(B)
    for b in range(B):
        meds[b] = np.median(ratio[rng.integers(0, n, n)])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", required=True)
    ap.add_argument("--adaptive", required=True)
    ap.add_argument("--metric", default="recall_p95_s")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    S = load_metric(a.static, a.metric)
    A = load_metric(a.adaptive, a.metric)
    common = sorted(set(S) & set(A))
    if len(common) < 2:
        print("[paired_compare] need >=2 matched replications; "
              f"static={len(S)} adaptive={len(A)} matched={len(common)}", file=sys.stderr)
        return 1

    s = np.array([S[i] for i in common], float)
    ad = np.array([A[i] for i in common], float)
    diff = s - ad                                   # >0 means adaptive lower (better)
    eps = 1e-9
    ratio = s / np.maximum(ad, eps)                 # >1 means adaptive lower (better)

    res = {
        "metric": a.metric,
        "n_matched_replications": len(common),
        "static_mean": float(s.mean()), "static_median": float(np.median(s)),
        "adaptive_mean": float(ad.mean()), "adaptive_median": float(np.median(ad)),
        "median_paired_difference_s": float(np.median(diff)),
        "median_ratio_static_over_adaptive": float(np.median(ratio)),
    }
    lo, hi = boot_ci_median_ratio(ratio)
    res["median_ratio_ci95"] = [lo, hi]

    # Wilcoxon signed-rank (paired, non-parametric) + rank-biserial effect size.
    try:
        from scipy.stats import wilcoxon
        w = wilcoxon(s, ad, alternative="greater")   # H1: static > adaptive
        res["wilcoxon_statistic"] = float(w.statistic)
        res["wilcoxon_p_one_sided"] = float(w.pvalue)
        nz = diff[diff != 0]
        if len(nz):
            ranks = np.argsort(np.argsort(np.abs(nz))) + 1
            rpos = ranks[nz > 0].sum(); rneg = ranks[nz < 0].sum()
            res["rank_biserial_effect_size"] = float((rpos - rneg) / (rpos + rneg))
    except Exception as e:
        res["wilcoxon_error"] = repr(e)

    print(json.dumps(res, indent=2))
    out = a.out or os.path.join(os.path.dirname(a.adaptive.rstrip("/")) or ".",
                                "paired_compare.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[paired_compare] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
