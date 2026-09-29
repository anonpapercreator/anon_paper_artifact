#!/usr/bin/env python3
"""
run_multitenant_paired.py -- reproduce the multi-tenant beta-sweep table.

This is the COMMITTED, self-contained reproduction of the multi-tenant
comparison (paper Table III / fig. 4). Earlier e04 drivers (run_comparison.py,
run_long.py) exercise only the single-stream workload and therefore do NOT
reproduce it; this script enables the multi-tenant noisy-neighbour workload
and sets the stress-test configuration under which the archive policy is
exercised (cache pressure within the observation window).

Three arms per beta:
  static   -- adaptive advisor OFF (fixed age threshold)
  hwm      -- reactive watermark-only advisor (the "one-line rule" ablation
              arm B; components/hwm_advisor.py -- no CTMC/MMFQ/IOM)
  adaptive -- full MMFQ/IOM/CTMC advisor (arm C)

Metrics per replication (definitions identical to slurm/run_block.py, which
produced the SLURM campaign behind the paper table):
  recall_p95_s      -- aggregate recall P95 (the "147x" metric)
  tenant_p95_max_s  -- worst-tenant P95 (the "54x" metric)
  tenant_p95_gini   -- Gini coefficient on per-tenant P95 (fairness)
  n_migrations      -- migration count (reported as a mean, like the table)

Stress-test configuration (deliberate, and stated in the paper's table
context):
  - multi_tenant.enabled = True, n_tenants = 128
  - disk_cache.capacity_bytes = 1 TB  (so the aggregate working set overflows;
    the production calibration uses the real 889 TiB tier instead)
  - workload.working_set_size = 64000 (aggregate WS >> cache -> real recall load)
  - policy_engine.cycle_period_s = 300 (migration cycles within the 4 h window;
    production is 6 h, used for trace-replay calibration only)
  - duration 14400 s (4 h), warmup 1800 s
  - heterogeneity sweep beta (noise_level) in {0.0, 0.5, 1.0}

Paired design: replication i of every arm shares the index; each arm draws its
seed as BASE_SEED + i + arm_offset (static 0, hwm 509, adaptive 1009). Note the
SLURM campaign pairs arms under true common random numbers (identical seed per
replication, arms differ only in the advisor flag); this standalone driver
keeps distinct per-arm seed streams, so treat its pairing as index-matched
rather than CRN.

The effect is large but HIGH-VARIANCE across seeds (the static worst-tenant P95
is bimodal: the cache either saturates or does not for a given arrival pattern).
A small number of replications is therefore unreliable; the paper uses n=200 and
reports medians with bootstrap CIs and a Wilcoxon signed-rank test. This script
defaults to n=200 but accepts --n-reps for quick checks (expect wide swings at
small n).

Usage:
  python experiments/e04_adaptive_vs_static/run_multitenant_paired.py \
      [--n-reps 200] [--betas 0.0,0.5,1.0] [--arms static,hwm,adaptive] \
      [--out results_mt]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from main import load_config, run_replication          # noqa: E402
from slurm.run_block import per_rep_kpis               # noqa: E402  (same defs as campaign)

DURATION = 14400
WARMUP = 1800
N_TENANTS = 128
WORKING_SET = 64000
CACHE_BYTES = 1_000_000_000_000          # 1 TB
CYCLE_PERIOD = 300.0
BASE_SEED = 42
ARM_SEED_OFFSET = {"static": 0, "hwm": 509, "adaptive": 1009}
ARM_ORDER = ("static", "hwm", "adaptive")
RATIO_METRICS = ("recall_p95_s", "tenant_p95_max_s", "tenant_p95_gini")
PAIRS = (("static", "adaptive"), ("static", "hwm"), ("hwm", "adaptive"))


def _loaded_cfg(policy: str, beta: float, seed: int):
    cfg = load_config(str(ROOT / "config" / "default_config.yaml"))
    cfg["simulation"].update(sim_duration_s=DURATION, warmup_s=WARMUP,
                             seed=seed, n_replications=1)
    cfg["workload"]["mode"] = "synthetic"
    cfg["workload"]["working_set_size"] = WORKING_SET
    mt = cfg["workload"].setdefault("multi_tenant", {})
    mt.update(enabled=True, n_tenants=N_TENANTS, noise_level=float(beta))
    cfg["disk_cache"]["capacity_bytes"] = int(CACHE_BYTES)
    cfg.setdefault("policy_engine", {})["cycle_period_s"] = CYCLE_PERIOD
    ap = cfg.setdefault("adaptive_policy", {})
    ap["enabled"] = policy in ("adaptive", "hwm")
    ap["advisor"] = "hwm" if policy == "hwm" else "full"
    return cfg


def _bootstrap_ci(ratios, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    r = np.asarray(ratios, float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return (float("nan"), float("nan"))
    boots = [np.median(rng.choice(r, r.size, replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def _metric_array(kpis: list, metric: str) -> np.ndarray:
    return np.array([float(k[metric]) if k.get(metric) is not None else np.nan
                     for k in kpis])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--betas", default="0.0,0.5,1.0")
    ap.add_argument("--arms", default=",".join(ARM_ORDER),
                    help="comma-separated subset of static,hwm,adaptive")
    ap.add_argument("--out", default=str(HERE.parent / "results_mt"))
    args = ap.parse_args()
    betas = [float(b) for b in args.betas.split(",")]
    arms = [a for a in ARM_ORDER if a in args.arms.split(",")]
    if not arms:
        raise SystemExit(f"--arms must name at least one of {ARM_ORDER}")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    try:
        from scipy.stats import wilcoxon
        have_scipy = True
    except Exception:
        have_scipy = False

    report = {"config": {
        "duration_s": DURATION, "warmup_s": WARMUP, "n_tenants": N_TENANTS,
        "working_set_size": WORKING_SET, "cache_bytes": CACHE_BYTES,
        "cycle_period_s": CYCLE_PERIOD, "n_reps": args.n_reps,
        "seed_base": BASE_SEED, "arm_seed_offsets": ARM_SEED_OFFSET,
        "arms": arms,
    }, "betas": {}}

    for beta in betas:
        kpis = {arm: [] for arm in arms}
        for i in range(args.n_reps):
            line = []
            for arm in arms:
                seed = BASE_SEED + i + ARM_SEED_OFFSET[arm]
                stats = run_replication(_loaded_cfg(arm, beta, seed),
                                        replication_id=i)
                kpis[arm].append(per_rep_kpis(stats))
                wt = kpis[arm][-1].get("tenant_p95_max_s")
                line.append(f"{arm}={wt:.0f}s" if wt is not None else f"{arm}=n/a")
            print(f"beta={beta} rep {i+1}/{args.n_reps}: " + " ".join(line),
                  flush=True)

        entry = {"per_arm": {}, "paired": {}}
        for arm in arms:
            entry["per_arm"][arm] = {
                "recall_p95_median_s":
                    float(np.nanmedian(_metric_array(kpis[arm], "recall_p95_s"))),
                "tenant_p95_max_median_s":
                    float(np.nanmedian(_metric_array(kpis[arm], "tenant_p95_max_s"))),
                "tenant_p95_gini_median":
                    float(np.nanmedian(_metric_array(kpis[arm], "tenant_p95_gini"))),
                "n_migrations_mean":
                    float(np.nanmean(_metric_array(kpis[arm], "n_migrations"))),
            }
        for hi, lo in PAIRS:
            if hi not in arms or lo not in arms:
                continue
            pair_key = f"{hi}_over_{lo}"
            entry["paired"][pair_key] = {}
            for metric in RATIO_METRICS:
                h = _metric_array(kpis[hi], metric)
                l = _metric_array(kpis[lo], metric)
                mask = np.isfinite(h) & np.isfinite(l) & (l > 0)
                if not mask.any():
                    entry["paired"][pair_key][metric] = {"n_matched": 0}
                    continue
                ratios = h[mask] / l[mask]
                m = {
                    "n_matched": int(mask.sum()),
                    "median_ratio": float(np.median(ratios)),
                    "median_ratio_ci95": _bootstrap_ci(ratios),
                    "ratio_min": float(np.min(ratios)),
                    "ratio_max": float(np.max(ratios)),
                }
                if have_scipy and mask.sum() >= 2:
                    try:
                        stat, p = wilcoxon(h[mask], l[mask], alternative="greater")
                        m["wilcoxon_stat"] = float(stat)
                        m["wilcoxon_p_one_sided"] = float(p)
                    except Exception as e:
                        m["wilcoxon_error"] = repr(e)
                entry["paired"][pair_key][metric] = m
        report["betas"][str(beta)] = entry

        for pair_key, metrics in entry["paired"].items():
            agg = metrics.get("recall_p95_s", {})
            wt = metrics.get("tenant_p95_max_s", {})
            print(f"  -> beta={beta} {pair_key}: "
                  f"agg P95 {agg.get('median_ratio', float('nan')):.1f}x "
                  f"CI{tuple(round(x, 1) for x in agg.get('median_ratio_ci95', (float('nan'),) * 2))}  "
                  f"worst-tenant {wt.get('median_ratio', float('nan')):.1f}x "
                  f"CI{tuple(round(x, 1) for x in wt.get('median_ratio_ci95', (float('nan'),) * 2))}")

    (outdir / "multitenant_paired_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {outdir/'multitenant_paired_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
