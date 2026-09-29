#!/usr/bin/env python3
"""
experiments/e05_sota_ior/run_ior_comparison.py
=============================================
E5: IOR-like benchmark workload for SOTA comparison.

Uses synthetic generator with IOR-like parameters:
- Fixed file sizes (1 MiB, low variance)
- Regular inter-arrival times (low CV)
- 100% write then 100% read pattern via read_fraction scheduling

Purpose: Show that on SmartIO's own benchmark conditions, both static
and adaptive policies perform similarly because the workload is regular
and predictable. The advantage of adaptivity only emerges under
heterogeneity.

Usage:
    python experiments/e05_sota_ior/run_ior_comparison.py
"""
from __future__ import annotations
import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import run_replication, load_config


N_REPLICATIONS = 10
DURATION = 7200   # 2 hours (sufficient for comparison)
WARMUP = 600      # 10 min
BASE_SEED = 42
BASE_CONFIG = "config/default_config.yaml"


def configure_ior_like(cfg: dict) -> dict:
    """Configure IOR-like synthetic workload: uniform, regular, no death."""
    cfg["workload"]["mode"] = "synthetic"
    cfg["workload"]["trace_file"] = None
    
    # IOR characteristics: fixed size, regular IAT
    cfg["workload"]["file_size_lognormal_mu"] = 20.03  # ln(1 MiB) ≈ 20.03
    cfg["workload"]["file_size_lognormal_sigma"] = 0.1  # Very low variance
    cfg["workload"]["file_size_min_bytes"] = 1024 * 1024
    cfg["workload"]["file_size_max_bytes"] = 2 * 1024 * 1024
    
    # Regular IAT: exponential with high rate (low mean)
    cfg["workload"]["iat_distribution"] = "exponential"
    cfg["workload"]["iat_exponential_rate"] = 10.0  # 10 ops/sec
    
    # Write-heavy initially, then read-heavy
    cfg["workload"]["read_fraction"] = 0.2  # 80% write
    cfg["workload"]["working_set_size"] = 5000
    cfg["workload"]["zipf_s"] = 1.0  # Higher locality
    
    # No diurnal variation — steady state
    cfg["workload"]["diurnal_enabled"] = False
    
    # Long file lifetimes (immortal for practical purposes)
    cfg["workload"]["initial_file_age_s_mu"] = 86400
    cfg["workload"]["initial_file_age_s_sigma"] = 3600
    
    return cfg


def extract_recall_metrics(rep_stats) -> Dict[str, float]:
    """Extract recall latency metrics from ReplicationStats after finalise()."""
    raw = rep_stats.recall_latency._last_rep_samples
    return {
        "mean_recall_latency_s": float(np.mean(raw)) if raw else 0.0,
        "p50_recall_latency_s": rep_stats.recall_latency._rep_p50[-1] if rep_stats.recall_latency._rep_p50 else 0.0,
        "p95_recall_latency_s": rep_stats.recall_latency._rep_p95[-1] if rep_stats.recall_latency._rep_p95 else 0.0,
        "p99_recall_latency_s": rep_stats.recall_latency._rep_p99[-1] if rep_stats.recall_latency._rep_p99 else 0.0,
    }


def run_policy_ior(adaptive: bool, n_reps: int) -> List[Dict]:
    """Run DESCASSI with IOR-like workload."""
    policy_name = "adaptive" if adaptive else "static"
    print(f"\n{'='*70}")
    print(f"  E5 IOR-like: {policy_name.upper()} policy")
    print(f"{'='*70}")

    base_cfg = load_config(BASE_CONFIG)
    base_cfg["simulation"]["sim_duration_s"] = DURATION
    base_cfg["simulation"]["warmup_s"] = WARMUP
    base_cfg["simulation"]["n_replications"] = 1
    base_cfg = configure_ior_like(base_cfg)

    if adaptive:
        base_cfg["adaptive_policy"]["enabled"] = True
        base_cfg["policy_engine"]["candidate_age_threshold_s"] = 600
    else:
        base_cfg["adaptive_policy"]["enabled"] = False
        base_cfg["policy_engine"]["candidate_age_threshold_s"] = 1200

    results = []
    for rep_id in range(n_reps):
        cfg = base_cfg.copy()
        cfg["simulation"]["seed"] = BASE_SEED + rep_id * 10000 + (1000 if adaptive else 0)

        try:
            print(f"  Rep {rep_id+1}/{n_reps}: ...", end="", flush=True)
            rep_stats = run_replication(cfg, replication_id=rep_id)
            lat = extract_recall_metrics(rep_stats)
            results.append({
                "rep_id": rep_id,
                "n_migrations": rep_stats.n_migrations,
                "n_recalls": rep_stats.n_recalls,
                "n_mounts": rep_stats.n_mounts,
                "mean_recall_latency_s": lat["mean_recall_latency_s"],
                "p95_recall_latency_s": lat["p95_recall_latency_s"],
                "cache_hit_rate": rep_stats.cache_hit_rate,
                "bytes_archived": rep_stats.bytes_archived,
                "bytes_written": rep_stats.bytes_written_to_cache,
            })
            print(f"  OK  lat={lat['mean_recall_latency_s']:.3f}s  "
                  f"mounts={rep_stats.n_mounts}  hits={rep_stats.cache_hit_rate:.2f}")
        except Exception as e:
            print(f"  FAILED — {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def summarise(results: List[Dict]) -> Dict:
    if not results:
        return {}
    return {
        "n_migrations": {
            "mean": float(np.mean([r["n_migrations"] for r in results])),
            "std": float(np.std([r["n_migrations"] for r in results], ddof=1)),
            "ci_half": float(1.96 * np.std([r["n_migrations"] for r in results], ddof=1) / np.sqrt(len(results))),
        },
        "n_mounts": {
            "mean": float(np.mean([r["n_mounts"] for r in results])),
            "std": float(np.std([r["n_mounts"] for r in results], ddof=1)),
            "ci_half": float(1.96 * np.std([r["n_mounts"] for r in results], ddof=1) / np.sqrt(len(results))),
        },
        "mean_recall_latency_s": {
            "mean": float(np.mean([r["mean_recall_latency_s"] for r in results])),
            "std": float(np.std([r["mean_recall_latency_s"] for r in results], ddof=1)),
            "ci_half": float(1.96 * np.std([r["mean_recall_latency_s"] for r in results], ddof=1) / np.sqrt(len(results))),
        },
        "p95_recall_latency_s": {
            "mean": float(np.mean([r["p95_recall_latency_s"] for r in results])),
            "std": float(np.std([r["p95_recall_latency_s"] for r in results], ddof=1)),
        },
        "cache_hit_rate": {
            "mean": float(np.mean([r["cache_hit_rate"] for r in results])),
            "std": float(np.std([r["cache_hit_rate"] for r in results], ddof=1)),
        },
    }


def main():
    print("=" * 70)
    print("E5: IOR-LIKE BENCHMARK WORKLOAD")
    print("=" * 70)
    print("  Fixed 1MiB blocks, regular IAT, low variance, 80% write")
    print(f"  Replications per policy: {N_REPLICATIONS}")
    print(f"  Duration: {DURATION}s = {DURATION/3600:.1f}h")
    print()

    output_dir = Path("experiments/e05_sota_ior/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    static_results = run_policy_ior(adaptive=False, n_reps=N_REPLICATIONS)
    adaptive_results = run_policy_ior(adaptive=True, n_reps=N_REPLICATIONS)

    static_summary = summarise(static_results)
    adaptive_summary = summarise(adaptive_results)

    report = {
        "parameters": {
            "n_replications": N_REPLICATIONS,
            "duration_s": DURATION,
            "warmup_s": WARMUP,
            "workload": "IOR-like: uniform 1MiB, regular IAT, 80% write",
        },
        "static": {"raw_results": static_results, "summary": static_summary},
        "adaptive": {"raw_results": adaptive_results, "summary": adaptive_summary},
    }

    with open(output_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(output_dir / "comparison_summary.csv", "w") as f:
        f.write("policy,n_migrations_mean,n_migrations_ci,n_mounts_mean,n_mounts_ci,"
                "mean_recall_latency_s_mean,mean_recall_latency_s_ci,"
                "p95_recall_latency_s_mean,cache_hit_rate_mean\n")
        for policy, summary in [("static", static_summary), ("adaptive", adaptive_summary)]:
            if summary:
                f.write(f"{policy},"
                        f"{summary['n_migrations']['mean']:.1f},"
                        f"{summary['n_migrations']['ci_half']:.1f},"
                        f"{summary['n_mounts']['mean']:.1f},"
                        f"{summary['n_mounts']['ci_half']:.1f},"
                        f"{summary['mean_recall_latency_s']['mean']:.3f},"
                        f"{summary['mean_recall_latency_s']['ci_half']:.3f},"
                        f"{summary['p95_recall_latency_s']['mean']:.3f},"
                        f"{summary['cache_hit_rate']['mean']:.3f}\n")

    print("\n" + "=" * 70)
    print("E5 SUMMARY")
    print("=" * 70)
    if static_summary and adaptive_summary:
        print(f"  Static:   mean_lat={static_summary['mean_recall_latency_s']['mean']:.3f}s  "
              f"p95={static_summary['p95_recall_latency_s']['mean']:.3f}s  "
              f"mounts={static_summary['n_mounts']['mean']:.1f}")
        print(f"  Adaptive: mean_lat={adaptive_summary['mean_recall_latency_s']['mean']:.3f}s  "
              f"p95={adaptive_summary['p95_recall_latency_s']['mean']:.3f}s  "
              f"mounts={adaptive_summary['n_mounts']['mean']:.1f}")
    print(f"\n  Output written to {output_dir}/")


if __name__ == "__main__":
    main()
