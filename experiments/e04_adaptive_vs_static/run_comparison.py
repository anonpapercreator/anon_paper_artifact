#!/usr/bin/env python3
"""
experiments/e04_adaptive_vs_static/run_comparison.py
====================================================
Compare CTMC-driven adaptive policy against static tau=3600s.
Both use a synthetic workload with phase transitions.

Usage:
    python experiments/e04_adaptive_vs_static/run_comparison.py
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
DURATION = 3600
WARMUP = 600
BASE_SEED = 42
BASE_CONFIG = "config/default_config.yaml"


def run_policy_variant(adaptive: bool, n_reps: int) -> List[Dict]:
    """Run N replications with either adaptive or static policy."""
    policy_name = "adaptive" if adaptive else "static"
    print(f"\n{'='*60}")
    print(f"  Policy: {policy_name.upper()}")
    print(f"{'='*60}")

    base_cfg = load_config(BASE_CONFIG)
    base_cfg["simulation"]["sim_duration_s"] = DURATION
    base_cfg["simulation"]["warmup_s"] = WARMUP
    base_cfg["simulation"]["n_replications"] = 1
    base_cfg["workload"]["mode"] = "synthetic"
    base_cfg["workload"]["trace_file"] = None

    if adaptive:
        base_cfg["adaptive_policy"]["enabled"] = True
        base_cfg["policy_engine"]["candidate_age_threshold_s"] = 1800
    else:
        base_cfg["adaptive_policy"]["enabled"] = False
        base_cfg["policy_engine"]["candidate_age_threshold_s"] = 3600

    results = []
    for rep_id in range(n_reps):
        cfg = base_cfg.copy()
        cfg["simulation"]["seed"] = BASE_SEED + rep_id * 1000 + (100 if adaptive else 0)

        try:
            rep_stats = run_replication(cfg, replication_id=rep_id)
            results.append({
                "rep_id": rep_id,
                "n_migrations": rep_stats.n_migrations,
                "n_recalls": rep_stats.n_recalls,
                "n_mounts": rep_stats.n_mounts,
                "bytes_archived": rep_stats.bytes_archived,
                "bytes_recalled": rep_stats.bytes_recalled,
                "bytes_written": rep_stats.bytes_written_to_cache,
                "cache_hit_rate": rep_stats.cache_hit_rate,
            })
            print(f"  Rep {rep_id+1}/{n_reps}: OK  "
                  f"migrations={rep_stats.n_migrations}  "
                  f"recalls={rep_stats.n_recalls}  "
                  f"mounts={rep_stats.n_mounts}")
        except Exception as e:
            print(f"  Rep {rep_id+1}/{n_reps}: FAILED — {e}")
            continue

    return results


def summarise(results: List[Dict]) -> Dict:
    """Compute mean ± std for key metrics."""
    if not results:
        return {}
    return {
        "n_migrations": {
            "mean": float(np.mean([r["n_migrations"] for r in results])),
            "std": float(np.std([r["n_migrations"] for r in results], ddof=1)),
            "ci_half": float(1.96 * np.std([r["n_migrations"] for r in results], ddof=1) / np.sqrt(len(results))),
        },
        "n_recalls": {
            "mean": float(np.mean([r["n_recalls"] for r in results])),
            "std": float(np.std([r["n_recalls"] for r in results], ddof=1)),
        },
        "n_mounts": {
            "mean": float(np.mean([r["n_mounts"] for r in results])),
            "std": float(np.std([r["n_mounts"] for r in results], ddof=1)),
        },
        "bytes_archived": {
            "mean": float(np.mean([r["bytes_archived"] for r in results])),
            "std": float(np.std([r["bytes_archived"] for r in results], ddof=1)),
        },
    }


def main():
    print("=" * 70)
    print("ADAPTIVE VS STATIC POLICY COMPARISON")
    print("=" * 70)
    print(f"  Replications per policy: {N_REPLICATIONS}")
    print(f"  Duration: {DURATION}s (warmup: {WARMUP}s)")
    print()

    static_results = run_policy_variant(adaptive=False, n_reps=N_REPLICATIONS)
    adaptive_results = run_policy_variant(adaptive=True, n_reps=N_REPLICATIONS)

    static_summary = summarise(static_results)
    adaptive_summary = summarise(adaptive_results)

    # Write outputs
    output_dir = Path("experiments/e04_adaptive_vs_static/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "parameters": {
            "n_replications": N_REPLICATIONS,
            "duration_s": DURATION,
            "warmup_s": WARMUP,
        },
        "static": {
            "raw_results": static_results,
            "summary": static_summary,
        },
        "adaptive": {
            "raw_results": adaptive_results,
            "summary": adaptive_summary,
        },
        "comparison": {
            "migrations_diff_pct": (
                (adaptive_summary["n_migrations"]["mean"] - static_summary["n_migrations"]["mean"])
                / max(static_summary["n_migrations"]["mean"], 1) * 100
                if static_summary else 0
            ),
            "mounts_diff_pct": (
                (adaptive_summary["n_mounts"]["mean"] - static_summary["n_mounts"]["mean"])
                / max(static_summary["n_mounts"]["mean"], 1) * 100
                if static_summary else 0
            ),
        },
    }

    with open(output_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # CSV for quick plotting
    with open(output_dir / "comparison_summary.csv", "w") as f:
        f.write("policy,n_migrations_mean,n_migrations_ci_half,n_mounts_mean,n_mounts_std\n")
        if static_summary:
            f.write(f"static,{static_summary['n_migrations']['mean']:.1f},"
                    f"{static_summary['n_migrations']['ci_half']:.1f},"
                    f"{static_summary['n_mounts']['mean']:.1f},"
                    f"{static_summary['n_mounts']['std']:.1f}\n")
        if adaptive_summary:
            f.write(f"adaptive,{adaptive_summary['n_migrations']['mean']:.1f},"
                    f"{adaptive_summary['n_migrations']['ci_half']:.1f},"
                    f"{adaptive_summary['n_mounts']['mean']:.1f},"
                    f"{adaptive_summary['n_mounts']['std']:.1f}\n")

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    if static_summary and adaptive_summary:
        print(f"  Static:   migrations={static_summary['n_migrations']['mean']:.1f} ± "
              f"{static_summary['n_migrations']['ci_half']:.1f}  "
              f"mounts={static_summary['n_mounts']['mean']:.1f}")
        print(f"  Adaptive: migrations={adaptive_summary['n_migrations']['mean']:.1f} ± "
              f"{adaptive_summary['n_migrations']['ci_half']:.1f}  "
              f"mounts={adaptive_summary['n_mounts']['mean']:.1f}")
        print(f"\n  Diff: migrations={report['comparison']['migrations_diff_pct']:+.1f}%  "
              f"mounts={report['comparison']['mounts_diff_pct']:+.1f}%")
    print(f"\n  Output written to {output_dir}/")


if __name__ == "__main__":
    main()
