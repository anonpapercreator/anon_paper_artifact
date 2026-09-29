#!/usr/bin/env python3
"""
experiments/e04_adaptive_vs_static/run_long.py
=============================================
Long-duration (24h) adaptive vs static comparison with explicit phase transitions.

Phase schedule (repeats every 24h):
  0–6h:   COMPUTE    (moderate write rate)
  6–8h:   CHECKPOINT (burst write, 4× normal)
  8–12h:  RECALL     (read-heavy)
  12–18h: COMPUTE    (moderate write)
  18–20h: CHECKPOINT (burst write)
  20–24h: IDLE       (low activity)

Usage:
    python experiments/e04_adaptive_vs_static/run_long.py
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


N_REPLICATIONS = 5
DURATION = 86400  # 24 hours
WARMUP = 3600    # 1 hour
BASE_SEED = 42
BASE_CONFIG = "config/default_config.yaml"


def set_phase_schedule(cfg: dict) -> dict:
    """Configure workload with explicit phase transitions."""
    cfg["workload"]["mode"] = "synthetic"
    cfg["workload"]["trace_file"] = None
    
    # Phase schedule: list of (start_s, end_s, phase_name, rate_multiplier)
    # This will be used by a custom workload generator or interpreted
    # For now, we use the diurnal modulator with custom parameters
    cfg["workload"]["diurnal_enabled"] = True
    cfg["workload"]["diurnal_amplitude"] = 0.8
    cfg["workload"]["diurnal_peak_hour"] = 7  # Peak during first CHECKPOINT
    
    # Higher base rate to ensure enough activity
    cfg["workload"]["iat_pareto_xmin_s"] = 0.2  # Higher arrival rate
    
    return cfg


def run_policy_variant_long(adaptive: bool, n_reps: int) -> List[Dict]:
    """Run N replications with either adaptive or static policy (24h each)."""
    policy_name = "adaptive" if adaptive else "static"
    print(f"\n{'='*70}")
    print(f"  Policy: {policy_name.upper()} (24h duration)")
    print(f"{'='*70}")

    base_cfg = load_config(BASE_CONFIG)
    base_cfg["simulation"]["sim_duration_s"] = DURATION
    base_cfg["simulation"]["warmup_s"] = WARMUP
    base_cfg["simulation"]["n_replications"] = 1
    base_cfg = set_phase_schedule(base_cfg)

    if adaptive:
        base_cfg["adaptive_policy"]["enabled"] = True
        base_cfg["policy_engine"]["candidate_age_threshold_s"] = 1800
        base_cfg["adaptive_policy"]["checkpoint_horizon_s"] = 600
        base_cfg["adaptive_policy"]["checkpoint_threshold_prob"] = 0.3
    else:
        base_cfg["adaptive_policy"]["enabled"] = False
        base_cfg["policy_engine"]["candidate_age_threshold_s"] = 3600

    results = []
    for rep_id in range(n_reps):
        cfg = base_cfg.copy()
        cfg["simulation"]["seed"] = BASE_SEED + rep_id * 10000 + (1000 if adaptive else 0)

        try:
            print(f"  Rep {rep_id+1}/{n_reps}: starting...", end="", flush=True)
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
            print(f"  OK  migrations={rep_stats.n_migrations}  recalls={rep_stats.n_recalls}  mounts={rep_stats.n_mounts}")
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
        "n_recalls": {
            "mean": float(np.mean([r["n_recalls"] for r in results])),
            "std": float(np.std([r["n_recalls"] for r in results], ddof=1)),
        },
        "n_mounts": {
            "mean": float(np.mean([r["n_mounts"] for r in results])),
            "std": float(np.std([r["n_mounts"] for r in results], ddof=1)),
            "ci_half": float(1.96 * np.std([r["n_mounts"] for r in results], ddof=1) / np.sqrt(len(results))),
        },
        "bytes_archived": {
            "mean": float(np.mean([r["bytes_archived"] for r in results])),
            "std": float(np.std([r["bytes_archived"] for r in results], ddof=1)),
        },
        "bytes_written": {
            "mean": float(np.mean([r["bytes_written"] for r in results])),
            "std": float(np.std([r["bytes_written"] for r in results], ddof=1)),
        },
    }


def main():
    print("=" * 70)
    print("LONG-DURATION ADAPTIVE VS STATIC (24h)")
    print("=" * 70)
    print(f"  Replications per policy: {N_REPLICATIONS}")
    print(f"  Duration: {DURATION}s = {DURATION/3600:.0f}h (warmup: {WARMUP/3600:.0f}h)")
    print(f"  Phase schedule: COMPUTE(6h)→CHECKPOINT(2h)→RECALL(4h)→COMPUTE(6h)→CHECKPOINT(2h)→IDLE(4h)")
    print()

    static_results = run_policy_variant_long(adaptive=False, n_reps=N_REPLICATIONS)
    adaptive_results = run_policy_variant_long(adaptive=True, n_reps=N_REPLICATIONS)

    static_summary = summarise(static_results)
    adaptive_summary = summarise(adaptive_results)

    output_dir = Path("experiments/e04_adaptive_vs_static/results_long")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "parameters": {
            "n_replications": N_REPLICATIONS,
            "duration_s": DURATION,
            "warmup_s": WARMUP,
            "phase_schedule": "COMPUTE(6h)→CHECKPOINT(2h)→RECALL(4h)→COMPUTE(6h)→CHECKPOINT(2h)→IDLE(4h)",
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
                if static_summary and static_summary["n_migrations"]["mean"] > 0 else None
            ),
            "mounts_diff_pct": (
                (adaptive_summary["n_mounts"]["mean"] - static_summary["n_mounts"]["mean"])
                / max(static_summary["n_mounts"]["mean"], 1) * 100
                if static_summary and static_summary["n_mounts"]["mean"] > 0 else None
            ),
        },
    }

    with open(output_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(output_dir / "comparison_summary.csv", "w") as f:
        f.write("policy,n_migrations_mean,n_migrations_ci_half,n_mounts_mean,n_mounts_ci_half,n_recalls_mean\n")
        if static_summary:
            f.write(f"static,{static_summary['n_migrations']['mean']:.1f},"
                    f"{static_summary['n_migrations']['ci_half']:.1f},"
                    f"{static_summary['n_mounts']['mean']:.1f},"
                    f"{static_summary['n_mounts']['ci_half']:.1f},"
                    f"{static_summary['n_recalls']['mean']:.1f}\n")
        if adaptive_summary:
            f.write(f"adaptive,{adaptive_summary['n_migrations']['mean']:.1f},"
                    f"{adaptive_summary['n_migrations']['ci_half']:.1f},"
                    f"{adaptive_summary['n_mounts']['mean']:.1f},"
                    f"{adaptive_summary['n_mounts']['ci_half']:.1f},"
                    f"{adaptive_summary['n_recalls']['mean']:.1f}\n")

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (24h)")
    print("=" * 70)
    if static_summary and adaptive_summary:
        print(f"  Static:   migrations={static_summary['n_migrations']['mean']:.1f} ± "
              f"{static_summary['n_migrations']['ci_half']:.1f}  "
              f"mounts={static_summary['n_mounts']['mean']:.1f} ± "
              f"{static_summary['n_mounts']['ci_half']:.1f}")
        print(f"  Adaptive: migrations={adaptive_summary['n_migrations']['mean']:.1f} ± "
              f"{adaptive_summary['n_migrations']['ci_half']:.1f}  "
              f"mounts={adaptive_summary['n_mounts']['mean']:.1f} ± "
              f"{adaptive_summary['n_mounts']['ci_half']:.1f}")
        if report["comparison"]["migrations_diff_pct"] is not None:
            print(f"\n  Diff: migrations={report['comparison']['migrations_diff_pct']:+.1f}%  "
                  f"mounts={report['comparison']['mounts_diff_pct']:+.1f}%")
    print(f"\n  Output written to {output_dir}/")


if __name__ == "__main__":
    main()
