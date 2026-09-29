#!/usr/bin/env python3
"""
experiments/e03_policy_sweep/run_long.py
======================================
Refined policy sweep with 12h duration to show tau→infinity divergence.

Tau values cover 100s to 24h (86400s), with 12h simulation so that:
- Small tau (<< 12h): migrations occur, cache controlled
- Large tau (> 12h): no migrations, cache fills, overflow events spike

Duration: 12h (43,200s) per replication
Replications: 5 per tau value
Tau values: 12 values from 100s to 86400s

Usage:
    python experiments/e03_policy_sweep/run_long.py
"""
from __future__ import annotations
import sys
import os
import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import run_replication, load_config


# Focused tau values — enough to show the curve and divergence
TAU_VALUES = [100, 300, 600, 1200, 1800, 3600, 7200, 14400, 28800, 43200, 57600, 86400]
N_REPLICATIONS = 5
DURATION = 43200  # 12 hours
WARMUP = 3600     # 1 hour
BASE_SEED = 42
BASE_CONFIG = "config/default_config.yaml"

# Weights for J(tau)
WP, WM, WR = 1.0, 0.5, 0.8


def modify_config_for_tau(base_cfg: dict, tau: int) -> dict:
    """Create config with specified age threshold."""
    cfg = base_cfg.copy()
    cfg["workload"]["mode"] = "synthetic"
    cfg["workload"]["trace_file"] = None
    cfg["policy_engine"]["candidate_age_threshold_s"] = tau
    cfg["simulation"]["sim_duration_s"] = DURATION
    cfg["simulation"]["warmup_s"] = WARMUP
    cfg["simulation"]["n_replications"] = 1
    cfg["simulation"]["progress_chunk_s"] = 300  # Less frequent updates = faster
    cfg["adaptive_policy"]["enabled"] = False
    return cfg


def run_tau_experiment(tau: int, n_reps: int, base_cfg: dict) -> List[Dict]:
    """Run n_replications for a single tau value."""
    print(f"\n{'='*60}")
    print(f"  tau = {tau}s ({tau/3600:.1f}h)")
    print(f"{'='*60}")

    results = []
    for rep_id in range(n_reps):
        cfg = modify_config_for_tau(base_cfg, tau)
        cfg["simulation"]["seed"] = BASE_SEED + rep_id * 10000 + tau

        try:
            print(f"  Rep {rep_id+1}/{n_reps}: starting...", end=" ", flush=True)
            rep_stats = run_replication(cfg, replication_id=rep_id)
            print(f"OK  migrations={rep_stats.n_migrations}  "
                  f"recalls={rep_stats.n_recalls}  mounts={rep_stats.n_mounts}  "
                  f"cache_hit={rep_stats.cache_hit_rate:.2f}")
            results.append({
                "rep_id": rep_id,
                "n_migrations": rep_stats.n_migrations,
                "n_recalls": rep_stats.n_recalls,
                "n_mounts": rep_stats.n_mounts,
                "bytes_archived": rep_stats.bytes_archived,
                "bytes_written": rep_stats.bytes_written_to_cache,
                "cache_hit_rate": rep_stats.cache_hit_rate,
            })
        except Exception as e:
            print(f"FAILED — {e}")
            continue

    return results


def compute_cost_components(all_results: List[Dict]) -> List[Dict]:
    """Compute Ploss, Mwaste, Rrisk, J with CIs."""
    processed = []
    
    # First pass: raw values
    raw = []
    for r in all_results:
        tau = r["tau"]
        n_mounts = [rep["n_mounts"] for rep in r["reps"]]
        bytes_archived = [rep["bytes_archived"] for rep in r["reps"]]
        bytes_written = [rep["bytes_written"] for rep in r["reps"]]
        n_migrations = [rep["n_migrations"] for rep in r["reps"]]
        
        raw.append({
            "tau": tau,
            "n_mounts_values": n_mounts,
            "mwaste_values": [ba / max(bw, 1) for ba, bw in zip(bytes_archived, bytes_written)],
            "rrisk_values": [(bw * tau / DURATION) / (1024**4) for bw in bytes_written],
            "n_migrations_values": n_migrations,
        })
    
    # Normalise
    max_mounts = max(np.mean(x["n_mounts_values"]) for x in raw) if raw else 1
    max_mwaste = max(np.mean(x["mwaste_values"]) for x in raw) if raw else 1
    max_rrisk = max(np.mean(x["rrisk_values"]) for x in raw) if raw else 1
    
    for r in raw:
        tau = r["tau"]
        n_mounts_arr = np.array(r["n_mounts_values"])
        mwaste_arr = np.array(r["mwaste_values"])
        rrisk_arr = np.array(r["rrisk_values"])
        
        n_mounts_mean = np.mean(n_mounts_arr)
        n_mounts_std = np.std(n_mounts_arr, ddof=1)
        n_mounts_ci = 1.96 * n_mounts_std / np.sqrt(len(n_mounts_arr))
        
        ploss_mean = (n_mounts_mean / max(max_mounts, 1)) * 100.0
        ploss_ci = (n_mounts_ci / max(max_mounts, 1)) * 100.0
        
        mwaste_mean = (np.mean(mwaste_arr) / max(max_mwaste, 1)) * 100.0
        mwaste_ci = (np.std(mwaste_arr, ddof=1) / np.sqrt(len(mwaste_arr)) / max(max_mwaste, 1)) * 100.0 * 1.96
        
        rrisk_mean = (np.mean(rrisk_arr) / max(max_rrisk, 1)) * 10.0
        rrisk_ci = (np.std(rrisk_arr, ddof=1) / np.sqrt(len(rrisk_arr)) / max(max_rrisk, 1)) * 10.0 * 1.96
        
        J_mean = WP * ploss_mean + WM * mwaste_mean + WR * rrisk_mean
        J_ci = np.sqrt((WP * ploss_ci)**2 + (WM * mwaste_ci)**2 + (WR * rrisk_ci)**2)
        
        processed.append({
            "tau": tau,
            "ploss_mean": ploss_mean,
            "ploss_ci": ploss_ci,
            "mwaste_mean": mwaste_mean,
            "mwaste_ci": mwaste_ci,
            "rrisk_mean": rrisk_mean,
            "rrisk_ci": rrisk_ci,
            "J_mean": J_mean,
            "J_ci": J_ci,
            "n_migrations_mean": float(np.mean(r["n_migrations_values"])),
            "n_migrations_std": float(np.std(r["n_migrations_values"], ddof=1)),
        })
    
    return processed


def write_sweep_csv_with_ci(processed: List[Dict], output_path: str) -> None:
    """Write sweep results with error bars for PGFPlots."""
    with open(output_path, "w") as f:
        f.write("tau,ploss,ploss_err,mwaste,mwaste_err,rrisk,rrisk_err,J,J_err\n")
        for p in processed:
            f.write(f"{p['tau']},{p['ploss_mean']:.4f},{p['ploss_ci']:.4f},"
                    f"{p['mwaste_mean']:.4f},{p['mwaste_ci']:.4f},"
                    f"{p['rrisk_mean']:.4f},{p['rrisk_ci']:.4f},"
                    f"{p['J_mean']:.4f},{p['J_ci']:.4f}\n")


def main():
    print("=" * 70)
    print("REFINED POLICY SWEEP (12h duration)")
    print("=" * 70)
    print(f"  tau values: {TAU_VALUES} ({len(TAU_VALUES)} values)")
    print(f"  Replications per tau: {N_REPLICATIONS}")
    print(f"  Duration: {DURATION}s = {DURATION/3600:.0f}h (warmup: {WARMUP/3600:.0f}h)")
    print(f"  Weights: wp={WP}, wm={WM}, wr={WR}")
    print(f"  Est. wall time: ~{len(TAU_VALUES) * N_REPLICATIONS * (DURATION/60) / 3600:.1f}h")
    print()

    base_cfg = load_config(BASE_CONFIG)

    all_results = []
    for tau in TAU_VALUES:
        reps = run_tau_experiment(tau, N_REPLICATIONS, base_cfg)
        if reps:
            all_results.append({"tau": tau, "reps": reps})

    if not all_results:
        print("\nERROR: No successful results. Aborting.")
        sys.exit(1)

    # Compute cost components with CIs
    processed = compute_cost_components(all_results)

    output_dir = Path("experiments/e03_policy_sweep/results_long")
    output_dir.mkdir(parents=True, exist_ok=True)

    write_sweep_csv_with_ci(processed, str(output_dir / "sweep_data.csv"))

    report = {
        "parameters": {
            "tau_values": TAU_VALUES,
            "n_replications": N_REPLICATIONS,
            "duration_s": DURATION,
            "warmup_s": WARMUP,
            "weights": {"wp": WP, "wm": WM, "wr": WR},
        },
        "results": processed,
        "optimal": {
            "tau_star": int(min(processed, key=lambda x: x["J_mean"])["tau"]),
            "J_min": float(min(p["J_mean"] for p in processed)),
        },
    }

    with open(output_dir / "sweep_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY (12h)")
    print("=" * 70)
    for p in processed:
        print(f"  tau={p['tau']:6d}s: Ploss={p['ploss_mean']:6.2f}±{p['ploss_ci']:5.2f}%  "
              f"Mwaste={p['mwaste_mean']:6.2f}±{p['mwaste_ci']:5.2f}%  "
              f"Rrisk={p['rrisk_mean']:7.4f}±{p['rrisk_ci']:6.4f}  "
              f"J={p['J_mean']:8.2f}±{p['J_ci']:6.2f}")

    optimal = report["optimal"]
    print(f"\n  Optimal: tau* = {optimal['tau_star']}s (J = {optimal['J_min']:.2f})")
    print(f"\n  Output written to {output_dir}/")


if __name__ == "__main__":
    main()
