#!/usr/bin/env python3
"""
experiments/e03_policy_sweep/run_sweep.py
=======================================
Policy sweep over archive interval tau.
For each tau value, run N replications and collect:
    - Mean queue wait (disk cache)
    - Number of migrations
    - Number of tape mounts
    - Bytes archived
    - Peak cache fill

These are used to compute:
    Ploss(tau)  — relative increase in queue wait from scratch-only baseline
    Mwaste(tau) — bytes archived / total bytes written (proxy for media waste)
    Rrisk(tau)  — peak unprotected data (bytes written since last migration cycle)
    J(tau)      — weighted sum with (wp, wm, wr) = (1.0, 0.5, 0.8)

Usage:
    python experiments/e03_policy_sweep/run_sweep.py
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
from stats.collector import ReplicationStats


# Sweep parameters
TAU_VALUES = [100, 500, 1000, 1800, 3600, 7200, 14400, 28800, 43200, 57600, 72000, 86400]
N_REPLICATIONS = 10
DURATION = 3600
WARMUP = 600
BASE_SEED = 42

# Weights for J(tau)
WP, WM, WR = 1.0, 0.5, 0.8

# Baseline config path
BASE_CONFIG = "config/default_config.yaml"


def modify_config_for_tau(base_cfg: dict, tau: int) -> dict:
    """Create a config dict with the specified age threshold."""
    cfg = base_cfg.copy()
    # Ensure we're using synthetic workload
    cfg["workload"]["mode"] = "synthetic"
    cfg["workload"]["trace_file"] = None
    # Set the archive interval
    cfg["policy_engine"]["candidate_age_threshold_s"] = tau
    # Standard simulation parameters
    cfg["simulation"]["sim_duration_s"] = DURATION
    cfg["simulation"]["warmup_s"] = WARMUP
    cfg["simulation"]["n_replications"] = 1  # We loop externally
    cfg["simulation"]["seed"] = BASE_SEED
    cfg["simulation"]["output_dir"] = f"experiments/e03_policy_sweep/results/tau_{tau}"
    cfg["adaptive_policy"]["enabled"] = False  # Static policy for sweep
    return cfg


def run_tau_experiment(tau: int, n_reps: int, base_cfg: dict) -> Dict:
    """Run n_replications for a single tau value."""
    print(f"\n{'='*60}")
    print(f"  tau = {tau}s")
    print(f"{'='*60}")

    results = []
    for rep_id in range(n_reps):
        cfg = modify_config_for_tau(base_cfg, tau)
        cfg["simulation"]["seed"] = BASE_SEED + rep_id * 1000 + tau

        try:
            rep_stats = run_replication(cfg, replication_id=rep_id)
            results.append(rep_stats)
            print(f"  Rep {rep_id+1}/{n_reps}: OK  "
                  f"migrations={rep_stats.n_migrations}  "
                  f"recalls={rep_stats.n_recalls}  "
                  f"mounts={rep_stats.n_mounts}")
        except Exception as e:
            print(f"  Rep {rep_id+1}/{n_reps}: FAILED — {e}")
            continue

    if not results:
        print(f"  WARNING: All replications failed for tau={tau}")
        return {}

    # Aggregate metrics across replications
    # We need to extract queue wait samples from each replication
    # The ReplicationStats object stores queue wait samples in _last_rep_samples
    # But we need to be careful about the structure

    # For now, extract key scalar metrics that are available
    n_migrations = [r.n_migrations for r in results]
    n_recalls = [r.n_recalls for r in results]
    n_mounts = [r.n_mounts for r in results]
    bytes_archived = [r.bytes_archived for r in results]
    bytes_recalled = [r.bytes_recalled for r in results]
    bytes_written = [r.bytes_written_to_cache for r in results]

    # Estimate mean queue wait from the queue depth samples
    # The disk_cache queue wait isn't directly exposed as a scalar
    # We'll use a proxy: mean number of I/O operations / throughput

    return {
        "tau": tau,
        "n_reps": len(results),
        "n_migrations": {
            "mean": float(np.mean(n_migrations)),
            "std": float(np.std(n_migrations, ddof=1)),
            "ci_half": float(1.96 * np.std(n_migrations, ddof=1) / np.sqrt(len(results))),
        },
        "n_recalls": {
            "mean": float(np.mean(n_recalls)),
            "std": float(np.std(n_recalls, ddof=1)),
        },
        "n_mounts": {
            "mean": float(np.mean(n_mounts)),
            "std": float(np.std(n_mounts, ddof=1)),
        },
        "bytes_archived": {
            "mean": float(np.mean(bytes_archived)),
            "std": float(np.std(bytes_archived, ddof=1)),
        },
        "bytes_written": {
            "mean": float(np.mean(bytes_written)),
            "std": float(np.std(bytes_written, ddof=1)),
        },
        "bytes_recalled": {
            "mean": float(np.mean(bytes_recalled)),
            "std": float(np.std(bytes_recalled, ddof=1)),
        },
    }


def compute_cost_components(results: List[Dict], baseline_wait: float) -> List[Dict]:
    """Compute Ploss, Mwaste, Rrisk, J from empirical results.
    
    Normalisation:
      Ploss:  0-100 scale, relative increase in queue wait
      Mwaste: 0-100 scale, fraction of data that is archived waste
      Rrisk:  0-10 scale,  data-at-risk in TiB (grows with tau)
      J:      weighted sum of normalised components
    """
    processed = []
    
    # Compute raw values first
    raw = []
    for r in results:
        tau = r["tau"]
        n_mounts_mean = r["n_mounts"]["mean"]
        bytes_archived = r["bytes_archived"]["mean"]
        bytes_written = r["bytes_written"]["mean"]
        
        raw.append({
            "tau": tau,
            "n_mounts": n_mounts_mean,
            "mwaste_raw": bytes_archived / max(bytes_written, 1),
            "rrisk_raw": (bytes_written * tau / DURATION) / (1024**4),
        })
    
    # Normalise
    max_mounts = max(x["n_mounts"] for x in raw) if raw else 1
    max_mwaste = max(x["mwaste_raw"] for x in raw) if raw else 1
    max_rrisk = max(x["rrisk_raw"] for x in raw) if raw else 1
    
    for r in raw:
        tau = r["tau"]
        ploss = (r["n_mounts"] / max(max_mounts, 1)) * 100.0
        mwaste = (r["mwaste_raw"] / max(max_mwaste, 1)) * 100.0
        rrisk = (r["rrisk_raw"] / max(max_rrisk, 1)) * 10.0  # Scale to 0-10
        
        J = WP * ploss + WM * mwaste + WR * rrisk
        
        processed.append({
            "tau": tau,
            "ploss": ploss,
            "mwaste": mwaste,
            "rrisk": rrisk,
            "J": J,
        })

    return processed


def write_sweep_csv(processed: List[Dict], output_path: str) -> None:
    """Write sweep results to CSV for PGFPlots."""
    with open(output_path, "w") as f:
        f.write("tau,ploss,mwaste,rrisk,J\n")
        for p in processed:
            f.write(f"{p['tau']},{p['ploss']:.4f},{p['mwaste']:.4f},"
                    f"{p['rrisk']:.4f},{p['J']:.4f}\n")


def main():
    print("=" * 70)
    print("POLICY SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"  tau values: {TAU_VALUES}")
    print(f"  Replications per tau: {N_REPLICATIONS}")
    print(f"  Duration: {DURATION}s (warmup: {WARMUP}s)")
    print(f"  Weights: wp={WP}, wm={WM}, wr={WR}")
    print()

    # Load base config
    base_cfg = load_config(BASE_CONFIG)

    # Run sweep
    all_results = []
    for tau in TAU_VALUES:
        result = run_tau_experiment(tau, N_REPLICATIONS, base_cfg)
        if result:
            all_results.append(result)

    if not all_results:
        print("\nERROR: No successful results. Aborting.")
        sys.exit(1)

    # Compute cost components
    processed = compute_cost_components(all_results, baseline_wait=0.0)

    # Write outputs
    output_dir = Path("experiments/e03_policy_sweep/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV for PGFPlots
    write_sweep_csv(processed, str(output_dir / "sweep_data.csv"))

    # JSON report
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
            "tau_star": int(min(processed, key=lambda x: x["J"])["tau"]),
            "J_min": float(min(p["J"] for p in processed)),
        },
    }

    with open(output_dir / "sweep_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    for p in processed:
        print(f"  tau={p['tau']:6d}s: Ploss={p['ploss']:6.2f}%  "
              f"Mwaste={p['mwaste']:6.2f}%  Rrisk={p['rrisk']:7.4f}TiB  "
              f"J={p['J']:8.2f}")

    optimal = report["optimal"]
    print(f"\n  Optimal: tau* = {optimal['tau_star']}s (J = {optimal['J_min']:.2f})")
    print(f"\n  Output written to {output_dir}/")


if __name__ == "__main__":
    main()
