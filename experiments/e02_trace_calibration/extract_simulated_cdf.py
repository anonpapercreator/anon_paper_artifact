#!/usr/bin/env python3
"""
Extract simulated recall latencies from trace replay and generate CDF data.
Uses the same pattern as validation.calibrate_trace_replay.evaluate_params.
"""
import sys
import os
import json
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import simpy
from core.rng import RNGManager
from stats.collector import ReplicationStats
from components.tape_subsystem import TapeSubsystem
from components.filesystem import Filesystem
from workload.generator import DMFTraceReader
from validation.calibrate_trace_replay import build_trace_replay_cfg, pre_populate_filesystem


def run_and_extract(trace_path: str) -> dict:
    """Run trace replay and return simulated latencies."""
    cfg = build_trace_replay_cfg(trace_path)

    sim_cfg = cfg["simulation"]
    duration = sim_cfg["sim_duration_s"]
    warmup_s = sim_cfg.get("warmup_s", duration * 0.1)
    master_seed = sim_cfg["seed"]
    chunk_s = sim_cfg.get("progress_chunk_s", 60)

    rng = RNGManager(master_seed=master_seed, replication=0)
    stats = ReplicationStats(replication_id=0, warmup_s=warmup_s)
    env = simpy.Environment()

    tape = TapeSubsystem(env, cfg, rng, stats)
    fs = Filesystem(env, cfg, tape, rng, stats)
    pre_populate_filesystem(trace_path, fs, tape)

    reader = DMFTraceReader(
        trace_path,
        offline_fraction=cfg["workload"].get("trace_offline_fraction", 0.30),
        warm_fraction=cfg["workload"].get("trace_warm_fraction", 0.20),
    )
    reader.load()
    env.process(reader.run(env, fs))

    def stats_snapshot():
        while True:
            interval = sim_cfg.get("trace_interval_s", 60)
            yield env.timeout(interval)
            stats.record_disk_usage(env.now, fs.disk.usage_fraction)
            for q in stats.queues.values():
                q.snapshot(env.now)

    env.process(stats_snapshot())

    t = 0.0
    while t < duration:
        t = min(t + chunk_s, duration)
        env.run(until=t)

    stats.finalise(sim_end_time=duration)

    # Extract recall samples
    sim_recalls = stats.recall_latency._last_rep_samples or []
    sim_migrates = stats.migrate_latency._last_rep_samples or []

    return {
        "sim_recalls": sim_recalls,
        "sim_migrates": sim_migrates,
        "n_files": len(fs.policy._file_registry) if hasattr(fs, "policy") else 0,
    }


def main(trace_path: str = "trace.jsonl"):
    print("Extracting simulated recall latencies from trace replay...")
    print(f"  Trace: {trace_path}")

    result = run_and_extract(trace_path)
    sim_recalls = result["sim_recalls"]
    print(f"  Simulated recalls: {len(sim_recalls)}")
    print(f"  Simulated migrates: {len(result['sim_migrates'])}")

    if len(sim_recalls) < 10:
        print("  ERROR: Too few simulated recalls. Check trace replay configuration.")
        return

    # Load observed
    from validation.calibration import load_trace_latencies
    obs_recalls, _, _ = load_trace_latencies(trace_path)
    print(f"  Observed recalls: {len(obs_recalls)}")

    # Compute component breakdowns
    def breakdown(samples):
        arr = np.array(samples)
        return {
            "fast_pct": float(np.sum(arr < 5) / len(arr) * 100),
            "warm_pct": float(np.sum((arr >= 5) & (arr < 20)) / len(arr) * 100),
            "cold_pct": float(np.sum(arr >= 20) / len(arr) * 100),
            "p50": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    sim_bd = breakdown(sim_recalls)
    obs_bd = breakdown(obs_recalls)

    print(f"\n  Simulated breakdown:")
    print(f"    Fast: {sim_bd['fast_pct']:.1f}%, Warm: {sim_bd['warm_pct']:.1f}%, Cold: {sim_bd['cold_pct']:.1f}%")
    print(f"    P50: {sim_bd['p50']:.2f}s, P95: {sim_bd['p95']:.2f}s")

    print(f"\n  Observed breakdown:")
    print(f"    Fast: {obs_bd['fast_pct']:.1f}%, Warm: {obs_bd['warm_pct']:.1f}%, Cold: {obs_bd['cold_pct']:.1f}%")
    print(f"    P50: {obs_bd['p50']:.2f}s, P95: {obs_bd['p95']:.2f}s")

    # KS test
    ks = float(scipy_stats.ks_2samp(sim_recalls, obs_recalls)[0])
    print(f"\n  KS distance: {ks:.4f}")

    # Generate CDF CSV
    output_dir = Path("experiments/e02_trace_calibration/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = np.linspace(0, 60, 200)

    def interp_cdf(samples, grid):
        sorted_vals = np.sort(samples)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        return np.interp(grid, sorted_vals, cdf, left=0.0, right=1.0)

    sim_cdf = interp_cdf(sim_recalls, grid)
    obs_cdf = interp_cdf(obs_recalls, grid)

    with open(output_dir / "cdf_comparison.csv", "w") as f:
        f.write("latency_s,simulated_cdf,observed_cdf\n")
        for i in range(len(grid)):
            f.write(f"{grid[i]:.3f},{sim_cdf[i]:.6f},{obs_cdf[i]:.6f}\n")

    # Save raw samples
    with open(output_dir / "simulated_recalls.json", "w") as f:
        json.dump(sim_recalls, f)

    # Report
    report = {
        "trace_path": trace_path,
        "n_observed_recalls": len(obs_recalls),
        "n_simulated_recalls": len(sim_recalls),
        "ks_distance": ks,
        "simulated_breakdown": sim_bd,
        "observed_breakdown": obs_bd,
        "parameters": {
            "mount_mu": 3.689,
            "mount_sigma": 0.3,
            "seek_mu": 3.5,
            "seek_sigma": 1.0,
        },
    }

    with open(output_dir / "calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Output written to {output_dir}/")
    print(f"    cdf_comparison.csv       — CDF data for figure")
    print(f"    simulated_recalls.json   — Raw simulated latencies")
    print(f"    calibration_report.json  — Summary report")


if __name__ == "__main__":
    trace = sys.argv[1] if len(sys.argv) > 1 else "trace.jsonl"
    main(trace)
