"""
validation/calibration_optimize.py — Inverse Calibration via KS Minimisation
==========================================================================

Instead of fitting a parametric distribution to observed latencies (which
fails because queue wait creates non-LogNormal tails), we treat calibration
as an inverse problem:

    Find service-time parameters θ = (mount_mu, mount_sigma, seek_mu, seek_sigma)
    that minimise the Kolmogorov-Smirnov statistic:

        D(θ) = sup_x | F_sim(x; θ) - F_obs(x) |

where F_sim is the CDF of latencies from a short trace_replay run and F_obs
is the empirical CDF of the observed trace.

Method:
    1. Define objective: KS statistic after a short trace_replay run
    2. Use scipy.optimize.minimize with bounds on θ
    3. Validate final parameters with a longer run

This is computationally expensive (each objective evaluation = one simulation
replication). We use a short sim_duration_s (60–120s) for the search, then
verify with a longer run.
"""
from __future__ import annotations
import copy
import json
import math
import numpy as np
from scipy import optimize, stats as scipy_stats
from typing import Dict, List, Tuple, Callable

from main import run_replication
from validation.calibration import load_trace_latencies


# ---------------------------------------------------------------------------
# Objective Function: KS Distance Between Simulated and Observed
# ---------------------------------------------------------------------------

def ks_objective(
    theta: np.ndarray,
    base_cfg: dict,
    trace_path: str,
    sim_duration_s: int = 120,
    seed: int = 42,
) -> float:
    """
    Evaluate KS statistic for a given parameter vector theta.

    theta = [mount_mu, mount_sigma, seek_mu, seek_sigma]

    Returns:
        Combined KS statistic (recall + migrate, weighted).
        Higher = worse fit. We minimise this.
    """
    mount_mu, mount_sigma, seek_mu, seek_sigma = theta

    # Build config with current parameters
    cfg = copy.deepcopy(base_cfg)
    cfg["workload"]["mode"] = "trace_replay"
    cfg["workload"]["trace_file"] = trace_path
    cfg["simulation"]["sim_duration_s"] = sim_duration_s
    cfg["simulation"]["warmup_s"] = min(10, sim_duration_s // 10)
    cfg["simulation"]["seed"] = seed
    cfg["adaptive_policy"]["enabled"] = False

    tape_cfg = cfg["tape_drives"]
    tape_cfg["mount_time_lognormal_mu_s"] = float(mount_mu)
    tape_cfg["mount_time_lognormal_sigma_s"] = max(0.01, float(mount_sigma))
    tape_cfg["seek_time_lognormal_mu_s"] = float(seek_mu)
    tape_cfg["seek_time_lognormal_sigma_s"] = max(0.01, float(seek_sigma))

    try:
        stats = run_replication(cfg, replication_id=0)
    except Exception as exc:
        # Simulation failure → return large penalty
        return 10.0

    # Extract simulated latencies
    sim_recalls = stats.recall_latency._last_rep_samples or []
    sim_migrates = stats.migrate_latency._last_rep_samples or []

    # Load observed latencies
    obs_recalls, obs_migrates, _ = load_trace_latencies(trace_path)

    # Filter to same time window (trace events within sim_duration)
    obs_recalls = [r for r in obs_recalls if r <= sim_duration_s * 5]
    obs_migrates = [m for m in obs_migrates if m <= sim_duration_s * 5]

    # Compute KS statistics
    ks_recall = _ks_stat(sim_recalls, obs_recalls)
    ks_migrate = _ks_stat(sim_migrates, obs_migrates)

    # Weighted combination: recall is more important for user-visible SLAs
    combined = 0.6 * ks_recall + 0.4 * ks_migrate
    return float(combined)


def _ks_stat(sim: List[float], obs: List[float]) -> float:
    """KS statistic; returns 1.0 if insufficient data."""
    if len(sim) < 5 or len(obs) < 5:
        return 1.0
    stat, _ = scipy_stats.ks_2samp(sim, obs)
    return float(stat)


# ---------------------------------------------------------------------------
# Search Strategy
# ---------------------------------------------------------------------------

def calibrate_by_ks_minimisation(
    trace_path: str,
    base_config: dict,
    sim_duration_s: int = 120,
    n_random_starts: int = 8,
    seed: int = 42,
) -> Dict[str, any]:
    """
    Multi-start optimisation to find service-time parameters that minimise
    the KS distance between simulated and observed latency distributions.

    Returns:
        Dict with best_params, best_ks, diagnostics, all_trials.
    """
    obs_recalls, obs_migrates, _ = load_trace_latencies(trace_path)
    print(f"[calibrate] Observed recalls: {len(obs_recalls)}, migrates: {len(obs_migrates)}")
    print(f"[calibrate] Obs recall mean/median: {np.mean(obs_recalls):.2f}s / {np.median(obs_recalls):.2f}s")
    print(f"[calibrate] Obs migrate mean/median: {np.mean(obs_migrates):.2f}s / {np.median(obs_migrates):.2f}s")

    # Parameter bounds (LogNormal mu/sigma for mount and seek)
    # Recall residual mean ≈ 8.3s, median ≈ 2.3s after subtracting transfer
    # This suggests total mount+seek ≈ LogNormal(1.47, 1.15)
    # We split this between mount and seek.
    bounds = [
        (0.0, 4.0),    # mount_mu  → ~1s to ~55s median mount
        (0.05, 2.0),   # mount_sigma
        (0.0, 4.0),    # seek_mu   → ~1s to ~55s median seek
        (0.05, 2.0),   # seek_sigma
    ]

    best_result = None
    best_ks = float("inf")
    all_trials = []

    rng = np.random.default_rng(seed)

    for trial in range(n_random_starts):
        # Random start within bounds
        x0 = [
            rng.uniform(bounds[0][0], bounds[0][1]),
            rng.uniform(bounds[1][0], bounds[1][1]),
            rng.uniform(bounds[2][0], bounds[2][1]),
            rng.uniform(bounds[3][0], bounds[3][1]),
        ]
        print(f"\n[calibrate] Trial {trial + 1}/{n_random_starts}: x0 = {[round(v, 3) for v in x0]}")

        result = optimize.minimize(
            ks_objective,
            x0=x0,
            args=(base_config, trace_path, sim_duration_s, seed + trial),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 15, "disp": False},
        )

        ks_val = float(result.fun)
        all_trials.append(
            {
                "trial": trial,
                "x0": [float(v) for v in x0],
                "best_x": [float(v) for v in result.x],
                "ks": ks_val,
                "success": result.success,
                "nfev": result.nfev,
            }
        )

        print(f"  → KS = {ks_val:.4f}, params = {[round(v, 3) for v in result.x]}")

        if ks_val < best_ks:
            best_ks = ks_val
            best_result = result

    if best_result is None:
        raise RuntimeError("All optimisation trials failed")

    mount_mu, mount_sigma, seek_mu, seek_sigma = best_result.x

    print(f"\n[calibrate] Best KS = {best_ks:.4f}")
    print(f"  mount: mu={mount_mu:.3f}, sigma={mount_sigma:.3f}")
    print(f"  seek:  mu={seek_mu:.3f}, sigma={seek_sigma:.3f}")

    return {
        "best_params": {
            "mount_time_lognormal_mu_s": float(mount_mu),
            "mount_time_lognormal_sigma_s": float(mount_sigma),
            "seek_time_lognormal_mu_s": float(seek_mu),
            "seek_time_lognormal_sigma_s": float(seek_sigma),
        },
        "best_ks": best_ks,
        "all_trials": all_trials,
    }


# ---------------------------------------------------------------------------
# Config Application
# ---------------------------------------------------------------------------

def apply_calibrated_params(cfg: dict, params: dict) -> dict:
    """Return a new config dict with calibrated tape parameters applied."""
    cfg = copy.deepcopy(cfg)
    tape_cfg = cfg.setdefault("tape_drives", {})
    tape_cfg.update(params)
    return cfg


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import yaml

    if len(sys.argv) < 2:
        print("Usage: python -m validation.calibration_optimize <trace.jsonl> [config.yaml]")
        sys.exit(1)

    trace_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/default_config.yaml"

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    report = calibrate_by_ks_minimisation(
        trace_path=trace_path,
        base_config=base_cfg,
        sim_duration_s=120,
        n_random_starts=4,
    )

    print("\n=== Calibration Report ===")
    print(json.dumps(report, indent=2, default=str))
