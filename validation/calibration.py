"""
validation/calibration.py — Trace-Driven Calibration Pipeline
===============================================================

Fits simulation service-time parameters from real DMF 7 trace observations
so that the simulated latency distribution matches the observed via KS test.

Methodology:
    1. Load trace.jsonl (normalised DMF 7 event stream)
    2. Separate latency samples by operation type (RECALL vs MIGRATE)
    3. For each operation, decompose total latency into:
         queue_wait + mount + seek + transfer + (unmount if volume switch)
    4. Fit LogNormal(mu, sigma) to each component via MLE
    5. Transfer time is deterministic: file_size / effective_rate
       → subtract from total to get residual (mount + seek + queue_wait)
    6. For the residual, fit LogNormal; use its parameters as mount/seek model
    7. Update config YAML with fitted parameters
    8. Run trace_replay simulation, compare simulated vs observed via KS test
    9. Iterate if KS rejects H₀

Reference:
    Aitchison, J. & Brown, J.A.C. (1957). The Lognormal Distribution.
    Law & Kelton (2000), §12.3: Input distribution selection and fitting.
"""
from __future__ import annotations
import json
import math
import numpy as np
from scipy import optimize, stats as scipy_stats
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Trace Loading and Component Extraction
# ---------------------------------------------------------------------------

def load_trace_latencies(trace_path: str) -> Tuple[List[float], List[float], List[Dict]]:
    """
    Load trace and return (recall_latencies, migrate_latencies, all_records).

    Returns:
        recalls: List of recall latencies (seconds)
        migrates: List of migrate latencies (seconds)
        records: All trace records for secondary analysis
    """
    recalls: List[float] = []
    migrates: List[float] = []
    records: List[Dict] = []

    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            lat = rec.get("latency_s")
            if lat is None or lat < 0:
                continue
            et = rec.get("event_type", "")
            if et == "RECALL":
                recalls.append(lat)
            elif et == "MIGRATE":
                migrates.append(lat)

    return recalls, migrates, records


# ---------------------------------------------------------------------------
# 2. Maximum Likelihood Estimation for LogNormal
# ---------------------------------------------------------------------------

def lognormal_mle(samples: List[float]) -> Tuple[float, float, Dict]:
    """
    Fit LogNormal distribution to positive samples via MLE.

    For X ~ LogNormal(mu, sigma):
        Y = ln(X) ~ Normal(mu, sigma^2)
        MLE: mu_hat = mean(ln(x)), sigma_hat = std(ln(x), ddof=0)

    Args:
        samples: Positive latency samples.

    Returns:
        (mu, sigma, diagnostics_dict)
    """
    arr = np.asarray([s for s in samples if s > 0], dtype=float)
    n = arr.size
    if n < 5:
        raise ValueError(f"Insufficient samples for MLE: {n} (need >= 5)")

    log_vals = np.log(arr)
    mu = float(np.mean(log_vals))
    sigma = float(np.std(log_vals, ddof=0))  # MLE uses population std

    # Goodness of fit: Shapiro-Wilk on log-transformed data (normality test)
    if n <= 5000:
        shapiro_stat, shapiro_p = scipy_stats.shapiro(log_vals[: min(n, 5000)])
    else:
        shapiro_stat, shapiro_p = scipy_stats.shapiro(log_vals[:5000])

    # Anderson-Darling on log-transformed data
    ad_stat, ad_crit, ad_sig = scipy_stats.anderson(log_vals, dist="norm")

    diagnostics = {
        "n_samples": n,
        "sample_mean": float(np.mean(arr)),
        "sample_median": float(np.median(arr)),
        "sample_std": float(np.std(arr, ddof=1)),
        "sample_cv": float(np.std(arr, ddof=1) / np.mean(arr)),
        "sample_p95": float(np.percentile(arr, 95)),
        "sample_p99": float(np.percentile(arr, 99)),
        "lognormal_mu": mu,
        "lognormal_sigma": sigma,
        "theoretical_mean": math.exp(mu + sigma**2 / 2),
        "theoretical_median": math.exp(mu),
        "shapiro_statistic": float(shapiro_stat),
        "shapiro_pvalue": float(shapiro_p),
        "anderson_statistic": float(ad_stat),
        "anderson_critical_5pct": float(ad_crit[2]) if len(ad_crit) > 2 else None,
        "lognormal_rejected": bool(ad_stat > ad_crit[2]) if len(ad_crit) > 2 else None,
    }
    return mu, sigma, diagnostics


def fit_lognormal_with_bounds(
    samples: List[float],
    mu_min: float = -5.0,
    mu_max: float = 10.0,
    sigma_min: float = 0.01,
    sigma_max: float = 5.0,
) -> Tuple[float, float, Dict]:
    """
    Constrained MLE for LogNormal with parameter bounds.

    Uses scipy.optimize.minimize on the negative log-likelihood with
    box constraints. Falls back to closed-form MLE if unconstrained optimum
    lies inside bounds.
    """
    arr = np.asarray([s for s in samples if s > 0], dtype=float)
    n = arr.size
    if n < 5:
        raise ValueError(f"Insufficient samples: {n}")

    log_vals = np.log(arr)
    mu_closed = float(np.mean(log_vals))
    sigma_closed = float(np.std(log_vals, ddof=0))

    # If closed-form is inside bounds, return it
    if (
        mu_min <= mu_closed <= mu_max
        and sigma_min <= sigma_closed <= sigma_max
    ):
        _, _, diag = lognormal_mle(samples)
        return mu_closed, sigma_closed, diag

    # Constrained optimisation
    def neg_log_likelihood(params):
        mu, sigma = params
        if sigma <= 0:
            return 1e18
        nll = (
            n * math.log(sigma)
            + np.sum((log_vals - mu) ** 2) / (2 * sigma**2)
            + np.sum(log_vals)
            + n * math.log(2 * math.pi) / 2
        )
        return float(nll)

    result = optimize.minimize(
        neg_log_likelihood,
        x0=[np.median(log_vals), max(np.std(log_vals, ddof=1), 0.1)],
        method="L-BFGS-B",
        bounds=[(mu_min, mu_max), (sigma_min, sigma_max)],
    )

    mu, sigma = float(result.x[0]), float(result.x[1])
    _, _, diag = lognormal_mle(samples)
    diag["lognormal_mu"] = mu
    diag["lognormal_sigma"] = sigma
    diag["theoretical_mean"] = math.exp(mu + sigma**2 / 2)
    diag["theoretical_median"] = math.exp(mu)
    return mu, sigma, diag


# ---------------------------------------------------------------------------
# 3. Component Decomposition (Subtract Deterministic Transfer)
# ---------------------------------------------------------------------------

def decompose_recall_latencies(
    trace_path: str,
    drive_rate_bps: float = 360_000_000,
    compression_ratio: float = 1.5,
) -> Tuple[List[float], List[float], List[float], Dict]:
    """
    Decompose observed recall latencies into (mount, seek, transfer, queue_wait).

    Strategy:
        transfer(file) = file_size / (drive_rate * compression)
        residual = observed_latency - transfer
        residual ≈ queue_wait + mount + seek + unmount

    Returns:
        transfer_times: Deterministic per-file transfer times
        residuals: residual latencies (mount+seek+queue)
        file_sizes: bytes for each observation
        summary: dict of statistics
    """
    transfer_times: List[float] = []
    residuals: List[float] = []
    file_sizes: List[int] = []

    with open(trace_path, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("event_type") != "RECALL":
                continue
            lat = rec.get("latency_s")
            size = rec.get("file_size_bytes", 0)
            if lat is None or size <= 0:
                continue

            eff_rate = drive_rate_bps * compression_ratio
            t_transfer = size / eff_rate
            residual = max(lat - t_transfer, 1e-6)  # floor at 1us

            transfer_times.append(t_transfer)
            residuals.append(residual)
            file_sizes.append(size)

    summary = {
        "n_decomposed": len(residuals),
        "mean_transfer": float(np.mean(transfer_times)),
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "p95_residual": float(np.percentile(residuals, 95)),
        "p99_residual": float(np.percentile(residuals, 99)),
        "max_residual": float(np.max(residuals)),
    }
    return transfer_times, residuals, file_sizes, summary


def decompose_migrate_latencies(
    trace_path: str,
    drive_rate_bps: float = 360_000_000,
    compression_ratio: float = 1.5,
) -> Tuple[List[float], List[float], List[int], Dict]:
    """
    Decompose observed migrate latencies.

    Migrate latency includes LS accrual wait + batch processing time.
    The residual is even more queue-dominated than recall.
    """
    transfer_times: List[float] = []
    residuals: List[float] = []
    file_sizes: List[int] = []

    with open(trace_path, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("event_type") != "MIGRATE":
                continue
            lat = rec.get("latency_s")
            size = rec.get("file_size_bytes", 0)
            if lat is None or size <= 0:
                continue

            eff_rate = drive_rate_bps * compression_ratio
            t_transfer = size / eff_rate
            residual = max(lat - t_transfer, 1e-6)

            transfer_times.append(t_transfer)
            residuals.append(residual)
            file_sizes.append(size)

    summary = {
        "n_decomposed": len(residuals),
        "mean_transfer": float(np.mean(transfer_times)),
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "p95_residual": float(np.percentile(residuals, 95)),
        "p99_residual": float(np.percentile(residuals, 99)),
        "max_residual": float(np.max(residuals)),
    }
    return transfer_times, residuals, file_sizes, summary


# ---------------------------------------------------------------------------
# 4. Fit Tape Service Time Components
# ---------------------------------------------------------------------------

def fit_tape_components(
    trace_path: str,
    drive_rate_bps: float = 360_000_000,
    compression_ratio: float = 1.5,
) -> Dict[str, Any]:
    """
    Fit all tape service-time components from trace data.

    Returns dict with:
        recall_total:  LogNormal MLE on raw recall latencies
        recall_residual: LogNormal MLE on (latency - transfer)
        migrate_total:   LogNormal MLE on raw migrate latencies
        migrate_residual: LogNormal MLE on (latency - transfer)
        transfer_stats:  Statistics on deterministic transfer times
        recommendations: Suggested config parameter updates
    """
    recalls, migrates, _ = load_trace_latencies(trace_path)

    # Raw fits
    r_mu, r_sigma, r_diag = lognormal_mle(recalls)
    m_mu, m_sigma, m_diag = lognormal_mle(migrates)

    # Decomposed fits
    _, r_residuals, _, r_decomp = decompose_recall_latencies(
        trace_path, drive_rate_bps, compression_ratio
    )
    _, m_residuals, _, m_decomp = decompose_migrate_latencies(
        trace_path, drive_rate_bps, compression_ratio
    )

    rr_mu, rr_sigma, rr_diag = lognormal_mle(r_residuals)
    mr_mu, mr_sigma, mr_diag = lognormal_mle(m_residuals)

    # Recommendations: map residual to (mount + seek) model
    # In the simulator:
    #   mount_time ~ LogNormal(mount_mu, mount_sigma)
    #   seek_time  ~ LogNormal(seek_mu, seek_sigma)
    #   Total mount+seek ~ sum of two independent LogNormals (no closed form)
    # Approximation: fit a single LogNormal to (mount+seek) residual.
    # Then set mount and seek parameters so their sum has this distribution.
    # Simple heuristic: split the mean 50/50 between mount and seek.

    recall_mount_seek_mu = rr_mu
    recall_mount_seek_sigma = rr_sigma

    # For config: mount gets ~60% of residual mean, seek ~40%
    mount_mu = recall_mount_seek_mu + math.log(0.6)
    seek_mu = recall_mount_seek_mu + math.log(0.4)
    mount_sigma = recall_mount_seek_sigma * 0.8
    seek_sigma = recall_mount_seek_sigma * 0.8

    return {
        "recall_total": {"mu": r_mu, "sigma": r_sigma, "diagnostics": r_diag},
        "recall_residual": {"mu": rr_mu, "sigma": rr_sigma, "diagnostics": rr_diag},
        "migrate_total": {"mu": m_mu, "sigma": m_sigma, "diagnostics": m_diag},
        "migrate_residual": {"mu": mr_mu, "sigma": mr_sigma, "diagnostics": mr_diag},
        "transfer_stats": r_decomp,
        "recommendations": {
            "mount_time_lognormal_mu_s": round(mount_mu, 4),
            "mount_time_lognormal_sigma_s": round(mount_sigma, 4),
            "seek_time_lognormal_mu_s": round(seek_mu, 4),
            "seek_time_lognormal_sigma_s": round(seek_sigma, 4),
            "native_rate_bytes_per_s": drive_rate_bps,
            "compression_ratio": compression_ratio,
        },
    }


# ---------------------------------------------------------------------------
# 5. Config Update
# ---------------------------------------------------------------------------

def update_config_from_calibration(
    config_path: str,
    fitted_params: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Update a YAML config file with fitted calibration parameters.

    Args:
        config_path: Path to existing config YAML.
        fitted_params: Output from fit_tape_components().
        output_path: Where to write updated config (default: overwrite input).

    Returns:
        Path to written config file.
    """
    import yaml

    path = Path(config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    rec = fitted_params["recommendations"]
    tape_cfg = cfg.setdefault("tape_drives", {})
    tape_cfg["mount_time_lognormal_mu_s"] = rec["mount_time_lognormal_mu_s"]
    tape_cfg["mount_time_lognormal_sigma_s"] = rec["mount_time_lognormal_sigma_s"]
    tape_cfg["seek_time_lognormal_mu_s"] = rec["seek_time_lognormal_mu_s"]
    tape_cfg["seek_time_lognormal_sigma_s"] = rec["seek_time_lognormal_sigma_s"]
    tape_cfg["native_rate_bytes_per_s"] = rec["native_rate_bytes_per_s"]

    out = Path(output_path or config_path)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return str(out)


# ---------------------------------------------------------------------------
# 6. KS Test Validation Loop
# ---------------------------------------------------------------------------

def validate_calibrated_model(
    trace_path: str,
    config_path: str,
    n_replications: int = 5,
    sim_duration_s: int = 600,
) -> Dict[str, Any]:
    """
    Run trace_replay with calibrated config and compare against observed via KS test.

    Returns:
        Dict with KS test results for recall and migrate latency distributions.
    """
    import yaml
    from main import run_replication
    from validation.analytical import ks_test_latency

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["workload"]["mode"] = "trace_replay"
    cfg["workload"]["trace_file"] = trace_path
    cfg["simulation"]["sim_duration_s"] = sim_duration_s
    cfg["simulation"]["warmup_s"] = min(60, sim_duration_s // 10)
    cfg["simulation"]["n_replications"] = 1
    cfg["adaptive_policy"]["enabled"] = False  # Calibrate baseline, not adaptive

    sim_recalls: List[float] = []
    sim_migrates: List[float] = []

    for rep in range(n_replications):
        cfg["simulation"]["seed"] = 42 + rep
        stats = run_replication(cfg, replication_id=rep)
        # Extract latencies from stats (they are in LatencyHistogram)
        # The histogram retains last replication's raw samples
        sim_recalls.extend(stats.recall_latency._last_rep_samples)
        sim_migrates.extend(stats.migrate_latency._last_rep_samples)

    # Load observed
    obs_recalls, obs_migrates, _ = load_trace_latencies(trace_path)

    return {
        "recall": ks_test_latency(sim_recalls, obs_recalls),
        "migrate": ks_test_latency(sim_migrates, obs_migrates),
        "n_simulated_recalls": len(sim_recalls),
        "n_simulated_migrates": len(sim_migrates),
        "n_observed_recalls": len(obs_recalls),
        "n_observed_migrates": len(obs_migrates),
    }


# ---------------------------------------------------------------------------
# 7. Full Calibration Orchestrator
# ---------------------------------------------------------------------------

def calibrate(
    trace_path: str,
    base_config_path: str = "config/default_config.yaml",
    output_config_path: Optional[str] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end calibration pipeline.

    Steps:
        1. Fit tape components from trace
        2. Update config YAML
        3. (Optional) Run trace_replay validation with KS test

    Returns:
        Full calibration report dict.
    """
    print(f"[calibrate] Loading trace: {trace_path}")
    recalls, migrates, _ = load_trace_latencies(trace_path)
    print(f"  Observed recalls: {len(recalls)}")
    print(f"  Observed migrates: {len(migrates)}")

    print("\n[calibrate] Fitting tape components...")
    fitted = fit_tape_components(trace_path)

    print("\n[calibrate] Fitted parameters:")
    for k, v in fitted["recommendations"].items():
        print(f"  {k}: {v}")

    print(f"\n[calibrate] Updating config: {base_config_path}")
    out_path = update_config_from_calibration(
        base_config_path, fitted, output_path=output_config_path
    )
    print(f"  Written to: {out_path}")

    validation = None
    if validate:
        print("\n[calibrate] Running trace_replay validation...")
        validation = validate_calibrated_model(trace_path, out_path)
        print(f"  Recall KS: {validation['recall']}")
        print(f"  Migrate KS: {validation['migrate']}")

    return {
        "fitted": fitted,
        "config_path": out_path,
        "validation": validation,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m validation.calibration <trace.jsonl> [config.yaml]")
        sys.exit(1)

    trace = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else "config/default_config.yaml"
    report = calibrate(trace, cfg)
    print("\n=== Calibration Report ===")
    print(json.dumps(report, indent=2, default=str))
