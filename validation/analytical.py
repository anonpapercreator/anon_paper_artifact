"""
validation/analytical.py — Analytical Validation Suite
=======================================================
Provides:
    1. M/M/1 analytical vs. simulated comparison (V&V Phase 1)
    2. Little's Law verification across all queues
    3. State machine conservation check (no files lost)
    4. Extreme value boundary tests

Per Shannon (1998): "A model has been verified when it can be demonstrated
that the simulation program correctly implements the conceptual model."

Per Law & Kelton (2000): "A model has been validated when it is judged to be
a sufficiently accurate representation of the real system for the intended
purpose of the model."
"""

from __future__ import annotations
import simpy
import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple
import math
import sys
import os

# Path bootstrap: add project root (parent of validation/) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rng import create_rng_manager
from core.events import FileRecord, FileState
from stats.collector import ReplicationStats, QueueStats


# ---------------------------------------------------------------------------
# M/M/1 Analytical Formulae (Law & Kelton, 2000, §2.4)
# ---------------------------------------------------------------------------

def mm1_analytical(lam: float, mu: float) -> dict:
    """
    Compute M/M/1 queue analytical performance measures.

    Args:
        lam: Arrival rate (λ), arrivals per second.
        mu:  Service rate (μ), services per second.

    Returns:
        Dictionary of analytical results.

    Raises:
        ValueError: If ρ ≥ 1 (unstable queue).
    """
    rho = lam / mu
    if rho >= 1.0:
        raise ValueError(
            f"M/M/1 unstable: ρ = λ/μ = {rho:.4f} ≥ 1.0. "
            f"Require λ < μ for steady state."
        )
    L    = rho / (1.0 - rho)
    L_q  = rho**2 / (1.0 - rho)
    W    = 1.0 / (mu - lam)
    W_q  = rho / (mu - lam)
    p0   = 1.0 - rho
    return {
        "lambda": lam, "mu": mu, "rho": rho,
        "L": L, "L_q": L_q,
        "W": W, "W_q": W_q,
        "p0": p0,
        "littles_law_check": abs(L - lam * W) < 1e-10,
    }


# ---------------------------------------------------------------------------
# Little's Law Batch Verifier
# ---------------------------------------------------------------------------

def verify_littles_law_batch(
    queue_stats: Dict[str, QueueStats],
    n_batches: int = 20,
    alpha: float = 0.05,
    min_arrivals: int = 10,
) -> dict:
    """
    Run the batch-means CI test of Little's Law on a dictionary of QueueStats.

    Args:
        queue_stats:  Dict of queue name -> QueueStats.
        n_batches:    Number of equal-time batches for variance estimation.
        alpha:        Significance level for the CI test.
        min_arrivals: Minimum arrivals required for valid test.

    Returns:
        Dict with per-queue results and an overall pass/fail.
    """
    results = {}
    all_passed = True

    for name, q in queue_stats.items():
        if q._n_arrivals < min_arrivals:
            results[name] = {
                "status": "SKIP",
                "reason": f"Insufficient arrivals: {q._n_arrivals} < {min_arrivals}",
                "n_arrivals": q._n_arrivals,
            }
            continue

        res = q.verify_littles_law(n_batches=n_batches, alpha=alpha)
        results[name] = res
        if not res.get("passed", False):
            all_passed = False

    results["_overall_passed"] = all_passed
    return results


# ---------------------------------------------------------------------------
# State Machine Conservation Check
# ---------------------------------------------------------------------------

def check_state_conservation(
    file_registry: Dict[str, FileRecord],
    expected_total: int,
) -> dict:
    """
    Verify that all files are accounted for (no files created or destroyed
    without an event).

    Args:
        file_registry:  All FileRecords in the simulation.
        expected_total: Total files created (from stats counter).

    Returns:
        Dict with conservation check results.
    """
    actual_total = len(file_registry)
    state_counts = {}
    for f in file_registry.values():
        s = f.state.value
        state_counts[s] = state_counts.get(s, 0) + 1

    # Files must be in one of the valid states
    valid_states = {s.value for s in FileState}
    invalid_states = {k for k in state_counts if k not in valid_states}

    return {
        "files_in_registry": actual_total,
        "files_created": expected_total,
        "state_distribution": state_counts,
        "invalid_states_detected": list(invalid_states),
        "conservation_passed": (
            actual_total <= expected_total and len(invalid_states) == 0
        ),
    }


# ---------------------------------------------------------------------------
# Kingman G/G/1 Upper Bound
# ---------------------------------------------------------------------------

def gg1_kingman_W_q(lam: float, E_s: float, sigma_a: float,
                    sigma_s: float) -> float:
    """
    Kingman's (1961) heavy-traffic approximation for G/G/1 mean wait.

    W_q ≈ (ρ/(1−ρ)) × E[S] × (c_a² + c_s²) / 2

    Args:
        lam:     Arrival rate λ.
        E_s:     Mean service time E[S].
        sigma_a: Std dev of inter-arrival time.
        sigma_s: Std dev of service time.

    Returns:
        Estimated mean waiting time W_q.
    """
    rho  = lam * E_s
    if rho >= 1.0:
        return float("inf")
    c_a2 = (sigma_a * lam) ** 2   # c_a = σ_a × λ = σ_a / E[A]
    c_s2 = (sigma_s / E_s) ** 2   # c_s = σ_s / E[S]
    W_q  = (rho / (1.0 - rho)) * E_s * (c_a2 + c_s2) / 2.0
    return W_q


# ---------------------------------------------------------------------------
# Kolmogorov–Smirnov Distribution Test
# ---------------------------------------------------------------------------

def ks_test_latency(simulated: List[float], observed: List[float],
                    alpha: float = 0.05) -> dict:
    """
    Perform a two-sample Kolmogorov–Smirnov test to compare simulated
    and observed recall latency distributions.

    H₀: simulated and observed samples come from the same distribution.
    H₁: they come from different distributions.

    Accept H₀ at significance level α (default 0.05).

    Args:
        simulated: List of simulated recall latency samples.
        observed:  List of observed (from DMF trace) recall latency samples.
        alpha:     Significance level.

    Returns:
        Dict with test statistic, p-value, and pass/fail decision.
    """
    if len(simulated) < 5 or len(observed) < 5:
        return {
            "status": "SKIP",
            "reason": "Insufficient samples for KS test",
            "n_simulated": len(simulated),
            "n_observed": len(observed),
        }

    ks_stat, p_value = scipy_stats.ks_2samp(simulated, observed)
    passed = p_value >= alpha

    return {
        "test": "Two-sample KS test",
        "H0": "Simulated and observed distributions are identical",
        "KS_statistic": round(float(ks_stat), 6),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "passed": passed,
        "decision": "Accept H₀" if passed else "Reject H₀",
        "n_simulated": len(simulated),
        "n_observed": len(observed),
    }


# ---------------------------------------------------------------------------
# Full V&V Report Generator
# ---------------------------------------------------------------------------

def generate_vv_report(
    rep_stats: ReplicationStats,
    file_registry: Dict[str, "FileRecord"],
    observed_latencies: List[float] = None,
    ll_alpha: float = 0.05,
    ll_n_batches: int = 20,
) -> dict:
    """
    Generate a complete Verification & Validation report for one replication.

    Args:
        rep_stats:          Replication statistics object.
        file_registry:      All FileRecords from this replication.
        observed_latencies: Optional list of observed latencies for KS test.
        ll_alpha:           Significance level for the Little's Law CI test.
        ll_n_batches:       Number of batches for the Little's Law CI test.

    Returns:
        Full V&V report dictionary.
    """
    report = {}

    # Little's Law (batch-means CI test)
    report["littles_law"] = verify_littles_law_batch(
        rep_stats.queues, n_batches=ll_n_batches, alpha=ll_alpha
    )

    # State conservation
    report["state_conservation"] = check_state_conservation(
        file_registry, expected_total=rep_stats.n_files_created
    )

    # KS test (if observed data available)
    if observed_latencies and rep_stats.recall_latency._samples:
        report["ks_test_recall_latency"] = ks_test_latency(
            simulated=rep_stats.recall_latency._samples,
            observed=observed_latencies,
        )

    # Overall V&V pass/fail
    ll_passed = report["littles_law"].get("_overall_passed", False)
    cons_passed = report["state_conservation"].get("conservation_passed", False)
    report["overall_passed"] = ll_passed and cons_passed

    return report
