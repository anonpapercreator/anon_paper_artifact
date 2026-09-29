#!/usr/bin/env python3
"""
experiments/e01_mm1_baseline/run_mm1.py
=====================================
Standalone M/M/1 baseline using SimPy with the simulator's RNG infrastructure.

Purpose: Verify DESCASSI queueing mechanics against closed-form M/M/1.
Theory (Law & Kelton, 2000):
    rho = lambda / mu
    Wq  = rho / (mu - lambda)
    L   = lambda * W
    Lq  = lambda * Wq

Parameters: lambda=0.8, mu=1.0  =>  rho=0.8, Wq=4.0s, W=5.0s
"""
from __future__ import annotations
import sys
import os
import json
import numpy as np
import simpy
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class MM1Result:
    """Results from one M/M/1 replication."""
    replication_id: int
    seed: int
    lambda_rate: float
    mu_rate: float
    rho: float
    n_arrivals: int
    n_served: int
    n_dropped: int
    W_mean: float           # mean sojourn time
    Wq_mean: float          # mean queue wait
    W_std: float
    Wq_std: float
    L_mean: float           # mean in system (from Little's Law)
    Lq_mean: float          # mean in queue (from Little's Law)
    littles_error_pct: float


@dataclass
class MM1Replication:
    """Track samples for one replication."""
    W_samples: List[float] = field(default_factory=list)
    Wq_samples: List[float] = field(default_factory=list)
    Lq_samples: List[float] = field(default_factory=list)
    n_served: int = 0
    n_dropped: int = 0


def run_mm1_replication(
    env: simpy.Environment,
    lambda_rate: float,
    mu_rate: float,
    duration: float,
    warmup: float,
    rng: np.random.Generator,
    server: simpy.Resource,
    rep: MM1Replication,
) -> None:
    """SimPy process: generate Poisson arrivals and serve them."""

    def arrival_process():
        """Generate arrivals according to Poisson process."""
        while env.now < duration:
            inter_arrival = rng.exponential(1.0 / lambda_rate)
            yield env.timeout(inter_arrival)
            if env.now < duration:
                env.process(service_process(env.now))

    def service_process(arrival_time: float):
        """Request server, record wait, serve, record sojourn."""
        with server.request() as req:
            yield req
            wait_time = env.now - arrival_time
            service_time = rng.exponential(1.0 / mu_rate)
            yield env.timeout(service_time)
            sojourn_time = env.now - arrival_time

            # Only record after warmup
            if env.now >= warmup:
                rep.W_samples.append(sojourn_time)
                rep.Wq_samples.append(wait_time)
                rep.n_served += 1

    env.process(arrival_process())


def run_experiment(
    lambda_rate: float = 0.8,
    mu_rate: float = 1.0,
    n_replications: int = 30,
    duration: float = 3600.0,
    warmup: float = 600.0,
    base_seed: int = 42,
    output_dir: str = "experiments/e01_mm1_baseline/results",
) -> List[MM1Result]:
    """Run M/M/1 experiment with n_replications."""

    theoretical_rho = lambda_rate / mu_rate
    theoretical_W = 1.0 / (mu_rate - lambda_rate)
    theoretical_Wq = theoretical_rho / (mu_rate - lambda_rate)
    theoretical_L = lambda_rate * theoretical_W
    theoretical_Lq = lambda_rate * theoretical_Wq

    print(f"M/M/1 Baseline Experiment")
    print(f"  lambda = {lambda_rate:.4f}, mu = {mu_rate:.4f}")
    print(f"  rho (theoretical) = {theoretical_rho:.4f}")
    print(f"  Wq (theoretical)  = {theoretical_Wq:.4f}s")
    print(f"  W  (theoretical)  = {theoretical_W:.4f}s")
    print(f"  Replications: {n_replications}")
    print(f"  Duration: {duration}s (warmup: {warmup}s)")
    print()

    results: List[MM1Result] = []

    for rep_id in range(n_replications):
        seed = base_seed + rep_id * 1000
        rng = np.random.default_rng(seed)
        env = simpy.Environment()
        server = simpy.Resource(env, capacity=1)
        rep = MM1Replication()

        run_mm1_replication(env, lambda_rate, mu_rate, duration, warmup, rng, server, rep)
        env.run(until=duration)

        if rep.n_served > 0:
            W_mean = np.mean(rep.W_samples)
            Wq_mean = np.mean(rep.Wq_samples)
            W_std = np.std(rep.W_samples, ddof=1)
            Wq_std = np.std(rep.Wq_samples, ddof=1)
            L_mean = lambda_rate * W_mean
            Lq_mean = lambda_rate * Wq_mean
            littles_error = abs(L_mean - len(rep.W_samples) / duration * W_mean) / max(L_mean, 1e-12) * 100
        else:
            W_mean = Wq_mean = W_std = Wq_std = L_mean = Lq_mean = 0.0
            littles_error = 100.0

        result = MM1Result(
            replication_id=rep_id,
            seed=base_seed + rep_id,
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            rho=theoretical_rho,
            n_arrivals=rep.n_served,
            n_served=rep.n_served,
            n_dropped=rep.n_dropped,
            W_mean=W_mean,
            Wq_mean=Wq_mean,
            W_std=W_std,
            Wq_std=Wq_std,
            L_mean=L_mean,
            Lq_mean=Lq_mean,
            littles_error_pct=littles_error,
        )
        results.append(result)

        print(f"  Rep {rep_id+1:2d}/{n_replications}: Wq={Wq_mean:.3f}s  "
              f"W={W_mean:.3f}s  served={rep.n_served}")

    # Summary statistics
    Wq_values = np.array([r.Wq_mean for r in results])
    W_values = np.array([r.W_mean for r in results])
    L_values = np.array([r.L_mean for r in results])
    Lq_values = np.array([r.Lq_mean for r in results])

    Wq_mean_overall = np.mean(Wq_values)
    Wq_std_overall = np.std(Wq_values, ddof=1)
    Wq_ci_half = 1.96 * Wq_std_overall / np.sqrt(n_replications)

    W_mean_overall = np.mean(W_values)
    W_std_overall = np.std(W_values, ddof=1)
    W_ci_half = 1.96 * W_std_overall / np.sqrt(n_replications)

    Wq_error_pct = (Wq_mean_overall - theoretical_Wq) / theoretical_Wq * 100
    W_error_pct = (W_mean_overall - theoretical_W) / theoretical_W * 100

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Theoretical:  rho={theoretical_rho:.4f}  Wq={theoretical_Wq:.4f}s  W={theoretical_W:.4f}s")
    print(f"  Simulated:    Wq={Wq_mean_overall:.4f}s ± {Wq_ci_half:.4f}s  "
          f"(error={Wq_error_pct:+.2f}%)")
    print(f"                W ={W_mean_overall:.4f}s ± {W_ci_half:.4f}s  "
          f"(error={W_error_pct:+.2f}%)")
    print(f"  L  (theoretical) = {theoretical_L:.4f}  "
          f"simulated = {np.mean(L_values):.4f}")
    print(f"  Lq (theoretical) = {theoretical_Lq:.4f}  "
          f"simulated = {np.mean(Lq_values):.4f}")
    print(f"  Little's Law error: mean={np.mean([r.littles_error_pct for r in results]):.2f}%")
    print("=" * 60)

    # Write outputs
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "theoretical": {
            "rho": theoretical_rho,
            "W": theoretical_W,
            "Wq": theoretical_Wq,
            "L": theoretical_L,
            "Lq": theoretical_Lq,
        },
        "simulated": {
            "W_mean": float(W_mean_overall),
            "W_std": float(W_std_overall),
            "W_ci95_halfwidth": float(W_ci_half),
            "Wq_mean": float(Wq_mean_overall),
            "Wq_std": float(Wq_std_overall),
            "Wq_ci95_halfwidth": float(Wq_ci_half),
            "L_mean": float(np.mean(L_values)),
            "Lq_mean": float(np.mean(Lq_values)),
            "W_error_pct": float(W_error_pct),
            "Wq_error_pct": float(Wq_error_pct),
        },
        "replications": [
            {
                "rep_id": r.replication_id,
                "seed": r.seed,
                "n_served": r.n_served,
                "Wq_mean": r.Wq_mean,
                "W_mean": r.W_mean,
                "littles_error_pct": r.littles_error_pct,
            }
            for r in results
        ],
        "metadata": {
            "lambda": lambda_rate,
            "mu": mu_rate,
            "n_replications": n_replications,
            "duration_s": duration,
            "warmup_s": warmup,
            "base_seed": base_seed,
        },
    }

    with open(Path(output_dir) / "mm1_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # CSV summary
    with open(Path(output_dir) / "mm1_replications.csv", "w") as f:
        f.write("rep_id,seed,n_served,Wq_mean,W_mean,littles_error_pct\n")
        for r in results:
            f.write(f"{r.replication_id},{r.seed},{r.n_served},"
                    f"{r.Wq_mean:.6f},{r.W_mean:.6f},{r.littles_error_pct:.4f}\n")

    print(f"\n  Output written to {output_dir}/")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="M/M/1 baseline experiment")
    parser.add_argument("--lambda", type=float, default=0.8, dest="lambda_rate")
    parser.add_argument("--mu", type=float, default=1.0, dest="mu_rate")
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--warmup", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/e01_mm1_baseline/results")
    args = parser.parse_args()

    run_experiment(
        lambda_rate=args.lambda_rate,
        mu_rate=args.mu_rate,
        n_replications=args.replications,
        duration=args.duration,
        warmup=args.warmup,
        base_seed=args.seed,
        output_dir=args.output_dir,
    )
