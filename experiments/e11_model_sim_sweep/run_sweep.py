#!/usr/bin/env python3
"""e11 / run_sweep.py -- model-versus-simulator sweep (see PLAN.md).

Same measurement as e07/run_vc.py (one replication per call to `one`), with the
lifetime distribution as a parameter and replications spread over worker
processes. Results are written per distribution, one row per replication.

usage: python3 run_sweep.py --dist E|L1|L2|A [--a ...] [--p ...] [--reps 50] [--workers 2] --out FILE
"""
from __future__ import annotations
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

DISTS = {
    "E":  dict(s_inf=0.5, kind="exp", mean_s=1800.0),
    "L1": dict(s_inf=0.5, kind="lognormal", mean_s=1800.0, sigma=1.0),
    "L2": dict(s_inf=0.5, kind="lognormal", mean_s=1800.0, sigma=2.0),
    "A":  dict(kind="empirical", curve=str(HERE / "survival_a.csv")),
}


def one(args):
    a, p, rep, window_s, seed, life, iat = args
    from main import load_config, run_replication
    from models.interval_model import Lifetime, tape_write_rate, unprotected_volume
    cfg = load_config(str(ROOT / "config" / "default_config.yaml"))
    warm = a + p + 3600.0
    cfg["simulation"].update(sim_duration_s=warm + window_s, warmup_s=warm,
                             seed=seed, n_replications=1)
    wl = cfg["workload"]
    wl.update(mode="synthetic", read_fraction=0.0, diurnal_enabled=False)
    if iat:
        wl["iat_distribution"] = iat
    wl["lifetime"] = dict(enabled=True, **life)
    cfg["policy_engine"].update(candidate_age_threshold_s=float(a), cycle_period_s=float(p))
    cfg["adaptive_policy"]["enabled"] = False
    cfg["tape_drives"]["stream_zone_writes"] = True
    s = run_replication(cfg, replication_id=rep)
    w = s.window
    span = w["t_end"] - w["t_start"]
    rho = w["bytes_created"] / span
    delta = w["bytes_x_pipeline_s"] / max(w["bytes_archived"], 1)
    L = Lifetime(**life)
    Cm = tape_write_rate(L, rho, a, p)
    Vm = unprotected_volume(L, rho, a + delta, p)
    return dict(a=a, p=p, rep=rep, rho=rho, delta=delta,
                V_hat=w["unprotected_avg_bytes"], V_model=Vm,
                C_hat=w["bytes_archived"] / span, C_model=Cm,
                garbage_rate=w["bytes_tape_garbage"] / span)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True, choices=sorted(DISTS))
    ap.add_argument("--a", default="0,300,600,1800,3600,7200")
    ap.add_argument("--p", default="60,300,600,1800,3600")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--window", type=float, default=4 * 3600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--iat", default="", help="override iat_distribution, e.g. exponential")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    life = DISTS[args.dist]
    jobs = [(float(a), float(p), r, args.window, args.seed, life, args.iat)
            for a in args.a.split(",") for p in args.p.split(",") for r in range(args.reps)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap(one, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"{args.dist}: {i}/{len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
    out = dict(dist=args.dist, iat=args.iat or "config default", lifetime=life, window_s=args.window, seed=args.seed,
               reps=args.reps, rows=rows, elapsed_s=time.time() - t0)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
