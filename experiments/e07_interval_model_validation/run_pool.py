#!/usr/bin/env python3
"""
e07 / run_pool.py -- validate the drive-pool queue approximation used by P.

Runs the multi-tenant stress configuration (128 tenants, 1 TB cache, working
set 64,000, cycle 300 s, 4 h + 30 min warm-up) at several static age
thresholds with file lifetimes enabled, and records every post-warm-up
drive job (recalls and write zones): arrival, wait and hold time.

For each replication the model's inputs are measured (recall and zone
arrival rates, first two moments of hold times) and the Lee-Longton M/G/c
mean wait is compared with the measured mean wait over all drive jobs and
over recalls only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main import load_config, run_replication      # noqa: E402
from models.interval_model import mgc_wait          # noqa: E402

LIFE = dict(s_inf=0.5, kind="exp", mean_s=1800.0)


def one(a, rep, seed, beta=0.0, lifetimes=True):
    cfg = load_config(str(ROOT / "config" / "default_config.yaml"))
    cfg["simulation"].update(sim_duration_s=14400, warmup_s=1800, seed=seed, n_replications=1)
    wl = cfg["workload"]
    wl.update(mode="synthetic", working_set_size=64000)
    wl.setdefault("multi_tenant", {}).update(enabled=True, n_tenants=128, noise_level=beta)
    if lifetimes:
        wl["lifetime"] = dict(enabled=True, **LIFE)
    cfg["disk_cache"]["capacity_bytes"] = 1_000_000_000_000
    cfg["policy_engine"].update(candidate_age_threshold_s=float(a), cycle_period_s=300.0)
    cfg["adaptive_policy"]["enabled"] = False
    cfg["tape_drives"]["stream_zone_writes"] = True
    s = run_replication(cfg, replication_id=rep)
    jobs = getattr(s, "drive_jobs", {"recall": [], "zone": []})
    span = 14400 - 1800
    out = dict(a=a, rep=rep)
    arrs = {}
    for k in ("recall", "zone"):
        j = np.array(jobs[k]) if jobs[k] else np.zeros((0, 4))
        j = j[j[:, 0] < 14400] if len(j) else j          # arrivals in the window
        arrs[k] = j
        out[f"n_{k}"] = int(len(j))
        out[f"lam_{k}"] = len(j) / span
        out[f"ES_{k}"] = float(j[:, 2].mean()) if len(j) else 0.0
        out[f"ES2_{k}"] = float((j[:, 2] ** 2).mean()) if len(j) else 0.0
        out[f"W_{k}"] = float(j[:, 1].mean()) if len(j) else 0.0
    lam = out["lam_recall"] + out["lam_zone"]
    ES = (out["lam_recall"] * out["ES_recall"] + out["lam_zone"] * out["ES_zone"]) / lam
    ES2 = (out["lam_recall"] * out["ES2_recall"] + out["lam_zone"] * out["ES2_zone"]) / lam
    out["util"] = lam * ES / 9.0
    out["W_model"] = float(mgc_wait(lam, ES, ES2, 9))
    allj = np.concatenate([arrs["recall"], arrs["zone"]])
    out["W_all"] = float(allj[:, 1].mean()) if len(allj) else 0.0
    out["recall_p95_s"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="300,1800,7200")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "pool_results.json"))
    args = ap.parse_args()
    rows = []
    for a in [float(x) for x in args.a.split(",")]:
        for r in range(args.reps):
            o = one(a, r, args.seed)
            rows.append(o)
            print(f"a={a:5.0f} rep={r}: util={o['util']:.2f} lam_r={o['lam_recall']:.4f} "
                  f"lam_z={o['lam_zone']:.4f} ES_r={o['ES_recall']:.1f} ES_z={o['ES_zone']:.1f} "
                  f"W_all={o['W_all']:.1f} W_model={o['W_model']:.1f} W_recall={o['W_recall']:.1f}",
                  flush=True)
    Path(args.out).write_text(json.dumps(dict(lifetime=LIFE, rows=rows), indent=1))


if __name__ == "__main__":
    main()
