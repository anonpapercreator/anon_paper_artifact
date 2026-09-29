#!/usr/bin/env python3
"""
slurm/aggregate.py -- pool all per-replication pickles into one report.

Loads every rep_*.pkl in --in-dir (in replication-ID order for determinism),
feeds each into a fresh SimulationStats via the SAME ingest_replication path
main.py uses, and emits produce_report(). The result is identical to a single
serial run over the same replication IDs (verified: blocked == serial).

Outputs:
    <in-dir>/aggregate_report.json   full report (mean + 95% bootstrap CI per KPI)
    <in-dir>/per_rep_kpis.csv        one row per replication (from the JSON sidecars)

Usage:
    python slurm/aggregate.py --in-dir runs/adaptive --policy adaptive \
        --seed 42 --duration 14400 --trace traces/new_trace.jsonl
"""
from __future__ import annotations
import sys, os, json, glob, pickle, argparse, csv
def _add_repo_to_path():
    """Locate the DESCASSI repo (the dir containing main.py) robustly, so this
    script works whether it lives inside the repo or in a separate runner dir.
    Priority: $DESCASSI_ROOT, then cwd, then the script's parent-parent."""
    import os, sys
    cands = [os.environ.get("DESCASSI_ROOT"), os.getcwd(),
             os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "main.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    raise SystemExit(
        "[error] cannot find the DESCASSI repo (no main.py in $DESCASSI_ROOT, "
        "cwd, or the script's parent). Set DESCASSI_ROOT=/path/to/DESCASSI_2026_CODE "
        "or run from inside the repo.")
_add_repo_to_path()

from stats.collector import SimulationStats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--policy", default="config")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--trace", default=None)
    a = ap.parse_args()

    pkls = sorted(glob.glob(os.path.join(a.in_dir, "rep_*.pkl")))
    if not pkls:
        print(f"[aggregate] no rep_*.pkl in {a.in_dir}", file=sys.stderr)
        return 1

    agg = SimulationStats()
    rep_ids = []
    for p in pkls:                     # sorted filenames == sorted replication IDs
        with open(p, "rb") as f:
            rs = pickle.load(f)
        agg.ingest_replication(rs)
        rep_ids.append(rs.replication_id)

    report = agg.produce_report()
    report["metadata"] = {
        "n_replications": len(pkls),
        "replication_ids": [min(rep_ids), max(rep_ids)],
        "policy": a.policy,
        "master_seed": a.seed,
        "sim_duration_s": a.duration,
        "trace_file": a.trace,
        "aggregation": "parallel blocks -> single ingest pass (== serial)",
    }
    out_json = os.path.join(a.in_dir, "aggregate_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Flat per-rep CSV from the JSON sidecars (handy for plotting / paired tests).
    sidecars = sorted(glob.glob(os.path.join(a.in_dir, "rep_*.json")))
    rows = [json.load(open(s)) for s in sidecars]
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(a.in_dir, "per_rep_kpis.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    k = report.get("kpis", {})
    def mean(d, *ks):
        for x in ks:
            d = d.get(x, {}) if isinstance(d, dict) else {}
        return d.get("mean") if isinstance(d, dict) else None
    print(f"[aggregate] {len(pkls)} reps pooled -> {out_json}")
    print(f"  recall P95 mean = {mean(k,'recall_latency_s','p95')}  "
          f"cache_hit_rate = {mean(k,'cache_hit_rate')}  "
          f"n_migrations = {mean(k,'n_migrations')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
