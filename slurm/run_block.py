#!/usr/bin/env python3
"""
slurm/run_block.py -- run a disjoint block of replication IDs.

Each SLURM array task runs replication IDs [rep_start, rep_start+rep_count).
Seeding is create_rng_manager(master_seed, replication), a pure deterministic
function of (master_seed, replication), so blocks run under ONE master seed are
globally independent AND reproducible. Aggregating the per-replication
ReplicationStats objects later reproduces the serial result exactly (verified).

For every replication this writes:
    <out_dir>/rep_<id:06d>.pkl    pickled ReplicationStats  (authoritative; for aggregate.py)
    <out_dir>/rep_<id:06d>.json   scalar KPIs               (human scan + paired_compare.py)

Usage (one array task):
    python slurm/run_block.py --config config/default_config.yaml \
        --rep-start 40 --rep-count 10 --out-dir runs/adaptive \
        --policy adaptive --mode trace_replay --trace-file traces/new_trace.jsonl \
        --seed 42 --duration 14400 --warmup 1800
"""
from __future__ import annotations
import sys, os, json, pickle, argparse, time
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

from main import load_config, run_replication  # reuse the exact serial run path


def apply_overrides(cfg: dict, a: argparse.Namespace) -> None:
    sim = cfg["simulation"]
    if a.seed is not None:     sim["seed"] = a.seed
    if a.duration is not None: sim["sim_duration_s"] = a.duration
    if a.warmup is not None:   sim["warmup_s"] = a.warmup
    if a.cycle_period is not None:
        cfg.setdefault("policy_engine", {})["cycle_period_s"] = float(a.cycle_period)
    if a.cache_bytes is not None:
        cfg.setdefault("disk_cache", {})["capacity_bytes"] = int(a.cache_bytes)
    if a.working_set_size is not None:
        cfg.setdefault("workload", {})["working_set_size"] = int(a.working_set_size)
    if a.agg_lambda is not None:
        cfg.setdefault("workload", {}).setdefault("multi_tenant", {})[
            "aggregate_lambda_files_per_s"] = float(a.agg_lambda)

    # Policy: static == adaptive advisor OFF; adaptive == full advisor ON;
    # hwm == reactive watermark-only advisor (ablation arm B).
    cfg.setdefault("adaptive_policy", {})
    if a.policy == "adaptive":
        cfg["adaptive_policy"]["enabled"] = True
        cfg["adaptive_policy"]["advisor"] = "full"
    elif a.policy == "hwm":
        cfg["adaptive_policy"]["enabled"] = True
        cfg["adaptive_policy"]["advisor"] = "hwm"
    elif a.policy == "static":
        cfg["adaptive_policy"]["enabled"] = False

    wl = cfg["workload"]
    if a.mode is not None:
        wl["mode"] = a.mode
    if a.trace_file is not None:
        wl["trace_file"] = a.trace_file
    if a.mode == "multi_tenant":
        # run_replication selects MT when mode != trace_replay and MT.enabled
        wl["mode"] = "synthetic"
        mt = wl.setdefault("multi_tenant", {})
        mt["enabled"] = True
        if a.beta is not None:
            mt["noise_level"] = float(a.beta)        # heterogeneity dial beta in [0,1]
        if a.n_tenants is not None:
            mt["n_tenants"] = int(a.n_tenants)


def per_rep_kpis(stats) -> dict:
    """Scalar KPIs for the JSON sidecar (no heavy arrays)."""
    stats.recall_latency.flush_replication()
    p = stats.recall_latency
    g = lambda lst: (lst[-1] if lst else None)
    rec = {
        "replication":   stats.replication_id,
        "n_migrations":  stats.n_migrations,
        "n_recalls":     stats.n_recalls,
        "n_mounts":      getattr(stats, "n_mounts", None),
        "cache_hit_rate": stats.cache_hit_rate,
        "recall_p50_s":  g(getattr(p, "_rep_p50", [])),
        "recall_p95_s":  g(getattr(p, "_rep_p95", [])),
        "recall_p99_s":  g(getattr(p, "_rep_p99", [])),
    }
    # Multi-tenant per-tenant tail fairness, if present.
    tlat = getattr(stats, "tenant_recall_latency", {}) or {}
    if tlat:
        p95s = []
        for h in tlat.values():
            h.flush_replication()
            v = g(getattr(h, "_rep_p95", []))
            if v is not None:
                p95s.append(v)
        if p95s:
            import numpy as np
            p95s = np.array(sorted(p95s), dtype=float)
            n = len(p95s)
            gini = (np.sum((2*np.arange(1, n+1) - n - 1) * p95s) /
                    (n * np.sum(p95s))) if p95s.sum() > 0 else 0.0
            rec.update(n_tenants=n, tenant_p95_min_s=float(p95s.min()),
                       tenant_p95_max_s=float(p95s.max()), tenant_p95_gini=float(gini))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rep-start", type=int, required=True)
    ap.add_argument("--rep-count", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--policy", choices=["static", "adaptive", "hwm", "config"],
                    default="config",
                    help="'config' leaves adaptive_policy.enabled as the file sets it; "
                         "'hwm' runs the reactive watermark-only advisor (ablation arm B).")
    ap.add_argument("--mode", choices=["synthetic", "trace_replay", "multi_tenant"], default=None)
    ap.add_argument("--trace-file", default=None)
    ap.add_argument("--beta", type=float, default=None,
                    help="multi_tenant heterogeneity dial (noise_level in [0,1]); MT mode only.")
    ap.add_argument("--n-tenants", type=int, default=None,
                    help="multi_tenant tenant count; MT mode only.")
    ap.add_argument("--cycle-period", type=float, default=None,
                    help="migration cycle period (s). Production is 21600 (6h); the "
                         "multi-tenant stress test overrides this to exercise the policy "
                         "within the observation window. Trace replay leaves it unset.")
    ap.add_argument("--cache-bytes", type=int, default=None,
                    help="disk cache capacity (bytes). MT stress test sizes this so the "
                         "synthetic working set overflows; trace replay leaves it unset "
                         "(production 889 TiB from config).")
    ap.add_argument("--working-set-size", type=int, default=None,
                    help="base working-set size (distinct files). Larger values make the "
                         "aggregate working set exceed the cache, producing recall load. "
                         "MT stress test only.")
    ap.add_argument("--agg-lambda", type=float, default=None,
                    help="aggregate multi-tenant file-arrival rate (files/s). Overrides "
                         "the single-stream-derived default; the contention dial for the "
                         "MT campaign.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--warmup", type=float, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    apply_overrides(cfg, a)
    os.makedirs(a.out_dir, exist_ok=True)

    rep_ids = range(a.rep_start, a.rep_start + a.rep_count)
    print(f"[run_block] policy={a.policy} mode={cfg['workload']['mode']} "
          f"seed={cfg['simulation']['seed']} dur={cfg['simulation']['sim_duration_s']}s "
          f"reps={a.rep_start}..{a.rep_start + a.rep_count - 1}", flush=True)

    for rid in rep_ids:
        t = time.time()
        stats = run_replication(cfg, replication_id=rid)
        with open(os.path.join(a.out_dir, f"rep_{rid:06d}.pkl"), "wb") as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(a.out_dir, f"rep_{rid:06d}.json"), "w") as f:
            json.dump(per_rep_kpis(stats), f, indent=2, default=str)
        print(f"[run_block]   rep {rid} done in {time.time() - t:.1f}s "
              f"(migrations={stats.n_migrations}, recalls={stats.n_recalls})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
