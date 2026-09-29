#!/usr/bin/env python3
"""
compare_solver_reps.py — paired per-replication diff of two campaign cells.

Stage 3 of the MMFQ solver-fix check. Point it at the committed (broken-solver)
cell and a freshly regenerated (corrected-solver) cell for the SAME (beta,policy)
and the SAME replication ids. Because the campaign uses common random numbers
(create_rng_manager(master_seed, replication)), rep k in both dirs sees the
identical workload realisation, so any KPI difference is the solver fix alone.

It reads only the campaign's own rep_<id:06d>.json sidecars (flat dicts with
recall_p95_s, tenant_p95_max_s, tenant_p95_gini, n_migrations, ...), matches by
the 'replication' field, prints per-rep deltas and median deltas, and flags any
adaptive recall_p95_s that leaves the Table II tolerance band (8 +- 3 s).

Usage (on a SLURM cluster):
    python compare_solver_reps.py runs/mt_b00_adaptive runs_fix/mt_b00_adaptive
    python compare_solver_reps.py OLD_DIR NEW_DIR --tol-recall 3 --table2 8
"""
from __future__ import annotations
import argparse, glob, json, math, os, statistics as st

KPIS = ["recall_p95_s", "tenant_p95_max_s", "tenant_p95_gini",
        "n_migrations", "n_mounts", "cache_hit_rate", "recall_p50_s"]


def load_cell(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "rep_*.json"))):
        if "._" in os.path.basename(f):
            continue
        with open(f) as fh:
            r = json.load(fh)
        out[int(r.get("replication", -1))] = r
    return out


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
    return float(st.median(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_dir", help="committed cell (broken solver), e.g. runs/mt_b00_adaptive")
    ap.add_argument("new_dir", help="regenerated cell (corrected solver), e.g. runs_fix/mt_b00_adaptive")
    ap.add_argument("--table2", type=float, default=8.0, help="locked Table II adaptive recall_p95 median")
    ap.add_argument("--tol-recall", type=float, default=3.0, help="Table II tolerance band half-width (s)")
    args = ap.parse_args()

    old, new = load_cell(args.old_dir), load_cell(args.new_dir)
    common = sorted(set(old) & set(new))
    if not common:
        print("No overlapping replication ids between the two directories.")
        print(f"  old ids: {sorted(old)[:10]}...  new ids: {sorted(new)[:10]}...")
        return 2
    print(f"Paired on {len(common)} replication(s): {common}\n")

    print(f"{'rep':>4}  {'recall_p95_s old->new':>26}  {'tenant_p95_max old->new':>26}  "
          f"{'n_migrations old->new':>24}")
    for k in common:
        o, n = old[k], new[k]
        def fmt(key, w=10):
            ov, nv = o.get(key), n.get(key)
            if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
                return f"{ov:>{w}.3f} -> {nv:<{w}.3f}"
            return f"{str(ov):>{w}} -> {str(nv):<{w}}"
        print(f"{k:>4}  {fmt('recall_p95_s'):>26}  {fmt('tenant_p95_max_s'):>26}  {fmt('n_migrations'):>24}")

    print("\nMEDIAN over paired reps (old -> new, delta):")
    moved = False
    for key in KPIS:
        mo, mn = med([old[k].get(key) for k in common]), med([new[k].get(key) for k in common])
        d = mn - mo
        tag = ""
        if key == "recall_p95_s":
            in_old = abs(mo - args.table2) <= args.tol_recall
            in_new = abs(mn - args.table2) <= args.tol_recall
            if in_old and not in_new:
                tag = "  <-- LEFT Table II band; re-run this cell"; moved = True
            elif in_new:
                tag = "  (within Table II 8+-3 band)"
        print(f"  {key:18} {mo:>12.4f} -> {mn:<12.4f}  delta={d:+.4f}{tag}")

    # Sanity gate: a healthy adaptive cell migrates and mounts. If the NEW runs
    # show no migration activity (or a perfect cache-hit rate) while the OLD ones
    # did, the comparison is invalid -- the new runs are broken or were generated
    # with a different workload/config, not "robust".
    def m(dirset, key):
        return med([dirset[k].get(key) for k in common])
    old_mig, new_mig = m(old, "n_migrations"), m(new, "n_migrations")
    old_mnt, new_mnt = m(old, "n_mounts"), m(new, "n_mounts")
    new_hit = m(new, "cache_hit_rate")
    broken = (
        (old_mig and old_mig > 10 and (not new_mig or new_mig < 0.05 * old_mig)) or
        (old_mnt and old_mnt > 0 and (not new_mnt or new_mnt == 0)) or
        (isinstance(new_hit, float) and new_hit >= 0.999)
    )
    if broken:
        print("\nVERDICT: INVALID COMPARISON — the new runs show no migration activity "
              "(n_migrations/n_mounts collapsed to ~0, cache_hit ~1.0).")
        print("  These runs did not exercise the policy normally; do NOT read recall_p95 "
              "as 'robust'. Re-generate the new cell with the SAME command as the "
              "original campaign, changing only the installed solver and --out-dir.")
        return 2

    print("\nVERDICT:", "KPIs moved — regenerate affected cells before submission."
          if moved else "Adaptive recall_p95 holds within the Table II band — results robust to the fix.")
    return 1 if moved else 0


if __name__ == "__main__":
    raise SystemExit(main())
