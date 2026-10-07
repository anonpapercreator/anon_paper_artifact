#!/usr/bin/env python3
"""e11 / run_cohort.py -- cohort (per-file) estimator for the model-versus-simulator sweep.

Why this exists (see PLAN.md, Addendum 2, and RESULTS.md). run_sweep.py divides the archive
write rate measured in a 4-hour window by the model's prediction computed from
the creation rate rho measured in the same window. The bytes written to tape in
the window were created up to a+p earlier, so numerator and denominator are sums
over different, partly overlapping sets of files. File sizes are lognormal with
sigma = 2.5, so these sums are dominated by a few large files and the ratio of
two partly independent sums is biased upward (E[X/Y] > E[X]/E[Y]). The bias grows
as the overlap shrinks, that is, with a and p.

The cohort estimator removes this. It follows every file created in the window
[warm, warm + window] until its outcome is known, and computes, over that one
set of files,
    c_sel = sum(s_i * 1[selected_i]) / sum(s_i)                 (C / rho)
    v_sel = sum(s_i * (min(sel_i, del_i) - c_i)) / sum(s_i)     (V / rho, protection at selection, A4)
    v_tap = sum(s_i * (min(tape_i, del_i) - c_i)) / sum(s_i)    (V / rho, protection at tape write)
Selection time, deletion time, and creation phase relative to the run schedule
are independent of the file size, so given the sizes each estimator has
expectation equal to the model's per-byte quantity: (G(a+p)-G(a))/p for c_sel
and (H(a+p)-H(a))/p for v_sel. The sizes cancel. v_tap is compared with the
model at a + delta, delta the byte-weighted mean selection-to-tape delay of the
same cohort, which is the e07 convention.

The simulation runs until warm + window + a + p + margin so that every cohort
file is resolved (selected or deleted, and its tape write finished). Unresolved
files are counted and reported.

usage: python3 run_cohort.py --dist E|L1|L2|A [--iat exponential] [--reps 50] --out FILE
"""
from __future__ import annotations
import argparse, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run_sweep import DISTS  # same lifetime definitions

_FILES: list = []
_PATCHED = False


def _patch():
    """Record every written file and the time each file is selected by a policy run."""
    global _PATCHED
    if _PATCHED:
        return
    import components.filesystem as fsm
    orig_write = fsm.Filesystem.handle_write
    orig_mig = fsm.PolicyEngine._migrate_file

    def handle_write(self, f):
        _FILES.append(f)
        yield from orig_write(self, f)

    def migrate_file(self, f):
        f.selected_at = self._env.now
        yield from orig_mig(self, f)

    fsm.Filesystem.handle_write = handle_write
    fsm.PolicyEngine._migrate_file = migrate_file
    _PATCHED = True


def one(args):
    a, p, rep, window_s, margin_s, seed, life, iat = args
    _patch()
    _FILES.clear()
    from main import load_config, run_replication
    from models.interval_model import Lifetime, tape_write_rate, unprotected_volume
    cfg = load_config(str(ROOT / "config" / "default_config.yaml"))
    warm = a + p + 3600.0
    t0, t1 = warm, warm + window_s
    end = t1 + a + p + margin_s
    cfg["simulation"].update(sim_duration_s=end, warmup_s=warm, seed=seed, n_replications=1)
    wl = cfg["workload"]
    wl.update(mode="synthetic", read_fraction=0.0, diurnal_enabled=False)
    if iat:
        wl["iat_distribution"] = iat
    wl["lifetime"] = dict(enabled=True, **life)
    cfg["policy_engine"].update(candidate_age_threshold_s=float(a), cycle_period_s=float(p))
    cfg["adaptive_policy"]["enabled"] = False
    cfg["tape_drives"]["stream_zone_writes"] = True
    run_replication(cfg, replication_id=rep)

    S = N = Ssel = Sv_sel = Sv_tap = Sdelay = Sarch = 0.0
    unresolved = 0
    for f in _FILES:
        c = f.created_at
        if not (t0 <= c < t1):
            continue
        s = float(f.size_bytes)
        sel = getattr(f, "selected_at", None)
        dl = f.deleted_at
        tape = f.migrated_at
        N += 1
        S += s
        if sel is None and dl is None:
            unresolved += 1
            continue
        if sel is not None:
            Ssel += s
            if tape is None:
                unresolved += 1
                continue
            Sarch += s
            Sdelay += s * (tape - sel)
        t_sel = min(x for x in (sel, dl) if x is not None)
        t_tap = min(x for x in (tape, dl) if x is not None)
        Sv_sel += s * (t_sel - c)
        Sv_tap += s * (t_tap - c)
    delta = Sdelay / Sarch if Sarch > 0 else 0.0
    L = Lifetime(**life)
    return dict(a=a, p=p, rep=rep, n_files=int(N), bytes=S, unresolved=unresolved,
                c_sel=Ssel / S, c_model=tape_write_rate(L, 1.0, a, p),
                v_sel=Sv_sel / S, v_model=unprotected_volume(L, 1.0, a, p),
                v_tap=Sv_tap / S, v_model_delta=unprotected_volume(L, 1.0, a + delta, p),
                delta=delta, rho=S / window_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True, choices=sorted(DISTS))
    ap.add_argument("--a", default="0,300,600,1800,3600,7200")
    ap.add_argument("--p", default="60,300,600,1800,3600")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--window", type=float, default=4 * 3600.0)
    ap.add_argument("--margin", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--iat", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    life = DISTS[args.dist]
    jobs = [(float(a), float(p), r, args.window, args.margin, args.seed, life, args.iat)
            for a in args.a.split(",") for p in args.p.split(",") for r in range(args.reps)]
    t0 = time.time()
    rows = []
    with Pool(args.workers, maxtasksperchild=25) as pool:
        for i, row in enumerate(pool.imap(one, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"{args.dist}: {i}/{len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
    out = dict(estimator="cohort", dist=args.dist, iat=args.iat or "config default", lifetime=life,
               window_s=args.window, margin_s=args.margin, seed=args.seed, reps=args.reps,
               rows=rows, elapsed_s=time.time() - t0)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
