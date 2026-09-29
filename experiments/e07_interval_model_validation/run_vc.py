#!/usr/bin/env python3
"""
e07 / run_vc.py -- validate the interval model's V and C against DESCASSI.

Write-only synthetic workload (read_fraction 0, so a file's age runs from its
creation, which is the model's assumption), diurnal modulation off, file
lifetimes enabled. For each (a, p) on a grid, n replications; per replication
we measure over [warmup, duration]:

  rho_hat  bytes created / window                      [bytes/s]
  V_hat    time-average bytes with no tape copy        [bytes]
  C_hat    bytes whose migration completed / window    [bytes/s]
  delta    byte-weighted mean migration pipeline latency [s]

and compare with the model, evaluated at rho_hat:
  C_model = rho * (G(a+p) - G(a)) / p                 (selection at age a+pU)
  V_model = rho * (H(a+d+p) - H(a+d)) / p,  d = delta  (protection at completion)

Output: JSON with per-replication ratios and a summary (mean ratio, 95% t-CI).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main import load_config, run_replication                      # noqa: E402
from models.interval_model import Lifetime, tape_write_rate, unprotected_volume  # noqa: E402

LIFE = dict(s_inf=0.5, kind="exp", mean_s=1800.0)


def one(a, p, rep, window_s, seed):
    cfg = load_config(str(ROOT / "config" / "default_config.yaml"))
    warm = a + p + 3600.0
    cfg["simulation"].update(sim_duration_s=warm + window_s, warmup_s=warm,
                             seed=seed, n_replications=1)
    wl = cfg["workload"]
    wl.update(mode="synthetic", read_fraction=0.0, diurnal_enabled=False)
    wl["lifetime"] = dict(enabled=True, **LIFE)
    cfg["policy_engine"].update(candidate_age_threshold_s=float(a), cycle_period_s=float(p))
    cfg["adaptive_policy"]["enabled"] = False
    cfg["tape_drives"]["stream_zone_writes"] = True
    s = run_replication(cfg, replication_id=rep)
    w = s.window
    span = w["t_end"] - w["t_start"]
    rho = w["bytes_created"] / span
    n = max(w["n_migrations"], 1)
    # V is byte-weighted, so the pipeline delay that shifts it is the
    # byte-weighted mean (large files take longer to copy).
    delta = w["bytes_x_pipeline_s"] / max(w["bytes_archived"], 1)
    life = Lifetime(**LIFE)
    Cm = tape_write_rate(life, rho, a, p)
    Vm = unprotected_volume(life, rho, a + delta, p)
    return dict(a=a, p=p, rep=rep, rho=rho, delta=delta,
                V_hat=w["unprotected_avg_bytes"], V_model=Vm,
                C_hat=w["bytes_archived"] / span, C_model=Cm,
                garbage_rate=w["bytes_tape_garbage"] / span)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="0,600,1800,3600,7200")
    ap.add_argument("--p", default="60,600")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--window", type=float, default=4 * 3600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "vc_results.json"))
    args = ap.parse_args()
    rows = []
    for a in [float(x) for x in args.a.split(",")]:
        for p in [float(x) for x in args.p.split(",")]:
            for r in range(args.reps):
                row = one(a, p, r, args.window, args.seed)
                rows.append(row)
                print(f"a={a:6.0f} p={p:4.0f} rep={r}: V {row['V_hat']/row['V_model']:.3f}  "
                      f"C {row['C_hat']/row['C_model']:.3f}  delta={row['delta']:.1f}s", flush=True)
    summary = []
    for a in sorted({r["a"] for r in rows}):
        for p in sorted({r["p"] for r in rows}):
            sub = [r for r in rows if r["a"] == a and r["p"] == p]
            out = dict(a=a, p=p, n=len(sub))
            for k in ("V", "C"):
                x = np.array([r[f"{k}_hat"] / r[f"{k}_model"] for r in sub])
                h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
                out[f"{k}_ratio_mean"], out[f"{k}_ratio_ci"] = float(x.mean()), [float(x.mean() - h), float(x.mean() + h)]
            summary.append(out)
            print(out)
    Path(args.out).write_text(json.dumps(dict(lifetime=LIFE, rows=rows, summary=summary), indent=1))


if __name__ == "__main__":
    main()
