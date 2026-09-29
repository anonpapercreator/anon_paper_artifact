"""
main.py — DMF 7 DES — CLI Entry Point and Simulation Orchestrator
=================================================================
Usage:
    python main.py run --config config/default_config.yaml
    python main.py run --config my_config.yaml --replications 5 --duration 3600
    python main.py validate-config --config config/default_config.yaml
    python main.py trace-template --output trace_template.csv
    python main.py analytical-baseline --arrival-rate 1.0 --service-rate 2.0
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Path bootstrap — MUST be the very first executable code.
# os.path.abspath resolves __file__ from the OS, not the shell cwd.
# This fixes "No module named 'core'" on every platform.
# ---------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simpy
import yaml
import json
import csv
import math
import time as wallclock
from pathlib import Path
from typing import Optional, Callable
import click

# Project imports (safe after path bootstrap)
from core.rng import create_rng_manager
from core.events import FileState
from components.filesystem import Filesystem
from components.tape_subsystem import TapeSubsystem
from workload.generator import SyntheticWorkloadGenerator, DMFTraceReader
from stats.collector import SimulationStats, ReplicationStats
from components.adaptive_policy import AdaptivePolicyAdvisor


# ---------------------------------------------------------------------------
# Terminal helpers — plain print, no Rich required
# ---------------------------------------------------------------------------

TERM_WIDTH = 78   # conservative default; works in all terminals

def _fmt_s(s: float) -> str:
    """Format seconds as Hh Mm Ss."""
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _bar(fraction: float, width: int = 25) -> str:
    """Simple ASCII progress bar."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(fraction * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_status(
    rep_id: int, n_reps: int,
    sim_time: float, duration: float,
    migrations: int, recalls: int,
    cache_pct: str, disk_pct: float,
    n_files: int,
    rep_elapsed: float,
    eta_s: float,
):
    """
    Print two overwriting status lines to stdout.
    Line 1: replication bar + simulated-time bar
    Line 2: live counters
    Uses \\r to overwrite in place so the terminal doesn't scroll.
    """
    rep_frac  = rep_id / max(n_reps, 1)
    sim_frac  = sim_time / max(duration, 1)

    rep_bar   = _bar(rep_frac, 20)
    sim_bar   = _bar(sim_frac, 20)

    eta_str   = _fmt_s(eta_s) if eta_s > 0 else "--"

    line1 = (
        f"  Reps {rep_bar} {rep_id}/{n_reps}  |  "
        f"SimTime {sim_bar} {sim_frac*100:5.1f}%  "
        f"elapsed={_fmt_s(rep_elapsed)}  ETA={eta_str}"
    )
    line2 = (
        f"  files={n_files:,}  migrated={migrations:,}  "
        f"recalled={recalls:,}  cache={cache_pct}  disk={disk_pct:.1f}%"
    )

    # Pad to terminal width so previous longer lines are fully overwritten
    line1 = line1.ljust(TERM_WIDTH)
    line2 = line2.ljust(TERM_WIDTH)

    # \033[F moves cursor up one line; \r goes to line start
    sys.stdout.write(f"\r{line1}\n\r{line2}")
    sys.stdout.flush()


def _separator(char: str = "-", width: int = TERM_WIDTH) -> None:
    print(char * width)


def _section(title: str) -> None:
    print(f"\n{'=' * TERM_WIDTH}")
    print(f"  {title}")
    print('=' * TERM_WIDTH)


# ---------------------------------------------------------------------------
# Configuration Loading and Validation
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    required = [
        "simulation", "disk_cache", "interconnect", "data_movers",
        "tape_robot", "tape_drives", "volume_groups", "library_server",
        "policy_engine", "workload",
    ]
    for s in required:
        if s not in cfg:
            raise ValueError(f"Missing required config section: '{s}'")

    sim = cfg["simulation"]
    if sim["n_replications"] < 1:
        raise ValueError("n_replications must be >= 1")
    if sim["n_replications"] < 30:
        print(f"  WARNING: n_replications={sim['n_replications']} < 30. "
              "95% CI validity requires >= 30 (CLT).")
    if sim["sim_duration_s"] <= 0:
        raise ValueError("sim_duration_s must be > 0")
    if sim.get("warmup_s", 0) >= sim["sim_duration_s"]:
        raise ValueError("warmup_s must be < sim_duration_s")

    disk = cfg["disk_cache"]
    if not (0 < disk["lwm_fraction"] < disk["hwm_fraction"] < 1.0):
        raise ValueError(
            f"Require 0 < LWM < HWM < 1.0. "
            f"Got LWM={disk['lwm_fraction']}, HWM={disk['hwm_fraction']}"
        )

    drives = cfg["tape_drives"]
    robot  = cfg["tape_robot"]
    if drives["n_drives"] != robot["n_drives"]:
        raise ValueError(
            f"tape_drives.n_drives ({drives['n_drives']}) != "
            f"tape_robot.n_drives ({robot['n_drives']})"
        )

    for key in ("cartridge_load_s_mu", "cartridge_load_s_sigma",
                "cartridge_unload_s_mu", "cartridge_unload_s_sigma"):
        if key not in robot:
            raise ValueError(f"tape_robot missing required field '{key}'")
        if robot[key] < 0:
            raise ValueError(f"tape_robot.{key} must be >= 0; got {robot[key]}")
    for key in ("cartridge_load_s_mu", "cartridge_unload_s_mu"):
        if robot[key] <= 0:
            raise ValueError(
                f"tape_robot.{key} is a linear-space mean (seconds) and must be > 0; "
                f"got {robot[key]}"
            )

    wl = cfg["workload"]
    if wl["iat_distribution"] == "pareto" and wl["iat_pareto_alpha"] <= 1.0:
        raise ValueError(
            f"Pareto alpha must be > 1.0. Got {wl['iat_pareto_alpha']}"
        )
    if not (0.0 <= wl["read_fraction"] <= 1.0):
        raise ValueError(f"read_fraction must be in [0,1]. Got {wl['read_fraction']}")


# ---------------------------------------------------------------------------
# Single Replication Runner
# ---------------------------------------------------------------------------

def run_replication(
    cfg: dict,
    replication_id: int,
    progress_callback: Optional[Callable] = None,
) -> ReplicationStats:
    """
    Run one simulation replication.

    SimPy is advanced in chunks of `progress_chunk_s` simulated seconds.
    After each chunk, progress_callback is called so the display can update.
    This is the only correct way to get live feedback from SimPy —
    env.run(until=duration) is a single blocking call with no hooks.
    """
    sim_cfg     = cfg["simulation"]
    duration    = sim_cfg["sim_duration_s"]
    warmup_s    = sim_cfg.get("warmup_s", duration * 0.1)
    master_seed = sim_cfg["seed"]
    chunk_s     = sim_cfg.get("progress_chunk_s", 60)

    rng   = create_rng_manager(master_seed=master_seed, replication=replication_id)
    stats = ReplicationStats(replication_id=replication_id, warmup_s=warmup_s)
    env   = simpy.Environment()

    tape = TapeSubsystem(env, cfg, rng, stats)
    fs   = Filesystem(env, cfg, tape, rng, stats)

    # Adaptive policy advisor (closes MMFQ/IOM/CTMC loop). advisor="hwm"
    # selects the reactive watermark-only controller (ablation arm B);
    # the default "full" runs the MMFQ/IOM/CTMC model.
    if cfg.get("adaptive_policy", {}).get("enabled", False):
        advisor_cls = AdaptivePolicyAdvisor
        if cfg["adaptive_policy"].get("advisor", "full") == "hwm":
            from components.hwm_advisor import HWMOnlyAdvisor
            advisor_cls = HWMOnlyAdvisor
        advisor = advisor_cls(
            env, cfg, fs.policy, tape.ls, stats, disk=fs.disk
        )
        env.process(advisor.run())

    wl_cfg = cfg["workload"]
    if wl_cfg["mode"] == "trace_replay" and wl_cfg.get("trace_file"):
        reader = DMFTraceReader(wl_cfg["trace_file"])
        reader.load()
        env.process(reader.run(env, fs))
    elif wl_cfg.get("multi_tenant", {}).get("enabled", False):
        # Multi-tenant noisy-neighbour mode (see workload/multi_tenant.py).
        from workload.multi_tenant import MultiTenantWorkloadGenerator
        mt_gen = MultiTenantWorkloadGenerator(env, cfg, rng, fs)
        env.process(mt_gen.run())
    else:
        generator = SyntheticWorkloadGenerator(env, wl_cfg, rng, fs)
        env.process(generator.run())

    def stats_snapshot():
        while True:
            interval = sim_cfg.get("trace_interval_s", 60)
            yield env.timeout(interval)
            stats.record_disk_usage(env.now, fs.disk.usage_fraction)
            for q in stats.queues.values():
                q.snapshot(env.now)

    env.process(stats_snapshot())

    # ------------------------------------------------------------------
    # Stepped execution: advance SimPy in chunks, yield to Python
    # between each so the progress display can update.
    # ------------------------------------------------------------------
    t = 0.0
    while t < duration:
        t = min(t + chunk_s, duration)
        env.run(until=t)
        if progress_callback:
            progress_callback(
                sim_time=env.now,
                duration=duration,
                migrations=stats.n_migrations,
                recalls=stats.n_recalls,
                cache_hit_rate=stats.cache_hit_rate,
                disk_pct=fs.disk.usage_fraction * 100.0,
                n_files=stats.n_files_created,
            )

    # Interval-model observables over the arrival window [warmup, duration],
    # captured before the drain so the window is the same for every quantity.
    stats.window = {
        "t_start": float(stats.warmup_s), "t_end": float(env.now),
        "unprotected_avg_bytes": stats.unprotected_time_average(env.now),
        "bytes_created": stats.lt_bytes_created,
        "bytes_archived": stats.lt_bytes_archived,
        "bytes_tape_garbage": stats.lt_bytes_tape_garbage,
        "bytes_deleted_unarchived": stats.lt_bytes_deleted_unarchived,
        "n_migrations": stats.n_migrations,
        "bytes_x_pipeline_s": stats.lt_bytes_x_pipeline_s,
        "migrate_latency_sum_s": float(sum(getattr(stats.migrate_latency, "_samples", []) or [])),
    }

    # ------------------------------------------------------------------
    # End-of-arrivals drain.
    #
    # The arrival window closes at `duration`, but migrations parked in the
    # LS accrual buffer and recalls/writes still queued on the tape drives
    # have NOT completed. Flush the accrual buffer and advance the clock until
    # the tape subsystem is quiescent, so late requests complete instead of
    # being censored at the cutoff. Without this, a bursty trace (e.g. a
    # checkpoint dump near the end of the span) has its tail mechanically
    # clipped and the migrate load severely under-counted.
    #
    # `duration` therefore acts as the arrival-admission window; the run then
    # drains however long the backlog takes (bounded by `max_drain_s`).
    # Chunked env.run(until=...) is used so perpetual helper processes
    # (stats snapshots, the adaptive advisor) do not prevent termination.
    # ------------------------------------------------------------------
    env.process(tape.ls.drain())
    max_drain_s = sim_cfg.get("max_drain_s", 10.0 * duration)
    drain_cap = duration + max_drain_s
    while t < drain_cap:
        t += chunk_s
        env.run(until=t)
        accrual_empty = not getattr(tape.ls, "_accrual", [])
        # Quiescence = no in-flight or queued tape work. Use the outstanding-work
        # counter, NOT drive-resource occupancy: recalls dispatched during a burst
        # are pending processes that have not yet requested a drive, so they do not
        # appear in drive.count/queue and would be lost if we stopped on idle drives.
        no_outstanding = getattr(tape.ls, "outstanding", 0) == 0
        if accrual_empty and no_outstanding:
            break

    stats.finalise(sim_end_time=env.now)
    return stats


# ---------------------------------------------------------------------------
# Output Writers
# ---------------------------------------------------------------------------

def write_report_json(report: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report, f, indent=2, default=str)


def write_report_csv(report: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(v, f"{prefix}{k}.")
        else:
            rows.append({"metric": prefix.rstrip("."), "value": obj})

    flatten(report.get("kpis", {}))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)


def _fmt(v) -> str:
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.4g}"
    return str(v) if v is not None else "n/a"


def print_summary_table(report: dict) -> None:
    _section("SIMULATION RESULTS")
    kpis = report.get("kpis", {})

    col_w = [32, 10, 12, 12, 6]
    header = (
        f"  {'KPI':<{col_w[0]}} {'Mean':>{col_w[1]}} "
        f"{'CI95 Lo':>{col_w[2]}} {'CI95 Hi':>{col_w[3]}} {'N':>{col_w[4]}}"
    )
    print(header)
    _separator()

    def row(name, d):
        if not isinstance(d, dict):
            return
        mean = _fmt(d.get("mean"))
        lo   = _fmt(d.get("ci95_lower"))
        hi   = _fmt(d.get("ci95_upper"))
        n    = str(d.get("n_replications", "?"))
        print(f"  {name:<{col_w[0]}} {mean:>{col_w[1]}} "
              f"{lo:>{col_w[2]}} {hi:>{col_w[3]}} {n:>{col_w[4]}}")

    rl = kpis.get("recall_latency_s", {})
    ml = kpis.get("migrate_latency_s", {})
    row("Recall latency P50 (s)",    rl.get("p50", {}))
    row("Recall latency P95 (s)",    rl.get("p95", {}))
    row("Recall latency P99 (s)",    rl.get("p99", {}))
    row("Migrate latency P50 (s)",   ml.get("p50", {}))
    row("Migrate latency P95 (s)",   ml.get("p95", {}))
    _separator("-")
    row("Cache hit rate",            kpis.get("cache_hit_rate", {}))
    row("Mean disk usage (%)",       kpis.get("mean_disk_usage_pct", {}))
    row("Total migrations",          kpis.get("n_migrations", {}))
    row("Total recalls",             kpis.get("n_recalls", {}))
    row("Tape mounts",               kpis.get("n_tape_mounts", {}))
    _separator()

    ll = report.get("littles_law_validation", {})
    if ll:
        header = "LITTLE'S LAW VALIDATION"
        print(f"\n  {header}")
        _separator("-")
        print(f"  {'Queue':<20} {'Pass Rate':>10} {'Reps':>6}")
        _separator("-")
        for qname, res in ll.items():
            pr = res.get("pass_rate", float("nan"))
            pr_s = f"{pr:.1%}" if isinstance(pr, float) and not math.isnan(pr) else "n/a"
            flag = "  OK" if isinstance(pr, float) and pr >= 0.9 else "  FAIL"
            print(f"  {qname:<20} {pr_s:>10} {res.get('n_replications','?'):>6}{flag}")
        _separator("-")

    mt = report.get("multi_tenant", {})
    if mt:
        print("\n  MULTI-TENANT WORKLOAD SUMMARY")
        _separator("-")
        print(f"  Tenants observed       : {mt.get('n_tenants_observed', 'n/a')}")
        gi = mt.get('p95_recall_gini', float('nan'))
        print(f"  P95 recall Gini        : {gi:.4f}"
              if isinstance(gi, float) and not math.isnan(gi)
              else "  P95 recall Gini        : n/a")
        pmin = mt.get('p95_recall_min', float('nan'))
        pmax = mt.get('p95_recall_max', float('nan'))
        if isinstance(pmin, float) and isinstance(pmax, float):
            print(f"  Per-tenant P95 range   : "
                  f"{pmin:.3f}s — {pmax:.3f}s  (ratio {pmax/max(pmin,1e-6):.1f}×)")
        _separator("-")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """HPE DMF 7 Discrete Event Simulator."""
    pass


@cli.command()
@click.option("--config", "-c", default="config/default_config.yaml",
              help="YAML configuration file.")
@click.option("--replications", "-r", default=None, type=int,
              help="Override n_replications.")
@click.option("--seed", "-s", default=None, type=int,
              help="Override master RNG seed.")
@click.option("--output-dir", "-o", default=None,
              help="Override output directory.")
@click.option("--duration", "-d", default=None, type=float,
              help="Override sim duration (seconds).")
@click.option("--chunk", default=60, type=float, show_default=True,
              help="Simulated seconds between progress updates.")
@click.option("--allow-partial", is_flag=True, default=False,
              help="Continue past replication errors instead of failing fast. "
                   "Confidence intervals in the report will be flagged as "
                   "partial and may be suppressed.")
def run(config, replications, seed, output_dir, duration, chunk, allow_partial):
    """Run the DMF 7 DES with live progress display.

    By default any exception raised by a replication aborts the entire
    run so a bad parameter or regression is surfaced immediately rather
    than silently producing a report over a subset of replications. Pass
    ``--allow-partial`` to preserve the previous best-effort behaviour;
    the report will record ``failed_replications`` and the aggregate
    CIs will be marked as computed over a partial sample.
    """
    _section("HPE DMF 7 Discrete Event Simulator")

    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    if replications is not None:
        cfg["simulation"]["n_replications"] = replications
    if seed is not None:
        cfg["simulation"]["seed"] = seed
    if output_dir is not None:
        cfg["simulation"]["output_dir"] = output_dir
    if duration is not None:
        cfg["simulation"]["sim_duration_s"] = duration
        # Re-validate so that a CLI duration smaller than the configured
        # warmup is caught immediately rather than producing a silent
        # zero-event run.
        try:
            validate_config(cfg)
        except ValueError as e:
            print(f"  ERROR after --duration override: {e}")
            sys.exit(1)
    cfg["simulation"]["progress_chunk_s"] = chunk

    n_reps  = cfg["simulation"]["n_replications"]
    out_dir = cfg["simulation"]["output_dir"]
    dur_s   = cfg["simulation"]["sim_duration_s"]

    print(f"  Config       : {config}")
    print(f"  Replications : {n_reps}")
    print(f"  Sim duration : {_fmt_s(dur_s)}  ({dur_s}s)")
    print(f"  Seed         : {cfg['simulation']['seed']}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Chunk size   : {chunk}s simulated per display refresh")
    _separator()

    # Print blank lines that the \r overwriting will use
    # (2 lines: bar line + counter line)
    print()
    print()

    agg_stats   = SimulationStats()
    rep_results = []
    failed_reps: list[dict] = []
    wall_start  = wallclock.time()

    for rep_id in range(n_reps):
        rep_wall_start = wallclock.time()

        def on_progress(sim_time, duration, migrations, recalls,
                        cache_hit_rate, disk_pct, n_files,
                        _rep_id=rep_id, _start=rep_wall_start):
            rep_elapsed = wallclock.time() - _start
            # ETA: extrapolate from current pace
            if sim_time > 0:
                pace    = rep_elapsed / sim_time          # wall-s per sim-s
                eta_s   = pace * (duration - sim_time)
            else:
                eta_s   = 0.0

            chr_s = "n/a" if math.isnan(cache_hit_rate) else f"{cache_hit_rate*100:.1f}%"

            _print_status(
                rep_id=_rep_id, n_reps=n_reps,
                sim_time=sim_time, duration=duration,
                migrations=migrations, recalls=recalls,
                cache_pct=chr_s, disk_pct=disk_pct,
                n_files=n_files,
                rep_elapsed=rep_elapsed,
                eta_s=eta_s,
            )

        try:
            rep_stats = run_replication(
                cfg,
                replication_id=rep_id,
                progress_callback=on_progress,
            )
            agg_stats.ingest_replication(rep_stats)
            rep_results.append({
                "replication":    rep_id,
                "n_migrations":   rep_stats.n_migrations,
                "n_recalls":      rep_stats.n_recalls,
                "cache_hit_rate": rep_stats.cache_hit_rate,
                "n_mounts":       rep_stats.n_mounts,
            })
        except Exception as e:
            # Move past the progress lines before printing error.
            print(f"\n\n  ERROR in replication {rep_id}: {e}")
            import traceback
            traceback.print_exc()
            if not allow_partial:
                print(
                    "\n  Aborting: a replication raised an exception and "
                    "--allow-partial was not set. Re-run with --allow-partial "
                    "to continue past failures (CIs will be flagged)."
                )
                sys.exit(1)
            failed_reps.append({
                "replication": rep_id,
                "error":       repr(e),
                "error_type":  type(e).__name__,
            })
            continue

        rep_elapsed = wallclock.time() - rep_wall_start
        # Move past the 2 progress lines, print a completion line
        print(f"\n\n  Replication {rep_id + 1}/{n_reps} done  "
              f"({_fmt_s(rep_elapsed)} wall-time,  "
              f"migrations={rep_stats.n_migrations},  "
              f"recalls={rep_stats.n_recalls})")
        # Reset blank lines for next replication's overwriting
        if rep_id < n_reps - 1:
            print()
            print()

    wall_elapsed = wallclock.time() - wall_start
    _separator()
    print(f"\n  Completed {n_reps} replications in "
          f"{_fmt_s(wall_elapsed)} "
          f"({wall_elapsed / max(n_reps, 1):.1f}s wall-time/rep)")

    report = agg_stats.produce_report()
    n_successful = n_reps - len(failed_reps)
    report["metadata"] = {
        "config_file":         config,
        "n_replications":      n_reps,
        "n_successful":        n_successful,
        "failed_replications": failed_reps,
        "partial_run":         bool(failed_reps),
        "master_seed":         cfg["simulation"]["seed"],
        "sim_duration_s":      dur_s,
        "wall_time_s":         round(wall_elapsed, 2),
        "simulator_version":   "1.0.0",
        "target_system":       "HPE DMF 7",
    }
    if failed_reps:
        report["metadata"]["ci_warning"] = (
            f"{len(failed_reps)} of {n_reps} replications failed. Aggregate "
            f"confidence intervals are computed over the {n_successful} "
            f"successful replications and should be treated as partial."
        )
        print(
            f"\n  WARNING: {len(failed_reps)} of {n_reps} replications "
            f"failed. Report CIs are marked partial."
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_report_json(report, str(Path(out_dir) / "simulation_report.json"))
    write_report_csv(report,  str(Path(out_dir) / "kpi_summary.csv"))
    with open(Path(out_dir) / "replication_results.jsonl", "w") as f:
        for r in rep_results:
            f.write(json.dumps(r) + "\n")

    print_summary_table(report)

    _separator()
    print(f"  Output written to: {out_dir}/")
    print(f"    simulation_report.json    — full KPI report with 95% CIs")
    print(f"    kpi_summary.csv           — flat CSV for Excel / R / Python")
    print(f"    replication_results.jsonl — per-replication raw data")
    _separator()


@cli.command(name="validate-config")
@click.option("--config", "-c", default="config/default_config.yaml")
def validate_config_cmd(config):
    """Validate a configuration file without running the simulation."""
    try:
        cfg = load_config(config)
        print(f"  Configuration valid: {config}")
        print(f"  Replications  : {cfg['simulation']['n_replications']}")
        print(f"  Duration      : {cfg['simulation']['sim_duration_s']}s")
        print(f"  Seed          : {cfg['simulation']['seed']}")
        print(f"  Workload mode : {cfg['workload']['mode']}")
        print(f"  Tape drives   : {cfg['tape_drives']['n_drives']}")
        print(f"  Disk HWM      : {cfg['disk_cache']['hwm_fraction']*100:.0f}%")
        print(f"  Disk LWM      : {cfg['disk_cache']['lwm_fraction']*100:.0f}%")
        print(f"  Policy cycle  : {cfg['policy_engine']['cycle_period_s']}s")
        print(f"  Candidate age : {cfg['policy_engine']['candidate_age_threshold_s']}s")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"  ERROR — configuration invalid: {e}")
        sys.exit(1)


@cli.command(name="trace-template")
@click.option("--output", "-o", default="trace_template.csv")
@click.option("--n-events", "-n", default=200, type=int)
@click.option("--format", "fmt", default="csv",
              type=click.Choice(["csv", "jsonl"]))
def trace_template(output, n_events, fmt):
    """Generate a DMF 7 trace template file for replay mode."""
    if fmt == "jsonl" and not output.endswith(".jsonl"):
        output = output.replace(".csv", ".jsonl")
    DMFTraceReader.create_template_trace(output, n_events=n_events)
    print(f"  Trace template written to: {output}")
    print("  Populate with real DMF log data, then in config set:")
    print("    workload:")
    print("      mode: trace_replay")
    print(f"      trace_file: {output}")


@cli.command(name="analytical-baseline")
@click.option("--arrival-rate", "-l", default=1.0, type=float)
@click.option("--service-rate", "-m", default=2.0, type=float)
def analytical_baseline(arrival_rate, service_rate):
    """Print M/M/1 analytical baseline (Law & Kelton, 2000)."""
    lam, mu = arrival_rate, service_rate
    if lam >= mu:
        print("  ERROR: lambda must be < mu for stable queue (rho < 1)")
        sys.exit(1)
    rho = lam / mu
    L   = rho / (1 - rho)
    Lq  = rho**2 / (1 - rho)
    W   = 1 / (mu - lam)
    Wq  = rho / (mu - lam)

    _section("M/M/1 Analytical Baseline  (Law & Kelton, 2000)")
    print(f"  lambda (arrival rate)   = {lam:.4f}")
    print(f"  mu     (service rate)   = {mu:.4f}")
    print(f"  rho    (utilisation)    = {rho:.4f}   [lambda/mu]")
    _separator("-")
    print(f"  L      (mean in system) = {L:.4f}   [rho/(1-rho)]")
    print(f"  Lq     (mean in queue)  = {Lq:.4f}   [rho^2/(1-rho)]")
    print(f"  W      (mean sojourn)   = {W:.4f}s  [1/(mu-lambda)]")
    print(f"  Wq     (mean wait)      = {Wq:.4f}s  [rho/(mu-lambda)]")
    _separator("-")
    print(f"  Little's Law check: L = lambda*W = {lam*W:.4f}  (should equal L={L:.4f})")
    _separator()
    print("  To validate: set iat_distribution=exponential and configure")
    print("  io_service_time to match, then compare simulator output to these values.")


if __name__ == "__main__":
    cli()
