"""
validation/calibrate_trace_replay.py — Trace Replay Calibration
===============================================================

Uses the actual trace in trace_replay mode with proper pre-population.
All unique files from the trace are created in the filesystem before
simulation starts, with states inferred from their first event:
    - RECALL  → OFL (must have tape copy)
    - MIGRATE → REG (online, no tape copy)

OFL files are assigned existing VSNs so the tape subsystem can recall them.
"""
from __future__ import annotations
import json
import copy
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats as scipy_stats

from main import run_replication
from validation.calibration import load_trace_latencies


def build_trace_replay_cfg(trace_path: str) -> dict:
    """Build config tuned for trace replay calibration."""
    import yaml
    with open("config/default_config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["simulation"]["n_replications"] = 1
    cfg["simulation"]["sim_duration_s"] = 60000
    cfg["simulation"]["warmup_s"] = 10
    cfg["simulation"]["seed"] = 42

    cfg["workload"]["mode"] = "trace_replay"
    cfg["workload"]["trace_file"] = trace_path
    cfg["workload"]["trace_format"] = "jsonl"

    cfg["disk_cache"]["capacity_bytes"] = 50 * 1024**3
    cfg["disk_cache"]["hwm_fraction"] = 0.80
    cfg["disk_cache"]["lwm_fraction"] = 0.70
    cfg["disk_cache"]["initial_fill_fraction"] = 0.05

    cfg["policy_engine"]["cycle_period_s"] = 30
    cfg["policy_engine"]["candidate_age_threshold_s"] = 1_000_000_000  # disable auto-migration
    cfg["policy_engine"]["space_release_on_migration"] = True

    cfg.setdefault("adaptive_policy", {})["enabled"] = False

    return cfg


def pre_populate_filesystem(trace_path: str, fs, tape):
    """Pre-populate filesystem with all trace files."""
    from core.events import FileRecord, FileState

    records = []
    with open(trace_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    records.sort(key=lambda r: r["timestamp_s"])

    vg = next(iter(tape.vgs.values()), None)
    if vg is None:
        raise RuntimeError("No volume groups configured")

    first_event: Dict[str, str] = {}
    for rec in records:
        fid = rec["file_id"]
        if fid not in first_event:
            first_event[fid] = rec["event_type"]

    vols = vg._volumes
    idx = 0
    vsn_map: Dict[str, str] = {}

    for fid, et in first_event.items():
        size = next((r["file_size_bytes"] for r in records if r["file_id"] == fid), 1024)
        if et == "RECALL":
            state = FileState.OFL
            vol = vols[idx % len(vols)]
            idx += 1
            vsn = vol.vsn
            vsn_map[fid] = vsn
        else:
            state = FileState.REG
            vsn = None

        frec = FileRecord(
            file_id=fid,
            size_bytes=int(size),
            state=state,
            created_at=0.0,
            last_accessed_at=0.0,
            last_modified_at=0.0,
            vsn=vsn,
            vg_name=vg.name if vsn else None,
        )
        fs.policy.register_file(frec)
        if state == FileState.REG:
            fs.disk.add_file(frec)

    print(f"[pre_populate] {len(first_event)} files: {len(vsn_map)} OFL, {len(first_event) - len(vsn_map)} REG")
    return vsn_map


def evaluate_params(
    cfg: dict,
    mount_mu: float,
    mount_sigma: float,
    seek_mu: float,
    seek_sigma: float,
    obs_recalls: List[float],
    obs_migrates: List[float],
    seed: int = 42,
) -> Tuple[float, dict]:
    """Run trace replay with given tape parameters."""
    c = copy.deepcopy(cfg)
    c["simulation"]["seed"] = seed
    tape_cfg = c["tape_drives"]
    tape_cfg["mount_time_lognormal_mu_s"] = mount_mu
    tape_cfg["mount_time_lognormal_sigma_s"] = max(0.01, mount_sigma)
    tape_cfg["seek_time_lognormal_mu_s"] = seek_mu
    tape_cfg["seek_time_lognormal_sigma_s"] = max(0.01, seek_sigma)

    try:
        import simpy
        from stats.collector import ReplicationStats
        from core.rng import RNGManager
        from components.tape_subsystem import TapeSubsystem
        from components.filesystem import Filesystem
        from workload.generator import DMFTraceReader

        sim_cfg = c["simulation"]
        duration = sim_cfg["sim_duration_s"]
        warmup_s = sim_cfg.get("warmup_s", duration * 0.1)
        master_seed = sim_cfg["seed"]
        chunk_s = sim_cfg.get("progress_chunk_s", 60)

        rng = RNGManager(master_seed=master_seed, replication=0)
        stats = ReplicationStats(replication_id=0, warmup_s=warmup_s)
        env = simpy.Environment()

        tape = TapeSubsystem(env, c, rng, stats)
        fs = Filesystem(env, c, tape, rng, stats)
        pre_populate_filesystem(c["workload"]["trace_file"], fs, tape)

        reader = DMFTraceReader(c["workload"]["trace_file"])
        reader.load()
        env.process(reader.run(env, fs))

        def stats_snapshot():
            while True:
                interval = sim_cfg.get("trace_interval_s", 60)
                yield env.timeout(interval)
                stats.record_disk_usage(env.now, fs.disk.usage_fraction)
                for q in stats.queues.values():
                    q.snapshot(env.now)

        env.process(stats_snapshot())

        t = 0.0
        while t < duration:
            t = min(t + chunk_s, duration)
            env.run(until=t)

        stats.finalise(sim_end_time=duration)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return 10.0, {"error": str(exc)}

    sim_r = stats.recall_latency._last_rep_samples or []
    sim_m = stats.migrate_latency._last_rep_samples or []

    ks_r = _ks(sim_r, obs_recalls)
    ks_m = _ks(sim_m, obs_migrates)
    combined = 0.6 * ks_r + 0.4 * ks_m

    return combined, {
        "ks_recall": ks_r,
        "ks_migrate": ks_m,
        "n_sim_recalls": len(sim_r),
        "n_sim_migrates": len(sim_m),
    }


def _ks(sim: List[float], obs: List[float]) -> float:
    if len(sim) < 5 or len(obs) < 5:
        return 1.0
    return float(scipy_stats.ks_2samp(sim, obs)[0])


if __name__ == "__main__":
    import sys
    trace = sys.argv[1] if len(sys.argv) > 1 else "trace.jsonl"

    obs_r, obs_m, _ = load_trace_latencies(trace)
    print(f"[calibrate] Observed: {len(obs_r)} recalls, {len(obs_m)} migrates")

    cfg = build_trace_replay_cfg(trace)

    best_ks = float("inf")
    best_params = None
    best_diag = None

    # Coarse grid
    for mmu in [2.0, 3.0, 4.0]:
        for msig in [0.2, 0.5, 1.0]:
            for smu in [2.0, 3.0, 4.0]:
                for ssig in [0.5, 1.0, 1.5]:
                    ks, diag = evaluate_params(
                        cfg, mmu, msig, smu, ssig, obs_r, obs_m, seed=42
                    )
                    print(f"  mount=({mmu:.1f},{msig:.1f}) seek=({smu:.1f},{ssig:.1f}) → KS={ks:.4f} {diag}")
                    if ks < best_ks:
                        best_ks = ks
                        best_params = (mmu, msig, smu, ssig)
                        best_diag = diag

    print("\n=== BEST ===")
    print(f"  mount_mu={best_params[0]:.2f}, mount_sigma={best_params[1]:.2f}")
    print(f"  seek_mu={best_params[2]:.2f}, seek_sigma={best_params[3]:.2f}")
    print(f"  KS={best_ks:.4f}")
    print(f"  {best_diag}")
