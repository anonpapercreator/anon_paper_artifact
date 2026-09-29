"""
stats/converter.py — ReplicationStats → SimulationResult Converter
====================================================================
Bridges the component-based DES (which returns ReplicationStats)
to the Pydantic SimulationResult expected by the sweeper and web UI.
"""
from __future__ import annotations
import math
from typing import Dict, Any, Optional

from stats.collector import ReplicationStats


def replication_stats_to_simulation_result(
    rep_stats: ReplicationStats,
    cfg: Dict[str, Any],
    replication_id: int = 0,
    wall_time_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Convert a ReplicationStats object (from one DES replication)
    into a flat dict matching the SimulationResult schema.

    Args:
        rep_stats:      Statistics collected during one replication.
        cfg:            The simulation configuration dict (YAML-loaded).
        replication_id: Which replication this is (0-based).
        wall_time_s:    Wall-clock time the replication took.

    Returns:
        Dict with fields: cache_hit_ratio, latency_p50/p95/p99,
        tape_mounts, bytes_migrated, archived_data_tb,
        tape_cartridges_used, tape_capacity_utilization_pct,
        migration_bandwidth_gbs, scratch_effective_bandwidth_gbs,
        scratch_bandwidth_degradation_pct, cache_resident_tb,
        cache_churn_tb, total_bytes_written, total_bytes_read,
        write_operations, read_operations, write_duration_s, read_duration_s.
    """
    # Flush latency histograms to compute percentiles
    rep_stats.recall_latency.flush_replication()
    rep_stats.migrate_latency.flush_replication()

    # Latency percentiles from the last replication's raw samples
    recall_p50 = _safe_percentile(rep_stats.recall_latency._last_rep_samples, 50.0)
    recall_p95 = _safe_percentile(rep_stats.recall_latency._last_rep_samples, 95.0)
    recall_p99 = _safe_percentile(rep_stats.recall_latency._last_rep_samples, 99.0)
    recall_mean = _safe_mean(rep_stats.recall_latency._last_rep_samples)

    # Tape subsystem capacity
    tape_cfg = cfg.get("tape_drives", {})
    tape_capacity_bytes = tape_cfg.get("tape_capacity_bytes", 12_000_000_000_000)
    n_drives = tape_cfg.get("n_drives", 8)
    native_rate = tape_cfg.get("native_rate_bytes_per_s", 360_000_000)
    total_tape_bw = n_drives * native_rate

    # Volume group totals
    vg_cfgs = cfg.get("volume_groups", [])
    total_volumes = sum(vg.get("n_volumes", 0) for vg in vg_cfgs)
    total_tape_capacity = total_volumes * tape_capacity_bytes if total_volumes else tape_capacity_bytes

    # Media consumption
    archived_bytes = rep_stats.bytes_archived
    archived_tb = archived_bytes / (1024**4)
    cartridges_used = archived_bytes / tape_capacity_bytes if tape_capacity_bytes > 0 else 0.0
    tape_util_pct = (
        (cartridges_used / max(total_volumes, 1)) * 100.0
        if total_volumes > 0
        else 0.0
    )

    # Bandwidth metrics (per-replication averages)
    duration_s = cfg.get("simulation", {}).get("sim_duration_s", 14400)
    migration_gbs = (archived_bytes / max(duration_s, 1)) / (1024**3)
    ess_peak_gbs = cfg.get("disk_cache", {}).get("aggregate_throughput_gib_s", 65.0)
    scratch_effective_gbs = max(0.0, ess_peak_gbs - migration_gbs)
    degradation_pct = (migration_gbs / max(ess_peak_gbs, 1e-9)) * 100.0

    # Cache metrics
    cache_resident_tb = _safe_mean(
        [v for _, v in rep_stats.disk_usage_series._values]
    ) / 100.0 * cfg.get("disk_cache", {}).get("capacity_bytes", 0) / (1024**4)
    cache_churn_tb = (
        rep_stats.bytes_written_to_cache + rep_stats.bytes_read_from_cache
    ) / (1024**4)

    # IOR-style timing
    write_dur = (
        float(rep_stats.last_write_time - rep_stats.first_write_time)
        if rep_stats.last_write_time and rep_stats.first_write_time
        else 0.0
    )
    read_dur = (
        float(rep_stats.last_read_time - rep_stats.first_read_time)
        if rep_stats.last_read_time and rep_stats.first_read_time
        else 0.0
    )

    return {
        "replication": replication_id,
        "cache_hit_ratio": rep_stats.cache_hit_rate,
        "latency_mean": recall_mean,
        "latency_p50": recall_p50,
        "latency_p95": recall_p95,
        "latency_p99": recall_p99,
        "tape_mounts": rep_stats.n_mounts,
        "bytes_migrated": archived_bytes,
        "archived_data_tb": archived_tb,
        "tape_cartridges_used": cartridges_used,
        "tape_capacity_utilization_pct": tape_util_pct,
        "migration_bandwidth_gbs": migration_gbs,
        "scratch_effective_bandwidth_gbs": scratch_effective_gbs,
        "scratch_bandwidth_degradation_pct": degradation_pct,
        "cache_resident_tb": cache_resident_tb,
        "cache_churn_tb": cache_churn_tb,
        "total_bytes_written": rep_stats.bytes_written_to_cache,
        "total_bytes_read": rep_stats.bytes_read_from_cache,
        "write_operations": rep_stats.write_operations,
        "read_operations": rep_stats.read_operations,
        "write_duration_s": write_dur,
        "read_duration_s": read_dur,
        "run_time_seconds": wall_time_s,
        "n_recalls": rep_stats.n_recalls,
        "n_migrations": rep_stats.n_migrations,
        "n_seeks": rep_stats.n_seeks,
        "n_policy_cycles": rep_stats.n_policy_cycles,
        "n_files_created": rep_stats.n_files_created,
        # Sweep parameters (for DB grouping)
        "age_threshold_minutes": cfg.get("policy_engine", {}).get("candidate_age_threshold_s", 1800) // 60,
        "hwm_fraction": cfg.get("disk_cache", {}).get("hwm_fraction", 0.9),
        "lwm_fraction": cfg.get("disk_cache", {}).get("lwm_fraction", 0.8),
        "cache_priority": cfg.get("adaptive_policy", {}).get("baseline_age_threshold_s", 1800) / (30 * 24 * 3600),
        "archiver_interval_minutes": cfg.get("policy_engine", {}).get("cycle_period_s", 300) // 60,
    }


def _safe_percentile(samples, p: float) -> float:
    import numpy as np

    if not samples:
        return 0.0
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))


def _safe_mean(values) -> float:
    import numpy as np

    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))
