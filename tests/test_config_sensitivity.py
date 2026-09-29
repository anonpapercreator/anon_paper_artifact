"""
tests/test_config_sensitivity.py — Config field sensitivity tests for the
unified component-based DES (main.run_replication).

Verifies that configuration fields propagate into observable simulation
behavior via the ReplicationStats output.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_replication


def _base_cfg(**overrides) -> dict:
    """Return a minimal fast config with optional overrides."""
    cfg = {
        "simulation": {
            "seed": 42,
            "n_replications": 1,
            "sim_duration_s": 60,
            "warmup_s": 5,
            "output_dir": "./results_test",
            "output_format": "jsonl",
            "trace_interval_s": 10,
        },
        "disk_cache": {
            "capacity_bytes": 50 * 1024**3,
            "hwm_fraction": 0.80,
            "lwm_fraction": 0.70,
            "initial_fill_fraction": 0.50,
            "io_service_time_lognormal_mu_s": -2.303,
            "io_service_time_lognormal_sigma_s": 1.0,
            "max_concurrent_io": 128,
            "metadata_lookup_latency_s": 0.005,
        },
        "interconnect": {
            "bandwidth_bytes_per_s": 10 * 1024**3,
            "fixed_latency_s": 0.0001,
            "max_concurrent_transfers": 64,
        },
        "data_movers": {
            "n_movers": 4,
            "node_bandwidth_bytes_per_s": 2.5 * 1024**3,
            "hba_bandwidth_bytes_per_s": 1.5 * 1024**3,
            "bandwidth_multiplier": 2.0,
            "mover_overhead_s": 0.050,
            "trickle_enabled": False,
            "dmmigrate_unack": 16,
        },
        "openvault": {
            "scheduling_latency_s": 1.0,
        },
        "tape_robot": {
            "n_arms": 1,
            "library_slots": 100,
            "n_drives": 4,
            "arm_horizontal_speed_m_per_s": 1.5,
            "arm_vertical_speed_m_per_s": 0.8,
            "library_height_m": 2.0,
            "library_width_m": 1.5,
            "cartridge_load_s_mu": 15.0,
            "cartridge_load_s_sigma": 2.5,
            "cartridge_unload_s_mu": 12.0,
            "cartridge_unload_s_sigma": 2.0,
        },
        "tape_drives": {
            "n_drives": 4,
            "drive_type": "LTO-8",
            "native_rate_bytes_per_s": 360_000_000,
            "compressed_rate_bytes_per_s": 720_000_000,
            "mount_time_lognormal_mu_s": 3.689,
            "mount_time_lognormal_sigma_s": 0.3,
            "unload_time_lognormal_mu_s": 3.258,
            "unload_time_lognormal_sigma_s": 0.25,
            "tape_capacity_bytes": 12_000_000_000_000,
            "seek_speed_factor": 1.5,
            "compression_ratio_lognormal_mu": 0.693,
            "compression_ratio_lognormal_sigma": 0.3,
            "cleaning_interval_mounts": 500,
            "cleaning_duration_s": 300,
        },
        "volume_groups": [
            {
                "name": "vg_primary",
                "n_volumes": 50,
                "drives_assigned": [0, 1, 2, 3],
                "volume_capacity_bytes": 12_000_000_000_000,
                "allocation_strategy": "sequential",
            },
        ],
        "library_server": {
            "ls_min_batch_bytes": 1 * 1024**3,
            "ls_max_wait_s": 60,
            "recall_priority_classes": [
                {"name": "interactive", "priority": 10, "sources": ["filesystem_read"]},
                {"name": "batch", "priority": 5, "sources": ["dmget", "api"]},
            ],
        },
        "policy_engine": {
            "cycle_period_s": 10,
            "candidate_age_threshold_s": 5,
            "migration_priority": "size_descending",
            "space_release_on_migration": False,
            "dual_copy": True,
            "n_policy_workers": 2,
            "recall_sort": "tape_order",
        },
        "workload": {
            "mode": "synthetic",
            "file_size_lognormal_mu": 16.118,
            "file_size_lognormal_sigma": 2.5,
            "file_size_min_bytes": 4096,
            "file_size_max_bytes": 1099511627776,
            "iat_distribution": "exponential",
            "iat_pareto_alpha": 1.8,
            "iat_pareto_xmin_s": 0.5,
            "iat_exponential_rate": 2.0,
            "working_set_size": 100,
            "zipf_s": 1.0,
            "read_fraction": 0.7,
            "diurnal_enabled": False,
            "diurnal_amplitude": 0.6,
            "diurnal_peak_hour": 10,
            "initial_file_age_s_mu": 7200,
            "initial_file_age_s_sigma": 3600,
        },
        "adaptive_policy": {
            "enabled": False,
        },
    }
    # Apply overrides by nested dict merge
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg:
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def test_hwm_affects_space_release_aggressiveness():
    """Lower HWM should trigger more DUL→OFL space releases.

    Migration is age-based and independent of the watermarks; the HWM only
    arms the event-driven free-space path (release dual-state files down to
    the LWM when usage crosses the HWM). A lower HWM therefore fires that
    path earlier and more often, while the migration count responds only
    through second-order stochastic knock-on effects.
    """
    cfg_low = _base_cfg(disk_cache={"hwm_fraction": 0.60})
    cfg_high = _base_cfg(disk_cache={"hwm_fraction": 0.95})

    stats_low = run_replication(cfg_low, replication_id=0)
    stats_high = run_replication(cfg_high, replication_id=0)

    assert stats_low.n_space_releases >= stats_high.n_space_releases, (
        f"Lower HWM produced fewer space releases: "
        f"low={stats_low.n_space_releases}, high={stats_high.n_space_releases}"
    )


def test_age_threshold_affects_migration_timing():
    """Shorter age threshold should produce more migrations."""
    cfg_short = _base_cfg(policy_engine={"candidate_age_threshold_s": 1})
    cfg_long = _base_cfg(policy_engine={"candidate_age_threshold_s": 3600})

    stats_short = run_replication(cfg_short, replication_id=0)
    stats_long = run_replication(cfg_long, replication_id=0)

    assert stats_short.n_migrations >= stats_long.n_migrations, (
        f"Shorter threshold produced fewer migrations: "
        f"short={stats_short.n_migrations}, long={stats_long.n_migrations}"
    )


def test_lsm_batch_size_affects_flush_frequency():
    """Smaller batch size should lead to more frequent LS flushes (more mounts)."""
    cfg_small = _base_cfg(library_server={"ls_min_batch_bytes": 1 * 1024**3})
    cfg_large = _base_cfg(library_server={"ls_min_batch_bytes": 50 * 1024**3})

    stats_small = run_replication(cfg_small, replication_id=0)
    stats_large = run_replication(cfg_large, replication_id=0)

    # Smaller batch → more flushes → more mounts (or at least not fewer)
    assert stats_small.n_mounts >= stats_large.n_mounts, (
        f"Smaller batch produced fewer mounts: "
        f"small={stats_small.n_mounts}, large={stats_large.n_mounts}"
    )


def test_disk_cache_capacity_limits_files():
    """Smaller cache capacity must throttle activity through the capacity
    gate. Migration is age-based, so it does not become *more* aggressive
    under pressure; instead the finite-capacity write gate (ENOSPC
    behaviour) blocks ingest, so a small cache serves strictly less work —
    fewer completed recalls and fewer migrations — than a large one."""
    cfg_small = _base_cfg(disk_cache={"capacity_bytes": 5 * 1024**3})
    cfg_large = _base_cfg(disk_cache={"capacity_bytes": 500 * 1024**3})

    stats_small = run_replication(cfg_small, replication_id=0)
    stats_large = run_replication(cfg_large, replication_id=0)

    served_small = stats_small.n_migrations + stats_small.n_recalls
    served_large = stats_large.n_migrations + stats_large.n_recalls
    assert served_small < served_large, (
        f"A 100x smaller cache should gate ingest and serve less work. "
        f"small={served_small} (mig={stats_small.n_migrations}, "
        f"rec={stats_small.n_recalls}); "
        f"large={served_large} (mig={stats_large.n_migrations}, "
        f"rec={stats_large.n_recalls})"
    )
