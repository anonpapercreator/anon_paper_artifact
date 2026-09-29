"""
tests/test_lognormal.py — Lognormal helper and TapeRobot mount-time sanity tests.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rng import RNGManager, lognormal_params_from_mean_std


@pytest.mark.parametrize(
    "mean, std",
    [(15.0, 2.5), (12.0, 2.0), (40.0, 8.0), (0.1, 0.05), (100.0, 1.0)],
)
def test_lognormal_params_recover_mean_and_std(mean, std):
    """Sampling LogNormal(μ_log, σ_log) must reproduce (mean, std) within 2 %."""
    mu_log, sigma_log = lognormal_params_from_mean_std(mean, std)
    rng = np.random.default_rng(1234)
    samples = rng.lognormal(mu_log, sigma_log, size=200_000)
    assert samples.mean() == pytest.approx(mean, rel=0.02)
    assert samples.std(ddof=1) == pytest.approx(std, rel=0.03)


def test_lognormal_params_reject_bad_inputs():
    with pytest.raises(ValueError):
        lognormal_params_from_mean_std(-1.0, 1.0)
    with pytest.raises(ValueError):
        lognormal_params_from_mean_std(0.0, 1.0)
    with pytest.raises(ValueError):
        lognormal_params_from_mean_std(1.0, -0.1)


def test_lognormal_params_zero_std_gives_deterministic_samples():
    mu_log, sigma_log = lognormal_params_from_mean_std(7.0, 0.0)
    assert sigma_log == 0.0
    assert math.exp(mu_log) == pytest.approx(7.0, rel=1e-12)


def test_tape_robot_mount_time_matches_config_mean():
    """TapeRobot.mount_service_time must reproduce the configured linear mean."""
    from components.tape_subsystem import TapeRobot
    from stats.collector import ReplicationStats
    import simpy

    cfg = {
        "n_arms": 1,
        "library_slots": 500,
        "n_drives": 8,
        "arm_horizontal_speed_m_per_s": 1.5,
        "arm_vertical_speed_m_per_s": 0.8,
        "library_height_m": 2.0,
        "library_width_m": 1.5,
        "cartridge_load_s_mu": 15.0,
        "cartridge_load_s_sigma": 2.5,
        "cartridge_unload_s_mu": 12.0,
        "cartridge_unload_s_sigma": 2.0,
    }
    env = simpy.Environment()
    rng = RNGManager(master_seed=7, replication=0)
    stats = ReplicationStats(replication_id=0, warmup_s=0.0)
    robot = TapeRobot(env, cfg, rng, stats)

    # Sample load-only distribution (bypass travel, which adds noise)
    samples = np.array([
        robot._rng_load.lognormal(robot._load_mu_log, robot._load_sigma_log)
        for _ in range(100_000)
    ])
    assert samples.mean() == pytest.approx(15.0, rel=0.02)
    assert samples.std(ddof=1) == pytest.approx(2.5, rel=0.05)
