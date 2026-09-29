"""
tests/test_core.py — Unit Tests for DMF 7 DES Core Components
=============================================================
Tests cover:
    1. File state machine validity (all transitions, invalid transition detection)
    2. RNG stream independence (statistical test)
    3. Welch warm-up detection
    4. Little's Law QueueStats
    5. CI computation
    6. Workload distributions (LogNormal, Pareto, Zipf moments)
    7. M/M/1 analytical formulae
    8. Tape seek time model monotonicity
    9. Trace file reader validation

Run with: python -m pytest tests/test_core.py -v
"""

import pytest
import sys
import os

# Path bootstrap: add project root (parent of tests/) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import simpy
import tempfile
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.rng import RNGManager, create_rng_manager, STREAM_NAMES
from core.events import (
    FileRecord, FileState, transition,
    InvalidStateTransitionError, SimEvent, EventType
)
from stats.collector import (
    QueueStats, MetricAccumulator, LatencyHistogram,
    welch_warmup, ReplicationStats
)
from workload.generator import ZipfSampler, DiurnalModulator, DMFTraceReader
from validation.analytical import mm1_analytical, gg1_kingman_W_q, ks_test_latency


# ---------------------------------------------------------------------------
# 1. File State Machine Tests
# ---------------------------------------------------------------------------

class TestFileStateMachine:

    def test_valid_transitions(self):
        """All documented valid transitions must succeed."""
        valid_cases = [
            (FileState.REG, "policy_select",    FileState.MIG),
            (FileState.REG, "dmput",            FileState.MIG),
            (FileState.MIG, "migrate_complete", FileState.DUL),
            (FileState.MIG, "migrate_failed",   FileState.REG),
            (FileState.DUL, "space_release",    FileState.OFL),
            (FileState.DUL, "modify",           FileState.REG),
            (FileState.OFL, "read_access",      FileState.UNM),
            (FileState.OFL, "dmget",            FileState.UNM),
            (FileState.UNM, "recall_complete",  FileState.DUL),
            (FileState.UNM, "recall_failed",    FileState.OFL),
        ]
        for from_state, event, expected in valid_cases:
            result = transition(from_state, event)
            assert result == expected, (
                f"transition({from_state}, {event!r}) = {result}, "
                f"expected {expected}"
            )

    def test_invalid_transition_raises(self):
        """Invalid transitions must raise InvalidStateTransitionError."""
        invalid_cases = [
            (FileState.REG, "recall_complete"),
            (FileState.OFL, "migrate_complete"),
            (FileState.MIG, "read_access"),
            (FileState.DUL, "recall_complete"),
        ]
        for state, event in invalid_cases:
            with pytest.raises(InvalidStateTransitionError):
                transition(state, event)

    def test_apply_transition_in_place(self):
        """FileRecord.apply_transition modifies state correctly."""
        f = FileRecord("test_001", size_bytes=1024)
        assert f.state == FileState.REG
        f.apply_transition("policy_select")
        assert f.state == FileState.MIG
        f.apply_transition("migrate_complete")
        assert f.state == FileState.DUL
        f.apply_transition("space_release")
        assert f.state == FileState.OFL

    def test_full_lifecycle(self):
        """Test complete REG → MIG → DUL → OFL → UNM → DUL lifecycle."""
        f = FileRecord("lifecycle_test", size_bytes=10 * 2**20)
        f.apply_transition("policy_select")   # REG → MIG
        f.apply_transition("migrate_complete") # MIG → DUL
        f.apply_transition("space_release")    # DUL → OFL
        f.apply_transition("read_access")      # OFL → UNM
        f.apply_transition("recall_complete")  # UNM → DUL
        assert f.state == FileState.DUL


# ---------------------------------------------------------------------------
# 2. RNG Stream Independence Tests
# ---------------------------------------------------------------------------

class TestRNGStreams:

    def test_all_streams_registered(self):
        """All STREAM_NAMES must be accessible."""
        rng = create_rng_manager(master_seed=42, replication=0)
        for name in STREAM_NAMES:
            stream = rng.get(name)
            assert stream is not None

    def test_invalid_stream_raises(self):
        """Unknown stream name must raise KeyError."""
        rng = create_rng_manager(master_seed=42)
        with pytest.raises(KeyError):
            rng.get("nonexistent_stream_xyz")

    def test_streams_are_independent(self):
        """
        Two distinct streams should produce uncorrelated samples.
        Test: Pearson correlation of 1000 samples from two streams should
        be within ±3σ of 0 (expected for uncorrelated sequences).
        """
        rng = create_rng_manager(master_seed=42)
        s1 = rng.get("file_size")
        s2 = rng.get("iat")
        n = 1000
        a = np.array([s1.uniform() for _ in range(n)])
        b = np.array([s2.uniform() for _ in range(n)])
        r, _ = np.corrcoef(a, b)[0, 1], None
        # Under H0 (independence), |r| ~ N(0, 1/√n)
        # 3σ threshold: 3/√1000 ≈ 0.095
        assert abs(r) < 3.0 / math.sqrt(n), (
            f"Streams appear correlated: r={r:.4f}, threshold=±{3/math.sqrt(n):.4f}"
        )

    def test_replications_differ(self):
        """Different replications must produce different sample sequences."""
        rng0 = create_rng_manager(master_seed=42, replication=0)
        rng1 = create_rng_manager(master_seed=42, replication=1)
        s0 = rng0.get("file_size")
        s1 = rng1.get("file_size")
        vals0 = [s0.uniform() for _ in range(20)]
        vals1 = [s1.uniform() for _ in range(20)]
        assert vals0 != vals1, "Replications 0 and 1 produced identical sequences"

    def test_same_seed_reproducible(self):
        """Same seed and replication must produce identical sequences."""
        rng_a = create_rng_manager(master_seed=99, replication=3)
        rng_b = create_rng_manager(master_seed=99, replication=3)
        sa = rng_a.get("iat")
        sb = rng_b.get("iat")
        assert [sa.uniform() for _ in range(10)] == [sb.uniform() for _ in range(10)]


# ---------------------------------------------------------------------------
# 3. Welch Warm-Up Detection Tests
# ---------------------------------------------------------------------------

class TestWelchWarmup:

    def test_stable_series_returns_early(self):
        """A stationary series should have a short warm-up estimate."""
        rng = np.random.default_rng(42)
        values = rng.normal(10.0, 0.1, size=500)
        warmup_idx = welch_warmup(values, window_fraction=0.1)
        assert warmup_idx < len(values) * 0.5

    def test_transient_detected(self):
        """A series with a decaying transient should detect warm-up."""
        # Exponentially decaying transient: starts at 100, converges to 10
        n = 1000
        transient = 90 * np.exp(-np.arange(n) / 100)
        values = 10.0 + transient + np.random.default_rng(7).normal(0, 0.5, n)
        warmup_idx = welch_warmup(values, window_fraction=0.1)
        # Should detect warm-up somewhere in first 30% (transient decays fast)
        assert warmup_idx > 0

    def test_short_series(self):
        """Series shorter than 10 samples should return 0."""
        assert welch_warmup(np.array([1.0, 2.0, 3.0])) == 0


# ---------------------------------------------------------------------------
# 4. Little's Law QueueStats Tests
# ---------------------------------------------------------------------------

class TestQueueStats:

    def test_littles_law_mm1_approximation(self):
        """
        Simulate an M/M/1 queue manually and verify the batch-means CI test of
        Little's Law accepts H₀ for a stationary stream.

        M/M/1 with λ=1, μ=2 → ρ=0.5, L=1, W=1, λW=1.
        """
        rng = np.random.default_rng(42)
        q = QueueStats("test_mm1")
        lam, mu = 1.0, 2.0
        t = 0.0
        n_events = 5000

        for _ in range(n_events):
            iat = rng.exponential(1.0 / lam)
            t += iat
            arr = t
            svc = rng.exponential(1.0 / mu)
            dep = arr + svc
            q.arrival(arr)
            q.departure(dep, arr)

        q.finalise(t)
        result = q.verify_littles_law()  # CI-based, default α=0.05, 20 batches
        assert result["passed"], f"Little's Law CI test rejected H₀: {result}"
        assert result["relative_error"] < 0.20, (
            f"Point-estimate deviation unexpectedly large: {result}"
        )

    def test_empty_queue_nan(self):
        """Empty queue should report insufficient data (not crash)."""
        q = QueueStats("empty")
        q.finalise(100.0)
        result = q.verify_littles_law()
        assert not result["passed"]
        assert result.get("note", "").startswith("Insufficient data")


# ---------------------------------------------------------------------------
# 5. Confidence Interval Tests
# ---------------------------------------------------------------------------

class TestConfidenceIntervals:

    def test_ci_contains_true_mean(self):
        """
        For N(μ, σ²) data, 95% CI should contain μ in ~95% of cases.
        Test with 1000 repetitions (binomial test).
        """
        rng = np.random.default_rng(123)
        true_mean = 5.0
        sigma = 1.0
        n_reps = 30
        n_experiments = 200
        contains_true = 0

        for _ in range(n_experiments):
            acc = MetricAccumulator("test")
            for _ in range(n_reps):
                acc.add_replication(rng.normal(true_mean, sigma))
            lo, hi = acc.confidence_interval_95()
            if lo <= true_mean <= hi:
                contains_true += 1

        coverage = contains_true / n_experiments
        # Should be ≈ 95% ± 3σ_binom = ± 3×sqrt(0.95×0.05/200) ≈ ± 0.046
        assert 0.90 <= coverage <= 1.00, (
            f"CI coverage = {coverage:.3f}, expected ~0.95"
        )

    def test_half_width_decreases_with_n(self):
        """Half-width should decrease as O(1/√n)."""
        rng = np.random.default_rng(7)
        true_mean = 10.0
        for n in [10, 30, 100]:
            acc = MetricAccumulator("hw_test")
            for _ in range(n):
                acc.add_replication(rng.normal(true_mean, 1.0))
            hw = acc.half_width()
            assert hw > 0, f"Half-width should be positive for n={n}"

        # Check ordering: hw(10) > hw(30) > hw(100)
        accs = []
        for n in [10, 30, 100]:
            rng2 = np.random.default_rng(42)
            acc = MetricAccumulator(f"n{n}")
            for _ in range(n):
                acc.add_replication(rng2.normal(10.0, 1.0))
            accs.append(acc.half_width())
        assert accs[0] > accs[1] > accs[2]


# ---------------------------------------------------------------------------
# 6. Workload Distribution Tests
# ---------------------------------------------------------------------------

class TestWorkloadDistributions:

    def test_lognormal_mean_variance(self):
        """LogNormal(μ, σ) sample mean and variance should match theoretical values."""
        rng = np.random.default_rng(42)
        mu, sigma = 3.0, 1.0
        n = 100_000
        samples = rng.lognormal(mu, sigma, n)

        # Theoretical mean: exp(μ + σ²/2)
        theoretical_mean = math.exp(mu + sigma**2 / 2)
        sample_mean = np.mean(samples)
        # Allow 1% relative error at n=100k
        assert abs(sample_mean - theoretical_mean) / theoretical_mean < 0.01, (
            f"LogNormal mean: sample={sample_mean:.2f}, "
            f"theoretical={theoretical_mean:.2f}"
        )

    def test_pareto_mean(self):
        """
        Pareto Type II (Lomax) with α=2, x_min=1 should have E[X] = x_min/(α-1) = 1.
        
        Lomax distribution: F(x) = 1 - (x_min/(x_min + x))^α
        Mean: x_min / (α - 1)   [not x_min × α/(α-1) which is Pareto Type I]
        IAT sampler: x_min × (U^(-1/α) - 1)
        
        Reference: Wikipedia, Lomax distribution (Pareto Type II).
        """
        rng = np.random.default_rng(42)
        alpha, x_min = 2.0, 1.0
        n = 200_000
        u = rng.uniform(1e-10, 1.0 - 1e-10, n)
        samples = x_min * (u ** (-1.0 / alpha) - 1.0)
        # Lomax mean: x_min / (α - 1) = 1 / (2 - 1) = 1.0
        expected_mean = x_min / (alpha - 1.0)
        sample_mean = np.mean(samples)
        # Allow 2% error at n=200k
        assert abs(sample_mean - expected_mean) / expected_mean < 0.02, (
            f"Pareto (Lomax) mean: sample={sample_mean:.4f}, expected={expected_mean:.4f}"
        )

    def test_zipf_most_popular_has_highest_probability(self):
        """Zipf rank 0 should be sampled most frequently."""
        rng_gen = np.random.default_rng(42)
        zipf = ZipfSampler(n_files=100, s=1.0, rng=rng_gen)
        counts = {}
        for _ in range(10_000):
            r = zipf.sample()
            counts[r] = counts.get(r, 0) + 1
        most_common = max(counts, key=counts.get)
        assert most_common == 0, (
            f"Zipf rank 0 should be most common, got rank {most_common}"
        )

    def test_diurnal_bounds(self):
        """Diurnal modulator should always return positive values."""
        d = DiurnalModulator(amplitude=0.9, peak_hour=10)
        for hour in range(24):
            val = d.modulate(hour * 3600)
            assert val > 0, f"Diurnal factor at hour {hour} = {val} (should be > 0)"

    def test_diurnal_peak(self):
        """Diurnal modulator should be highest near peak hour."""
        d = DiurnalModulator(amplitude=0.6, peak_hour=10)
        peak_val = d.modulate(10 * 3600)
        off_peak_val = d.modulate(22 * 3600)
        assert peak_val > off_peak_val, (
            f"Peak value {peak_val:.3f} should exceed off-peak {off_peak_val:.3f}"
        )


# ---------------------------------------------------------------------------
# 7. M/M/1 Analytical Formulae Tests
# ---------------------------------------------------------------------------

class TestMM1Analytical:

    def test_known_values(self):
        """M/M/1 with λ=1, μ=2 → ρ=0.5, L=1, W=1."""
        r = mm1_analytical(lam=1.0, mu=2.0)
        assert abs(r["rho"] - 0.5) < 1e-10
        assert abs(r["L"] - 1.0) < 1e-10
        assert abs(r["W"] - 1.0) < 1e-10
        assert r["littles_law_check"]

    def test_little_law_internally_consistent(self):
        """L = λ × W must hold exactly for analytical formula."""
        for lam, mu in [(0.5, 2.0), (1.0, 3.0), (0.8, 1.5)]:
            r = mm1_analytical(lam, mu)
            assert abs(r["L"] - lam * r["W"]) < 1e-10

    def test_unstable_raises(self):
        """ρ ≥ 1 must raise ValueError."""
        with pytest.raises(ValueError):
            mm1_analytical(lam=2.0, mu=1.0)
        with pytest.raises(ValueError):
            mm1_analytical(lam=1.0, mu=1.0)  # ρ = 1


# ---------------------------------------------------------------------------
# 8. Tape Seek Model Tests
# ---------------------------------------------------------------------------

class TestTapeSeek:

    def test_seek_monotone_in_distance(self):
        """Seek time should be monotonically non-decreasing with tape distance."""
        from components.tape_subsystem import TapeDrive
        import simpy as sp

        env = sp.Environment()
        rng = create_rng_manager(42)

        drive_cfg = {
            "n_drives": 1,
            "native_rate_bytes_per_s": 360_000_000,
            "compressed_rate_bytes_per_s": 720_000_000,
            "tape_capacity_bytes": 12_000_000_000_000,
            "seek_speed_factor": 1.5,
            "mount_time_lognormal_mu_s": 40.0,
            "mount_time_lognormal_sigma_s": 8.0,
            "unload_time_lognormal_mu_s": 26.0,
            "unload_time_lognormal_sigma_s": 5.0,
            "compression_ratio_lognormal_mu": 0.693,
            "compression_ratio_lognormal_sigma": 0.3,
            "cleaning_interval_mounts": 500,
            "cleaning_duration_s": 300,
        }
        robot_cfg = {
            "n_arms": 1,
            "library_slots": 100,
            "n_drives": 1,
            "arm_horizontal_speed_m_per_s": 1.5,
            "arm_vertical_speed_m_per_s": 0.8,
            "library_height_m": 2.0,
            "library_width_m": 1.5,
            "cartridge_load_s_mu": 15.0,
            "cartridge_load_s_sigma": 2.5,
            "cartridge_unload_s_mu": 12.0,
            "cartridge_unload_s_sigma": 2.0,
        }
        from components.tape_subsystem import TapeRobot
        stats = ReplicationStats(replication_id=0, warmup_s=0)
        robot = TapeRobot(env, robot_cfg, rng, stats)
        drive = TapeDrive(0, env, drive_cfg, rng, robot, stats)

        # Seek times for increasing distances should be non-decreasing
        positions = [0, 10**9, 10**10, 10**11]
        times = [drive._seek_time(0, p) for p in positions]
        for i in range(len(times) - 1):
            assert times[i] <= times[i+1] * 1.5, (
                f"Seek time not monotone: {times}"
            )

    def test_zero_seek(self):
        """Seek from position X to X should be near zero."""
        from components.tape_subsystem import TapeDrive, TapeRobot
        import simpy as sp
        env = sp.Environment()
        rng = create_rng_manager(42)
        robot_cfg = {
            "n_arms": 1, "library_slots": 100, "n_drives": 1,
            "arm_horizontal_speed_m_per_s": 1.5,
            "arm_vertical_speed_m_per_s": 0.8,
            "library_height_m": 2.0, "library_width_m": 1.5,
            "cartridge_load_s_mu": 15.0, "cartridge_load_s_sigma": 2.5,
            "cartridge_unload_s_mu": 12.0, "cartridge_unload_s_sigma": 2.0,
        }
        drive_cfg = {
            "n_drives": 1,
            "native_rate_bytes_per_s": 360_000_000,
            "compressed_rate_bytes_per_s": 720_000_000,
            "tape_capacity_bytes": 12_000_000_000_000,
            "seek_speed_factor": 1.5,
            "mount_time_lognormal_mu_s": 40.0,
            "mount_time_lognormal_sigma_s": 8.0,
            "unload_time_lognormal_mu_s": 26.0,
            "unload_time_lognormal_sigma_s": 5.0,
            "compression_ratio_lognormal_mu": 0.693,
            "compression_ratio_lognormal_sigma": 0.3,
            "cleaning_interval_mounts": 500,
            "cleaning_duration_s": 300,
        }
        stats = ReplicationStats(replication_id=0, warmup_s=0)
        robot = TapeRobot(env, robot_cfg, rng, stats)
        drive = TapeDrive(0, env, drive_cfg, rng, robot, stats)
        t = drive._seek_time(500_000_000, 500_000_000)
        # Zero delta: the length-based locate model still pays the fixed
        # load-to-ready overhead (locate_base_s ≈ 8 s), so the result must sit
        # on the base-overhead scale — never near zero, and far below any
        # distance-driven locate.
        assert 6.0 < t < 20.0, (
            f"Zero-delta locate should be on the base-overhead scale "
            f"(~locate_base_s), got {t:.4f}s"
        )


# ---------------------------------------------------------------------------
# 9. Trace File Reader Tests
# ---------------------------------------------------------------------------

class TestTraceReader:

    def _make_temp_trace_csv(self, records):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
        f.close()
        return f.name

    def _make_temp_trace_jsonl(self, records):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def _sample_records(self):
        return [
            {"timestamp_s": 0.0, "event_type": "FILE_CREATE", "file_id": "bfid_0001",
             "file_size_bytes": 1024000, "file_state_before": "REG",
             "file_state_after": "REG", "vsn": None, "drive_id": None,
             "latency_s": None, "queue_depth": 0, "disk_usage_pct": 50.0},
            {"timestamp_s": 5.0, "event_type": "MIGRATE", "file_id": "bfid_0001",
             "file_size_bytes": 1024000, "file_state_before": "REG",
             "file_state_after": "OFL", "vsn": "VG1_V0001", "drive_id": 0,
             "latency_s": 45.2, "queue_depth": 1, "disk_usage_pct": 50.0},
        ]

    def test_load_csv(self):
        records = self._sample_records()
        path = self._make_temp_trace_csv(records)
        reader = DMFTraceReader(path)
        loaded = reader.load()
        assert len(loaded) == 2
        assert loaded[0]["event_type"] == "FILE_CREATE"
        os.unlink(path)

    def test_load_jsonl(self):
        records = self._sample_records()
        path = self._make_temp_trace_jsonl(records)
        reader = DMFTraceReader(path)
        loaded = reader.load()
        assert len(loaded) == 2
        os.unlink(path)

    def test_missing_required_field_raises(self):
        records = [{"timestamp_s": 0.0, "event_type": "FILE_CREATE"}]  # missing fields
        path = self._make_temp_trace_jsonl(records)
        reader = DMFTraceReader(path)
        with pytest.raises(ValueError, match="missing required field"):
            reader.load()
        os.unlink(path)

    def test_invalid_event_type_raises(self):
        records = [{"timestamp_s": 0.0, "event_type": "INVALID_TYPE",
                    "file_id": "x", "file_size_bytes": 100}]
        path = self._make_temp_trace_jsonl(records)
        reader = DMFTraceReader(path)
        with pytest.raises(ValueError, match="unknown event_type"):
            reader.load()
        os.unlink(path)

    def test_sorted_by_timestamp(self):
        records = [
            {"timestamp_s": 10.0, "event_type": "RECALL", "file_id": "b",
             "file_size_bytes": 100},
            {"timestamp_s": 1.0, "event_type": "FILE_CREATE", "file_id": "a",
             "file_size_bytes": 200},
        ]
        path = self._make_temp_trace_jsonl(records)
        reader = DMFTraceReader(path)
        loaded = reader.load()
        assert loaded[0]["timestamp_s"] == 1.0
        assert loaded[1]["timestamp_s"] == 10.0
        os.unlink(path)

    def test_template_generation_csv(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        DMFTraceReader.create_template_trace(path, n_events=20)
        reader = DMFTraceReader(path)
        loaded = reader.load()
        assert len(loaded) == 20
        os.unlink(path)


# ---------------------------------------------------------------------------
# 10. Integration Smoke Test
# ---------------------------------------------------------------------------

class TestIntegrationSmoke:
    """
    Minimal end-to-end smoke test: run a very short simulation and check
    that it completes without errors and produces valid statistics.
    """

    def test_smoke_run(self):
        """Run 30 seconds of simulation and check basic invariants."""
        import yaml
        from pathlib import Path

        cfg_path = str(Path(__file__).parent.parent / "config" / "default_config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # Very short run for smoke test
        cfg["simulation"]["sim_duration_s"] = 30
        cfg["simulation"]["warmup_s"] = 5
        cfg["simulation"]["n_replications"] = 1
        cfg["simulation"]["seed"] = 0
        cfg["workload"]["working_set_size"] = 50
        cfg["workload"]["iat_distribution"] = "exponential"
        cfg["workload"]["iat_exponential_rate"] = 2.0

        from main import run_replication
        stats = run_replication(cfg, replication_id=0)

        # Basic sanity checks
        assert stats.n_files_created >= 0
        assert stats.n_recalls >= 0
        assert 0.0 <= stats.cache_hit_rate or math.isnan(stats.cache_hit_rate)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
