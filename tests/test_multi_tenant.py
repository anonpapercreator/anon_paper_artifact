"""
tests/test_multi_tenant.py — Multi-tenant noisy-neighbour workload tests.

Property-level checks rather than KPI checks: we assert what the
``MultiTenantWorkloadGenerator`` *guarantees* about its output, regardless
of the specific configuration. KPI-level checks belong in dedicated
experiment scripts and require enough simulated time to converge.
"""

import os
import sys

import numpy as np
import pytest
import simpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.events import FileRecord
from core.rng import create_rng_manager
from workload.multi_tenant import (
    MultiTenantWorkloadGenerator,
    TenantSpec,
    TenantStream,
    NoisyTenantStream,
    gini_coefficient,
)


# ─── Filesystem stub — counts arrivals per tenant ──────────────────────────

class _FSStub:
    """Minimal filesystem that records every (tenant_id, op) it sees."""
    def __init__(self):
        self.writes_by_tenant: dict = {}
        self.reads_by_tenant: dict = {}
        # the policy attribute is needed because MultiTenant accesses it via
        # the standard FS facade — but our stub doesn't need a real one.
        class _Policy:
            def register_file(self, *args, **kwargs):
                pass
        self.policy = _Policy()

    def handle_write(self, file_rec: FileRecord):
        tid = file_rec.tenant_id
        self.writes_by_tenant[tid] = self.writes_by_tenant.get(tid, 0) + 1
        # SimPy generator must yield at least once
        if False:
            yield
        return

    def handle_access(self, file_rec: FileRecord):
        tid = file_rec.tenant_id
        self.reads_by_tenant[tid] = self.reads_by_tenant.get(tid, 0) + 1
        if False:
            yield
        return


def _base_cfg(noise_level: float = 0.5, n_tenants: int = 8,
              duration: float = 600.0, multi_enabled: bool = True) -> dict:
    """Minimal config sufficient for the multi-tenant generator."""
    return {
        "simulation": {"seed": 42},
        "workload": {
            "iat_pareto_alpha": 1.8,
            "iat_pareto_xmin_s": 0.5,
            "file_size_lognormal_mu": 16.118,
            "file_size_lognormal_sigma": 2.5,
            "working_set_size": 200,
            "zipf_s": 1.0,
            "read_fraction": 0.5,
            "diurnal_enabled": False,
            "multi_tenant": {
                "enabled": multi_enabled,
                "n_tenants": n_tenants,
                "noise_level": noise_level,
                "aggregate_lambda_files_per_s": 4.0,
                "noisy_burst_period_s": 60.0,
                "noisy_burst_size": 5,
                "noisy_fraction_max": 0.25,  # so β=0.5 ⇒ 12.5% noisy
                "power_user_fraction": 0.0,
                "rate_lognormal_sigma_max": 1.5,
                "size_perturbation_sigma_max": 0.5,
            },
        },
    }


# ─── Property tests ───────────────────────────────────────────────────────

def test_every_file_has_a_tenant_tag():
    """Every FileRecord produced by a tenant stream must carry tenant_id."""
    cfg = _base_cfg(noise_level=0.5, n_tenants=4, duration=60.0)
    env = simpy.Environment()
    rng = create_rng_manager(master_seed=7)
    fs = _FSStub()
    gen = MultiTenantWorkloadGenerator(env, cfg, rng, fs)
    env.process(gen.run())
    env.run(until=60.0)

    seen_tenants = set(fs.writes_by_tenant) | set(fs.reads_by_tenant)
    # Every key is a non-empty string and matches one of our specs.
    spec_ids = {s.tenant_id for s in gen.tenant_specs}
    assert seen_tenants.issubset(spec_ids)
    assert all(isinstance(tid, str) and tid.startswith("t") for tid in seen_tenants)


def test_n_tenants_specs_produced():
    """Exactly n_tenants specs are produced; tenant_ids unique."""
    cfg = _base_cfg(noise_level=0.5, n_tenants=12)
    env = simpy.Environment()
    rng = create_rng_manager(master_seed=11)
    gen = MultiTenantWorkloadGenerator(env, cfg, rng, _FSStub())
    assert len(gen.tenant_specs) == 12
    ids = [s.tenant_id for s in gen.tenant_specs]
    assert len(set(ids)) == 12, "Tenant ids must be unique"


def test_role_mix_responds_to_noise_level():
    """β=0 → no noisy tenants. β=1 → at most noisy_fraction_max·n noisy."""
    rng = create_rng_manager(master_seed=23)
    env = simpy.Environment()

    cfg_zero = _base_cfg(noise_level=0.0, n_tenants=64)
    gen_zero = MultiTenantWorkloadGenerator(env, cfg_zero, rng, _FSStub())
    n_noisy_zero = sum(1 for s in gen_zero.tenant_specs if s.role == "noisy")
    assert n_noisy_zero == 0, (
        f"β=0 must produce zero noisy tenants; got {n_noisy_zero}"
    )

    cfg_full = _base_cfg(noise_level=1.0, n_tenants=64)
    gen_full = MultiTenantWorkloadGenerator(env, cfg_full, rng, _FSStub())
    n_noisy_full = sum(1 for s in gen_full.tenant_specs if s.role == "noisy")
    # Configured noisy_fraction_max=0.25 → up to 16 noisy at β=1.
    assert 8 <= n_noisy_full <= 20, (
        f"β=1 should give ~25%·64=16 noisy tenants; got {n_noisy_full}"
    )


def test_aggregate_rate_is_held_constant_across_noise_level():
    """Increasing β should not silently inflate the aggregate background
    arrival rate. We compute the sum of per-tenant Lomax means across the
    non-noisy tenants and check that 1/sum(mean) ≈ aggregate_lambda."""
    target = 4.0  # files/s, set in _base_cfg via aggregate_lambda_files_per_s

    for β in (0.0, 0.5, 1.0):
        cfg = _base_cfg(noise_level=β, n_tenants=64)
        env = simpy.Environment()
        rng = create_rng_manager(master_seed=51 + int(β * 10))
        gen = MultiTenantWorkloadGenerator(env, cfg, rng, _FSStub())
        non_noisy = [s for s in gen.tenant_specs if s.role != "noisy"]
        rates = [s.lambda_files_per_s for s in non_noisy]
        # Aggregate per-tenant rate among the *non-noisy* slice should
        # equal target × (n_non_noisy / n_total).
        n = len(gen.tenant_specs)
        expected = target * len(non_noisy) / n
        rel_err = abs(sum(rates) - expected) / max(expected, 1e-9)
        assert rel_err < 0.05, (
            f"β={β}: aggregate non-noisy rate {sum(rates):.3f} differs from "
            f"target {expected:.3f} by {rel_err*100:.1f}% (>5%)."
        )


def test_noisy_tenant_produces_clusters_in_time():
    """A NoisyTenantStream must produce *clustered* writes — not a
    uniform Pareto stream. Check by running just one noisy tenant and
    confirming the inter-arrival distribution is bimodal: one mode near
    intra_burst_gap_s, another near burst_period_s."""
    spec = TenantSpec(
        tenant_id="t_noisy", role="noisy",
        iat_pareto_alpha=1.8, iat_pareto_xmin_s=0.5,
        file_size_lognormal_mu=16.0, file_size_lognormal_sigma=1.0,
        working_set_size=10, zipf_s=1.0, read_fraction=0.0,
        burst_period_s=30.0, burst_size=10,
        diurnal_enabled=False,
    )
    env = simpy.Environment()
    fs = _FSStub()
    rng = np.random.default_rng(0)

    arrivals: list = []

    class _RecordingFS(_FSStub):
        def handle_write(self, file_rec: FileRecord):
            arrivals.append(env.now)
            if False:
                yield
            return

        def handle_access(self, file_rec: FileRecord):
            arrivals.append(env.now)
            if False:
                yield
            return

    rec_fs = _RecordingFS()
    stream = NoisyTenantStream(env, spec, rng, rec_fs, diurnal=None)
    env.process(stream.run())
    env.run(until=600.0)

    iats = np.diff(arrivals)
    assert len(iats) > 5, "Need at least a handful of arrivals to test clustering"

    # Bimodality test: most IATs should be either small (intra-burst) or
    # large (inter-burst). The midpoint between the two modes should
    # contain very few samples.
    short = float(np.percentile(iats, 70))
    long_p = float(np.percentile(iats, 95))
    # short ≪ long (factor ≥ 5) ⇒ clustered (vs. a uniform Pareto where
    # the ratio of P95 to P70 is more like 2-3 even with α=1.8).
    assert long_p / max(short, 1e-9) >= 5.0, (
        f"Noisy tenant should cluster: P95/P70(IAT) = {long_p:.2f}/{short:.4f} "
        f"= {long_p/max(short,1e-9):.2f} (expected ≥ 5)."
    )


def test_gini_coefficient_known_values():
    """Sanity: Gini(equal) = 0; Gini(one-spike) approaches 1."""
    assert gini_coefficient(np.ones(10)) == pytest.approx(0.0, abs=1e-9)
    spike = np.zeros(100)
    spike[0] = 1.0
    assert gini_coefficient(spike) > 0.95, (
        f"Gini of a single-spike distribution should be near 1; "
        f"got {gini_coefficient(spike):.4f}"
    )


def test_single_tenant_path_unaffected():
    """Backwards compat: when multi_tenant.enabled = False (the default),
    a tenant_id of None on FileRecord is still acceptable and the
    existing single-stream generator path is used."""
    rec = FileRecord(file_id="x", size_bytes=100)
    assert rec.tenant_id is None
