"""
workload/multi_tenant.py — Multi-tenant noisy-neighbour workload model
======================================================================

Why this exists
---------------
A real DMF-fronted HSM never serves a single user. ORNL Atlas/
Spider, NCSA Blue Waters, every site DMF 7 ships into — they each support
hundreds to thousands of concurrent tenants. Modelling that with one
``SyntheticWorkloadGenerator`` is unreasonable for two reasons:

1. **Aggregation effects matter.** The superposition of many independent
   renewal streams converges (slowly) toward a Poisson process — but the
   convergence is incomplete on the timescales that actually matter for
   cache and tape, and the leftover heavy-tail / autocorrelation is
   precisely what a single-stream Pareto generator over- or under-states
   depending on its tuning.

2. **Noisy neighbours exist and dominate tail latency.** A single tenant
   running checkpoint-restart at the end of every compute step submits a
   *clustered* burst of writes that has nothing to do with the long-run
   Pareto tail of background traffic. These bursts inflate the P99 / P99.9
   recall latency for every other tenant sharing the cache, and a single-
   stream model cannot produce them at all because it has only one
   parameterisation.

Model
-----
``MultiTenantWorkloadGenerator`` runs N independent ``TenantStream``
processes inside one SimPy environment. Each tenant has:

  - Its own renewal IAT distribution (Pareto with diurnal time-rescaling),
    parameterised by a per-tenant rate drawn from a meta-distribution.
  - Its own LogNormal file-size distribution (slightly perturbed across
    tenants).
  - Its own Zipf working set (smaller than the aggregate catalogue —
    each tenant has their own "hot files").
  - A ``tenant_id`` tag carried on every produced ``FileRecord`` so
    downstream stats can compute per-tenant KPIs.

Tenants come in three classes (cf. real HSM workloads):

  - **Background** (default): low rate, smooth Pareto IATs.
  - **Power-user**: higher rate (~10× background), larger files,
    smaller working set.
  - **Noisy-neighbour**: low average rate but produces *clustered bursts*
    every ``burst_period_s`` seconds — modelling the checkpoint-storm
    pattern. Inside a burst, ``burst_size`` files are written in fast
    succession; between bursts the tenant is silent. The burst arrival
    process is independent of diurnal modulation by default (checkpoint
    cadence is set by the application's compute loop, not time of day).

Noise dial β ∈ [0, 1]
---------------------
β controls heterogeneity across tenants:

  - The σ of the per-tenant rate LogNormal: σ_rate(β) = β · σ_rate_max.
  - The fraction of tenants that are noisy-neighbours: f_noise(β) =
    β · f_noise_max.
  - The σ of the per-tenant file-size perturbation: σ_size(β) = β · σ_size_max.

β=0 produces a homogeneous farm of identical tenants (same aggregate
behaviour as the single-stream baseline). β=1 produces the maximally
heterogeneous farm.

References
----------
Whitt, W. (1985). "Approximations for departure processes and queues in
    series." Naval Research Logistics Quarterly 31.
Krishnamurthy, A. & Suri, R. (1995). "Quantifying the burstiness of arrival
    processes." Queueing Systems.
Schroeder, B. & Gibson, G.A. (2007). "Disk failures in the real world."
    FAST '07.
Schroeder, B. & Harchol-Balter, M. (2003). "Web servers under overload:
    How scheduling can help." ACM TOIT.
"""
from __future__ import annotations
import math
import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Any

from core.rng import RNGManager
from core.events import FileRecord, FileState

from workload.generator import (
    SyntheticWorkloadGenerator,
    DiurnalModulator,
    ZipfSampler,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tenant configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TenantSpec:
    """Per-tenant parameter bundle. The ``MultiTenantWorkloadGenerator``
    samples one of these per tenant at sim start."""
    tenant_id: str
    role: str                            # "background", "power", "noisy"
    iat_pareto_alpha: float
    iat_pareto_xmin_s: float
    file_size_lognormal_mu: float
    file_size_lognormal_sigma: float
    working_set_size: int
    zipf_s: float
    read_fraction: float
    # Noisy-neighbour bursts (only used when role == "noisy"):
    burst_period_s: float = 0.0
    burst_size: int = 0
    # Diurnal modulation (shared per-cluster default; tenants can opt out):
    diurnal_enabled: bool = True

    @property
    def lambda_files_per_s(self) -> float:
        """Long-run mean file-arrival rate of this tenant's IAT process.
        For Lomax(α, x_min) the mean IAT is x_min/(α−1)."""
        denom = max(self.iat_pareto_alpha - 1.0, 1e-9)
        return denom / max(self.iat_pareto_xmin_s, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Per-tenant generator (single tenant's stream)
# ─────────────────────────────────────────────────────────────────────────────

class TenantStream:
    """A single tenant's renewal arrival stream. Tagged with a tenant_id
    on every produced FileRecord so downstream stats can attribute load."""

    def __init__(self, env: simpy.Environment, spec: TenantSpec,
                 rng: np.random.Generator, filesystem: Any,
                 diurnal: Optional[DiurnalModulator]) -> None:
        self._env = env
        self._spec = spec
        self._rng = rng
        self._fs = filesystem
        self._diurnal = diurnal if spec.diurnal_enabled else None

        # Working set kept in a list so eviction picks from the unpopular
        # tail (matching the single-stream generator's fix).
        self._zipf = ZipfSampler(spec.working_set_size, spec.zipf_s, rng)
        self._working_set: List[FileRecord] = []
        self._file_counter = 0

    # ---- IAT sampling (time-rescaling on a Pareto base) ---------------------

    def _sample_base_iat(self) -> float:
        u = self._rng.uniform(0.0, 1.0 - 1e-10)
        return self._spec.iat_pareto_xmin_s * (
            u ** (-1.0 / self._spec.iat_pareto_alpha) - 1.0
        )

    def _sample_iat(self, t: float) -> float:
        base = self._sample_base_iat()
        if self._diurnal is None:
            return max(base, 1e-6)
        t_next = self._diurnal.invert_Lambda(t, base)
        return max(t_next - t, 1e-6)

    # ---- File generation ----------------------------------------------------

    def _new_file(self) -> FileRecord:
        fid = f"{self._spec.tenant_id}_f{self._file_counter:010d}"
        self._file_counter += 1
        size = int(np.clip(
            self._rng.lognormal(self._spec.file_size_lognormal_mu,
                                self._spec.file_size_lognormal_sigma),
            4096, 1 << 40,
        ))
        now = self._env.now
        return FileRecord(
            file_id=fid, size_bytes=size, state=FileState.REG,
            created_at=now, last_accessed_at=now, last_modified_at=now,
            tenant_id=self._spec.tenant_id,
        )

    # ---- Main run loop ------------------------------------------------------

    def run(self) -> Iterator:
        """SimPy process: emit reads/writes for this tenant.

        Reads are sampled by Zipf rank from the tenant's own working set;
        writes create a new file and evict from the unpopular tail (lower
        half of ranks) to preserve the popularity head.
        """
        while True:
            iat = self._sample_iat(self._env.now)
            yield self._env.timeout(iat)

            is_read = self._rng.uniform() < self._spec.read_fraction
            if is_read and self._working_set:
                rank = self._zipf.sample() % len(self._working_set)
                f = self._working_set[rank]
                if f.deleted_at is not None:
                    continue   # deleted files are not accessed
                f.last_accessed_at = self._env.now
                yield from self._fs.handle_access(f)
            else:
                f = self._new_file()
                ws = self._spec.working_set_size
                if len(self._working_set) < ws:
                    self._working_set.append(f)
                else:
                    tail_lo = ws // 2
                    rank = int(self._rng.integers(tail_lo, ws))
                    self._working_set[rank] = f
                yield from self._fs.handle_write(f)


# ─────────────────────────────────────────────────────────────────────────────
# Noisy-neighbour tenant — the checkpoint-storm pattern
# ─────────────────────────────────────────────────────────────────────────────

class NoisyTenantStream(TenantStream):
    """A tenant that produces clustered bursts of writes at irregular
    intervals — modelling the per-step checkpoint pattern of HPC jobs.

    Inter-burst gap is exponential with mean ``burst_period_s`` (so the
    burst-arrival process is Poisson at the burst level, which matches
    independence of compute steps across user jobs); within a burst,
    ``burst_size`` files are written back-to-back with a small jitter.
    Diurnal modulation does NOT apply to bursts — checkpoint cadence is
    set by the compute loop, not the time of day.
    """

    def run(self) -> Iterator:
        spec = self._spec
        burst_period = max(spec.burst_period_s, 1.0)
        burst_size = max(spec.burst_size, 1)
        # Jitter inside a burst — small random gaps so the burst doesn't
        # look perfectly synchronous (which would be a model artefact).
        intra_burst_gap_s = 0.05

        while True:
            # Wait for the next burst. Exponential gap → Poisson bursts.
            gap = self._rng.exponential(burst_period)
            yield self._env.timeout(gap)

            # Fire the burst: a clustered run of writes (mostly) plus a
            # smattering of reads.
            for _ in range(burst_size):
                if self._rng.uniform() < spec.read_fraction and self._working_set:
                    rank = self._zipf.sample() % len(self._working_set)
                    f = self._working_set[rank]
                    if f.deleted_at is None:   # deleted files are not accessed
                        f.last_accessed_at = self._env.now
                        yield from self._fs.handle_access(f)
                else:
                    f = self._new_file()
                    ws = spec.working_set_size
                    if len(self._working_set) < ws:
                        self._working_set.append(f)
                    else:
                        tail_lo = ws // 2
                        rank = int(self._rng.integers(tail_lo, ws))
                        self._working_set[rank] = f
                    yield from self._fs.handle_write(f)

                yield self._env.timeout(
                    max(self._rng.exponential(intra_burst_gap_s), 1e-6)
                )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-tenant orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MultiTenantWorkloadGenerator:
    """Orchestrates N tenant streams sharing a single filesystem.

    Tenant specs are sampled from meta-distributions parameterised by the
    noise dial ``β`` and the per-class fractions specified in config. The
    aggregate file-arrival rate is held approximately fixed across β by
    re-normalising the per-tenant rates so their sum equals
    ``aggregate_lambda_files_per_s``. This is the right comparison to do:
    we want to study what happens at a *fixed* aggregate load when the
    *composition* of that load becomes more heterogeneous, not what
    happens when we add more tenants on top.
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 rng: RNGManager, filesystem: Any) -> None:
        self._env = env
        self._cfg = cfg
        self._rng_mgr = rng
        self._fs = filesystem

        wl_cfg = cfg["workload"]
        mt_cfg = wl_cfg.get("multi_tenant", {})
        if not mt_cfg.get("enabled", False):
            raise ValueError("multi_tenant.enabled must be True to use "
                             "MultiTenantWorkloadGenerator")

        self._n_tenants = int(mt_cfg.get("n_tenants", 32))
        self._noise_level = float(np.clip(mt_cfg.get("noise_level", 0.5), 0.0, 1.0))
        # Honour explicit ``null`` in YAML by falling back to the single-tenant
        # Pareto config: ``mt_cfg.get(key, default)`` returns ``None`` when the
        # key is present-with-null, so we have to handle that case ourselves.
        agg = mt_cfg.get("aggregate_lambda_files_per_s")
        if agg is None:
            agg = self._aggregate_lambda_from_cfg(wl_cfg)
        self._aggregate_lambda = float(agg)
        self._noisy_fraction_max = float(mt_cfg.get("noisy_fraction_max", 0.10))
        self._power_fraction = float(mt_cfg.get("power_user_fraction", 0.05))
        self._sigma_rate_max = float(mt_cfg.get("rate_lognormal_sigma_max", 1.5))
        self._sigma_size_max = float(mt_cfg.get("size_perturbation_sigma_max", 0.5))
        self._burst_period_s = float(mt_cfg.get("noisy_burst_period_s", 600.0))
        self._burst_size = int(mt_cfg.get("noisy_burst_size", 50))

        # Diurnal modulator — tenants share it (time-of-day affects
        # everyone the same way, modulo opt-out).
        if wl_cfg.get("diurnal_enabled", True):
            self._diurnal = DiurnalModulator(
                amplitude=wl_cfg.get("diurnal_amplitude", 0.6),
                peak_hour=wl_cfg.get("diurnal_peak_hour", 10),
            )
        else:
            self._diurnal = None

        # Master RNG used for tenant-spec sampling — separate from the
        # per-tenant RNG streams used during run-time.
        self._spec_rng = rng.get("iat")  # reuse iat stream for spec spawn

        self._specs: List[TenantSpec] = self._sample_tenant_specs(wl_cfg)
        self._streams: List[TenantStream] = []

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _aggregate_lambda_from_cfg(wl_cfg: dict) -> float:
        """Aggregate file rate from the existing single-tenant Pareto config."""
        alpha = wl_cfg.get("iat_pareto_alpha", 1.8)
        xmin = wl_cfg.get("iat_pareto_xmin_s", 0.5)
        denom = max(alpha - 1.0, 1e-9)
        return denom / max(xmin, 1e-9)

    def _sample_tenant_specs(self, wl_cfg: dict) -> List[TenantSpec]:
        n = self._n_tenants
        β = self._noise_level

        # Class-fraction interpolation by β.
        f_noise = β * self._noisy_fraction_max
        n_noisy = int(round(f_noise * n))
        n_power = int(round(self._power_fraction * n))
        n_back = max(0, n - n_noisy - n_power)
        roles = (
            ["background"] * n_back +
            ["power"] * n_power +
            ["noisy"] * n_noisy
        )
        # Shuffle so tenant indices don't carry role information.
        roles = list(roles)
        self._spec_rng.shuffle(roles)

        # Per-tenant rates (files/s). Sample dimensionless factors from a
        # LogNormal centered at 1, apply class-level multipliers, then
        # renormalise so the *non-noisy* contribution sums to the
        # configured aggregate rate scaled by the non-noisy fraction.
        # That way β doesn't silently inflate background load.
        sigma_rate = β * self._sigma_rate_max
        non_noisy_idx = np.array([r != "noisy" for r in roles])
        n_non_noisy = int(non_noisy_idx.sum())
        if sigma_rate < 1e-9:
            raw = np.ones(n)
        else:
            raw = self._spec_rng.lognormal(
                mean=-0.5 * sigma_rate**2, sigma=sigma_rate, size=n,
            )
        # Class multiplier: power users get ~10× the background mean.
        class_mul = np.array([
            10.0 if r == "power" else (1.0 if r == "background" else 1.0)
            for r in roles
        ])
        raw = raw * class_mul

        # Convert to files-per-second per tenant. Each tenant slot is
        # worth 1/n of the aggregate; the non-noisy slice therefore sums
        # to ``aggregate × (n_non_noisy/n)`` while noisy slots replace
        # their Pareto share with bursts (rate computed separately
        # downstream from burst_period × burst_size). This makes the
        # noise dial a *substitution* knob rather than an additive one,
        # and keeps the total Pareto load well-defined as β changes.
        rates = np.full(n, np.nan)
        if n_non_noisy > 0:
            non_noisy_target_sum = self._aggregate_lambda * (n_non_noisy / n)
            current = float(raw[non_noisy_idx].sum())
            if current <= 0:
                rates[non_noisy_idx] = non_noisy_target_sum / n_non_noisy
            else:
                share = raw[non_noisy_idx] / current   # sums to 1
                rates[non_noisy_idx] = non_noisy_target_sum * share

        # File-size mu perturbation (per tenant).
        sigma_size = β * self._sigma_size_max
        size_mu_offset = (
            self._spec_rng.normal(0, sigma_size, size=n) if sigma_size > 0
            else np.zeros(n)
        )

        base_alpha = float(wl_cfg.get("iat_pareto_alpha", 1.8))
        base_xmin = float(wl_cfg.get("iat_pareto_xmin_s", 0.5))
        base_size_mu = float(wl_cfg.get("file_size_lognormal_mu", 16.118))
        base_size_sigma = float(wl_cfg.get("file_size_lognormal_sigma", 2.5))
        base_zipf_s = float(wl_cfg.get("zipf_s", 1.0))
        base_read_frac = float(wl_cfg.get("read_fraction", 0.7))
        base_ws = int(wl_cfg.get("working_set_size", 10000))

        specs: List[TenantSpec] = []
        for i, role in enumerate(roles):
            tid = f"t{i:04d}"

            # Convert per-tenant rate (files/s) to a Lomax xmin so the
            # tenant's IAT distribution has the right long-run mean.
            #   For Lomax(α, xmin): rate = (α − 1) / xmin
            #     ⟹ xmin = (α − 1) / rate
            # Noisy tenants have rate=NaN from the assignment above; they
            # use a placeholder xmin (never sampled, since the noisy
            # stream samples bursts, not Pareto IATs).
            if role == "noisy":
                tenant_xmin = base_xmin
            else:
                rate = max(float(rates[i]), 1e-9)
                tenant_xmin = max(base_alpha - 1.0, 1e-6) / rate

            # Per-tenant file-size mu perturbation; power users larger files.
            size_mu = base_size_mu + size_mu_offset[i]
            if role == "power":
                size_mu += 1.5    # ~4.5× larger median
            elif role == "noisy":
                size_mu += 0.7    # ~2× larger (checkpoint files)

            # Per-tenant working-set size (smaller for power, smallest for noisy).
            if role == "power":
                ws = max(100, int(base_ws * 0.1))
            elif role == "noisy":
                ws = max(10, int(base_ws * 0.02))
            else:
                ws = max(50, int(base_ws / max(n, 1)))

            spec = TenantSpec(
                tenant_id=tid, role=role,
                iat_pareto_alpha=base_alpha,
                iat_pareto_xmin_s=max(tenant_xmin, 1e-3),
                file_size_lognormal_mu=size_mu,
                file_size_lognormal_sigma=base_size_sigma,
                working_set_size=ws,
                zipf_s=base_zipf_s,
                read_fraction=base_read_frac,
                burst_period_s=self._burst_period_s if role == "noisy" else 0.0,
                burst_size=self._burst_size if role == "noisy" else 0,
                diurnal_enabled=(role != "noisy"),
            )
            specs.append(spec)
        return specs

    # -- public API -----------------------------------------------------------

    @property
    def tenant_specs(self) -> List[TenantSpec]:
        return list(self._specs)

    def run(self) -> Iterator:
        """Spawn one SimPy process per tenant, then yield once so the
        environment registers them and starts dispatching."""
        master_seed = self._rng_mgr.master_seed
        ss = np.random.SeedSequence(master_seed).spawn(self._n_tenants + 1)
        # Last seed slot reserved for any future cross-tenant coordination.
        for i, spec in enumerate(self._specs):
            tenant_rng = np.random.default_rng(ss[i])
            cls = NoisyTenantStream if spec.role == "noisy" else TenantStream
            stream = cls(self._env, spec, tenant_rng, self._fs, self._diurnal)
            self._streams.append(stream)
            self._env.process(stream.run())
        # Keep the orchestrator alive for the lifetime of the simulation
        # so any later supervision (rebalancing, joining tenants, …) has
        # a place to live. For now we just sleep forever.
        while True:
            yield self._env.timeout(86400.0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-tenant fairness / KPI helpers
# ─────────────────────────────────────────────────────────────────────────────

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative sample. 0 = perfectly equal,
    1 = maximally unequal. Useful as a fairness summary across tenants
    (e.g. on per-tenant P95 recall latency: high Gini ⇒ a small set of
    victims absorbs most of the noisy-neighbour pain)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0 or arr.min() < 0:
        return float("nan")
    if arr.sum() == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * sorted_arr).sum() / (n * sorted_arr.sum()))
