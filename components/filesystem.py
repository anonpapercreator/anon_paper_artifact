"""
components/filesystem.py — Disk Cache, Interconnect, Data Movers, Policy Engine
================================================================================
Models the disk-side infrastructure of DMF 7 and the policy engine.

    1. DiskCache       — G/G/1 queue with capacity, watermark tracking
    2. Interconnect    — M/D/1 queue (bandwidth-limited SAN)
    3. DataMoverPool   — N parallel M/G/1 data mover servers
    4. PolicyEngine    — DMF 7 faithful policy: cycles, watermarks, accrual

See MATHEMATICAL_MODEL.md §4.1, §4.2, §4.3, §5.
"""

from __future__ import annotations
import simpy
import numpy as np
import math
from typing import Dict, List, Optional, Any, Iterator
from collections import deque

from core.rng import RNGManager
from core.events import FileRecord, FileState, SimEvent, EventType
from stats.collector import ReplicationStats


# ---------------------------------------------------------------------------
# Disk Cache (G/G/1 with finite buffer)
# ---------------------------------------------------------------------------

class DiskCache:
    """
    Models the Lustre/HPC disk filesystem tier as a G/G/1 queue with a
    hard finite-capacity gate.

    Service time distribution: LogNormal(μ, σ) — see MATHEMATICAL_MODEL.md §4.1.

    Capacity gate:
        A real DMF-fronted Lustre cache cannot grow past its configured
        capacity. Writes that would cause overflow block on the underlying
        filesystem until space is freed (DMF's policy engine triggers
        ``dmfsfree`` to release oldest DUL → OFL). We model this with a
        ``simpy.Container`` whose level represents *free* bytes; writes
        ``get`` the size they need (blocking when no room) and space
        releases ``put`` bytes back. ``_used_bytes`` is therefore derived
        from the container level and is bounded by ``[0, capacity]`` by
        construction.

    Tracks:
        - Disk usage (bytes) against HWM/LWM thresholds (derived).
        - Cache hit / miss (file in cache vs. offline recall needed).
        - Queue depth.
        - Cumulative write blocking time (back-pressure indicator).
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 rng: RNGManager, stats: ReplicationStats) -> None:
        self._env = env
        self._cfg = cfg
        self._stats = stats

        self._capacity    = cfg["capacity_bytes"]
        self._hwm         = cfg["hwm_fraction"] * self._capacity
        self._lwm         = cfg["lwm_fraction"] * self._capacity

        # Capacity gate. Container level is the free-byte budget; an
        # initial_fill_fraction of c reserves c·capacity bytes at sim start
        # so the policy engine has a realistic starting watermark state.
        initial_used = int(cfg["initial_fill_fraction"] * self._capacity)
        self._capacity_gate = simpy.Container(
            env, init=self._capacity - initial_used, capacity=self._capacity,
        )

        # Disk I/O resource (max concurrent IOs)
        max_io = cfg.get("max_concurrent_io", 128)
        self._io_resource = simpy.Resource(env, capacity=max_io)

        # Service time distribution
        self._svc_mu    = cfg["io_service_time_lognormal_mu_s"]
        self._svc_sigma = cfg["io_service_time_lognormal_sigma_s"]
        self._meta_lat  = cfg.get("metadata_lookup_latency_s", 0.005)
        self._rng_disk  = rng.get("disk_service")

        # Aggregate disk throughput for size-dependent service times (bytes/s)
        # Default 65 GiB/s = ~70 GB/s (ESS 3500 aggregate)
        self._agg_bw = cfg.get("aggregate_throughput_gib_s", 65.0) * 1024**3

        # Registered policy engine callback (set by PolicyEngine after construction)
        self._policy_notify_callback = None

        # Cumulative wall-time spent blocked on the capacity gate.
        self._total_write_block_s: float = 0.0

    @property
    def used_bytes(self) -> int:
        # Derived from the capacity gate: free + used = capacity.
        return int(self._capacity - self._capacity_gate.level)

    @property
    def usage_fraction(self) -> float:
        return self.used_bytes / self._capacity if self._capacity > 0 else 0.0

    @property
    def above_hwm(self) -> bool:
        return self.used_bytes >= self._hwm

    @property
    def above_lwm(self) -> bool:
        return self.used_bytes >= self._lwm

    @property
    def total_write_block_s(self) -> float:
        """Total seconds writes were blocked waiting for space across this
        replication. Useful as a write-back-pressure / SLA-violation signal."""
        return self._total_write_block_s

    def set_policy_callback(self, cb) -> None:
        self._policy_notify_callback = cb

    def _sample_service_time(self, size_bytes: int = 0) -> float:
        """
        Sample disk I/O service time from LogNormal distribution.

        When ``size_bytes`` is provided, the service time scales with file
        size to model realistic disk throughput:
            t_svc = (size_bytes / aggregate_bw) + lognormal_jitter

        When ``size_bytes`` is zero (metadata-only ops), only jitter is used.
        """
        base = max(1e-6, self._rng_disk.lognormal(self._svc_mu, self._svc_sigma))
        if size_bytes > 0 and self._agg_bw > 0:
            # Add size-dependent component bounded by aggregate throughput
            size_component = size_bytes / self._agg_bw
            return base + size_component
        return base

    def add_file(self, file_rec: FileRecord) -> None:
        """Reserve a file's space on disk.

        Use ``handle_write`` for the full blocking-aware path. This helper is
        kept for diagnostic / test injection only — it does *not* consult
        the capacity gate, so external callers are responsible for not
        over-filling the cache. The recall path uses it because by then
        the file already had a disk allocation that was released on
        space-release; refilling on recall is bounded by the same gate via
        ``recall_alloc``.
        """
        # Container.get / put may not be safe to invoke from sync code in
        # all SimPy versions; use the level math but fall back to clamping.
        sz = max(0, int(file_rec.size_bytes))
        avail = int(self._capacity_gate.level)
        take = min(sz, avail)
        if take > 0:
            self._capacity_gate.get(take)
        self._stats.record_disk_usage(self._env.now, self.usage_fraction)

    def remove_file(self, file_rec: FileRecord) -> None:
        """Release a file's space from disk (space release after migration)."""
        sz = max(0, int(file_rec.size_bytes))
        # Don't put back more than was used; otherwise the gate level can
        # exceed capacity (an exception in simpy.Container).
        used_now = int(self._capacity - self._capacity_gate.level)
        give = min(sz, used_now)
        if give > 0:
            self._capacity_gate.put(give)
        self._stats.record_disk_usage(self._env.now, self.usage_fraction)

    def recall_alloc(self, file_rec: FileRecord) -> Iterator:
        """SimPy process: reserve disk space for a recall (UNM → DUL),
        blocking on the capacity gate when the cache is full."""
        size = max(0, int(file_rec.size_bytes))
        if size == 0:
            return
        # Recalls do NOT check free space (faithful to the real system): they
        # acquire space and complete. The capacity gate still enforces a true
        # ENOSPC block if the filesystem is genuinely full, but in normal
        # operation the independent free-space trigger keeps usage below that.
        block_start = self._env.now
        yield self._capacity_gate.get(size)
        self._total_write_block_s += self._env.now - block_start
        self._stats.record_disk_usage(self._env.now, self.usage_fraction)
        # Usage just rose -> fire the event-driven free-space trigger.
        if self._policy_notify_callback is not None and self.above_hwm:
            self._policy_notify_callback(self._env.now)

    def handle_write(self, file_rec: FileRecord) -> Iterator:
        """SimPy process: handle a file write to disk cache.

        First reserves capacity (blocking on the gate when the cache is
        full); then runs the I/O service path. Modelling write blocking on
        a finite cache is what keeps ``used_bytes`` bounded by capacity
        and propagates space-release pressure back to the workload.
        """
        size = max(0, int(file_rec.size_bytes))

        # 1) Capacity reservation. Tracks blocking time so the report can
        #    expose write back-pressure directly.
        arrival_time = self._env.now
        self._stats.queues["disk_cache"].arrival(self._env.now)
        block_start = self._env.now
        if size > 0:
            yield self._capacity_gate.get(size)
        self._total_write_block_s += self._env.now - block_start
        # Usage just rose -> fire the event-driven free-space trigger.
        if self._policy_notify_callback is not None and self.above_hwm:
            self._policy_notify_callback(self._env.now)

        # 2) I/O service path.
        req = self._io_resource.request()
        yield req
        yield self._env.timeout(self._meta_lat)
        t_svc = self._sample_service_time(file_rec.size_bytes)
        yield self._env.timeout(t_svc)
        self._io_resource.release(req)
        self._stats.queues["disk_cache"].departure(self._env.now, arrival_time)

        # 3) State transition + telemetry.
        self._stats.record_disk_usage(self._env.now, self.usage_fraction)
        file_rec.state = FileState.REG

    def handle_access(self, file_rec: FileRecord) -> Iterator:
        """
        SimPy process: handle a file access.
        - If file is online (REG or DUL): cache hit, serve from disk.
        - If file is offline (OFL): cache miss, trigger recall (handled by caller).
        """
        arrival_time = self._env.now

        if file_rec.state in (FileState.REG, FileState.DUL):
            # Cache hit
            self._stats.record_cache_event(hit=True, time=self._env.now)
            self._stats.queues["disk_cache"].arrival(self._env.now)

            req = self._io_resource.request()
            yield req
            yield self._env.timeout(self._meta_lat)
            t_svc = self._sample_service_time(file_rec.size_bytes)
            yield self._env.timeout(t_svc)
            self._io_resource.release(req)

            self._stats.queues["disk_cache"].departure(self._env.now, arrival_time)

        else:
            # Cache miss (file is OFL or other non-accessible state)
            self._stats.record_cache_event(hit=False, time=self._env.now)
            # Recall is handled by the caller (filesystem.handle_access)

    def handle_explicit_migrate(self, file_rec: FileRecord) -> Iterator:
        """No-op at disk layer; migration is dispatched to tape subsystem."""
        yield self._env.timeout(0)


# ---------------------------------------------------------------------------
# Interconnect (M/D/1 — bandwidth-limited)
# ---------------------------------------------------------------------------

class Interconnect:
    """
    Models the SAN interconnect (InfiniBand / 100GbE) between filesystem
    nodes and DMF data movers as an M/D/1 queue.

    Service time is deterministic for a given file size:
        S(f) = f.size_bytes / bandwidth + fixed_latency

    See MATHEMATICAL_MODEL.md §4.2.
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 stats: ReplicationStats) -> None:
        self._env   = env
        self._stats = stats
        self._bw    = cfg["bandwidth_bytes_per_s"]
        self._lat   = cfg["fixed_latency_s"]
        max_xfer    = cfg.get("max_concurrent_transfers", 64)
        self._res   = simpy.Resource(env, capacity=max_xfer)

    def transfer(self, file_rec: FileRecord) -> Iterator:
        """SimPy process: transfer a file across the interconnect."""
        arrival_time = self._env.now
        self._stats.queues["interconnect"].arrival(self._env.now)

        req = self._res.request()
        yield req

        t_svc = file_rec.size_bytes / max(self._bw, 1) + self._lat
        yield self._env.timeout(t_svc)

        self._res.release(req)
        self._stats.queues["interconnect"].departure(self._env.now, arrival_time)


# ---------------------------------------------------------------------------
# Data Mover Pool (N parallel M/G/1 servers)
# ---------------------------------------------------------------------------

class DataMoverPool:
    """
    Models N parallel DMF data mover nodes as M/G/1 servers.

    Service time per mover:
        S(f) = f.size_bytes / effective_bandwidth + overhead

    Effective bandwidth accounts for NODE_BANDWIDTH, HBA_BANDWIDTH,
    and BANDWIDTH_MULTIPLIER (compression).

    See MATHEMATICAL_MODEL.md §4.3.
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 stats: ReplicationStats) -> None:
        self._env    = env
        self._stats  = stats
        self._n      = cfg["n_movers"]
        self._overhead = cfg["mover_overhead_s"]

        # Effective bandwidth: min(node_bw, hba_bw × compression)
        node_bw = cfg["node_bandwidth_bytes_per_s"]
        hba_bw  = cfg["hba_bandwidth_bytes_per_s"]
        mult    = cfg["bandwidth_multiplier"]
        self._eff_bw = min(node_bw, hba_bw * mult)

        # Pool of mover resources (one per mover node)
        self._movers = [simpy.Resource(env, capacity=1) for _ in range(self._n)]

    def _select_mover(self) -> int:
        """Select the least-loaded mover (shortest queue)."""
        return min(range(self._n), key=lambda i: len(self._movers[i].queue))

    def transfer(self, file_rec: FileRecord) -> Iterator:
        """SimPy process: transfer a file through the mover pool."""
        mover_id = self._select_mover()
        res = self._movers[mover_id]
        arrival_time = self._env.now
        self._stats.queues["mover_pool"].arrival(self._env.now)

        req = res.request()
        yield req

        t_svc = (file_rec.size_bytes / max(self._eff_bw, 1)) + self._overhead
        yield self._env.timeout(t_svc)

        res.release(req)
        self._stats.queues["mover_pool"].departure(self._env.now, arrival_time)


# ---------------------------------------------------------------------------
# DMF Policy Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    Models the DMF 7 policy engine.

    Behaviour is faithful to DMF 7:
        - Runs periodically (T_cycle_period)
        - Selects migration candidates (age ≥ t_age AND state == REG)
        - Applies watermark hysteresis (HWM → migrate until LWM)
        - Sorts candidates by configured priority (size_desc, age_desc, combined)
        - Dispatches to tape subsystem via LS accrual buffer
        - Handles trickle migration (DMMIGRATE_TRICKLE / DMMIGRATE_UNACK)

    See MATHEMATICAL_MODEL.md §5.
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 disk_cache: DiskCache, tape_subsystem: Any,
                 interconnect: Interconnect, movers: DataMoverPool,
                 rng: RNGManager, stats: ReplicationStats) -> None:
        self._env    = env
        self._cfg    = cfg
        self._disk   = disk_cache
        self._tape   = tape_subsystem
        self._net    = interconnect
        self._movers = movers
        self._stats  = stats

        self._cycle_period  = cfg["cycle_period_s"]
        self._age_threshold = cfg["candidate_age_threshold_s"]
        self._priority      = cfg.get("migration_priority", "size_descending")
        self._space_release = cfg.get("space_release_on_migration", False)
        self._n_workers     = cfg.get("n_policy_workers", 4)
        self._recall_sort   = cfg.get("recall_sort", "tape_order")

        # Trickle semaphore (DMMIGRATE_UNACK).
        # The DMF policy engine doesn't own the trickle config — it lives in
        # data_movers.* in the YAML — so PolicyEngine just exposes the field
        # for tests / instrumentation but does not gate dispatch on it. The
        # actual trickle limit is enforced at the LibraryServer / data-mover
        # layers where parallelism is allocated. Leaving the field as None
        # by default avoids creating a no-op SimPy resource.
        self._trickle_sem = None

        # File registry: maps file_id -> FileRecord
        self._file_registry: Dict[str, FileRecord] = {}

        # Small jitter on policy cycle to avoid thundering herd
        self._rng_jitter = rng.get("policy_jitter")

    def register_file(self, file_rec: FileRecord) -> None:
        """Register a file with the policy engine."""
        self._file_registry[file_rec.file_id] = file_rec

    def deregister_file(self, file_id: str) -> None:
        """Remove a file from the policy engine's scope."""
        self._file_registry.pop(file_id, None)

    def set_age_threshold(self, val: float) -> None:
        """Dynamically adjust the migration age threshold (seconds)."""
        self._age_threshold = max(0.0, float(val))

    def set_cycle_period(self, val: float) -> None:
        """Dynamically adjust the policy cycle period (seconds)."""
        self._cycle_period = max(1.0, float(val))

    def run(self) -> Iterator:
        """
        SimPy process: the MIGRATION loop, independent of free space.

        Faithful to the real policy: a migration run starts every
        ``cycle_period_s`` (6 h) measured from the PREVIOUS run's START, and a
        migration cannot overlap another on the same filesystem (this single
        process is the per-filesystem serialisation; it does not re-enter until
        its tape work has been dispatched). If a run overruns the 6 h window the
        next starts as soon as it finishes. Migration is purely age-based and is
        NOT gated by watermarks -- free space is handled separately by an
        event-driven trigger (see ``free_space_check``).
        """
        while True:
            start = self._env.now
            self._stats.n_policy_cycles += 1
            yield from self._migration_cycle()
            # Next migration 6 h after THIS one started; if already past, go now.
            next_at = start + self._cycle_period
            yield self._env.timeout(max(0.0, next_at - self._env.now))

    def _migration_cycle(self) -> Iterator:
        """One migration run: migrate REG files older than the age threshold
        (REG -> MIG -> DUL). Age-based only; no watermark gating, no space
        release (space release is the independent free-space trigger)."""
        now = self._env.now
        candidates = [
            f for f in self._file_registry.values()
            if f.state == FileState.REG
            and (now - f.last_touched_at) >= self._age_threshold
        ]
        if not candidates:
            return
        candidates = self._sort_candidates(candidates, now)
        n_dispatched = 0
        for f in candidates:
            if f.state != FileState.REG:
                continue
            # Dispatch as independent processes (concurrent migrations, capped
            # at the LS/drive layer) -- not serialised on each flush wait.
            self._env.process(self._migrate_file(f))
            n_dispatched += 1
        if n_dispatched:
            yield self._env.timeout(0)

    def free_space_check(self, now: float) -> None:
        """Event-driven free-space trigger (the dmfsfree-equivalent).

        Fired whenever disk usage rises (recall lands, write completes). When
        usage crosses the high watermark, release oldest DUL files (DUL -> OFL)
        until usage drops to the low watermark. DUL-only: if there are no
        dual-state files to release, no space is freed and I/O will block at the
        capacity gate (a real, rare ENOSPC under a recall storm). Independent of
        the migration loop.
        """
        if self._disk.above_hwm:
            self._release_dul_until_lwm(now)

    def _release_dul_until_lwm(self, now: float) -> None:
        """Transition oldest-touch DUL files to OFL until the disk drops
        below LWM. Selection is by ``last_touched_at`` ascending (LRU-style),
        which matches ``dmfsfree``'s default age-based release order."""
        dul_candidates = [
            f for f in self._file_registry.values()
            if f.state == FileState.DUL
        ]
        if not dul_candidates:
            return

        # Oldest-touched first. Tie-break by size descending so big files
        # release more space per call when ages match.
        dul_candidates.sort(key=lambda f: (f.last_touched_at, -f.size_bytes))

        for f in dul_candidates:
            if not self._disk.above_lwm:
                break
            if f.state != FileState.DUL:
                continue   # changed state mid-loop
            f.apply_transition("space_release")  # DUL → OFL
            self._disk.remove_file(f)
            if self._env.now >= self._stats.warmup_s:
                self._stats.n_space_releases += 1

    def _sort_candidates(self, candidates: List[FileRecord], now: float) -> List[FileRecord]:
        """Sort candidates by migration priority."""
        if self._priority == "size_descending":
            return sorted(candidates, key=lambda f: f.size_bytes, reverse=True)
        elif self._priority == "age_descending":
            return sorted(
                candidates,
                key=lambda f: now - f.last_touched_at,
                reverse=True,
            )
        elif self._priority == "combined":
            # Combined: normalised rank of (size + age)
            max_sz  = max(f.size_bytes for f in candidates) or 1
            max_age = max(now - f.last_touched_at for f in candidates) or 1
            def score(f):
                sz_score  = f.size_bytes / max_sz
                age_score = (now - f.last_touched_at) / max_age
                return sz_score + age_score
            return sorted(candidates, key=score, reverse=True)
        else:
            return candidates

    def _migrate_file(self, file_rec: FileRecord) -> Iterator:
        """
        Execute one file migration:
            1. Transition file state: REG → MIG
            2. Transfer across interconnect
            3. Process through data mover
            4. Dispatch to LS (tape write)
            5. Transition: MIG → DUL (→ OFL if space_release_on_migration)
        """
        start_time = self._env.now
        file_rec.apply_transition("policy_select")  # REG → MIG

        # Interconnect
        yield from self._net.transfer(file_rec)

        # Data mover
        yield from self._movers.transfer(file_rec)

        # Tape subsystem (LS accrual + drive write)
        yield from self._tape.migrate(file_rec)

        # State transition: MIG → DUL
        file_rec.apply_transition("migrate_complete")
        file_rec.migrated_at = self._env.now

        # Interval-model accounting: the tape copy now exists.
        now = self._env.now
        if now >= self._stats.warmup_s:
            self._stats.lt_bytes_archived += file_rec.size_bytes
            # byte-weighted pipeline delay (selection -> tape copy complete)
            self._stats.lt_bytes_x_pipeline_s += file_rec.size_bytes * (now - start_time)
        if file_rec.deleted_at is not None:
            # Deleted while the copy was in flight: the copy is garbage on
            # arrival. The unprotected level was already reduced at deletion.
            if now >= self._stats.warmup_s:
                self._stats.lt_bytes_tape_garbage += file_rec.size_bytes
            self._disk.remove_file(file_rec)
            self.deregister_file(file_rec.file_id)
            self._stats.record_migrate_latency(now - start_time, time=now)
            return
        self._stats.unprot_change(now, -file_rec.size_bytes)

        # If configured for immediate space release: DUL → OFL
        if self._space_release:
            file_rec.apply_transition("space_release")
            self._disk.remove_file(file_rec)

        # Releasable bytes just appeared. The usage-rise trigger alone cannot
        # see this: if the capacity gate has blocked all writes while no DUL
        # existed, no write ever completes again, free_space_check is never
        # invoked, and the filesystem deadlocks full even as DUL accumulates.
        # dmfsfree monitors the filesystem continuously, so a DUL landing on
        # an over-threshold filesystem must be eligible for release now.
        self.free_space_check(self._env.now)

        latency = self._env.now - start_time
        self._stats.record_migrate_latency(latency, time=self._env.now)

    def trigger_recall(self, file_rec: FileRecord,
                       priority: int = 5) -> Iterator:
        """
        Execute a recall for an offline file:
            1. Transition: OFL → UNM
            2. Submit to tape subsystem (LS recall queue)
            3. Transfer across interconnect (tape → disk)
            4. Transition: UNM → DUL
            5. Add space back to disk cache
        """
        if file_rec.state not in (FileState.OFL, FileState.PAR):
            return

        file_rec.apply_transition("read_access")  # OFL → UNM
        file_rec.recall_request_at = self._env.now
        file_rec.recall_priority = priority

        # Tape recall
        yield from self._tape.recall(file_rec, priority)

        # Interconnect (tape → disk)
        yield from self._net.transfer(file_rec)

        # Data mover
        yield from self._movers.transfer(file_rec)

        # Restore to disk via the capacity gate (blocks if cache is full,
        # forcing space release pressure to propagate before the recall
        # completes). This keeps ``used_bytes <= capacity`` invariantly.
        yield from self._disk.recall_alloc(file_rec)
        file_rec.apply_transition("recall_complete")  # UNM → DUL
        file_rec.recalled_at = self._env.now
        file_rec.n_recalls += 1
        if file_rec.deleted_at is not None:
            # Deleted while the recall was in flight: discard the restored copy.
            self._disk.remove_file(file_rec)
            self.deregister_file(file_rec.file_id)


# ---------------------------------------------------------------------------
# Filesystem Façade (top-level component)
# ---------------------------------------------------------------------------

class Filesystem:
    """
    Top-level façade for the disk-side DMF 7 components.
    Provides the interface used by the workload generator.
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 tape_subsystem: Any, rng: RNGManager,
                 stats: ReplicationStats) -> None:
        self._env    = env
        self._stats  = stats

        self.disk       = DiskCache(env, cfg["disk_cache"], rng, stats)
        self.net        = Interconnect(env, cfg["interconnect"], stats)
        self.movers     = DataMoverPool(env, cfg["data_movers"], stats)
        self.policy     = PolicyEngine(
            env, cfg["policy_engine"], self.disk, tape_subsystem,
            self.net, self.movers, rng, stats
        )

        # HWM-triggered eviction: let the disk cache call the policy's space-release
        # (DUL -> OFL down to LWM) synchronously when an allocation finds no room.
        self.disk.set_policy_callback(self.policy.free_space_check)

        # Start policy engine process
        env.process(self.policy.run())

        # Optional file lifetimes (workload.lifetime). When disabled, files are
        # never deleted, which reproduces the behaviour of earlier versions.
        self._rng_life = None
        self._lifetime = None
        lt = cfg.get("workload", {}).get("lifetime", {}) or {}
        if lt.get("enabled", False):
            from models.interval_model import Lifetime
            self._lifetime = Lifetime(s_inf=float(lt.get("s_inf", 1.0)),
                                      kind=str(lt.get("kind", "exp")),
                                      mean_s=float(lt.get("mean_s", 3600.0)),
                                      sigma=float(lt.get("sigma", 1.0)))
            self._rng_life = rng.extra("file_lifetime")

    def handle_write(self, file_rec: FileRecord) -> Iterator:
        """Handle a new file write to the filesystem."""
        yield from self.disk.handle_write(file_rec)
        self.policy.register_file(file_rec)
        self._stats.n_files_created += 1
        now = self._env.now
        self._stats.unprot_change(now, file_rec.size_bytes)
        if now >= self._stats.warmup_s:
            self._stats.lt_bytes_created += file_rec.size_bytes
        if self._lifetime is not None:
            L = float(self._lifetime.sample(self._rng_life, 1)[0])
            if math.isfinite(L):
                # Lifetime runs from creation; the write itself may have waited
                # on the capacity gate, so the remaining life is L - elapsed.
                remaining = max(0.0, file_rec.created_at + L - now)
                self._env.process(self._delete_after(file_rec, remaining))
        self._stats.record_cache_write(
            file_rec.size_bytes, time=self._env.now,
            tenant_id=file_rec.tenant_id,
        )

    def _delete_after(self, file_rec: FileRecord, delay: float) -> Iterator:
        yield self._env.timeout(delay)
        self.delete_file(file_rec)

    def delete_file(self, file_rec: FileRecord) -> None:
        """Delete a file. REG bytes leave the unprotected pool without a tape
        copy; bytes with a tape copy (DUL, OFL, PAR, UNM) become tape garbage;
        MIG and UNM deletions are finished by their completion handlers."""
        if file_rec.deleted_at is not None:
            return
        now = self._env.now
        file_rec.deleted_at = now
        post = now >= self._stats.warmup_s
        size = file_rec.size_bytes
        st = file_rec.state
        if st == FileState.REG:
            self.disk.remove_file(file_rec)
            self.policy.deregister_file(file_rec.file_id)
            self._stats.unprot_change(now, -size)
            if post:
                self._stats.lt_bytes_deleted_unarchived += size
        elif st == FileState.MIG:
            self._stats.unprot_change(now, -size)
        elif st == FileState.DUL:
            self.disk.remove_file(file_rec)
            self.policy.deregister_file(file_rec.file_id)
            if post:
                self._stats.lt_bytes_tape_garbage += size
        elif st in (FileState.OFL, FileState.PAR):
            self.policy.deregister_file(file_rec.file_id)
            if post:
                self._stats.lt_bytes_tape_garbage += size
        elif st == FileState.UNM:
            if post:
                self._stats.lt_bytes_tape_garbage += size

    def handle_access(self, file_rec: FileRecord) -> Iterator:
        """Handle a file access (read). Records latency for all recall paths."""
        start_time = self._env.now
        if file_rec.state in (FileState.OFL, FileState.PAR):
            # Cache miss: account for it explicitly (the disk-side miss path
            # is unreachable from this branch) and trigger the recall.
            self._stats.record_cache_event(hit=False, time=self._env.now)
            yield from self.policy.trigger_recall(file_rec, priority=10)
        else:
            yield from self.disk.handle_access(file_rec)
            self._stats.record_cache_read(
                file_rec.size_bytes, time=self._env.now,
                tenant_id=file_rec.tenant_id,
            )
        # Record total access latency (includes both fast disk and slow tape paths)
        self._stats.record_recall_latency(
            self._env.now - start_time, time=self._env.now,
            tenant_id=file_rec.tenant_id,
        )

    def handle_explicit_migrate(self, file_rec: FileRecord) -> Iterator:
        """Handle an explicit dmput migration request."""
        if file_rec.state == FileState.REG:
            yield from self.policy._migrate_file(file_rec)
