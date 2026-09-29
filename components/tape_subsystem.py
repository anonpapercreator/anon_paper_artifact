"""
components/tape_subsystem.py — Tape Robot, Drives, Volume Manager, LS Accrual Buffer
=====================================================================================
Models the complete tape-side infrastructure of DMF 7:

    1. TapeVolume       — A single tape cartridge (VSN) with position tracking
    2. VolumeGroup      — A VG containing N volumes, assigned to specific drives
    3. TapeRobot        — Robotic arm (single or dual) with geometry model
    4. TapeDrive        — Individual drive with mount/seek/transfer/unload model
    5. LibraryServer    — LS accrual buffer (DMF 7-specific batching behaviour)
    6. TapeSubsystem    — Top-level coordinator

Mathematical models documented in MATHEMATICAL_MODEL.md §4.5 and §4.6.
"""

from __future__ import annotations
import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

from core.rng import RNGManager, lognormal_params_from_mean_std
from core.events import FileRecord, FileState, SimEvent, EventType
from stats.collector import ReplicationStats


# ---------------------------------------------------------------------------
# Tape Volume (VSN)
# ---------------------------------------------------------------------------

@dataclass
class TapeVolume:
    """A single tape cartridge managed by a VG."""
    vsn: str
    vg_name: str
    capacity_bytes: int
    used_bytes: int = 0
    reserved_bytes: int = 0           # Space promised to in-flight (not yet committed) writes
    current_position_bytes: int = 0   # Current tape head position (for seek model)
    slot_index: int = 0               # Physical slot in the library
    n_mounts: int = 0                 # Total mount count (for cleaning trigger)
    in_drive: Optional[int] = None    # Drive index if currently loaded

    @property
    def free_bytes(self) -> int:
        # Subtract reservations so concurrent selectors do not double-allocate a
        # volume during the gap between selection and the (yielding) write.
        return self.capacity_bytes - self.used_bytes - self.reserved_bytes

    @property
    def fill_fraction(self) -> float:
        return self.used_bytes / self.capacity_bytes if self.capacity_bytes > 0 else 0.0

    def reserve(self, size_bytes: int) -> None:
        """Reserve space at selection time (before the yielding write)."""
        self.reserved_bytes += size_bytes

    def release_reservation(self, size_bytes: int) -> None:
        """Release a reservation that will not be committed (e.g. write failed)."""
        self.reserved_bytes = max(0, self.reserved_bytes - size_bytes)

    def write_file(self, size_bytes: int, reserved: bool = False) -> int:
        """
        Write a file to the volume. Returns the block offset written.

        If ``reserved`` is True the space was already accounted for by reserve()
        at selection time; this call commits it (moves reserved -> used). Raises
        ValueError only if the (unreserved) request genuinely does not fit.
        """
        if reserved:
            # Commit a previously reserved write: reserved -> used.
            self.reserved_bytes = max(0, self.reserved_bytes - size_bytes)
        elif size_bytes > self.free_bytes:
            raise ValueError(
                f"Volume {self.vsn} full: "
                f"requested {size_bytes}, available {self.free_bytes}"
            )
        offset = self.current_position_bytes
        self.current_position_bytes += size_bytes
        self.used_bytes += size_bytes
        return offset


# ---------------------------------------------------------------------------
# Volume Group
# ---------------------------------------------------------------------------

class VolumeGroup:
    """
    Manages a pool of tape volumes (VSNs) assigned to a set of drives.
    Implements volume selection policy: sequential (append to current) or round-robin.
    """

    def __init__(self, name: str, cfg: dict) -> None:
        self.name = name
        self._drives: List[int] = cfg["drives_assigned"]
        self._capacity_per_vol = cfg["volume_capacity_bytes"]
        self._n_volumes = cfg["n_volumes"]
        self._strategy = cfg.get("allocation_strategy", "sequential")

        # Create volumes
        self._volumes: List[TapeVolume] = [
            TapeVolume(
                vsn=f"{name}_V{i:04d}",
                vg_name=name,
                capacity_bytes=self._capacity_per_vol,
                slot_index=i,
            )
            for i in range(self._n_volumes)
        ]
        self._current_write_vol_idx: int = 0

    @property
    def drive_indices(self) -> List[int]:
        return self._drives

    def get_write_volume(self, required_bytes: int) -> TapeVolume:
        """
        Find a volume with sufficient free space for a write.
        Advances to next volume if current is full.

        Raises:
            RuntimeError: If all volumes are full.
        """
        start = self._current_write_vol_idx
        for _ in range(self._n_volumes):
            vol = self._volumes[self._current_write_vol_idx]
            if vol.free_bytes >= required_bytes:
                vol.reserve(required_bytes)   # atomic: later selectors see it consumed
                return vol
            self._current_write_vol_idx = (
                (self._current_write_vol_idx + 1) % self._n_volumes
            )
            if self._current_write_vol_idx == start:
                raise RuntimeError(f"VG {self.name}: all volumes full.")
        raise RuntimeError(f"VG {self.name}: no available volume found.")

    def get_volume(self, vsn: str) -> Optional[TapeVolume]:
        """Look up a volume by VSN."""
        for vol in self._volumes:
            if vol.vsn == vsn:
                return vol
        return None

    @property
    def total_capacity_bytes(self) -> int:
        return sum(v.capacity_bytes for v in self._volumes)

    @property
    def total_used_bytes(self) -> int:
        return sum(v.used_bytes for v in self._volumes)


# ---------------------------------------------------------------------------
# Tape Robot
# ---------------------------------------------------------------------------

class TapeRobot:
    """
    Models the robotic arm with geometry-aware service time.

    Service time model (M/G/1 queue):
        S_robot(src_slot, dst_drive) = t_base + d(src, dst) / v_arm + t_load

    For dual-robot libraries (Spectra TFinity), two independent arms serve
    separate zones; cross-zone handoff incurs an additional penalty.

    See MATHEMATICAL_MODEL.md §4.5.
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 rng: RNGManager, stats: ReplicationStats) -> None:
        self._env = env
        self._cfg = cfg
        self._stats = stats

        n_arms = cfg.get("n_arms", 1)
        # Robot arm = SimPy Resource with capacity = n_arms
        self.arm = simpy.Resource(env, capacity=n_arms)

        self._n_slots      = cfg["library_slots"]
        self._n_drives     = cfg["n_drives"]
        self._h_speed      = cfg["arm_horizontal_speed_m_per_s"]
        self._v_speed      = cfg["arm_vertical_speed_m_per_s"]
        self._lib_height   = cfg["library_height_m"]
        self._lib_width    = cfg["library_width_m"]
        self._n_arms       = n_arms

        self._rng_robot    = rng.get("robot_travel")
        self._rng_load     = rng.get("mount_time")

        # Load / unload time distributions.
        # Config supplies linear-space (mean, std) in seconds; convert to the
        # log-space parameters expected by numpy Generator.lognormal().
        self._load_mu_log, self._load_sigma_log = lognormal_params_from_mean_std(
            cfg["cartridge_load_s_mu"], cfg["cartridge_load_s_sigma"],
        )
        self._unload_mu_log, self._unload_sigma_log = lognormal_params_from_mean_std(
            cfg["cartridge_unload_s_mu"], cfg["cartridge_unload_s_sigma"],
        )

        # Cumulative busy time (for utilisation)
        self._busy_time: float = 0.0
        self._last_busy_start: float = 0.0

    def _slot_position(self, slot_idx: int) -> Tuple[float, float]:
        """Map slot index to (x, y) position in library (simplified uniform grid)."""
        cols = max(1, self._n_slots // 10)
        row = slot_idx // cols
        col = slot_idx % cols
        x = (col / max(cols - 1, 1)) * self._lib_width
        y = (row / max(10 - 1, 1)) * self._lib_height
        return x, y

    def _drive_position(self, drive_idx: int) -> Tuple[float, float]:
        """Map drive index to (x, y) position in library (drives at right edge)."""
        y = (drive_idx / max(self._n_drives - 1, 1)) * self._lib_height
        return self._lib_width, y

    def travel_time(self, from_slot: int, to_drive: int) -> float:
        """
        Compute arm travel time from slot to drive.

        t_travel = |Δx| / v_h + |Δy| / v_v + noise
        """
        sx, sy = self._slot_position(from_slot)
        dx, dy = self._drive_position(to_drive)
        t_h = abs(dx - sx) / max(self._h_speed, 1e-6)
        t_v = abs(dy - sy) / max(self._v_speed, 1e-6)
        # Add small lognormal noise to model arm acceleration / deceleration
        noise = self._rng_robot.lognormal(0.0, 0.1)
        return t_h + t_v + noise

    def mount_service_time(self, slot_idx: int, drive_idx: int) -> float:
        """Total service time for a mount operation."""
        t_travel = self.travel_time(slot_idx, drive_idx)
        t_load = self._rng_load.lognormal(self._load_mu_log, self._load_sigma_log)
        return t_travel + t_load

    def unmount_service_time(self, slot_idx: int, drive_idx: int) -> float:
        """Total service time for an unmount operation."""
        t_travel = self.travel_time(drive_idx, slot_idx)
        t_unload = self._rng_load.lognormal(self._unload_mu_log, self._unload_sigma_log)
        return t_travel + t_unload

    def mount(self, volume: TapeVolume, drive_idx: int) -> Any:
        """
        SimPy process: request the robot arm, travel to slot, load cartridge.

        Returns a context manager for use with 'with self.arm.request()'.
        """
        return self._do_mount(volume, drive_idx)

    def _do_mount(self, volume: TapeVolume, drive_idx: int):
        req = self.arm.request()
        arrival_time = self._env.now
        self._stats.queues["robot"].arrival(self._env.now)
        yield req

        t_service = self.mount_service_time(volume.slot_index, drive_idx)
        self._last_busy_start = self._env.now
        yield self._env.timeout(t_service)
        self._busy_time += t_service

        volume.in_drive = drive_idx
        volume.n_mounts += 1
        if self._env.now >= self._stats.warmup_s:
            self._stats.n_mounts += 1
        self._stats.queues["robot"].departure(self._env.now, arrival_time)
        self._stats.mount_latency.record(t_service, time=self._env.now)
        self.arm.release(req)


# ---------------------------------------------------------------------------
# Tape Drive
# ---------------------------------------------------------------------------

class TapeDrive:
    """
    Models a single tape drive as an M/G/1 server.

    Service time components:
        S = S_mount + S_seek + S_transfer + S_unload

    See MATHEMATICAL_MODEL.md §4.6.
    """

    def __init__(self, drive_id: int, env: simpy.Environment, cfg: dict,
                 rng: RNGManager, robot: TapeRobot,
                 stats: ReplicationStats) -> None:
        self._id = drive_id
        self._env = env
        self._cfg = cfg
        self._rng = rng
        self._robot = robot
        self._stats = stats

        # Drive resource (capacity=1: only one operation at a time).
        # Plain FIFO resource: arbitration between recalls and migrations is by
        # DRIVE SELECTION at the library level (recalls take an idle drive; a
        # drive streaming a write zone is never interrupted), not by per-request
        # priority. See LibraryServer._flush (zone hold) and _do_recall (idle-first).
        self.resource = simpy.Resource(env, capacity=1)
        self._loaded_volume: Optional[TapeVolume] = None
        self._current_tape_pos: int = 0   # Current tape head position (bytes)

        # Timing parameters
        self._native_rate   = cfg["native_rate_bytes_per_s"]
        self._tape_capacity = cfg["tape_capacity_bytes"]
        self._seek_factor   = cfg["seek_speed_factor"]
        # Length-based locate model (TS1160): seek time scales with tape LENGTH
        # traversed at high-speed search, not with bytes at the read rate.
        # Defaults derived from IBM 3592/TS1160 published specs (20 TB native on
        # ~1140 m of tape; enterprise high-speed search class ~10 m/s); exact
        # locate-seconds are not published, so these are stated assumptions.
        self._tape_length_m   = cfg.get("tape_length_m", 1140.0)
        self._search_speed_m_s = cfg.get("search_speed_m_s", 10.0)
        self._locate_base_s   = cfg.get("locate_base_s", 8.0)
        # IPDPS 2027 revision: files after the first in a write zone stream at
        # end-of-data without a locate. False reproduces earlier versions.
        self._stream_zone_writes = bool(cfg.get("stream_zone_writes", False))
        self._bytes_per_m     = max(1.0, self._tape_capacity / max(self._tape_length_m, 1.0))
        self._mount_mu      = cfg["mount_time_lognormal_mu_s"]
        self._mount_sigma   = cfg["mount_time_lognormal_sigma_s"]
        self._comp_mu       = cfg["compression_ratio_lognormal_mu"]
        self._comp_sigma    = cfg["compression_ratio_lognormal_sigma"]

        self._rng_seek      = rng.get("seek_time")
        self._rng_comp      = rng.get("compression_ratio")
        self._rng_mount     = rng.get("mount_time")

        # Cleaning state
        self._n_mounts_since_clean: int = 0
        self._cleaning_interval   = cfg.get("cleaning_interval_mounts", 500)
        self._cleaning_duration   = cfg.get("cleaning_duration_s", 300)
        self._is_cleaning: bool = False

        # Statistics
        self._busy_time: float = 0.0
        self._idle_time: float = 0.0
        self._last_state_change: float = 0.0

    def _seek_time(self, from_pos: int, to_pos: int) -> float:
        """
        Compute tape locate (seek) time using a LENGTH-based model.

        Enterprise tape positions the head by moving along the physical tape at a
        high-speed search rate; locate time is governed by the LENGTH of tape
        traversed, not by the number of bytes at the streaming read rate. Modelling
        it as bytes/read_rate (the old formula) overstates locate on a full 20 TB
        cartridge by orders of magnitude (multi-hour "seeks").

            distance_m = |Δbytes| / bytes_per_m
            t_seek     = locate_base + distance_m / search_speed_m_s   (+ jitter)

        Parameters derived from IBM 3592/TS1160 published specs (20 TB on ~1140 m;
        high-speed search ~10 m/s); seek_speed_factor remains an adjustable
        overhead multiplier on the traversal term.
        """
        delta_bytes = abs(to_pos - from_pos)
        distance_m = delta_bytes / self._bytes_per_m
        t_base = self._locate_base_s + (distance_m / max(self._search_speed_m_s, 1e-6)) * self._seek_factor
        # Multiplicative +/-5% jitter. lognormal(0, 0.05) has median 1.0, so
        # ADDING the product to t_base doubles every locate (the mean of the
        # old form was ~2.0 * t_base); the jitter must scale t_base, not add
        # a second copy of it.
        return t_base * self._rng_seek.lognormal(0.0, 0.05)

    def _transfer_time(self, size_bytes: int) -> float:
        """Compute transfer time with stochastic compression ratio."""
        comp = self._rng_comp.lognormal(self._comp_mu, self._comp_sigma)
        comp = max(1.0, comp)  # Can't compress to less than original
        effective_rate = self._native_rate * comp
        return size_bytes / effective_rate

    def write_file(self, file_rec: FileRecord, volume: TapeVolume) -> Any:
        """
        SimPy process: write a file to tape.
        Includes: mount (if volume not loaded), seek, transfer, optional unload.
        """
        return self._do_write(file_rec, volume)

    def write_batch(self, items):
        """Write a whole zone (batch of files to the SAME cartridge) holding the
        drive for the entire zone. A cart streaming writes is never interrupted
        mid-zone: the drive is acquired once and released only after the last
        file in the batch. ``items`` is a list of (file_rec, volume, evt).
        Files are written in arrival order; each file's evt is fired as its
        write completes, but the drive is not released between files."""
        if not items:
            return
        req = self.resource.request()
        arrival_time = self._env.now
        self._stats.queues["drive_pool"].arrival(self._env.now)
        yield req
        granted = self._env.now
        try:
            prev_vsn = None
            for file_rec, volume, evt in items:
                # Within a zone the head is already at end-of-data after the
                # previous file on the same cartridge, so no locate is needed.
                streaming = (self._stream_zone_writes and prev_vsn == volume.vsn
                             and self._loaded_volume is not None
                             and self._loaded_volume.vsn == volume.vsn)
                yield from self._write_one(file_rec, volume, streaming=streaming)
                prev_vsn = volume.vsn
                if evt is not None and not evt.triggered:
                    evt.succeed()
        finally:
            self._stats.queues["drive_pool"].departure(self._env.now, arrival_time)
            self._stats.record_drive_job("zone", arrival_time, granted, self._env.now,
                                         sum(it[0].size_bytes for it in items),
                                         items[0][1].vsn if items else None)
            self.resource.release(req)

    def _write_one(self, file_rec: FileRecord, volume: TapeVolume,
                   streaming: bool = False):
        """Write a single file to ``volume`` assuming the drive is ALREADY held
        (called only from write_batch). Does not request/release the resource."""
        # Mount volume if not already loaded
        if self._loaded_volume is None or self._loaded_volume.vsn != volume.vsn:
            if self._loaded_volume is not None:
                # Unload current volume
                t_unload = self._robot.unmount_service_time(
                    self._loaded_volume.slot_index, self._id
                )
                yield self._env.timeout(t_unload)
                self._loaded_volume.in_drive = None
                self._loaded_volume = None

            # Mount new volume via robot
            yield from self._robot._do_mount(volume, self._id)
            self._loaded_volume = volume
            self._current_tape_pos = volume.current_position_bytes
            self._n_mounts_since_clean += 1

        # Seek to end of written data (skipped when streaming within a zone)
        if streaming and self._current_tape_pos == volume.current_position_bytes:
            t_seek = 0.0
        else:
            t_seek = self._seek_time(self._current_tape_pos,
                                      volume.current_position_bytes)
            yield self._env.timeout(t_seek)

        # Transfer
        t_transfer = self._transfer_time(file_rec.size_bytes)
        yield self._env.timeout(t_transfer)

        # Update tape position
        tape_offset = volume.write_file(file_rec.size_bytes, reserved=True)
        self._current_tape_pos = volume.current_position_bytes

        # Update file record
        file_rec.vsn = volume.vsn
        file_rec.vg_name = volume.vg_name
        file_rec.tape_block_offset = tape_offset

        # Track tape I/O in statistics
        self._stats.record_tape_write(file_rec.size_bytes)
        if t_seek > 0.0:
            self._stats.record_seek()

        total_service = t_seek + t_transfer
        self._busy_time += total_service

        # Check cleaning trigger (drive remains held across the zone)
        if self._n_mounts_since_clean >= self._cleaning_interval:
            self._env.process(self._do_cleaning())

    def read_file(self, file_rec: FileRecord, volume: TapeVolume) -> Any:
        """SimPy process: read a file from tape (recall)."""
        return self._do_read(file_rec, volume)

    def _do_read(self, file_rec: FileRecord, volume: TapeVolume):
        req = self.resource.request()
        arrival_time = self._env.now
        self._stats.queues["drive_pool"].arrival(self._env.now)
        yield req
        granted = self._env.now

        # Check for "warm" recall — tape already mounted (recently used volume)
        is_warm = getattr(file_rec, '_warm_recall', False)

        # Mount if needed (skip for warm recalls)
        if not is_warm and (self._loaded_volume is None or self._loaded_volume.vsn != volume.vsn):
            if self._loaded_volume is not None:
                t_unload = self._robot.unmount_service_time(
                    self._loaded_volume.slot_index, self._id
                )
                yield self._env.timeout(t_unload)
                self._loaded_volume.in_drive = None
                self._loaded_volume = None

            yield from self._robot._do_mount(volume, self._id)
            self._loaded_volume = volume
            self._current_tape_pos = 0
            self._n_mounts_since_clean += 1

        # Seek to file's tape position
        target_pos = file_rec.tape_block_offset or 0
        t_seek = self._seek_time(self._current_tape_pos, target_pos)
        yield self._env.timeout(t_seek)
        self._current_tape_pos = target_pos

        # Transfer
        t_transfer = self._transfer_time(file_rec.size_bytes)
        yield self._env.timeout(t_transfer)
        self._current_tape_pos += file_rec.size_bytes

        # Track tape I/O in statistics
        self._stats.record_tape_read(file_rec.size_bytes)
        self._stats.record_seek()

        total_service = t_seek + t_transfer
        self._busy_time += total_service
        self._stats.queues["drive_pool"].departure(self._env.now, arrival_time)
        self._stats.record_drive_job("recall", arrival_time, granted, self._env.now,
                                     file_rec.size_bytes, volume.vsn)

        self.resource.release(req)

    def _do_cleaning(self):
        """SimPy process: take drive offline for cleaning."""
        self._is_cleaning = True
        self._n_mounts_since_clean = 0
        yield self._env.timeout(self._cleaning_duration)
        self._is_cleaning = False

    @property
    def utilisation(self) -> float:
        total = self._env.now
        return self._busy_time / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Library Server (LS) Accrual Buffer
# ---------------------------------------------------------------------------

class LibraryServer:
    """
    Models DMF 7's Library Server (LS) accrual behaviour.

    Per the DMF 7 architecture: "For outbound data, the LS accrues requests
    until the volume of data justifies a volume mount."

    Implementation:
        - Incoming migration requests enter an accrual buffer
        - Buffer flushes when: total bytes ≥ LS_MIN_BATCH or wait ≥ LS_MAX_WAIT
        - Flushed batches are dispatched to available drives via VGs

    See MATHEMATICAL_MODEL.md §5.2.
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 vgs: Dict[str, VolumeGroup], drives: List[TapeDrive],
                 stats: ReplicationStats, rng: RNGManager) -> None:
        self._env = env
        self._cfg = cfg
        self._vgs = vgs
        self._drives = drives
        self._stats = stats

        self._min_batch_bytes = cfg["ls_min_batch_bytes"]
        self._max_wait_s      = cfg["ls_max_wait_s"]

        # Accrual buffer: list of (FileRecord, arrival_time)
        self._accrual: List[Tuple[FileRecord, float]] = []
        self._accrual_bytes: int = 0
        self._outstanding: int = 0   # in-flight recalls + migrations (for drain quiescence)
        self._flush_event: Optional[simpy.Event] = None

        # Trickle migration semaphore
        dm_cfg = cfg.get("trickle", {})
        trickle_enabled = dm_cfg.get("trickle_enabled", False)
        max_concurrent = dm_cfg.get("dmmigrate_unack", 16)
        if trickle_enabled:
            self._trickle_sem = simpy.Resource(env, capacity=max_concurrent)
        else:
            self._trickle_sem = None

        # Dual-copy configuration
        self._dual_copy = cfg.get("dual_copy", True)
        self._vg_names = list(vgs.keys())

        # Drive pools (p4_ls topology): all drives serve recalls; a subset also
        # serves writes. n_write_drives defaults to all drives when unset.
        n_all = len(drives)
        n_write = max(1, min(int(cfg.get("n_write_drives", n_all)), n_all))
        self._read_drive_idx  = list(range(n_all))
        self._write_drive_idx = list(range(n_write))

    def set_min_batch_bytes(self, val: int) -> None:
        """Dynamically adjust the minimum batch size for accrual."""
        self._min_batch_bytes = max(1, int(val))

    def set_max_wait_s(self, val: float) -> None:
        """Dynamically adjust the maximum wait time before flushing."""
        self._max_wait_s = max(1.0, float(val))

    def submit_migration(self, file_rec: FileRecord) -> Any:
        """Submit a file for migration.

        Returns a SimPy process that completes only after the file's tape
        write has finished — i.e. after this batch (or the batch this file
        rolls into) has been flushed and written. This is what lets the
        upstream DMF state-machine transition MIG → DUL only when the tape
        copy actually exists, instead of as soon as the file lands in the
        accrual buffer.
        """
        return self._env.process(self._tracked(self._accrue_and_wait(file_rec)))

    def _accrue_and_wait(self, file_rec: FileRecord):
        """Add a file to the accrual buffer and yield until *that file's*
        tape write has completed.

        The buffer carries one ``simpy.Event`` per file (the file's
        ``written_evt``); the flush logic fires that event after the file's
        write completes. This decouples per-file completion from per-batch
        flushing while still serialising via the LS accrual queue.
        """
        arrival_time = self._env.now
        self._stats.queues["ls_accrual"].arrival(self._env.now)

        # Per-file completion event — fired by _flush after the tape write.
        written_evt = self._env.event()
        self._accrual.append((file_rec, arrival_time, written_evt))
        self._accrual_bytes += file_rec.size_bytes

        # Trigger an immediate flush if the byte threshold is met; otherwise
        # ensure exactly one outstanding wait-and-flush timer is pending so
        # accruals can drain on time even when the threshold is never reached.
        if self._accrual_bytes >= self._min_batch_bytes:
            self._env.process(self._flush())
        elif self._flush_event is None or self._flush_event.processed:
            self._flush_event = self._env.process(self._wait_and_flush())

        self._stats.queues["ls_accrual"].departure(self._env.now, arrival_time)
        # Block the caller until *this* file is on tape.
        yield written_evt

    def _wait_and_flush(self):
        """Flush accrual buffer after the maximum wait time."""
        yield self._env.timeout(self._max_wait_s)
        if self._accrual:
            yield from self._flush()

    def drain(self):
        """Flush any files still in the accrual buffer at end of arrivals.

        Called once the workload stops (see run_replication) so migrations
        parked below the byte threshold reach tape and are not censored by the
        simulation cutoff. Idempotent: a no-op when the buffer is empty.
        """
        if self._accrual:
            yield from self._flush()

    def _flush(self):
        """Flush all pending migration requests to drives, signalling each
        file's per-file ``written_evt`` after its tape write completes."""
        if not self._accrual:
            return

        batch = list(self._accrual)
        self._accrual = []
        self._accrual_bytes = 0

        # Single shared volume group for this library.
        primary_vg = self._vgs[self._vg_names[0]] if self._vg_names else None
        if primary_vg is None:
            for _, _, evt in batch:
                if not evt.triggered:
                    evt.succeed()
            return

        # Writes use the WRITE-capable drive pool (subset of all drives).
        write_drives = [self._drives[i] for i in self._write_drive_idx
                        if i < len(self._drives)]
        if not write_drives:
            for _, _, evt in batch:
                if not evt.triggered:
                    evt.succeed()
            return

        # Reserve a volume for each file, building the zone's write list. A cart
        # streaming writes is not interrupted mid-zone: we hold ONE drive for the
        # whole batch (write_batch), rather than re-acquiring per file (which
        # would let a recall interleave on a write drive between files).
        items = []
        for file_rec, _, evt in batch:
            try:
                volume = primary_vg.get_write_volume(file_rec.size_bytes)
            except RuntimeError:
                if evt is not None and not evt.triggered:
                    evt.succeed()
                continue
            items.append((file_rec, volume, evt))
        if not items:
            return
        # Pick the least-busy write-capable drive ONCE for this zone. Prefer an
        # idle write drive; else the one with the shortest queue.
        idle = [d for d in write_drives if d.resource.count == 0]
        drive = (idle[0] if idle
                 else min(write_drives, key=lambda d: len(d.resource.queue)))
        yield from drive.write_batch(items)

    def submit_recall(self, file_rec: FileRecord, priority: int = 5) -> Any:
        """Submit a recall request. Returns a SimPy process."""
        return self._env.process(self._tracked(self._do_recall(file_rec, priority)))

    def _tracked(self, gen):
        """Wrap a recall/migration process so outstanding work is counted.

        The drain in run_replication waits on `outstanding == 0` rather than on
        drive-resource occupancy: a recall dispatched during a burst is a pending
        SimPy process that has not yet called resource.request(), so it would not
        show up in drive.count/queue and the run could stop while it is still
        queued, censoring slow recalls from the latency histogram.
        """
        self._outstanding += 1
        try:
            yield from gen
        finally:
            self._outstanding -= 1

    @property
    def outstanding(self) -> int:
        return self._outstanding

    def _do_recall(self, file_rec: FileRecord, priority: int):
        """Execute a recall: find volume, select drive, read file."""
        arrival_time = self._env.now
        self._stats.queues["recall_queue"].arrival(self._env.now)

        if file_rec.vsn is None:
            # File not recorded as migrated — cannot recall
            self._stats.queues["recall_queue"].departure(self._env.now, arrival_time)
            return

        # Find volume
        vg = self._vgs.get(file_rec.vg_name)
        if vg is None:
            self._stats.queues["recall_queue"].departure(self._env.now, arrival_time)
            return
        volume = vg.get_volume(file_rec.vsn)
        if volume is None:
            self._stats.queues["recall_queue"].departure(self._env.now, arrival_time)
            return

        # Select drive from the full READ pool (all drives serve recalls).
        # Prefer a genuinely IDLE drive (a spare): the real library loads the
        # recall media into any free drive rather than displacing an active
        # write. If none is idle, fall back to least-queued and wait for a free
        # slot (a drive streaming a write zone is never interrupted).
        avail_drives = [self._drives[i] for i in self._read_drive_idx
                        if i < len(self._drives)]
        if not avail_drives:
            self._stats.queues["recall_queue"].departure(self._env.now, arrival_time)
            return

        # Prefer drive with volume already mounted (avoids remount)
        mounted_drive = next(
            (d for d in avail_drives
             if d._loaded_volume and d._loaded_volume.vsn == file_rec.vsn),
            None
        )
        idle_drives = [d for d in avail_drives if d.resource.count == 0]
        drive = (mounted_drive
                 or (idle_drives[0] if idle_drives else None)
                 or min(avail_drives, key=lambda d: len(d.resource.queue)))

        yield from drive.read_file(file_rec, volume)

        self._stats.queues["recall_queue"].departure(self._env.now, arrival_time)
        # NOTE: total recall latency is recorded by Filesystem.handle_access
        # so that both fast (disk) and slow (tape) paths are captured.


# ---------------------------------------------------------------------------
# Top-Level Tape Subsystem
# ---------------------------------------------------------------------------

class TapeSubsystem:
    """
    Top-level coordinator for all tape-side components:
        - VolumeGroups
        - TapeRobot
        - TapeDrives
        - LibraryServer
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 rng: RNGManager, stats: ReplicationStats) -> None:
        self._env = env
        self._stats = stats

        # Build volume groups
        self.vgs: Dict[str, VolumeGroup] = {}
        for vg_cfg in cfg.get("volume_groups", []):
            vg = VolumeGroup(vg_cfg["name"], vg_cfg)
            self.vgs[vg.name] = vg

        # Build robot
        self.robot = TapeRobot(env, cfg["tape_robot"], rng, stats)

        # Build drives
        drive_cfg = cfg["tape_drives"]
        n_drives = drive_cfg["n_drives"]
        self.drives: List[TapeDrive] = [
            TapeDrive(i, env, drive_cfg, rng, self.robot, stats)
            for i in range(n_drives)
        ]

        # Build LS
        ls_cfg = {**cfg["library_server"],
                  "dual_copy": cfg["policy_engine"].get("dual_copy", True),
                  "n_write_drives": drive_cfg.get("n_write_drives", n_drives),
                  "trickle": cfg.get("data_movers", {})}
        self.ls = LibraryServer(env, ls_cfg, self.vgs, self.drives, stats, rng)

    def migrate(self, file_rec: FileRecord) -> Iterator:
        """Enqueue a file for migration to tape."""
        yield self.ls.submit_migration(file_rec)

    def recall(self, file_rec: FileRecord, priority: int = 5) -> Iterator:
        """Recall a file from tape to disk."""
        yield self.ls.submit_recall(file_rec, priority)

    def drive_utilisations(self) -> Dict[int, float]:
        return {d._id: d.utilisation for d in self.drives}
