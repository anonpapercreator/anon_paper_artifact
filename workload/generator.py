"""
workload/generator.py — Workload Generator and DMF Trace Reader
===============================================================
Implements:
    1. Synthetic workload generator using empirically-grounded distributions
       - File size: LogNormal (Smirni & Reed, 1998)
       - Inter-arrival time: Pareto (self-similar, Leland et al., 1994)
       - Access frequency: Zipf's law
       - Diurnal modulation

    2. DMF 7 trace file reader for replay mode
       - Accepts normalised CSV/JSONL trace format
       - Extracts arrival pattern from real DMF logs

Scientific basis for distributions:
    - LogNormal file sizes: Barrera et al.; Schroeder & Gibson (2007)
    - Pareto IAT: Leland et al. (1994), Willinger et al. (1997)
    - Zipf access: Breslau et al. (1999), Arlitt & Williamson (1997)
"""

from __future__ import annotations
import simpy
import numpy as np
import math
from typing import Iterator, Optional, List, Dict, Any
from pathlib import Path
import csv
import json

from core.rng import RNGManager
from core.events import FileRecord, FileState, SimEvent, EventType


# ---------------------------------------------------------------------------
# Zipf Distribution Sampler
# ---------------------------------------------------------------------------

class ZipfSampler:
    """
    Samples file indices from a Zipf(s) distribution over N files.

    P(rank r) ∝ 1/r^s,  r = 1, 2, ..., N

    Uses inverse CDF (pre-computed) for O(1) sampling.

    Reference: Breslau et al. (1999), "Web caching and Zipf-like distributions:
    Evidence and implications."
    """

    def __init__(self, n_files: int, s: float, rng: np.random.Generator) -> None:
        """
        Args:
            n_files: Number of distinct files in working set.
            s:       Zipf exponent. s=1.0 is classical Zipf.
            rng:     Independent RNG stream.
        """
        self._n = n_files
        self._s = s
        self._rng = rng

        # Pre-compute CDF
        ranks = np.arange(1, n_files + 1, dtype=np.float64)
        weights = 1.0 / (ranks ** s)
        self._cdf = np.cumsum(weights) / np.sum(weights)

    def sample(self) -> int:
        """Return a 0-indexed file rank (0 = most popular)."""
        u = self._rng.uniform(0.0, 1.0)
        idx = int(np.searchsorted(self._cdf, u, side='right'))
        return min(idx, self._n - 1)


# ---------------------------------------------------------------------------
# Diurnal Modulator
# ---------------------------------------------------------------------------

class DiurnalModulator:
    """
    Applies a sinusoidal time-of-day modulation to arrival rates.

        φ(t) = max(φ_floor, 1 + A · sin(2π(t mod T - τ_peak) / T))

    where the unmodulated time-average of φ is 1 (so the long-run rate is
    preserved when φ is used as a multiplicative intensity factor on the
    underlying renewal process). ``φ_floor`` is a small positive clamp that
    keeps the cumulative intensity Λ strictly increasing (and therefore
    invertible) even at A=1.

    Both the closed-form cumulative ``Lambda`` and its inverse
    ``invert_Lambda`` are exposed so callers can apply the time-rescaling
    theorem (Çinlar 1975 §6.5; Karr 1991 §3.3) to any renewal base —
    including non-Poisson processes (Pareto), where simple thinning
    (Lewis–Shedler 1979) does *not* produce a process with the desired
    instantaneous rate.
    """

    def __init__(self, amplitude: float, peak_hour: float,
                 day_length_s: float = 86400.0,
                 phi_floor: float = 0.01) -> None:
        self._A = float(amplitude)
        self._tau_peak = peak_hour * 3600.0
        self._T = float(day_length_s)
        self._phi_floor = float(phi_floor)

    def modulate(self, t: float) -> float:
        """Return modulation factor φ(t) at simulation time t."""
        phase = 2.0 * math.pi * (t % self._T - self._tau_peak) / self._T
        factor = 1.0 + self._A * math.sin(phase)
        return max(self._phi_floor, factor)

    @property
    def max_factor(self) -> float:
        """Upper bound on φ(t) — sup over any t."""
        return max(1.0 + self._A, self._phi_floor)

    @property
    def min_factor(self) -> float:
        """Lower bound on φ(t) — used to bracket the inverse Λ."""
        # The clamp could activate only when 1 - A < phi_floor.
        return max(1.0 - self._A, self._phi_floor)

    def Lambda(self, t: float) -> float:
        """Cumulative intensity Λ(t) = ∫₀^t φ(s) ds.

        Closed form (valid when 1 - A ≥ φ_floor, i.e. the floor never
        activates within a period):

            Λ(t) = t + (A·T / 2π) · [cos(2π·τ_peak/T) − cos(2π·(t − τ_peak)/T)]

        When the floor *is* active for some range of t, falls back to a
        numerical integral via scipy.integrate.quad — accurate but slower.
        """
        if 1.0 - self._A >= self._phi_floor:
            two_pi = 2.0 * math.pi
            return t + (self._A * self._T / two_pi) * (
                math.cos(two_pi * self._tau_peak / self._T)
                - math.cos(two_pi * (t - self._tau_peak) / self._T)
            )
        # Floor is active for part of the cycle — fall back to quad.
        from scipy.integrate import quad
        val, _ = quad(self.modulate, 0.0, max(t, 0.0), limit=200)
        return float(val)

    def invert_Lambda(self, t0: float, dU: float,
                      tol: float = 1e-9) -> float:
        """Solve Δ ≥ 0 such that Λ(t0 + Δ) − Λ(t0) = dU, returning t0+Δ.

        Λ is strictly increasing (φ ≥ φ_floor > 0), so the equation has a
        unique non-negative root. We use Brent's method on a guaranteed
        bracket [0, dU / φ_min] derived from the lower bound on φ.
        """
        if dU <= 0.0:
            return t0
        from scipy.optimize import brentq
        L0 = self.Lambda(t0)
        phi_min = self.min_factor
        upper = dU / max(phi_min, 1e-12)
        # Generous safety factor so a perfectly minimal-rate window inside
        # [t0, t0+upper] still leaves the upper bracket strictly above the
        # target. brentq will refine the root regardless.
        upper *= 1.5
        f = lambda dt: self.Lambda(t0 + dt) - L0 - dU
        # f(0) = -dU < 0; f(upper) ≥ 0 by construction. brentq will find Δ.
        delta = brentq(f, 0.0, upper, xtol=tol)
        return t0 + delta


# ---------------------------------------------------------------------------
# Synthetic Workload Generator
# ---------------------------------------------------------------------------

class SyntheticWorkloadGenerator:
    """
    Generates a sequence of file arrival events using statistically grounded
    distributions appropriate for HPC/DMF 7 workloads.

    Produces SimPy processes that inject FileRecord objects into the simulation.
    """

    def __init__(self, env: simpy.Environment, cfg: dict,
                 rng: RNGManager, filesystem: Any) -> None:
        """
        Args:
            env:        SimPy environment.
            cfg:        Workload configuration dict (from YAML 'workload' section).
            rng:        RNGManager (pre-initialised for this replication).
            filesystem: Disk cache / filesystem component to receive file arrivals.
        """
        self._env = env
        self._cfg = cfg
        self._rng = rng
        self._fs = filesystem

        # RNG streams
        self._rng_size    = rng.get("file_size")
        self._rng_iat     = rng.get("iat")
        self._rng_access  = rng.get("access_freq")
        self._rng_rw      = rng.get("rw_ratio")
        self._rng_age     = rng.get("file_age_init")

        # File size distribution parameters
        self._size_mu    = cfg["file_size_lognormal_mu"]
        self._size_sigma = cfg["file_size_lognormal_sigma"]
        self._size_min   = cfg["file_size_min_bytes"]
        self._size_max   = cfg["file_size_max_bytes"]

        # IAT distribution
        self._iat_dist = cfg["iat_distribution"]
        self._pareto_alpha = cfg["iat_pareto_alpha"]
        self._pareto_xmin  = cfg["iat_pareto_xmin_s"]
        self._exp_rate     = cfg["iat_exponential_rate"]

        # Read/write ratio
        self._read_frac = cfg["read_fraction"]

        # Zipf sampler
        n_files = cfg["working_set_size"]
        zipf_s  = cfg["zipf_s"]
        self._zipf = ZipfSampler(n_files, zipf_s, rng.get("access_freq"))

        # Diurnal modulator
        self._diurnal_enabled = cfg.get("diurnal_enabled", True)
        if self._diurnal_enabled:
            self._diurnal = DiurnalModulator(
                amplitude=cfg.get("diurnal_amplitude", 0.6),
                peak_hour=cfg.get("diurnal_peak_hour", 10),
            )

        # Working set: maps rank -> FileRecord
        self._working_set: Dict[int, FileRecord] = {}
        self._file_counter: int = 0

    def _sample_file_size(self) -> int:
        """Sample file size from LogNormal distribution."""
        size = self._rng_size.lognormal(self._size_mu, self._size_sigma)
        return int(np.clip(size, self._size_min, self._size_max))

    def _sample_base_iat(self) -> float:
        """Sample from the base (unmodulated) IAT distribution.

        Mean of the returned IAT is 1/λ_base:
            Pareto (Lomax): E[IAT] = x_min / (α − 1)
            Exponential:    E[IAT] = 1 / λ_exp
        """
        if self._iat_dist == "pareto":
            u = self._rng_iat.uniform(0.0, 1.0 - 1e-10)
            return self._pareto_xmin * (u ** (-1.0 / self._pareto_alpha) - 1.0)
        return self._rng_iat.exponential(1.0 / self._exp_rate)

    def _base_iat_mean(self) -> float:
        """Long-run mean of ``_sample_base_iat`` — needed to express each
        sample in unit-rate Λ-space for the time-rescaling transform."""
        if self._iat_dist == "pareto":
            return self._pareto_xmin / max(self._pareto_alpha - 1.0, 1e-9)
        return 1.0 / max(self._exp_rate, 1e-9)

    def _sample_iat(self, t: float = 0.0) -> float:
        """Draw the inter-arrival time to the next arrival under diurnal
        modulation.

        Method — time rescaling (Çinlar 1975 §6.5; Karr 1991 §3.3).

        Let ``X_i`` be a sample from the base renewal distribution with
        mean ``E[X] = 1/λ_base``. Let φ(t) be the diurnal multiplicative
        factor (long-run mean 1). Define the cumulative intensity factor
        Λ_φ(t) = ∫₀^t φ(s) ds. The next physical arrival time is the
        unique solution of

            Λ_φ(t_{i+1}) − Λ_φ(t_i) = X_i                     (*)

        Because ⟨φ⟩ = 1, the long-run mean of T_{i+1} − T_i equals E[X],
        so the long-run physical rate is 1/E[X] = λ_base — exactly the
        condition we want. The instantaneous physical rate at time t is
        λ_base · φ(t), so the IATs are short during peak hours and long
        during off-peak hours, while still inheriting the heavy-tail /
        autocorrelation structure of the base renewal process.

        For the exponential base, this construction is equivalent to the
        textbook NHPP via inverse-CDF. For Pareto / heavy-tail bases it
        is the *only* consistent construction — Lewis–Shedler thinning
        (1979) is a Poisson result and does not yield a process with
        rate λ_base·φ(t) at the file level when applied to a non-Poisson
        candidate stream.
        """
        base_iat = self._sample_base_iat()
        if not self._diurnal_enabled:
            return max(base_iat, 1e-6)

        # Solve (*) for the next physical time.
        t_next = self._diurnal.invert_Lambda(t, base_iat)
        return max(t_next - t, 1e-6)

    def _new_file(self) -> FileRecord:
        """Create a new FileRecord with generated attributes."""
        fid = f"file_{self._file_counter:010d}"
        self._file_counter += 1
        size = self._sample_file_size()
        now = self._env.now
        return FileRecord(
            file_id=fid,
            size_bytes=size,
            state=FileState.REG,
            created_at=now,
            last_accessed_at=now,
            last_modified_at=now,
        )

    def _pre_populate_working_set(self) -> None:
        """
        Populate the working set with files of pre-existing ages.
        Simulates a filesystem that has been running before simulation start.
        Files are assigned initial ages drawn from a normal distribution.
        """
        n = self._cfg["working_set_size"]
        mu_age    = self._cfg.get("initial_file_age_s_mu", 7200.0)
        sigma_age = self._cfg.get("initial_file_age_s_sigma", 3600.0)

        for rank in range(n):
            f = self._new_file()
            # Assign a pre-existing age (file was created before sim start)
            age = float(np.clip(
                self._rng_age.normal(mu_age, sigma_age),
                0.0, mu_age * 5
            ))
            f.created_at = -age
            f.last_accessed_at = -age
            f.last_modified_at = -age
            self._working_set[rank] = f

    def run(self) -> Iterator:
        """
        SimPy process: generate file access events continuously.

        Yields SimPy timeout events to advance simulation clock. Each event
        is either a read (sampled by Zipf rank from the working set) or a
        write (creates a new file).

        Working-set replacement strategy:
            New writes evict files from the *unpopular tail* of the working
            set, so the Zipf-ranked popularity structure is preserved. The
            previous code used ``rank = file_counter % working_set_size``,
            which rotated through ranks uniformly and continually overwrote
            the popular head — that effectively turned popularity into
            recency, defeating the purpose of the Zipf sampler.
        """
        self._pre_populate_working_set()
        # Random stream dedicated to eviction choice so it doesn't perturb
        # the existing Zipf / size / rw streams.
        rng_evict = self._rng.get("ws_eviction")

        ws_size = self._cfg["working_set_size"]

        while True:
            # Sample inter-arrival time
            iat = self._sample_iat(self._env.now)
            yield self._env.timeout(iat)

            # Decide: read existing file or create new file
            is_read = self._rng_rw.uniform() < self._read_frac

            if is_read and self._working_set:
                rank = self._zipf.sample()
                rank = rank % len(self._working_set)  # Bounds-safe
                file_rec = self._working_set[rank]
                if file_rec.deleted_at is not None:
                    continue   # deleted files are not accessed
                file_rec.last_accessed_at = self._env.now
                yield from self._fs.handle_access(file_rec)

            else:
                # Write a new file. Evict from the unpopular *tail* of the
                # working set (lower half of ranks) so the Zipf head is
                # stable. If the working set is below capacity, fill in
                # the next free rank.
                file_rec = self._new_file()
                if len(self._working_set) < ws_size:
                    rank = len(self._working_set)
                else:
                    tail_lo = ws_size // 2
                    rank = int(rng_evict.integers(tail_lo, ws_size))
                self._working_set[rank] = file_rec
                yield from self._fs.handle_write(file_rec)


# ---------------------------------------------------------------------------
# DMF 7 Trace File Reader
# ---------------------------------------------------------------------------

class DMFTraceReader:
    """
    Reads a normalised DMF 7 trace file and replays events into the simulation.

    Trace format (CSV or JSON Lines) — see MATHEMATICAL_MODEL.md §10 for
    the full field specification.

    The reader extracts the arrival pattern (timestamps + file sizes) from real
    DMF logs while stochastic service times (tape seek, mount, etc.) are still
    modelled probabilistically.

    Supports two normalised input sources:
        1. Direct export from dmqview / dmstat logs (see parse_dmstat_line)
        2. Generic normalised CSV/JSONL (see MATHEMATICAL_MODEL.md)
    """

    REQUIRED_FIELDS = {"timestamp_s", "event_type", "file_id", "file_size_bytes"}
    OPTIONAL_FIELDS = {"file_state_before", "file_state_after", "vsn",
                       "drive_id", "latency_s", "queue_depth", "disk_usage_pct"}
    VALID_EVENT_TYPES = {"MIGRATE", "RECALL", "POLICY_CYCLE", "SPACE_RELEASE",
                         "REQUEST_COMPLETED",
                         "FILE_CREATE", "FILE_DELETE", "FILE_MODIFY"}

    def __init__(self, trace_path: str, offline_fraction: float = 0.30,
                 warm_fraction: float = 0.20) -> None:
        self._path = Path(trace_path)
        self._format = self._detect_format()
        self._offline_fraction = max(0.0, min(1.0, float(offline_fraction)))
        self._warm_fraction = max(0.0, min(1.0, float(warm_fraction)))
        self._records: List[dict] = []

    def _detect_format(self) -> str:
        """Detect trace file format from extension."""
        suffix = self._path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            return "jsonl"
        elif suffix in (".csv",):
            return "csv"
        else:
            # Try to detect from content
            with open(self._path, "r") as f:
                first_line = f.readline().strip()
            if first_line.startswith("{"):
                return "jsonl"
            return "csv"

    def load(self) -> List[dict]:
        """
        Load and validate all trace records.

        Returns:
            List of validated record dicts, sorted by timestamp_s.

        Raises:
            ValueError: If required fields are missing or data is malformed.
        """
        if self._format == "jsonl":
            self._records = self._load_jsonl()
        else:
            self._records = self._load_csv()

        # Validate
        self._validate(self._records)

        # Sort by timestamp
        self._records.sort(key=lambda r: r["timestamp_s"])

        return self._records

    def _load_jsonl(self) -> List[dict]:
        records = []
        with open(self._path, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no}: {e}")
        return records

    def _load_csv(self) -> List[dict]:
        records = []
        with open(self._path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row_no, row in enumerate(reader, 1):
                rec = {}
                for k, v in row.items():
                    # Type coercion
                    k = k.strip()
                    v = v.strip() if v else None
                    if k == "timestamp_s" or k == "latency_s":
                        rec[k] = float(v) if v else None
                    elif k in ("file_size_bytes", "queue_depth", "drive_id"):
                        rec[k] = int(v) if v else None
                    elif k == "disk_usage_pct":
                        rec[k] = float(v) if v else None
                    else:
                        rec[k] = v
                records.append(rec)
        return records

    def _validate(self, records: List[dict]) -> None:
        """Validate all records against schema."""
        if not records:
            raise ValueError("Trace file contains no records.")

        for i, rec in enumerate(records):
            # Check required fields
            for field in self.REQUIRED_FIELDS:
                if field not in rec or rec[field] is None:
                    raise ValueError(
                        f"Record {i}: missing required field '{field}'. "
                        f"Present fields: {list(rec.keys())}"
                    )

            # Validate timestamp
            ts = rec["timestamp_s"]
            if not isinstance(ts, (int, float)) or ts < 0:
                raise ValueError(
                    f"Record {i}: timestamp_s must be a non-negative number, got {ts}"
                )

            # Validate event type
            et = rec["event_type"]
            if et not in self.VALID_EVENT_TYPES:
                raise ValueError(
                    f"Record {i}: unknown event_type '{et}'. "
                    f"Valid types: {self.VALID_EVENT_TYPES}"
                )

            # Validate file size
            sz = rec["file_size_bytes"]
            if not isinstance(sz, (int, float)) or sz < 0:
                raise ValueError(
                    f"Record {i}: file_size_bytes must be non-negative, got {sz}"
                )

    def _ensure_file_in_registry(self, file_registry: Dict, rec: dict, env):
        """Create or retrieve FileRecord, pre-populating tape metadata for recalls."""
        fid = rec["file_id"]
        if fid in file_registry:
            return file_registry[fid]

        et = rec["event_type"]
        init_state = FileState.OFL if et == "RECALL" else FileState.REG

        f = FileRecord(
            file_id=fid,
            size_bytes=int(rec["file_size_bytes"]),
            state=init_state,
            created_at=env.now,
            last_accessed_at=env.now,
            last_modified_at=env.now,
        )
        file_registry[fid] = f
        return f

    def run(self, env: simpy.Environment, filesystem: Any) -> Iterator:
        """
        SimPy process: replay trace events in simulation time order.

        Time scaling: trace timestamps are replayed as-is relative to
        simulation epoch (t=0 = first event in trace).

        Args:
            env:        SimPy environment.
            filesystem: Disk cache component to receive events.
        """
        if not self._records:
            self.load()

        t_epoch = self._records[0]["timestamp_s"]
        file_registry: Dict[str, FileRecord] = {}

        # Pre-pass: build registry with correct initial states
        for rec in self._records:
            self._ensure_file_in_registry(file_registry, rec, env)

        # Pre-populate VSNs for OFL files so recalls work
        self._assign_vsns(file_registry, filesystem)

        for rec in self._records:
            # Advance clock to event time
            sim_time = rec["timestamp_s"] - t_epoch
            wait = max(0.0, sim_time - env.now)
            yield env.timeout(wait)

            f = file_registry[rec["file_id"]]
            et = rec["event_type"]

            # Dispatch based on event type.
            # Spawn each request as an INDEPENDENT process so the arrival stream
            # advances at trace timestamps and concurrent requests contend for
            # the tape drives (the source of queue-driven tail latency). Using
            # `yield from` here would serialise the replay and suppress
            # contention entirely.
            if et == "FILE_CREATE":
                env.process(filesystem.handle_write(f))
            elif et == "RECALL":
                f.last_accessed_at = env.now
                env.process(filesystem.handle_access(f))
            elif et == "MIGRATE":
                # Explicit migration request (dmput)
                env.process(filesystem.handle_explicit_migrate(f))
            elif et == "FILE_MODIFY":
                f.last_modified_at = env.now
                env.process(filesystem.handle_write(f))
            elif et == "REQUEST_COMPLETED":
                # Completion notification after recall/migrate; no-op in replay
                pass
            # POLICY_CYCLE and SPACE_RELEASE are handled by the policy engine;
            # trace replay of these events is informational only.

    def _assign_vsns(self, file_registry: Dict, filesystem):
        """Assign VSNs and resolve file states for trace replay pre-population.

        Since the trace only records events (not full state history), we infer
        initial states.  Files whose first event is RECALL are treated as
        offline (OFL) with probability ``trace_offline_fraction`` (default 0.30),
        and as online (DUL) otherwise.  This matches the observed bimodal
        latency distribution where ~65% of recalls complete in <5s (disk hit)
        and ~35% require tape access.
        """
        tape = getattr(filesystem.policy, '_tape', None)
        if tape is None:
            return
        vg = next(iter(tape.vgs.values()), None)
        if vg is None:
            return

        vols = vg._volumes
        if not vols:
            return

        # Files that were initially marked OFL because first event was RECALL
        candidate_files = [f for f in file_registry.values()
                           if f.state == FileState.OFL and f.vsn is None]

        # Shuffle deterministically so we pick a random subset to keep as OFL
        import numpy as np
        rng = np.random.default_rng(42)
        rng.shuffle(candidate_files)

        # Fraction of RECALL files that are actually offline (need tape mount)
        offline_frac = getattr(self, '_offline_fraction', 0.30)
        n_offline = int(len(candidate_files) * offline_frac)
        offline_files = candidate_files[:n_offline]
        online_files = candidate_files[n_offline:]

        # Among offline files, a subset have tape already mounted ("warm").
        # This creates the intermediate latency component (5-15s).
        warm_frac = getattr(self, '_warm_fraction', 0.20)
        n_warm = int(len(offline_files) * warm_frac)
        warm_files = offline_files[:n_warm]
        cold_files = offline_files[n_warm:]

        # Assign VSNs to all offline files
        files_per_vol = max(1, len(offline_files) // max(1, len(vols)))
        for i, f in enumerate(offline_files):
            vol = vols[(i // files_per_vol) % len(vols)]
            f.vsn = vol.vsn
            f.vg_name = vg.name

        # Mark warm files as having pre-mounted tape (zero mount latency)
        for f in warm_files:
            f.state = FileState.OFL
            # Use a special flag to indicate pre-mounted tape
            f._warm_recall = True

        # Convert online files to DUL (dual state: tape copy exists, disk copy too)
        for f in online_files:
            f.state = FileState.DUL
            f.vsn = "ONLINE_NO_TAPE"
            f.vg_name = vg.name

    @staticmethod
    def create_template_trace(output_path: str, n_events: int = 100) -> None:
        """
        Write a template trace file with example records.
        Useful for users who want to format their DMF logs for replay.
        """
        import random
        records = []
        t = 0.0
        for i in range(n_events):
            t += random.uniform(0.1, 5.0)
            et = random.choice(["FILE_CREATE", "RECALL", "MIGRATE", "FILE_MODIFY"])
            records.append({
                "timestamp_s": round(t, 3),
                "event_type": et,
                "file_id": f"bfid_{i:08x}",
                "file_size_bytes": int(10**random.uniform(6, 12)),
                "file_state_before": random.choice(["REG", "OFL", "DUL"]),
                "file_state_after": "MIG" if et == "MIGRATE" else "REG",
                "vsn": f"C0{random.randint(1000,9999)}",
                "drive_id": random.randint(0, 7),
                "latency_s": round(random.uniform(0.1, 120.0), 3),
                "queue_depth": random.randint(0, 50),
                "disk_usage_pct": round(random.uniform(60, 95), 2),
            })

        out = Path(output_path)
        if out.suffix == ".csv":
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
        else:
            with open(out, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
