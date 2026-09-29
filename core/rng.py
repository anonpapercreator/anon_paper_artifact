"""
core/rng.py — Independent RNG Stream Management
================================================
Uses numpy's SeedSequence + PCG64 to generate independent, non-overlapping
random number streams, one per stochastic component.

Scientific basis:
    Law & Kelton (2000), §7.2: "Each stochastic input to the model must
    be driven by an independent random number stream to avoid spurious
    correlations between components."

    PCG64 (O'Neill, 2014) provides 2^128 period with provably independent
    sub-streams when spawned from SeedSequence.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple


def lognormal_params_from_mean_std(mean: float, std: float) -> Tuple[float, float]:
    """Convert linear-space (mean, std) → log-space (μ_log, σ_log).

    The returned values are the parameters of the underlying normal
    distribution, suitable for numpy Generator.lognormal(mean=μ_log, sigma=σ_log).

    Derivation (standard lognormal moments):
        σ_log = sqrt( ln(1 + std²/mean²) )
        μ_log = ln(mean) - σ_log² / 2

    Raises ValueError for non-positive mean or negative std.
    """
    if mean <= 0.0:
        raise ValueError(f"lognormal mean must be > 0; got {mean}")
    if std < 0.0:
        raise ValueError(f"lognormal std must be >= 0; got {std}")
    cv2 = (std / mean) ** 2
    sigma_log = math.sqrt(math.log1p(cv2))
    mu_log = math.log(mean) - 0.5 * sigma_log * sigma_log
    return mu_log, sigma_log

# ---------------------------------------------------------------------------
# Stream registry — one entry per stochastic component.
# Adding a new stream here automatically allocates an independent sub-stream.
# ---------------------------------------------------------------------------
STREAM_NAMES = [
    "file_size",          # 0: File size distribution samples
    "iat",                # 1: Inter-arrival time samples
    "access_freq",        # 2: Zipf access frequency
    "rw_ratio",           # 3: Read/write decision
    "mount_time",         # 4: Tape mount time samples
    "seek_time",          # 5: Tape seek time perturbation
    "transfer_time",      # 6: Tape transfer time (compression ratio)
    "disk_service",       # 7: Disk cache service time
    "robot_travel",       # 8: Robot arm travel time
    "ov_latency",         # 9: OpenVault scheduling latency
    "compression_ratio",  # 10: Tape compression ratio
    "diurnal",            # 11: Diurnal pattern noise
    "policy_jitter",      # 12: Small jitter on policy cycle timing
    "file_age_init",      # 13: Initial file ages at sim start
    "replication",        # 14: Per-replication seed differentiation
    "ws_eviction",        # 15: Working-set eviction choice in workload gen
]

N_STREAMS = len(STREAM_NAMES)


class RNGManager:
    """
    Manages independent random number streams using numpy SeedSequence.

    Each call to get(name) returns the same Generator object for that stream,
    allowing callers to draw samples from it without worrying about stream
    independence.

    Usage:
        rng_mgr = RNGManager(seed=42)
        file_size_rng = rng_mgr.get("file_size")
        size = file_size_rng.lognormal(mu, sigma)
    """

    def __init__(self, master_seed: int, replication: int = 0) -> None:
        """
        Args:
            master_seed:  Master integer seed for the simulation run.
            replication:  Replication index (0-based). Each replication
                          gets a completely independent set of streams.
        """
        self._master_seed = master_seed
        self._replication = replication

        # Two-level spawn: first spawn N_STREAMS+1 children from master_seed,
        # then use child[N_STREAMS] to spawn replication-specific overrides.
        ss_master = np.random.SeedSequence(master_seed)
        # Spawn one extra child to differentiate replications
        children = ss_master.spawn(N_STREAMS + 1)

        # The last child is used to differentiate replications
        rep_ss = children[N_STREAMS].spawn(replication + 1)[replication]

        self._streams: Dict[str, np.random.Generator] = {}
        for i, name in enumerate(STREAM_NAMES):
            if name == "replication":
                # Replication stream uses replication-specific seed
                self._streams[name] = np.random.default_rng(rep_ss)
            else:
                # Mix base stream with replication index for independence
                mixed = children[i].spawn(replication + 1)[replication]
                self._streams[name] = np.random.default_rng(mixed)

    def get(self, name: str) -> np.random.Generator:
        """Return the Generator for the named stream.

        Args:
            name: Stream name from STREAM_NAMES.

        Returns:
            numpy.random.Generator instance.

        Raises:
            KeyError: If name is not a registered stream.
        """
        if name not in self._streams:
            raise KeyError(
                f"Unknown RNG stream: '{name}'. "
                f"Registered streams: {STREAM_NAMES}"
            )
        return self._streams[name]

    _EXTRA_STREAMS = {"file_lifetime": 1001}

    def extra(self, name: str) -> np.random.Generator:
        """Return an additional named stream that is independent of, and does
        not perturb, the registered STREAM_NAMES streams. Deterministic in
        (master_seed, replication, name)."""
        if not hasattr(self, "_extra"):
            self._extra: Dict[str, np.random.Generator] = {}
        if name not in self._extra:
            key = self._EXTRA_STREAMS[name]
            ss = np.random.SeedSequence(entropy=self._master_seed,
                                        spawn_key=(key, self._replication))
            self._extra[name] = np.random.default_rng(ss)
        return self._extra[name]

    @property
    def master_seed(self) -> int:
        return self._master_seed

    @property
    def replication(self) -> int:
        return self._replication


def create_rng_manager(master_seed: int, replication: int = 0) -> RNGManager:
    """Factory function for creating an RNGManager for a given replication."""
    return RNGManager(master_seed=master_seed, replication=replication)
