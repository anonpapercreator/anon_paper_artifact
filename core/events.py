"""
core/events.py — Event Types and DMF 7 File State Machine
==========================================================
Defines all discrete events in the simulation and the DMF 7 file state machine.

DMF 7 file states are faithfully reproduced from the official DMF architecture:
    REG  — Regular (fully online, no tape copy)
    MIG  — Migrating (copy-to-tape in progress, online copy intact)
    DUL  — Dual-state (tape copy exists, online copy retained)
    OFL  — Offline (tape copy exists, online space released; stub in inode)
    UNM  — Unmigrating (recall from tape in progress)
    PAR  — Partial (mixed online/offline regions)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


# ---------------------------------------------------------------------------
# DMF 7 File State Machine
# ---------------------------------------------------------------------------

class FileState(Enum):
    """DMF 7 file states as defined in the DMF architecture documentation."""
    REG = "REG"   # Regular: fully online, no tape copy
    MIG = "MIG"   # Migrating: copy-to-tape in progress
    DUL = "DUL"   # Dual-state: tape copy + online copy
    OFL = "OFL"   # Offline: tape copy only, stub in inode
    UNM = "UNM"   # Unmigrating: recall in progress
    PAR = "PAR"   # Partial: some regions online, some offline


# ---------------------------------------------------------------------------
# Valid state transitions (δ function)
# Enforced at runtime to catch model defects.
# ---------------------------------------------------------------------------
VALID_TRANSITIONS: dict[tuple[FileState, str], FileState] = {
    # (from_state, event) -> to_state
    (FileState.REG, "policy_select"):      FileState.MIG,
    (FileState.REG, "dmput"):              FileState.MIG,
    (FileState.MIG, "migrate_complete"):   FileState.DUL,
    (FileState.MIG, "migrate_failed"):     FileState.REG,   # Rollback on failure
    (FileState.DUL, "space_release"):      FileState.OFL,
    (FileState.DUL, "modify"):             FileState.REG,   # File modified; BFID soft-deleted
    (FileState.DUL, "delete"):             FileState.REG,   # File deleted
    (FileState.OFL, "read_access"):        FileState.UNM,   # Transparent recall
    (FileState.OFL, "dmget"):              FileState.UNM,   # Explicit recall
    (FileState.OFL, "dmunput"):            FileState.REG,   # Remove tape copy
    (FileState.UNM, "recall_complete"):    FileState.DUL,
    (FileState.UNM, "recall_failed"):      FileState.OFL,   # Rollback on failure
    (FileState.PAR, "recall_complete"):    FileState.DUL,
    (FileState.PAR, "partial_release"):    FileState.OFL,
    # Direct migration path (space_release_on_migration=true in config)
    (FileState.MIG, "space_release"):      FileState.OFL,
}


class InvalidStateTransitionError(Exception):
    """Raised when an invalid DMF file state transition is attempted."""
    pass


def transition(current: FileState, event: str) -> FileState:
    """
    Apply a state transition event to a file.

    Args:
        current: Current file state.
        event:   Transition event name (see VALID_TRANSITIONS).

    Returns:
        New file state.

    Raises:
        InvalidStateTransitionError: If the transition is not valid.
    """
    key = (current, event)
    if key not in VALID_TRANSITIONS:
        raise InvalidStateTransitionError(
            f"Invalid transition: {current.value} --[{event}]--> ?"
            f"\nValid transitions from {current.value}: "
            f"{[e for (s, e) in VALID_TRANSITIONS if s == current]}"
        )
    return VALID_TRANSITIONS[key]


# ---------------------------------------------------------------------------
# File Record
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    """
    Represents a single file managed by the DMF 7 simulator.

    All timing values are in simulation seconds (float).
    """
    file_id: str                         # Unique identifier (BFID analogue)
    size_bytes: int                      # File size in bytes
    state: FileState = FileState.REG     # Current DMF state
    created_at: float = 0.0             # Simulation time of creation
    last_accessed_at: float = 0.0       # Simulation time of last access
    last_modified_at: float = 0.0       # Simulation time of last modification
    migrated_at: Optional[float] = None # Simulation time migration completed
    recalled_at: Optional[float] = None # Simulation time of last recall

    # Tape placement (set when migrated)
    vsn: Optional[str] = None           # Volume serial number
    vg_name: Optional[str] = None       # Volume group name
    tape_block_offset: Optional[int] = None  # Block position on tape (for seek model)
    copy2_vsn: Optional[str] = None     # Second copy VSN (dual_copy mode)

    # Recall tracking
    recall_request_at: Optional[float] = None  # When recall was triggered
    recall_priority: int = 5            # 5=batch, 10=interactive

    # Multi-tenant tag. Optional — None for single-tenant runs (back-compat
    # with existing tests / configs). Set by the multi-tenant workload
    # generator to identify which tenant's stream produced this file.
    tenant_id: Optional[str] = None

    # File lifetime (IPDPS 2027 revision). Set when the file is deleted by the
    # lifetime process; a deleted file is never accessed, selected or recalled.
    deleted_at: Optional[float] = None

    # Statistics counters
    n_migrations: int = 0
    n_recalls: int = 0

    @property
    def last_touched_at(self) -> float:
        """Timestamp (sim seconds) of the most recent touch — the later of
        ``last_accessed_at`` and ``created_at``. This is a *timestamp*, not
        an elapsed age; callers compute the age as ``now - last_touched_at``.
        """
        return max(self.last_accessed_at, self.created_at)

    @property
    def age(self) -> float:
        """Deprecated alias for ``last_touched_at``.

        The name is historical and misleading: this property returns the
        timestamp of the last touch, not an elapsed duration. Existing
        callers compute ``now - file.age`` to recover the actual age. New
        code should use ``last_touched_at`` directly to make the intent
        explicit; this alias is kept so external scripts and tests keep
        working.
        """
        return self.last_touched_at

    def apply_transition(self, event: str) -> None:
        """Apply a state transition in-place, enforcing validity."""
        new_state = transition(self.state, event)
        self.state = new_state


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

class EventType(Enum):
    """All discrete event types in the DMF 7 simulator."""
    # Workload events
    FILE_ARRIVE        = auto()   # New file written to filesystem
    FILE_ACCESS        = auto()   # File accessed (read/open)
    FILE_MODIFY        = auto()   # File modified
    FILE_DELETE        = auto()   # File deleted

    # Policy engine events
    POLICY_CYCLE_START = auto()   # Policy cycle begins
    POLICY_CYCLE_END   = auto()   # Policy cycle completes
    POLICY_SELECT_FILE = auto()   # Policy selects a file for migration

    # Migration path events
    LS_ACCRUE          = auto()   # File enters LS accrual buffer
    LS_FLUSH           = auto()   # LS accrual buffer flushes to drive
    MOVER_START        = auto()   # Data mover begins transferring file
    MOVER_COMPLETE     = auto()   # Data mover completes transfer
    MIGRATE_COMPLETE   = auto()   # File fully migrated to tape
    SPACE_RELEASE      = auto()   # Online copy released (DUL → OFL)

    # Recall path events
    RECALL_TRIGGER     = auto()   # Recall initiated (read access or dmget)
    RECALL_QUEUE       = auto()   # Recall request enters tape queue
    MOUNT_START        = auto()   # Robot begins mount sequence
    MOUNT_COMPLETE     = auto()   # Tape mounted and positioned
    RECALL_TRANSFER    = auto()   # Tape-to-disk transfer begins
    RECALL_COMPLETE    = auto()   # File fully recalled to disk

    # Robot events
    ROBOT_IDLE         = auto()   # Robot arm becomes available
    ROBOT_BUSY         = auto()   # Robot arm begins operation

    # Drive events
    DRIVE_MOUNT        = auto()   # Drive mounting a volume
    DRIVE_IDLE         = auto()   # Drive becomes available
    DRIVE_CLEANING     = auto()   # Drive enters cleaning cycle

    # Statistics events
    STATS_SNAPSHOT     = auto()   # Periodic statistics snapshot


@dataclass
class SimEvent:
    """
    A simulation event on the Future Event List (FEL).

    The FEL is managed by SimPy's event mechanism; this dataclass carries
    payload data alongside the event type for logging and statistics.
    """
    event_type: EventType
    time: float                          # Simulation time (seconds)
    file_id: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_state_before: Optional[FileState] = None
    file_state_after: Optional[FileState] = None
    vsn: Optional[str] = None
    drive_id: Optional[int] = None
    mover_id: Optional[int] = None
    latency_s: Optional[float] = None
    queue_depth: Optional[int] = None
    disk_usage_pct: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a dictionary for output."""
        return {
            "event_type": self.event_type.name,
            "time": round(self.time, 6),
            "file_id": self.file_id,
            "file_size_bytes": self.file_size_bytes,
            "file_state_before": self.file_state_before.value if self.file_state_before else None,
            "file_state_after": self.file_state_after.value if self.file_state_after else None,
            "vsn": self.vsn,
            "drive_id": self.drive_id,
            "mover_id": self.mover_id,
            "latency_s": round(self.latency_s, 6) if self.latency_s is not None else None,
            "queue_depth": self.queue_depth,
            "disk_usage_pct": round(self.disk_usage_pct, 4) if self.disk_usage_pct is not None else None,
        }
