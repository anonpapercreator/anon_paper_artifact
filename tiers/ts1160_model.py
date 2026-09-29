"""
IBM TS1160 Tape Drive Model
============================
Models IBM TS1160 enterprise tape drive with mathematically accurate
service time calculations based on IBM specifications.

IBM TS1160 Specifications:
- Native capacity: 20 TB (JE media)
- Sustained read/write: 400 MB/s
- Search speed: 12.4 m/s
- Load time: ~15 seconds
- Unload time: ~3 seconds
- MTBF: 250,000 hours

Mathematical Model:
- Total recall time: S_recall = t_mount + t_seek + t_read + t_unmount
- Mount time: LogNormal(μ=3.0, σ=0.4) ≈ 20-30s median

Author: Simulation Team
"""

from __future__ import annotations
import simpy
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum


class TapeState(Enum):
    """Tape cartridge states."""
    IDLE = "idle"
    MOUNTED = "mounted"
    UNMOUNTING = "unmounting"


@dataclass
class TapeCartridge:
    """Represents a tape cartridge."""
    vsn: str                    # Volume Serial Number
    capacity_bytes: int          # 20 TB native
    current_position: int = 0   # Current byte position
    is_mounted: bool = False
    last_access_time: float = 0.0
    
    
@dataclass
class TapeMetrics:
    """Metrics for tape subsystem performance."""
    mounts: int = 0
    unmounts: int = 0
    reads: int = 0
    writes: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    total_mount_time: float = 0.0
    total_seek_time: float = 0.0
    total_transfer_time: float = 0.0
    mount_wait_times: List[float] = field(default_factory=list)
    drive_utilization: List[float] = field(default_factory=list)
    
    @property
    def mean_mount_time(self) -> float:
        return self.total_mount_time / self.mounts if self.mounts > 0 else 0.0


class TS1160Drive:
    """
    IBM TS1160 Tape Drive Model.
    
    Mathematical Model:
    
    Total Service Time:
        S_total = S_mount + S_seek + S_transfer + S_unmount
        
    Where:
        - S_mount ~ LogNormal(μ=3.0, σ=0.4) ≈ 20-30s
        - S_seek = position / v_search
        - S_transfer = Size / R_read
        - S_unmount ~ LogNormal(μ=1.1, σ=0.3) ≈ 3s
    """
    
    def __init__(
        self,
        drive_id: int,
        env: simpy.Environment,
        native_rate_bps: int = 419430400,      # 400 MB/s
        mount_time_mu: float = 3.0,            # LogNormal mu (~20s median)
        mount_time_sigma: float = 0.4,          # LogNormal sigma
        unload_time_mu: float = 1.1,            # LogNormal mu (~3s median)
        unload_time_sigma: float = 0.3,
        seek_speed_mps: float = 12.4,           # 12.4 m/s
        tape_capacity_bytes: int = 21990232555520,  # 20 TB
        rng: Optional[np.random.Generator] = None,
    ):
        self.drive_id = drive_id
        self.env = env
        self.native_rate_bps = native_rate_bps
        self.mount_time_mu = mount_time_mu
        self.mount_time_sigma = mount_time_sigma
        self.unload_time_mu = unload_time_mu
        self.unload_time_sigma = unload_time_sigma
        self.seek_speed_mps = seek_speed_mps
        self.tape_capacity_bytes = tape_capacity_bytes
        self.rng = rng or np.random.default_rng()
        
        # State
        self.current_cartridge: Optional[TapeCartridge] = None
        self.is_busy = False
        self.busy_time = 0.0
        self.total_time = 0.0
        
        # Queue for requests
        self.queue = simpy.Store(env)
        
        # Start processing
        self.env.process(self._process_requests())
    
    def calculate_mount_time(self) -> float:
        """Calculate mount time using LogNormal distribution."""
        return self.rng.lognormal(self.mount_time_mu, self.mount_time_sigma)
    
    def calculate_unload_time(self) -> float:
        """Calculate unload time using LogNormal distribution."""
        return self.rng.lognormal(self.unload_time_mu, self.unload_time_sigma)
    
    def calculate_seek_time(self, from_pos: int, to_pos: int, file_size: int) -> float:
        """
        Calculate seek time based on tape position.
        
        Mathematical relationship:
        - Short seek (< 10% of tape): ~seconds
        - Long seek (across tape): proportional to distance
        
        For simplicity, we model as proportional to distance:
        S_seek = |to_pos - from_pos| / v_search
        """
        distance_bytes = abs(to_pos - from_pos)
        # Convert to meters (assume 600m tape length)
        tape_length_m = 600
        bytes_per_meter = self.tape_capacity_bytes / tape_length_m
        distance_m = distance_bytes / bytes_per_meter
        
        return distance_m / self.seek_speed_mps
    
    def calculate_transfer_time(self, file_size: int) -> float:
        """Calculate transfer time at native rate."""
        return file_size / self.native_rate_bps
    
    def _process_requests(self):
        """SimPy process: handle tape requests."""
        while True:
            request = yield self.queue.get()
            
            self.is_busy = True
            request_start = self.env.now
            
            # Extract request data
            file_size = request['file_size_bytes']
            target_position = request.get('target_position', 0)
            is_read = request.get('is_read', True)
            callback = request.get('callback')
            
            # Mount cartridge if needed
            if self.current_cartridge is None:
                mount_time = self.calculate_mount_time()
                yield self.env.timeout(mount_time)
                
                # Create new cartridge
                self.current_cartridge = TapeCartridge(
                    vsn=request.get('vsn', 'UNKNOWN'),
                    capacity_bytes=self.tape_capacity_bytes,
                    current_position=0,
                    is_mounted=True
                )
            
            # Seek to position
            from_pos = self.current_cartridge.current_position
            seek_time = self.calculate_seek_time(from_pos, target_position, file_size)
            yield self.env.timeout(seek_time)
            
            # Transfer data
            transfer_time = self.calculate_transfer_time(file_size)
            yield self.env.timeout(transfer_time)
            
            # Update cartridge position
            self.current_cartridge.current_position = target_position + file_size
            
            # Unmount if requested (or keep mounted for next access)
            if request.get('unmount_after', False):
                unload_time = self.calculate_unload_time()
                yield self.env.timeout(unload_time)
                self.current_cartridge.is_mounted = False
                self.current_cartridge = None
            
            request_end = self.env.now
            
            # Record metrics
            self.busy_time += (request_end - request_start)
            
            # Callback
            if callback:
                callback(request_end)
            
            self.is_busy = False
    
    def submit_request(
        self,
        file_size_bytes: int,
        vsn: str,
        is_read: bool = True,
        target_position: int = 0,
        unmount_after: bool = False,
        callback=None
    ) -> float:
        """Submit a read/write request to the drive."""
        request = {
            'file_size_bytes': file_size_bytes,
            'vsn': vsn,
            'is_read': is_read,
            'target_position': target_position,
            'unmount_after': unmount_after,
            'callback': callback,
            'submit_time': self.env.now,
        }
        self.queue.put(request)
        return request['submit_time']
    
    @property
    def utilization(self) -> float:
        """Drive utilization."""
        if self.env.now <= 0:
            return 0.0
        return self.busy_time / self.env.now


class TapeLibrary:
    """
    Tape Library with robot arm and multiple drives.
    
    Mathematical Model:
    - M/G/k queue where k = number of drives
    - Robot arm as separate M/G/1 queue
    - Volume groups for logical organization
    """
    
    def __init__(
        self,
        env: simpy.Environment,
        n_drives: int,
        n_slots: int = 500,
        robot_arm_speed: float = 1.5,        # m/s
        library_height: float = 2.0,          # meters
        library_width: float = 1.5,           # meters
        mount_time_mu: float = 3.0,
        mount_time_sigma: float = 0.4,
        unload_time_mu: float = 1.1,
        unload_time_sigma: float = 0.3,
        native_rate_bps: int = 419430400,     # 400 MB/s
        tape_capacity_bytes: int = 21990232555520,  # 20 TB
        volume_groups: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.env = env
        self.n_drives = n_drives
        self.n_slots = n_slots
        self.rng = rng or np.random.default_rng()
        
        # Create drives
        self.drives: List[TS1160Drive] = []
        for i in range(n_drives):
            drive = TS1160Drive(
                drive_id=i,
                env=env,
                native_rate_bps=native_rate_bps,
                mount_time_mu=mount_time_mu,
                mount_time_sigma=mount_time_sigma,
                unload_time_mu=unload_time_mu,
                unload_time_sigma=unload_time_sigma,
                tape_capacity_bytes=tape_capacity_bytes,
                rng=rng,
            )
            self.drives.append(drive)
        
        # Robot arm
        self.robot_busy = False
        self.robot_busy_time = 0.0
        
        # Cartridge inventory
        self.cartridges: Dict[str, TapeCartridge] = {}
        self.free_slots = list(range(n_slots))
        
        # Volume groups
        self.volume_groups = volume_groups or {}
        
        # Metrics
        self.metrics = TapeMetrics()
    
    def get_available_drive(self) -> Optional[TS1160Drive]:
        """Get an available drive (not busy)."""
        for drive in self.drives:
            if not drive.is_busy:
                return drive
        return None
    
    def get_drive_for_vg(self, vg_name: str) -> Optional[TS1160Drive]:
        """Get available drive assigned to volume group."""
        vg = self.volume_groups.get(vg_name, {})
        drives_assigned = vg.get('drives_assigned', list(range(self.n_drives)))
        
        for drive_idx in drives_assigned:
            if drive_idx < len(self.drives) and not self.drives[drive_idx].is_busy:
                return self.drives[drive_idx]
        
        return None
    
    def mount_cartridge(self, vsn: str) -> float:
        """
        Mount a cartridge (robot arm operation).
        
        Mathematical model:
        S_mount = t_base + distance / v_arm + t_load
        """
        # Simplified: use LogNormal from drive
        if self.cartridges.get(vsn):
            return 0.0  # Already mounted
            
        mount_time = self.rng.lognormal(3.0, 0.4)  # ~20-30s
        self.robot_busy = True
        
        yield self.env.timeout(mount_time)
        
        self.robot_busy = False
        self.metrics.mounts += 1
        self.metrics.total_mount_time += mount_time
        
        return mount_time
    
    def recall_file(
        self,
        file_id: str,
        file_size_bytes: int,
        vsn: str,
        callback=None
    ) -> float:
        """
        Recall a file from tape.
        
        Mathematical relationship:
        Total recall time = mount_wait + mount + seek + read + (unmount)
        
        Returns submission time.
        """
        # Get available drive
        drive = self.get_available_drive()
        
        if drive is None:
            # Wait for drive
            # This is handled by the drive queue
            drive = self.drives[0]  # Will queue
        
        # Submit to drive
        submit_time = drive.submit_request(
            file_size_bytes=file_size_bytes,
            vsn=vsn,
            is_read=True,
            callback=callback
        )
        
        self.metrics.reads += 1
        self.metrics.bytes_read += file_size_bytes
        
        return submit_time
    
    def migrate_file(
        self,
        file_id: str,
        file_size_bytes: int,
        vsn: str,
        callback=None
    ) -> float:
        """Migrate (write) a file to tape."""
        drive = self.get_available_drive()
        
        if drive is None:
            drive = self.drives[0]
        
        submit_time = drive.submit_request(
            file_size_bytes=file_size_bytes,
            vsn=vsn,
            is_read=False,
            callback=callback
        )
        
        self.metrics.writes += 1
        self.metrics.bytes_written += file_size_bytes
        
        return submit_time
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        total_drive_time = sum(d.busy_time for d in self.drives)
        elapsed = self.env.now
        
        return {
            'total_mounts': self.metrics.mounts,
            'total_unmounts': self.metrics.unmounts,
            'total_reads': self.metrics.reads,
            'total_writes': self.metrics.writes,
            'bytes_read': self.metrics.bytes_read,
            'bytes_written': self.metrics.bytes_written,
            'mean_mount_time': self.metrics.mean_mount_time,
            'drive_utilization': [d.utilization for d in self.drives],
            'robot_utilization': self.robot_busy_time / elapsed if elapsed > 0 else 0,
            'queued_requests': sum(len(d.queue.items) for d in self.drives),
        }
