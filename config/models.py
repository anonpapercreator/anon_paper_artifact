"""
Configuration Models for HSM Simulator
====================================
Pydantic models for type-safe configuration.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from pathlib import Path


class ESS3500Config(BaseModel):
    """IBM ESS 3500 Parallel Filesystem Configuration."""
    
    # Performance
    nsd_nodes: int = Field(default=4, ge=1, le=32, description="Number of NSD nodes")
    aggregate_throughput_gib_s: float = Field(default=65.0, ge=1.0, description="Aggregate throughput GiB/s")
    network_type: str = Field(default="hdr_infiniband", description="Network type")
    network_bandwidth_gb_s: int = Field(default=200, ge=10, description="Network bandwidth GB/s")
    
    # Striping
    stripe_across_all_nsds: bool = Field(default=True, description="Stripe across all NSDs")
    effective_parallelism: int = Field(default=4, description="Effective parallelism (stripe width)")
    
    # Capacity
    usable_capacity_tib: float = Field(default=809.0, ge=1.0, description="Usable capacity TiB")

    # Per-file size cap (truncates drawn lognormal samples in HSMSimulator)
    max_file_size_bytes: int = Field(default=50 * 1024 ** 3, ge=4096, description="Per-file size cap (bytes)")
    
    # Cost
    cost_per_tb: float = Field(default=15000.0, description="Cost per TiB")
    power_watts_per_node: int = Field(default=1500, description="Power per node Watts")
    
    @validator('effective_parallelism')
    def validate_parallelism(cls, v, values):
        """Ensure parallelism doesn't exceed NSD nodes."""
        nsd = values.get('nsd_nodes', 4)
        return min(v, nsd)


class TS1160Config(BaseModel):
    """IBM TS1160 Tape Drive Configuration."""
    
    n_drives: int = Field(default=16, ge=1, le=64, description="Number of tape drives")
    drive_type: str = Field(default="TS1160", description="Drive type")
    
    # Transfer rates
    native_rate_bytes_per_s: int = Field(default=419430400, description="Native rate bytes/s")
    compressed_rate_bytes_per_s: int = Field(default=943718400, description="Compressed rate bytes/s")
    
    # Mount/unmount times (LogNormal parameters)
    mount_time_mu: float = Field(default=3.0, description="Mount time LogNormal mu")
    mount_time_sigma: float = Field(default=0.4, description="Mount time LogNormal sigma")
    unload_time_mu: float = Field(default=1.1, description="Unload time LogNormal mu")
    unload_time_sigma: float = Field(default=0.3, description="Unload time LogNormal sigma")
    
    # Tape parameters
    tape_capacity_bytes: int = Field(default=21990232555520, description="Tape capacity (20TB)")
    seek_speed_m_per_s: float = Field(default=12.4, description="Seek speed m/s")
    
    # Seek time parameters (LogNormal for realistic variation)
    seek_time_mu: float = Field(default=3.5, description="Seek time LogNormal mu (~35s mean)")
    seek_time_sigma: float = Field(default=0.5, description="Seek time LogNormal sigma")
    
    # Namespace lookup overhead
    namespace_lookup_mu: float = Field(default=1.5, description="Namespace lookup LogNormal mu (~5s mean)")
    namespace_lookup_sigma: float = Field(default=0.6, description="Namespace lookup LogNormal sigma")
    
    # Cost
    cost_per_drive: float = Field(default=40000.0, description="Cost per drive")
    power_watts_drive: int = Field(default=30, description="Power per drive Watts")

    # Library-level cartridge pool (used for tape_capacity_utilization_pct denominator)
    cartridge_pool_size: int = Field(default=100, ge=1, description="Total cartridges in library pool")


class DMFPolicyConfig(BaseModel):
    """DMF Policy Configuration."""
    
    # Age thresholds
    age_threshold_minutes: int = Field(default=1, ge=1, description="File age threshold in minutes")
    
    # Space management
    hwm_fraction: float = Field(default=0.75, ge=0.5, le=0.99, description="High water mark")
    lwm_fraction: float = Field(default=0.50, ge=0.3, le=0.95, description="Low water mark")
    
    # Size-based thresholds
    large_file_threshold_gb: int = Field(default=100, ge=1, description="Large file threshold GB")
    
    # Batching
    min_migration_batch_mb: int = Field(default=10240, ge=1024, description="Min batch MB")
    max_migration_batch_mb: int = Field(default=1048576, ge=1024, description="Max batch MB")

    # Per-cycle migration batch size (number of files moved per archiver tick)
    batch_size: int = Field(default=50, ge=1, description="Files migrated per archiver cycle")

    # Archiver timing
    archiver_interval_minutes: int = Field(default=5, ge=1, le=60, description="Archiver scan interval in minutes")


class DeclarativePolicyConfig(BaseModel):
    """Declarative Policy Configuration."""
    
    preset: Literal["maximum_performance", "balanced", "archive_optimized", "user_defined"] = \
        Field(default="balanced", description="Policy preset")
    
    cache_priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Cache priority (0-1)")
    tolerate_tape_delays: bool = Field(default=True, description="Allow tape batching")
    prefetch_enabled: bool = Field(default=True, description="Enable prefetching")
    prefetch_aggression: float = Field(default=0.5, ge=0.0, le=1.0, description="Prefetch aggression")
    
    # Derived (set automatically)
    age_threshold_seconds: Optional[int] = Field(default=None, description="Derived age threshold")
    migration_batch_mb: Optional[int] = Field(default=None, description="Derived batch size")
    
    def derive_parameters(self):
        """Derive operational parameters from declarative goals."""
        base_age = 30 * 24 * 3600  # 30 days in seconds
        base_batch = 20 * 1024**2  # 20 GB
        
        if self.preset == "maximum_performance":
            self.cache_priority = 1.0
            self.prefetch_aggression = 1.0
        elif self.preset == "balanced":
            self.cache_priority = 0.7
            self.prefetch_aggression = 0.5
        elif self.preset == "archive_optimized":
            self.cache_priority = 0.3
            self.prefetch_aggression = 0.0
        
        # Derive
        self.age_threshold_seconds = int(base_age * (self.cache_priority ** 0.5))
        self.migration_batch_mb = int(base_batch / (1 + self.cache_priority * 2))
        
        # Ensure bounds
        self.age_threshold_seconds = max(86400, self.age_threshold_seconds or 86400)


class CostConfig(BaseModel):
    """Cost Model Configuration."""
    
    power_cost_per_kw_month: float = Field(default=427.0, ge=0.0, description="Power cost $/kW/month")
    rack_space_cost_month: float = Field(default=200.0, description="Rack space $/month")
    admin_cost_annual: float = Field(default=50000.0, description="Admin cost $/year")


class SimulationConfig(BaseModel):
    """Main Simulation Configuration."""
    
    name: str = Field(default="default", description="Configuration name")
    description: str = Field(default="", description="Description")
    
    # Subsystems
    ess3500: ESS3500Config = Field(default_factory=ESS3500Config)
    ts1160: TS1160Config = Field(default_factory=TS1160Config)
    dmf_policy: DMFPolicyConfig = Field(default_factory=DMFPolicyConfig)
    declarative_policy: DeclarativePolicyConfig = Field(default_factory=DeclarativePolicyConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    
    # Simulation parameters
    seed: int = Field(default=42, description="Random seed")
    n_replications: int = Field(default=30, ge=1, le=1000, description="Number of replications")
    sim_duration_hours: float = Field(default=24.0, ge=0.1, description="Simulation duration hours")
    warmup_hours: float = Field(default=0.5, ge=0.0, description="Warmup period hours")
    
    # Workload
    workload_mode: Literal["synthetic", "trace_replay"] = \
        Field(default="trace_replay", description="Workload mode")
    trace_file: Optional[Path] = Field(default=None, description="Path to trace file")
    
    # Output
    output_dir: Path = Field(default=Path("./results"), description="Output directory")
    verbose: bool = Field(default=False, description="Verbose output")
    
    def derive_policy_parameters(self):
        """Derive all policy parameters."""
        self.declarative_policy.derive_parameters()
    
    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to database-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "ess_nodes": self.ess3500.nsd_nodes,
            "ess_throughput_gib_s": self.ess3500.aggregate_throughput_gib_s,
            "ess_capacity_tib": self.ess3500.usable_capacity_tib,
            "network_type": self.ess3500.network_type,
            "ts1160_drives": self.ts1160.n_drives,
            "age_threshold_minutes": self.dmf_policy.age_threshold_minutes,
            "hwm_fraction": self.dmf_policy.hwm_fraction,
            "lwm_fraction": self.dmf_policy.lwm_fraction,
            "large_file_threshold_gb": self.dmf_policy.large_file_threshold_gb,
            "cache_priority": self.declarative_policy.cache_priority,
            "policy_preset": self.declarative_policy.preset,
            "prefetch_enabled": self.declarative_policy.prefetch_enabled,
            "power_cost_per_kw_month": self.cost.power_cost_per_kw_month,
        }


class SweepConfig(BaseModel):
    """Parameter Sweep Configuration."""
    
    name: str = Field(default="", description="Sweep name")
    description: str = Field(default="", description="Description")
    
    # Parameters to sweep
    sweep_age_threshold: bool = Field(default=False, description="Sweep age threshold")
    age_threshold_values: List[int] = Field(
        default=[5, 10, 20, 30, 40, 50, 60, 120, 240, 480, 1440],
        description="Age threshold values (minutes)"
    )
    
    sweep_hwm: bool = Field(default=False, description="Sweep HWM")
    hwm_values: List[float] = Field(
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        description="HWM values"
    )
    
    sweep_cache_priority: bool = Field(default=False, description="Sweep cache priority")
    cache_priority_values: List[float] = Field(
        default=[0.1, 0.3, 0.5, 0.7, 1.0],
        description="Cache priority values"
    )
    
    sweep_archiver_interval: bool = Field(default=True, description="Sweep archiver interval")
    archiver_interval_values: List[int] = Field(
        default=[1, 5, 15, 30, 60, 120],
        description="Archiver interval values (minutes)"
    )
    
    # Execution
    replications: int = Field(default=30, ge=1, description="Replications per config")
    n_workers: int = Field(default=4, ge=1, description="Parallel workers")
    output_dir: Path = Field(default=Path("./results"), description="Output directory")
    
    def get_sweep_params(self) -> Dict[str, List[Any]]:
        """Get parameters being swept."""
        params = {}
        if self.sweep_age_threshold:
            params["age_threshold_minutes"] = self.age_threshold_values
        if self.sweep_hwm:
            params["hwm_fraction"] = self.hwm_values
        if self.sweep_cache_priority:
            params["cache_priority"] = self.cache_priority_values
        if self.sweep_archiver_interval:
            params["archiver_interval_minutes"] = self.archiver_interval_values
        return params
    
    def get_combinations(self) -> List[Dict[str, Any]]:
        """Get all parameter combinations."""
        import itertools
        
        params = self.get_sweep_params()
        if not params:
            return [{}]
        
        keys = list(params.keys())
        values = list(params.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations


class SimulationResult(BaseModel):
    """Simulation Result."""
    
    # Configuration used
    config: SimulationConfig
    
    # Performance Metrics
    cache_hit_ratio: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    
    # Tape Metrics
    tape_mounts: int = 0
    bytes_migrated: int = 0
    
    # Cost Metrics
    total_cost: float = 0.0
    cost_per_recall: float = 0.0
    
    # NEW: Media Consumption Metrics
    archived_data_tb: float = 0.0          # Total data written to tape
    tape_cartridges_used: float = 0.0       # Cartridges consumed
    tape_capacity_utilization_pct: float = 0.0  # % of tape capacity used
    
    # NEW: Bandwidth Metrics
    migration_bandwidth_gbs: float = 0.0    # GB/s used for migration
    scratch_effective_bandwidth_gbs: float = 0.0  # Remaining scratch bandwidth
    scratch_bandwidth_degradation_pct: float = 0.0  # % bandwidth degraded
    
    # NEW: Cache Metrics
    cache_resident_tb: float = 0.0         # Current data on scratch
    cache_churn_tb: float = 0.0             # Total data written to scratch
    
    # NEW: IOR Metrics
    ior_config_name: str = ""              # e.g., "ior_4k_65536k"
    ior_iteration: int = 0                 # iteration number
    total_bytes_written: int = 0           # from WRITE events
    total_bytes_read: int = 0              # from READ events
    write_operations: int = 0              # count of WRITE events
    read_operations: int = 0               # count of READ events
    write_duration_s: float = 0.0           # time from first to last WRITE
    read_duration_s: float = 0.0            # time from first to last READ
    
    # Run info
    run_time_seconds: float = 0.0
    replication: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for database."""
        return {
            # Performance
            "cache_hit_ratio": self.cache_hit_ratio,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "latency_mean": self.latency_mean,
            
            # Tape
            "tape_mounts": self.tape_mounts,
            "bytes_migrated": self.bytes_migrated,
            
            # Cost
            "total_cost": self.total_cost,
            "cost_per_recall": self.cost_per_recall,
            
            # NEW: Media Consumption
            "archived_data_tb": self.archived_data_tb,
            "tape_cartridges_used": self.tape_cartridges_used,
            "tape_capacity_utilization_pct": self.tape_capacity_utilization_pct,
            
            # NEW: Bandwidth
            "migration_bandwidth_gbs": self.migration_bandwidth_gbs,
            "scratch_effective_bandwidth_gbs": self.scratch_effective_bandwidth_gbs,
            "scratch_bandwidth_degradation_pct": self.scratch_bandwidth_degradation_pct,
            
            # NEW: Cache
            "cache_resident_tb": self.cache_resident_tb,
            "cache_churn_tb": self.cache_churn_tb,
            
            # NEW: IOR Metrics
            "ior_config_name": self.ior_config_name,
            "ior_iteration": self.ior_iteration,
            "total_bytes_written": self.total_bytes_written,
            "total_bytes_read": self.total_bytes_read,
            "write_operations": self.write_operations,
            "read_operations": self.read_operations,
            "write_duration_s": self.write_duration_s,
            "read_duration_s": self.read_duration_s,
            
            # Run info
            "run_time_seconds": self.run_time_seconds,
            "replication": self.replication,
            "age_threshold_minutes": self.config.dmf_policy.age_threshold_minutes,
            "hwm_fraction": self.config.dmf_policy.hwm_fraction,
            "lwm_fraction": self.config.dmf_policy.lwm_fraction,
            "cache_priority": self.config.declarative_policy.cache_priority,
            "archiver_interval_minutes": self.config.dmf_policy.archiver_interval_minutes,
        }
