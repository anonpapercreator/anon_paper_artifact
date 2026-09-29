"""
Migration Policy with Declarative Control
==========================================
Implements time-based migration with access pattern prefetching.

Declarative Policy System:
- Users specify WHAT they want (performance vs cost)
- System derives HOW to achieve it (cache vs tape usage)
- Multiple presets for common use cases

Mathematical Model:
- Age threshold: T_age = user_specified
- Selection: f* = argmin(age(f)) for age(f) > T_age
- Prefetch: Benefit > 0 with confidence > threshold

Author: Simulation Team
"""

from __future__ import annotations
import simpy
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class PolicyMode(Enum):
    """Policy execution mode."""
    DECLARATIVE = "declarative"  # User specifies goals, system derives config
    STATIC = "static"            # Fixed parameters (legacy DMF behavior)


class PolicyPreset(Enum):
    """Predefined policy presets."""
    MAXIMUM_PERFORMANCE = "maximum_performance"
    BALANCED = "balanced"
    ARCHIVE_OPTIMIZED = "archive_optimized"
    USER_DEFINED = "user_defined"


@dataclass
class PolicyConfig:
    """
    Declarative Policy Configuration.
    
    Users specify their goals, and the system derives parameters.
    """
    # Mode
    mode: PolicyMode = PolicyMode.DECLARATIVE
    preset: PolicyPreset = PolicyPreset.BALANCED
    
    # Declarative parameters (what user wants)
    cache_priority: float = 0.5        # 0.0-1.0 (1.0 = use all cache)
    tolerate_tape_delays: bool = True  # Allow batching for tape efficiency
    max_recall_latency_s: Optional[float] = None  # User SLA
    max_cost_per_month: Optional[float] = None    # Budget constraint
    
    # Derived parameters (how to achieve it) - set by system
    derived_age_threshold_seconds: float = 2592000  # 30 days
    derived_migration_batch_mb: int = 10240          # 10 GB
    derived_prefetch_lookahead_s: float = 86400     # 24 hours
    derived_prefetch_aggression: float = 0.5         # 0.0-1.0
    
    # High/low water marks
    hwm_fraction: float = 0.90
    lwm_fraction: float = 0.80
    
    # Batching
    min_migration_batch_mb: int = 10240
    max_migration_batch_mb: int = 1048576
    
    # Prefetch
    prefetch_enabled: bool = True
    prefetch_lookahead_s: float = 86400
    prefetch_confidence_threshold: float = 0.75
    
    @classmethod
    def from_preset(cls, preset: PolicyPreset) -> 'PolicyConfig':
        """Create config from predefined preset."""
        configs = {
            PolicyPreset.MAXIMUM_PERFORMANCE: cls(
                mode=PolicyMode.DECLARATIVE,
                preset=PolicyPreset.MAXIMUM_PERFORMANCE,
                cache_priority=1.0,
                tolerate_tape_delays=False,
                derived_age_threshold_seconds=604800,    # 7 days (less aggressive)
                derived_migration_batch_mb=5120,         # 5 GB (smaller batches)
                derived_prefetch_lookahead_s=43200,      # 12 hours
                derived_prefetch_aggression=1.0,
                prefetch_enabled=True,
            ),
            PolicyPreset.BALANCED: cls(
                mode=PolicyMode.DECLARATIVE,
                preset=PolicyPreset.BALANCED,
                cache_priority=0.7,
                tolerate_tape_delays=True,
                derived_age_threshold_seconds=2592000,  # 30 days
                derived_migration_batch_mb=10240,        # 10 GB
                derived_prefetch_lookahead_s=86400,      # 24 hours
                derived_prefetch_aggression=0.5,
                prefetch_enabled=True,
            ),
            PolicyPreset.ARCHIVE_OPTIMIZED: cls(
                mode=PolicyMode.DECLARATIVE,
                preset=PolicyPreset.ARCHIVE_OPTIMIZED,
                cache_priority=0.3,
                tolerate_tape_delays=True,
                derived_age_threshold_seconds=604800,   # 7 days (more aggressive)
                derived_migration_batch_mb=20480,        # 20 GB (larger batches)
                derived_prefetch_lookahead_s=172800,     # 48 hours
                derived_prefetch_aggression=0.2,
                prefetch_enabled=False,
            ),
        }
        return configs.get(preset, cls())
    
    def derive_from_declarative(self):
        """
        Derive operational parameters from declarative goals.
        
        Mathematical relationships:
        
        cache_priority → age_threshold:
            Higher priority = longer retention = larger threshold
            age_threshold = base × cache_priority^(0.5)
            
        cache_priority → batch_size:
            Higher priority = smaller batches (more responsive)
            batch_size = max_batch / (1 + cache_priority)
            
        cache_priority → prefetch_aggression:
            Higher priority = more prefetching
            aggression = cache_priority
        """
        if self.mode != PolicyMode.DECLARATIVE:
            return
            
        base_age = 30 * 24 * 3600  # 30 days
        base_batch = 20 * 1024**2  # 20 GB
        
        self.derived_age_threshold_seconds = base_age * (self.cache_priority ** 0.5)
        self.derived_migration_batch_mb = int(
            base_batch / (1 + self.cache_priority * 2)
        )
        self.derived_prefetch_aggression = self.cache_priority
        
        # Ensure bounds
        self.derived_age_threshold_seconds = max(86400, self.derived_age_threshold_seconds)
        self.derived_migration_batch_mb = max(
            self.min_migration_batch_mb,
            min(self.max_migration_batch_mb, self.derived_migration_batch_mb)
        )


@dataclass
class MigrationMetrics:
    """Metrics for migration policy."""
    policy_evaluations: int = 0
    files_migrated: int = 0
    bytes_migrated: int = 0
    files_evicted: int = 0
    prefetch_triggered: int = 0
    prefetch_hits: int = 0
    batch_waits: int = 0
    
    # Latency impact
    recalls_served_from_cache: int = 0
    recalls_required_tape: int = 0


class MigrationPolicy:
    """
    Migration Policy Engine with Declarative Control.
    
    Mathematical Model:
    
    1. Time-based trigger:
       ∀f ∈ Cache: migrate if (now - f.last_access) > T_age
       
    2. LRU selection within threshold:
       f* = argmin_{f: age(f) > T_age} (age(f))
       
    3. Space-based trigger (hybrid):
       If utilization > HWM:
          Migrate oldest files until LWM reached
          
    4. Prefetch:
       If pattern detected and confidence > threshold:
          Pre-migrate likely-to-be-accessed files
    """
    
    def __init__(
        self,
        env: simpy.Environment,
        cache_tier: Any,  # NVMeCacheTier
        tape_library: Any,  # TapeLibrary
        config: Optional[PolicyConfig] = None,
        callback: Optional[Callable] = None,
    ):
        self.env = env
        self.cache = cache_tier
        self.tape_library = tape_library
        self.config = config or PolicyConfig()
        self.callback = callback
        
        self.metrics = MigrationMetrics()
        
        # Derived configuration
        self.config.derive_from_declarative()
        
        # Batch accumulator
        self.pending_migrations: List[Dict] = []
        self.pending_bytes = 0
        
        # Start policy evaluation process
        self.env.process(self._policy_loop())
    
    def _policy_loop(self):
        """Main policy evaluation loop."""
        interval = self.config.derived_age_threshold_seconds / 100  # Fraction of threshold
        interval = max(60, min(3600, interval))  # Clamp to 1 min - 1 hour
        
        while True:
            yield self.env.timeout(interval)
            self._evaluate_policy()
    
    def _evaluate_policy(self):
        """Evaluate migration policy."""
        self.metrics.policy_evaluations += 1
        
        cache_state = self.cache.get_cache_state()
        current_util = cache_state['utilization']
        
        # Check space-based trigger
        if current_util > self.config.hwm_fraction:
            self._evict_to_lwm()
        
        # Check time-based trigger
        if self.config.tolerate_tape_delays:
            self._check_time_based_migration()
        
        # Process pending batch
        self._process_pending_batch()
        
        # Prefetch if enabled
        if self.config.prefetch_enabled:
            self._check_prefetch()
    
    def _check_time_based_migration(self):
        """Check for files eligible for time-based migration."""
        threshold = self.config.derived_age_threshold_seconds
        
        # Get files older than threshold
        eligible_files = self.cache.get_files_for_migration(threshold)
        
        if not eligible_files:
            return
        
        # Sort by age (oldest first)
        eligible_files.sort(
            key=lambda f: self.env.now - f.last_access_time,
            reverse=True
        )
        
        # Add to pending batch
        for file in eligible_files:
            self.pending_migrations.append({
                'file_id': file.file_id,
                'size_bytes': file.size_bytes,
                'vsn': getattr(file, 'vsn', 'default'),
                'age': self.env.now - file.last_access_time,
            })
            self.pending_bytes += file.size_bytes
            
            # Mark as migrating (DUL state)
            file.on_tape = True  # Now has tape copy
    
    def _evict_to_lwm(self):
        """Evict files to reach LWM."""
        cache_state = self.cache.get_cache_state()
        target_bytes = int(cache_state['capacity_bytes'] * self.config.lwm_fraction)
        current_bytes = cache_state['bytes_used']
        
        if current_bytes <= target_bytes:
            return
        
        bytes_to_free = current_bytes - target_bytes
        freed_bytes = 0
        
        # Get LRU files
        candidates = self.cache.get_least_recently_used(1000)
        
        for file in candidates:
            if freed_bytes >= bytes_to_free:
                break
                
            self.pending_migrations.append({
                'file_id': file.file_id,
                'size_bytes': file.size_bytes,
                'vsn': getattr(file, 'vsn', 'default'),
                'age': self.env.now - file.last_access_time,
                'forced': True,  # Space pressure migration
            })
            self.pending_bytes += file.size_bytes
            freed_bytes += file.size_bytes
            self.metrics.files_evicted += 1
    
    def _process_pending_batch(self):
        """Process pending migration batch."""
        if not self.pending_migrations:
            return
        
        batch_bytes = self.pending_bytes
        min_batch = self.config.min_migration_batch_mb * 1024 * 1024
        
        # Check if batch is ready
        if batch_bytes < min_batch and not self._should_force_migration():
            return
        
        # Process batch
        batch_size = 0
        to_migrate = []
        
        for migration in self.pending_migrations:
            to_migrate.append(migration)
            batch_size += migration['size_bytes']
            
            if batch_size >= min_batch:
                break
        
        # Execute migrations
        for migration in to_migrate:
            self._migrate_file(
                migration['file_id'],
                migration['size_bytes'],
                migration.get('vsn', 'default')
            )
            self.pending_migrations.remove(migration)
            self.pending_bytes -= migration['size_bytes']
            
            self.metrics.files_migrated += 1
            self.metrics.bytes_migrated += migration['size_bytes']
    
    def _should_force_migration(self) -> bool:
        """Check if migration should be forced regardless of batch size."""
        cache_state = self.cache.get_cache_state()
        return cache_state['utilization'] > 0.95
    
    def _migrate_file(self, file_id: str, size_bytes: int, vsn: str):
        """Execute actual file migration to tape."""
        def on_complete(end_time):
            if self.callback:
                self.callback(file_id, end_time)
        
        self.tape_library.migrate_file(
            file_id=file_id,
            file_size_bytes=size_bytes,
            vsn=vsn,
            callback=on_complete
        )
    
    def _check_prefetch(self):
        """Check for prefetch opportunities."""
        if not hasattr(self.cache, 'prefetch_predictor'):
            return
        
        predictor = self.cache.prefetch_predictor
        
        # Get recent cache misses (potential prefetch candidates)
        # Simplified: prefetch most popular files periodically
        pass  # Placeholder for full implementation
    
    def record_access(self, file_id: str):
        """Record file access for prefetch prediction."""
        if hasattr(self.cache, 'prefetch_predictor'):
            self.cache.prefetch_predictor.record_access(
                file_id, self.env.now
            )
    
    def request_recall(
        self,
        file_id: str,
        file_size_bytes: int,
        vsn: str,
        callback: Optional[Callable] = None
    ) -> float:
        """
        Handle recall request.
        
        Mathematical relationship:
        - If in cache: serve immediately (cache hit)
        - If not in cache: recall from tape (cache miss → tape)
        """
        # Record access for prefetch
        self.record_access(file_id)
        
        # Check if in cache
        if self.cache.check_hit(file_id):
            self.metrics.recalls_served_from_cache += 1
            if callback:
                callback(self.env.now)
            return self.env.now
        
        # Need to recall from tape
        self.metrics.recalls_required_tape += 1
        
        # Submit to tape library
        submit_time = self.tape_library.recall_file(
            file_id=file_id,
            file_size_bytes=file_size_bytes,
            vsn=vsn,
            callback=callback
        )
        
        return submit_time
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy statistics."""
        cache_state = self.cache.get_cache_state()
        
        return {
            'policy_mode': self.config.mode.value,
            'policy_preset': self.config.preset.value,
            'cache_priority': self.config.cache_priority,
            'age_threshold_days': self.config.derived_age_threshold_seconds / 86400,
            'batch_size_mb': self.config.derived_migration_batch_mb,
            'prefetch_enabled': self.config.prefetch_enabled,
            'evaluations': self.metrics.policy_evaluations,
            'files_migrated': self.metrics.files_migrated,
            'bytes_migrated': self.metrics.bytes_migrated,
            'files_evicted': self.metrics.files_evicted,
            'cache_hit_recalls': self.metrics.recalls_served_from_cache,
            'tape_recalls': self.metrics.recalls_required_tape,
            'cache_utilization': cache_state['utilization'],
            'pending_migrations': len(self.pending_migrations),
        }
    
    def set_policy_preset(self, preset: PolicyPreset):
        """Change policy preset at runtime."""
        self.config = PolicyConfig.from_preset(preset)
        self.config.derive_from_declarative()
    
    def set_custom_policy(
        self,
        cache_priority: float,
        tolerate_tape_delays: bool = True,
        max_recall_latency: Optional[float] = None,
        max_cost: Optional[float] = None,
    ):
        """Set custom declarative policy."""
        self.config.mode = PolicyMode.DECLARATIVE
        self.config.preset = PolicyPreset.USER_DEFINED
        self.config.cache_priority = cache_priority
        self.config.tolerate_tape_delays = tolerate_tape_delays
        self.config.max_recall_latency_s = max_recall_latency
        self.config.max_cost_per_month = max_cost
        self.config.derive_from_declarative()
