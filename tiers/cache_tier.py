"""
NVMe Cache Tier Model
=====================
Models a high-performance NVMe cache tier for hierarchical storage.

Mathematical Model:
- M/G/1 queue with IOPS limits
- LRU cache replacement policy
- Service time: min(Size/R_seq, 1/IOPS)

Author: Simulation Team
"""

from __future__ import annotations
import simpy
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import heapq


@dataclass
class CacheFile:
    """Represents a file in the cache."""
    file_id: str
    size_bytes: int
    last_access_time: float = 0.0
    access_count: int = 0
    created_time: float = 0.0
    on_tape: bool = True  # True if backup exists on tape
    
    
@dataclass 
class CacheMetrics:
    """Metrics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_served: int = 0
    bytes_written: int = 0
    total_latency: float = 0.0
    queue_wait_times: List[float] = field(default_factory=list)
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def mean_latency(self) -> float:
        return self.total_latency / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0


class LRUCache:
    """
    LRU Cache implementation with configurable capacity.
    
    Mathematical relationship:
    - Cache hit probability P(hit) depends on:
      - Cache size C (bytes)
      - Working set size W (bytes)  
      - Access pattern (Zipf parameter α)
    - For Zipf distribution: P(hit) ≈ Σ_{i=1}^{C/avg_file_size} (1/i^α)
    """
    
    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self._cache: OrderedDict[str, CacheFile] = OrderedDict()
    
    def contains(self, file_id: str) -> bool:
        return file_id in self._cache
    
    def get(self, file_id: str) -> Optional[CacheFile]:
        """Get file from cache, updating LRU order."""
        if file_id not in self._cache:
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(file_id)
        return self._cache[file_id]
    
    def put(self, file: CacheFile) -> List[str]:
        """
        Add file to cache. Evict LRU files if needed.
        Returns list of evicted file IDs.
        """
        evicted = []
        
        # If file already in cache, just update
        if file.file_id in self._cache:
            old_size = self._cache[file.file_id].size_bytes
            self.current_bytes -= old_size
            self._cache.move_to_end(file.file_id)
            self._cache[file.file_id] = file
            self.current_bytes += file.size_bytes
            return evicted
        
        # While we need space, evict LRU
        while self.current_bytes + file.size_bytes > self.capacity_bytes and self._cache:
            # Pop oldest (first) item
            oldest_id, oldest_file = self._cache.popitem(last=False)
            self.current_bytes -= oldest_file.size_bytes
            evicted.append(oldest_id)
        
        # Add new file
        if file.size_bytes <= self.capacity_bytes:
            self._cache[file.file_id] = file
            self.current_bytes += file.size_bytes
            
        return evicted
    
    def get_lru_files(self, n: int) -> List[CacheFile]:
        """Get N least recently used files."""
        # Return from oldest (start of OrderedDict)
        return list(self._cache.values())[:n]
    
    def get_oldest_files(self, age_threshold: float, current_time: float) -> List[CacheFile]:
        """Get files older than age_threshold (for migration)."""
        return [
            f for f in self._cache.values()
            if current_time - f.last_access_time > age_threshold
        ]
    
    @property
    def utilization(self) -> float:
        return self.current_bytes / self.capacity_bytes if self.capacity_bytes > 0 else 0.0
    
    @property
    def file_count(self) -> int:
        return len(self._cache)


class NVMeCacheTier:
    """
    NVMe Cache Tier Model with M/G/1 queueing and IOPS limits.
    
    Mathematical Model:
    
    Service Time:
        S(f) = min(Size / R_seq, 1/IOPS_random)
        
    Queueing Delay (Kingman approximation):
        W_q ≈ (ρ / (1-ρ)) × E[S] × (c_a² + c_s²) / 2
        
    Where:
        - ρ = λ × E[S] (utilization)
        - c_a = σ_A / E[A] (arrival CV)
        - c_s = σ_S / E[S] (service CV)
    """
    
    def __init__(
        self,
        env: simpy.Environment,
        capacity_bytes: int,
        sequential_read_mbps: float = 5000,
        sequential_write_mbps: float = 4000,
        random_read_iops: int = 750000,
        random_write_iops: int = 600000,
        latency_us: float = 15,
        service_time_cv: float = 0.35,
        rng: Optional[np.random.Generator] = None,
    ):
        self.env = env
        self.capacity_bytes = capacity_bytes
        
        # Performance parameters
        self.sequential_read_mbps = sequential_read_mbps
        self.sequential_write_mbps = sequential_write_mbps
        self.random_read_iops = random_read_iops
        self.random_write_iops = random_write_iops
        self.latency_us = latency_us
        self.service_time_cv = service_time_cv
        
        # Derived limits
        self.sequential_read_bps = sequential_read_mbps * 1024 * 1024
        self.sequential_write_bps = sequential_write_mbps * 1024 * 1024
        
        # Cache and queue
        self.cache = LRUCache(capacity_bytes)
        self.queue = simpy.Store(env)
        self.resource = simpy.Resource(env, capacity=1)  # Single server
        
        # Metrics
        self.metrics = CacheMetrics()
        
        # RNG
        self.rng = rng or np.random.default_rng()
        
        # Start processing
        self.env.process(self._process_requests())
    
    def calculate_service_time(self, file_size_bytes: int, is_write: bool = False) -> float:
        """
        Calculate service time for a file.
        
        Mathematical relationship:
        S = min(Size / R_sequential, 1/IOPS) + base_latency
        
        Returns service time in seconds.
        """
        # Sequential time
        rate = self.sequential_write_bps if is_write else self.sequential_read_bps
        seq_time = file_size_bytes / rate
        
        # Random IOPS time (time per IO assuming 4KB block)
        block_size = 4096
        num_ios = file_size_bytes / block_size
        iops_limit = self.random_write_iops if is_write else self.random_read_iops
        iops_time = num_ios / iops_limit
        
        # Take the slower (limiting) path
        service_time = min(seq_time, iops_time)
        
        # Add base latency (microseconds to seconds)
        service_time += self.latency_us / 1_000_000
        
        # Add stochastic variation (CV)
        if self.service_time_cv > 0:
            # LogNormal with specified CV
            # For LogNormal: CV = sqrt(e^σ² - 1)
            # So: σ = sqrt(log(1 + CV²))
            sigma = np.sqrt(np.log(1 + self.service_time_cv**2))
            mu = np.log(service_time) - sigma**2 / 2
            service_time = self.rng.lognormal(mu, sigma)
        
        return max(service_time, self.latency_us / 1_000_000)  # Min latency
    
    def _process_requests(self):
        """SimPy process: serve requests from queue."""
        while True:
            request = yield self.queue.get()
            
            with self.resource.request() as req:
                yield req
                request_start = self.env.now
                
                # Calculate service time
                service_time = self.calculate_service_time(
                    request['file_size_bytes'],
                    is_write=request.get('is_write', False)
                )
                
                yield self.env.timeout(service_time)
                
                request_end = self.env.now
                queue_wait = request_start - request['arrival_time']
                
                # Update metrics
                self.metrics.queue_wait_times.append(queue_wait)
                self.metrics.bytes_served += request['file_size_bytes']
                self.metrics.total_latency += (request_end - request['arrival_time'])
                
                # Callback
                if request.get('callback'):
                    request['callback'](request_end)
    
    def access(
        self,
        file_id: str,
        file_size_bytes: int,
        is_write: bool = False,
        callback=None
    ) -> float:
        """
        Access a file from the cache tier.
        Returns the time the request was enqueued.
        """
        arrival_time = self.env.now
        
        # Check cache hit
        file = self.cache.get(file_id)
        
        if file:
            # Cache hit
            self.metrics.hits += 1
            # Update access time
            file.last_access_time = arrival_time
            file.access_count += 1
            
            if callback:
                # Immediate callback for cache hits
                callback(arrival_time)
        else:
            # Cache miss
            self.metrics.misses += 1
            
            # Add to cache
            new_file = CacheFile(
                file_id=file_id,
                size_bytes=file_size_bytes,
                last_access_time=arrival_time,
                created_time=arrival_time,
                on_tape=True
            )
            evicted = self.cache.put(new_file)
            self.metrics.evictions += len(evicted)
            self.metrics.bytes_written += file_size_bytes
            
            # Enqueue for service
            request = {
                'file_id': file_id,
                'file_size_bytes': file_size_bytes,
                'is_write': is_write,
                'arrival_time': arrival_time,
                'callback': callback,
            }
            self.queue.put(request)
        
        return arrival_time
    
    def check_hit(self, file_id: str) -> bool:
        """Check if file is in cache (without updating LRU)."""
        return self.cache.contains(file_id)
    
    def get_cache_state(self) -> Dict[str, Any]:
        """Get current cache state for policy decisions."""
        return {
            'utilization': self.cache.utilization,
            'file_count': self.cache.file_count,
            'bytes_used': self.cache.current_bytes,
            'capacity_bytes': self.capacity_bytes,
            'hits': self.metrics.hits,
            'misses': self.metrics.misses,
            'hit_ratio': self.metrics.hit_ratio,
        }
    
    def get_files_for_migration(self, age_threshold: float) -> List[CacheFile]:
        """Get files eligible for migration based on age threshold."""
        return self.cache.get_oldest_files(age_threshold, self.env.now)
    
    def get_least_recently_used(self, n: int) -> List[CacheFile]:
        """Get N least recently used files."""
        return self.cache.get_lru_files(n)


class PrefetchPredictor:
    """
    Access pattern-based prefetch predictor.
    
    Mathematical Model:
    - Detects periodic access patterns via autocorrelation
    - Calculates prefetch benefit:
      Benefit = E[future_access_latency] - prefetch_cost
    - Activates if benefit > 0 with confidence > threshold
    """
    
    def __init__(
        self,
        lookahead_seconds: float = 86400,
        confidence_threshold: float = 0.75,
        rng: Optional[np.random.Generator] = None,
    ):
        self.lookahead_seconds = lookahead_seconds
        self.confidence_threshold = confidence_threshold
        self.rng = rng or np.random.default_rng()
        
        # Access history for pattern detection
        self.access_history: List[tuple] = []  # (timestamp, file_id)
        
        # Detected patterns
        self.periodic_jobs: Dict[str, float] = {}  # file_id -> detected period
        
    def record_access(self, file_id: str, timestamp: float):
        """Record an access for pattern detection."""
        self.access_history.append((timestamp, file_id))
        
        # Keep history manageable
        if len(self.access_history) > 10000:
            self.access_history = self.access_history[-5000:]
    
    def detect_periodic_access(self, file_id: str) -> Optional[float]:
        """
        Detect periodic access pattern for a file.
        Uses autocorrelation to find periodicity.
        
        Returns detected period in seconds, or None if no pattern.
        """
        # Get all access times for this file
        file_accesses = [
            ts for ts, fid in self.access_history 
            if fid == file_id
        ]
        
        if len(file_accesses) < 4:
            return None
        
        # Calculate inter-arrival times
        inter_arrivals = [
            file_accesses[i+1] - file_accesses[i]
            for i in range(len(file_accesses) - 1)
        ]
        
        if not inter_arrivals:
            return None
            
        mean_ia = np.mean(inter_arrivals)
        std_ia = np.std(inter_arrivals)
        
        # Low CV indicates periodic (CV < 0.5)
        cv = std_ia / mean_ia if mean_ia > 0 else float('inf')
        
        if cv < 0.5:
            return mean_ia
        
        return None
    
    def should_prefetch(
        self,
        file_id: str,
        current_time: float,
        prefetch_cost: float,
        base_latency: float
    ) -> bool:
        """
        Decide if file should be prefetched.
        
        Mathematical relationship:
            Benefit = E[future_latency] - prefetch_cost
            Prefetch if Benefit > 0 and confidence > threshold
            
        Where:
            - E[future_latency] estimated from pattern
            - prefetch_cost = service time to bring file to cache
            - confidence = based on pattern consistency
        """
        # Detect pattern
        period = self.detect_periodic_access(file_id)
        
        if period is None:
            return False
        
        # Calculate confidence (inverse of CV)
        file_accesses = [
            ts for ts, fid in self.access_history 
            if fid == file_id
        ]
        
        if len(file_accesses) < 4:
            return False
            
        inter_arrivals = [
            file_accesses[i+1] - file_accesses[i]
            for i in range(len(file_accesses) - 1)
        ]
        
        mean_ia = np.mean(inter_arrivals)
        std_ia = np.std(inter_arrivals)
        cv = std_ia / mean_ia if mean_ia > 0 else float('inf')
        
        confidence = max(0, 1 - cv)  # Higher confidence for lower CV
        
        if confidence < self.confidence_threshold:
            return False
        
        # Calculate next access time
        last_access = file_accesses[-1]
        next_access = last_access + period
        
        # If next access is within lookahead
        if next_access - current_time <= self.lookahead_seconds:
            # Benefit: avoiding future tape recall
            benefit = base_latency - prefetch_cost
            return benefit > 0
        
        return False
