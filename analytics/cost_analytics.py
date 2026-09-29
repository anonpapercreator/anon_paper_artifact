"""
Cost Analytics Module
=====================
Full TCO (Total Cost of Ownership) model for hierarchical storage.

Cost Components:
1. Capital Cost (CAPEX): NVMe + Tape infrastructure
2. Operational Cost (OPEX): Power, cooling, maintenance
3. Wear Cost: SSD endurance usage

Mathematical Model:

Total Cost:
    C_total = C_capex + C_opex + C_wear
    
Where:
    C_capex = (Cost_NVMe × Cache_TB + Cost_Tape × Tape_TB) / Amortization_years
    C_opex = (Power_NVMe × Hours + Power_Tape × Hours) × $/kWh
    C_wear = (TBW_written / TBW_rated) × Replacement_Cost

Per-Operation Cost:
    $/operation = C_total / total_operations

Author: Simulation Team
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CostConfig:
    """Cost model configuration."""
    # NVMe Cache costs
    nvme_cost_per_tb: float = 250.0           # $/TB
    nvme_tbw_rating: int = 600               # TB written endurance
    nvme_power_watts: float = 10.0             # Average (idle + active)
    nvme_replacement_cost: float = 8000.0     # Per NVMe failure
    nvme_amortization_years: int = 3
    
    # Tape costs
    tape_cost_per_cartridge: float = 150.0     # Per 20TB cartridge
    tape_drive_cost: float = 40000.0          # Per TS1160 drive
    tape_power_watts_drive: float = 30.0       # Per drive (active)
    tape_power_watts_idle: float = 8.0        # Per drive (idle)
    tape_robot_power_watts: float = 500.0     # Robot arm
    tape_library_maintenance_annual: float = 10000.0  # Annual maintenance
    
    # Operational costs
    power_cost_per_kw_month: float = 427.0     # Your actual cost: $/kW/month
    rack_space_cost_month: float = 200.0       # Per rack unit/month
    admin_cost_annual: float = 50000.0         # FTE for administration
    
    # Derived
    @property
    def power_cost_per_kw_year(self) -> float:
        return self.power_cost_per_kw_month * 12
    
    @property
    def power_cost_per_watt_year(self) -> float:
        return self.power_cost_per_kw_year / 1000  # $/W/year = $/kW_year / 1000


@dataclass
class CostMetrics:
    """Accumulated cost metrics."""
    # NVMe metrics
    nvme_bytes_written: int = 0
    nvme_bytes_read: int = 0
    nvme_operations_write: int = 0
    nvme_operations_read: int = 0
    
    # Tape metrics
    tape_mounts: int = 0
    tape_bytes_read: int = 0
    tape_bytes_written: int = 0
    tape_drive_active_time: float = 0.0
    tape_robot_active_time: float = 0.0
    
    # Simulation
    simulation_hours: float = 0.0
    total_operations: int = 0
    
    # Historical tracking
    cost_history: List[Dict] = field(default_factory=list)
    
    @property
    def nvme_tbw_written_tb(self) -> float:
        """Total TB written to NVMe."""
        return self.nvme_bytes_written / (1024**4)
    
    @property
    def nvme_wear_percentage(self) -> float:
        """Wear as percentage of rated TBW."""
        return 0.0  # Will be calculated with config


class CostAnalytics:
    """
    Cost Analytics for Hierarchical Storage.
    
    Mathematical Models:
    
    1. NVMe Wear Cost:
       C_wear_NVMe = (TBW_written / TBW_rated) × Cost_replacement
       
    2. Power Cost:
       C_power = Σ(Power_i × Hours_i) × $/kWh
       
    3. Per-Operation Cost:
       $/op = C_total / total_operations
       
    4. Cost per Recall (key metric for paper):
       $/recall = (C_NVMe + C_Tape) / total_recalls
    """
    
    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()
        self.metrics = CostMetrics()
        
        # Calculate annual costs
        self._calculate_annual_costs()
    
    def _calculate_annual_costs(self):
        """Pre-calculate annual costs."""
        # Capital costs (annualized)
        # Assuming 1 PiB NVMe cache
        cache_tb = 1024  # 1 PiB = 1024 TiB
        self.annual_nvme_capex = (self.config.nvme_cost_per_tb * cache_tb) / \
                                  self.config.nvme_amortization_years
        
        # Operational costs (power cost is already annual, just multiply by watts)
        # $5.124/W/year means: 1 watt costs $5.124 per year
        nvme_power_annual = self.config.nvme_power_watts * self.config.power_cost_per_watt_year
        self.annual_power = nvme_power_annual
        
        # Tape annual costs (estimated for 16 drives)
        # Average power = active% × active_watts + idle% × idle_watts
        tape_avg_power = (self.config.tape_power_watts_drive * 16 * 0.3 +  # active
                        self.config.tape_power_watts_idle * 16 * 0.7)  # idle
        self.annual_tape_power = tape_avg_power * self.config.power_cost_per_watt_year
        
        self.annual_tape_maintenance = self.config.tape_library_maintenance_annual
    
    def record_nvme_write(self, bytes_written: int):
        """Record NVMe write operation."""
        self.metrics.nvme_bytes_written += bytes_written
        self.metrics.nvme_operations_write += 1
        self.metrics.total_operations += 1
    
    def record_nvme_read(self, bytes_read: int):
        """Record NVMe read operation."""
        self.metrics.nvme_bytes_read += bytes_read
        self.metrics.nvme_operations_read += 1
        self.metrics.total_operations += 1
    
    def record_tape_mount(self, mount_time: float = 0.0):
        """Record tape mount."""
        self.metrics.tape_mounts += 1
    
    def record_tape_read(self, bytes_read: int):
        """Record tape read."""
        self.metrics.tape_bytes_read += bytes_read
    
    def record_tape_write(self, bytes_written: int):
        """Record tape write."""
        self.metrics.tape_bytes_written += bytes_written
    
    def set_simulation_time(self, hours: float):
        """Set simulation elapsed time."""
        self.metrics.simulation_hours = hours
    
    def calculate_nvme_wear_cost(self) -> float:
        """
        Calculate NVMe wear cost.
        
        Mathematical relationship:
        C_wear = (TBW_annual / TBW_rated) × Cost_replacement
        
        For the simulation period:
        C_wear_sim = (TBW_written / TBW_rated_annualized) × Cost_replacement
        """
        tbw_written_tb = self.metrics.nvme_tbw_written_tb
        
        # Scale to annual (if simulation is less than a year)
        if self.metrics.simulation_hours > 0 and self.metrics.simulation_hours < 8760:
            scale_factor = 8760 / self.metrics.simulation_hours
            tbw_written_tb *= scale_factor
        
        wear_ratio = tbw_written_tb / self.config.nvme_tbw_rating
        
        # Cost proportional to wear
        wear_cost = wear_ratio * self.config.nvme_replacement_cost
        
        return wear_cost
    
    def calculate_power_cost(self) -> float:
        """
        Calculate power cost.
        
        Mathematical relationship:
        C_power = (NVMe_power + Tape_power) × Hours × $/kWh
        """
        nvme_power_cost = (
            self.config.nvme_power_watts * 
            self.metrics.simulation_hours * 
            self.config.power_cost_per_watt_year / 8760
        )
        
        tape_power_cost = (
            self.annual_tape_power * 
            self.metrics.simulation_hours / 8760
        )
        
        return nvme_power_cost + tape_power_cost
    
    def calculate_annualized_costs(self) -> Dict[str, float]:
        """Calculate all annualized costs."""
        # Scale simulation costs to annual
        if self.metrics.simulation_hours > 0:
            scale = 8760 / self.metrics.simulation_hours
        else:
            scale = 1.0
        
        return {
            'nvme_capex': self.annual_nvme_capex,
            'nvme_wear': self.calculate_nvme_wear_cost() * scale,
            'power': self.calculate_power_cost() * scale,
            'tape_maintenance': self.annual_tape_maintenance,
            'total_annual': (
                self.annual_nvme_capex +
                self.calculate_nvme_wear_cost() * scale +
                self.calculate_power_cost() * scale +
                self.annual_tape_maintenance
            )
        }
    
    def calculate_per_operation_cost(self) -> Dict[str, float]:
        """Calculate cost per operation."""
        total_ops = self.metrics.total_operations
        if total_ops == 0:
            return {
                'cost_per_operation': 0.0,
                'cost_per_recall': 0.0,
                'cost_per_migration': 0.0,
            }
        
        total_cost = (
            self.calculate_nvme_wear_cost() +
            self.calculate_power_cost()
        )
        
        recalls = self.metrics.tape_mounts  # Approximate
        migrations = self.metrics.nvme_operations_write
        
        return {
            'cost_per_operation': total_cost / total_ops,
            'cost_per_recall': total_cost / max(recalls, 1),
            'cost_per_migration': total_cost / max(migrations, 1),
        }
    
    def calculate_cost_by_policy(
        self,
        cache_hit_ratio: float,
        tape_mounts: int,
        total_recalls: int
    ) -> Dict[str, float]:
        """
        Calculate cost impact of different policies.
        
        Mathematical relationship:
        
        Higher cache hit ratio:
        - More NVMe wear (more writes)
        - Fewer tape mounts (saves tape power)
        - Net effect depends on workload
        
        This allows policy comparison:
        - Maximum Performance: High cache priority → Higher NVMe cost, lower tape cost
        - Archive Optimized: Low cache priority → Lower NVMe cost, higher tape cost
        """
        # Estimate costs
        nvme_cost = self.annual_nvme_capex + self.calculate_nvme_wear_cost()
        tape_cost = self.annual_tape_power + self.annual_tape_maintenance
        
        # Adjust based on cache hit ratio
        # Higher hit ratio = more cache usage = more NVMe wear
        adjusted_nvme = nvme_cost * (0.5 + 0.5 * cache_hit_ratio)
        
        # Higher hit ratio = fewer tape mounts
        adjusted_tape = tape_cost * (1.0 - 0.5 * cache_hit_ratio)
        
        total = adjusted_nvme + adjusted_tape
        
        return {
            'adjusted_nvme_cost': adjusted_nvme,
            'adjusted_tape_cost': adjusted_tape,
            'total_adjusted_cost': total,
            'cost_per_recall': total / max(total_recalls, 1),
            'cost_per_tb_stored': total / 1024,  # Per TiB
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        annualized = self.calculate_annualized_costs()
        per_op = self.calculate_per_operation_cost()
        
        return {
            'annual_costs': annualized,
            'per_operation_costs': per_op,
            'metrics': {
                'simulation_hours': self.metrics.simulation_hours,
                'nvme_tbw_written_tb': self.metrics.nvme_tbw_written_tb,
                'nvme_wear_percentage': (
                    self.metrics.nvme_tbw_written_tb / self.config.nvme_tbw_rating * 100
                ),
                'tape_mounts': self.metrics.tape_mounts,
                'total_operations': self.metrics.total_operations,
            },
            'configuration': {
                'nvme_cost_per_tb': self.config.nvme_cost_per_tb,
                'power_cost_per_kw_month': self.config.power_cost_per_kw_month,
            }
        }
    
    def compare_policies(
        self,
        policy_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare costs across different policy configurations.
        
        Input: List of policy result dicts with:
        - policy_name
        - cache_hit_ratio
        - tape_mounts
        - total_recalls
        
        Output: Comparison table
        """
        comparison = []
        
        for result in policy_results:
            costs = self.calculate_cost_by_policy(
                cache_hit_ratio=result.get('cache_hit_ratio', 0.5),
                tape_mounts=result.get('tape_mounts', 0),
                total_recalls=result.get('total_recalls', 1)
            )
            
            comparison.append({
                'policy': result.get('policy_name', 'unknown'),
                'cache_hit_ratio': result.get('cache_hit_ratio', 0),
                'total_cost': costs['total_adjusted_cost'],
                'cost_per_recall': costs['cost_per_recall'],
                'cost_per_tb': costs['cost_per_tb_stored'],
                'tape_mounts': result.get('tape_mounts', 0),
            })
        
        return {
            'comparison': comparison,
            'recommendation': self._recommend_policy(comparison)
        }
    
    def _recommend_policy(self, comparison: List[Dict]) -> str:
        """Recommend policy based on cost-performance tradeoff."""
        if not comparison:
            return "No data"
        
        # Find policies at different performance levels
        max_perf = max(comparison, key=lambda x: x['cache_hit_ratio'])
        min_cost = min(comparison, key=lambda x: x['total_cost'])
        
        return (
            f"Maximum Performance: {max_perf['policy']} "
            f"(cache hit: {max_perf['cache_hit_ratio']:.1%}, "
            f"cost: ${max_perf['total_cost']:,.0f}/year)\n"
            f"Minimum Cost: {min_cost['policy']} "
            f"(cache hit: {min_cost['cache_hit_ratio']:.1%}, "
            f"cost: ${min_cost['total_cost']:,.0f}/year)"
        )
