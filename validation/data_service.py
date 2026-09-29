"""
Validation Data Service
======================
Data access layer for validation.
Provides simulation results and real system metrics.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from db.database import SweepDB


class ValidationDataService:
    """Service for retrieving data for validation."""
    
    def __init__(self):
        self.sweep_db = SweepDB()
    
    def get_simulation_results(self, sweep_id: int = None) -> pd.DataFrame:
        """
        Get simulation results from database.
        
        Args:
            sweep_id: Optional sweep ID to filter results
            
        Returns:
            DataFrame with simulation results
        """
        if sweep_id:
            results = self.sweep_db.get_sweep_results(sweep_id)
        else:
            results = self.sweep_db.get_all_results()
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def get_sweeps(self) -> List[Dict[str, Any]]:
        """Get all sweeps."""
        return self.sweep_db.list_sweeps()
    
    def get_latencies_from_trace(self, trace_path: str) -> List[float]:
        """
        Get latencies from parsed trace file.
        
        Args:
            trace_path: Path to JSONL trace file
            
        Returns:
            List of latency values
        """
        latencies = []
        trace_path = Path(trace_path)
        
        if not trace_path.exists():
            return latencies
        
        with open(trace_path, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if 'latency_s' in event:
                        latencies.append(event['latency_s'])
                except json.JSONDecodeError:
                    continue
        
        return latencies
    
    def get_metrics_from_dmf_trace(self, trace_path: str) -> Dict[str, Any]:
        """
        Extract metrics from DMF trace.
        
        Args:
            trace_path: Path to DMF trace file
            
        Returns:
            Dict with aggregated metrics
        """
        trace_path = Path(trace_path)
        
        if not trace_path.exists():
            return {}
        
        events = []
        with open(trace_path, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        if not events:
            return {}
        
        # Calculate metrics
        recalls = [e for e in events if e.get('event_type') == 'RECALL']
        migrations = [e for e in events if e.get('event_type') == 'MIGRATE']
        
        recall_sizes = [e.get('file_size_bytes', 0) for e in recalls]
        migrate_sizes = [e.get('file_size_bytes', 0) for e in migrations]
        
        latencies = [e.get('latency_s', 0) for e in events if 'latency_s' in e]
        
        # Time range
        timestamps = [e.get('timestamp_s', 0) for e in events]
        duration_hours = (max(timestamps) - min(timestamps)) / 3600 if timestamps else 0
        
        return {
            'total_events': len(events),
            'recall_count': len(recalls),
            'migrate_count': len(migrations),
            'total_recall_bytes': sum(recall_sizes),
            'total_migrate_bytes': sum(migrate_sizes),
            'avg_recall_size_bytes': sum(recall_sizes) / len(recall_sizes) if recall_sizes else 0,
            'avg_migrate_size_bytes': sum(migrate_sizes) / len(migrate_sizes) if migrate_sizes else 0,
            'latency_mean': sum(latencies) / len(latencies) if latencies else 0,
            'latency_p50': sorted(latencies)[len(latencies)//2] if latencies else 0,
            'latency_p95': sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
            'duration_hours': duration_hours,
            'throughput_gbs': (sum(migrate_sizes) / 1024**3) / (duration_hours * 3600) if duration_hours > 0 else 0,
        }
    
    def get_metrics_from_p4_trace(self, trace_path: str) -> Dict[str, Any]:
        """
        Extract metrics from P4/MSP trace.
        
        Args:
            trace_path: Path to P4 trace file
            
        Returns:
            Dict with aggregated metrics
        """
        trace_path = Path(trace_path)
        
        if not trace_path.exists():
            return {}
        
        events = []
        with open(trace_path, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        if not events:
            return {}
        
        # Calculate metrics
        recalls = [e for e in events if e.get('event_type') == 'RECALL']
        migrations = [e for e in events if e.get('event_type') == 'MIGRATE']
        
        recall_sizes = [e.get('file_size_bytes', 0) for e in recalls]
        migrate_sizes = [e.get('file_size_bytes', 0) for e in migrations]
        
        # Time range
        timestamps = [e.get('timestamp_s', 0) for e in events]
        duration_hours = (max(timestamps) - min(timestamps)) / 3600 if timestamps else 0
        
        return {
            'total_events': len(events),
            'recall_count': len(recalls),
            'migrate_count': len(migrations),
            'total_recall_bytes': sum(recall_sizes),
            'total_migrate_bytes': sum(migrate_sizes),
            'avg_recall_size_bytes': sum(recall_sizes) / len(recall_sizes) if recall_sizes else 0,
            'avg_migrate_size_bytes': sum(migrate_sizes) / len(migrate_sizes) if migrate_sizes else 0,
            'duration_hours': duration_hours,
            'throughput_gbs': (sum(migrate_sizes) / 1024**3) / (duration_hours * 3600) if duration_hours > 0 else 0,
        }
    
    def get_metrics_from_ior_summary(self, summary_csv_path: str = "./traces/ior_summary.csv") -> Dict[str, Any]:
        """
        Extract metrics from IOR summary CSV.
        
        Args:
            summary_csv_path: Path to IOR summary CSV file
            
        Returns:
            Dict with aggregated IOR metrics suitable for comparison
        """
        summary_path = Path(summary_csv_path)
        
        if not summary_path.exists():
            return {}
        
        df = pd.read_csv(summary_path)
        
        if df.empty:
            return {}
        
        writes = df[df['operation'] == 'write']
        reads = df[df['operation'] == 'read']
        
        return {
            'total_events': len(df),
            'write_count': len(writes),
            'read_count': len(reads),
            'write_bw_mean_mibs': writes['bw_mib_s'].mean() if len(writes) > 0 else 0,
            'read_bw_mean_mibs': reads['bw_mib_s'].mean() if len(reads) > 0 else 0,
            'write_iops_mean': writes['iops'].mean() if len(writes) > 0 else 0,
            'read_iops_mean': reads['iops'].mean() if len(reads) > 0 else 0,
            'write_latency_mean_s': writes['latency_s'].mean() if len(writes) > 0 else 0,
            'read_latency_mean_s': reads['latency_s'].mean() if len(reads) > 0 else 0,
            'total_bytes': df['total_bytes'].sum(),
        }
    
    def compare_parameter_sweep(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze parameter sweep results for validation.
        
        Args:
            results_df: DataFrame with simulation results
            
        Returns:
            Dict with analysis results
        """
        if results_df.empty:
            return {}
        
        analysis = {}
        
        # Group by age_threshold
        if 'age_threshold_minutes' in results_df.columns:
            age_groups = results_df.groupby('age_threshold_minutes').agg({
                'archived_data_tb': ['mean', 'std'],
                'migration_bandwidth_gbs': ['mean', 'std'],
                'cache_hit_ratio': ['mean', 'std'],
                'latency_p95': ['mean', 'std'],
            }).reset_index()
            
            analysis['age_threshold_trends'] = age_groups.to_dict('records')
            
            # Check monotonic relationships
            if len(age_groups) > 1:
                ages = sorted(age_groups['age_threshold_minutes'].tolist())
                archived = [float(age_groups[age_groups['age_threshold_minutes'] == a][('archived_data_tb', 'mean')].iloc[0]) for a in ages]
                bandwidth = [float(age_groups[age_groups['age_threshold_minutes'] == a][('migration_bandwidth_gbs', 'mean')].iloc[0]) for a in ages]
                
                # Check: higher age -> less archived (monotonic decreasing)
                archived_trend = "decreasing" if all(archived[i] >= archived[i+1] for i in range(len(archived)-1)) else "non-monotonic"
                
                # Check: higher age -> less bandwidth (monotonic decreasing)
                bandwidth_trend = "decreasing" if all(bandwidth[i] >= bandwidth[i+1] for i in range(len(bandwidth)-1)) else "non-monotonic"
                
                analysis['archived_data_trend'] = archived_trend
                analysis['bandwidth_trend'] = bandwidth_trend
                analysis['age_thresholds_tested'] = ages
        
        # Group by archiver_interval_minutes
        if 'archiver_interval_minutes' in results_df.columns:
            arch_groups = results_df.groupby('archiver_interval_minutes').agg({
                'archived_data_tb': ['mean', 'std'],
                'migration_bandwidth_gbs': ['mean', 'std'],
                'tape_cartridges_used': ['mean', 'std'],
                'scratch_bandwidth_degradation_pct': ['mean', 'std'],
                'cache_hit_ratio': ['mean', 'std'],
                'latency_p95': ['mean', 'std'],
            }).reset_index()
            
            analysis['archiver_interval_trends'] = arch_groups.to_dict('records')
            
            # Check monotonic relationships
            if len(arch_groups) > 1:
                intervals = sorted(arch_groups['archiver_interval_minutes'].tolist())
                archived = [float(arch_groups[arch_groups['archiver_interval_minutes'] == i][('archived_data_tb', 'mean')].iloc[0]) for i in intervals]
                bandwidth = [float(arch_groups[arch_groups['archiver_interval_minutes'] == i][('migration_bandwidth_gbs', 'mean')].iloc[0]) for i in intervals]
                cartridges = [float(arch_groups[arch_groups['archiver_interval_minutes'] == i][('tape_cartridges_used', 'mean')].iloc[0]) for i in intervals]
                
                # Note: intervals are sorted ascending, so 1min < 120min
                # Expected: 1min = HIGH archived, 120min = LOW archived
                # So archived should DECREASE as interval increases
                archived_trend = "decreasing" if all(archived[i] >= archived[i+1] for i in range(len(archived)-1)) else "non-monotonic"
                
                # Check: more frequent (lower interval) -> more bandwidth (monotonic decreasing as interval increases)
                bandwidth_trend = "decreasing" if all(bandwidth[i] >= bandwidth[i+1] for i in range(len(bandwidth)-1)) else "non-monotonic"
                
                # Check: more frequent (lower interval) -> more cartridges used
                cartridges_trend = "decreasing" if all(cartridges[i] >= cartridges[i+1] for i in range(len(cartridges)-1)) else "non-monotonic"
                
                analysis['archiver_archived_trend'] = archived_trend
                analysis['archiver_bandwidth_trend'] = bandwidth_trend
                analysis['archiver_cartridges_trend'] = cartridges_trend
                analysis['archiver_intervals_tested'] = intervals
        
        return analysis


def get_available_traces(directory: str = "./traces") -> List[str]:
    """Get list of available trace files."""
    traces_dir = Path(directory)
    if not traces_dir.exists():
        return []
    
    return [str(f) for f in traces_dir.glob("*.jsonl")]
