"""
IOR Validation Service
=====================
Compares IOR benchmark results with simulation results
using statistical significance testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from scipy import stats


class IORValidationService:
    """
    Service for validating simulation results against IOR benchmarks.
    
    Uses statistical significance testing (p-value) to determine
    if simulation results match IOR benchmarks.
    """
    
    def __init__(self, ior_summary_file: str = "./traces/ior_summary.csv"):
        self.ior_summary_file = Path(ior_summary_file)
        self.ior_data = None
        
        if self.ior_summary_file.exists():
            self.ior_data = pd.read_csv(self.ior_summary_file)
    
    def load_simulation_results(self, sweep_db, sweep_id: int = None) -> pd.DataFrame:
        """Load simulation results from database."""
        if sweep_id:
            results = sweep_db.get_sweep_results(sweep_id)
        else:
            results = sweep_db.get_all_results()
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def compare_results(self, ior_df: pd.DataFrame, sim_df: pd.DataFrame, 
                       alpha: float = 0.05) -> pd.DataFrame:
        """
        Compare IOR benchmarks with simulation results.
        
        Args:
            ior_df: DataFrame with IOR results
            sim_df: DataFrame with simulation results
            alpha: Significance level (default 0.05)
            
        Returns:
            DataFrame with comparison results
        """
        comparisons = []
        
        # Group IOR data by config
        ior_configs = ior_df['config'].unique() if 'config' in ior_df.columns else []
        
        for config in ior_configs:
            ior_config_data = ior_df[ior_df['config'] == config]
            
            # Get corresponding simulation data
            sim_config_data = sim_df[sim_df['ior_config_name'] == config] if 'ior_config_name' in sim_df.columns else pd.DataFrame()
            
            for _, ior_row in ior_config_data.iterrows():
                iteration = ior_row.get('iteration', 0)
                operation = ior_row.get('operation', 'unknown')
                
                # Find matching simulation result
                sim_match = sim_config_data[
                    (sim_config_data['ior_iteration'] == iteration)
                ] if not sim_config_data.empty else pd.DataFrame()
                
                if sim_match.empty:
                    continue
                
                sim_row = sim_match.iloc[0]
                
                # Compare metrics based on operation type
                if operation == 'write':
                    # Compare write metrics
                    ior_bw = ior_row.get('bw_mib_s', 0)
                    sim_bw = self._calculate_simulated_bw(
                        sim_row.get('total_bytes_written', 0),
                        sim_row.get('write_duration_s', 0)
                    ) / (1024**2)  # Convert to MiB/s
                    
                    comp = self._compare_metric(
                        config=config,
                        iteration=iteration,
                        operation=operation,
                        metric='bandwidth',
                        ior_value=ior_bw,
                        sim_value=sim_bw,
                        alpha=alpha
                    )
                    comparisons.append(comp)
                    
                elif operation == 'read':
                    # Compare read metrics
                    ior_bw = ior_row.get('bw_mib_s', 0)
                    sim_bw = self._calculate_simulated_bw(
                        sim_row.get('total_bytes_read', 0),
                        sim_row.get('read_duration_s', 0)
                    ) / (1024**2)  # Convert to MiB/s
                    
                    comp = self._compare_metric(
                        config=config,
                        iteration=iteration,
                        operation=operation,
                        metric='bandwidth',
                        ior_value=ior_bw,
                        sim_value=sim_bw,
                        alpha=alpha
                    )
                    comparisons.append(comp)
        
        return pd.DataFrame(comparisons)
    
    def _calculate_simulated_bw(self, bytes_count: int, duration_s: float) -> float:
        """Calculate bandwidth from simulation metrics."""
        if duration_s <= 0:
            return 0.0
        return bytes_count / duration_s
    
    def _compare_metric(self, config: str, iteration: int, operation: str,
                      metric: str, ior_value: float, sim_value: float,
                      alpha: float) -> Dict[str, Any]:
        """Compare a single metric using statistical test."""
        # Calculate difference
        diff_pct = 0.0
        if ior_value > 0:
            diff_pct = ((sim_value - ior_value) / ior_value) * 100
        
        # For now, use simple t-test approximation
        # In a full implementation, we'd have multiple samples
        # Here we treat single values but calculate significance
        
        # Use coefficient of variation to estimate variance
        # Assume 10% CV for IOR measurements
        cv = 0.10
        std_ior = ior_value * cv if ior_value > 0 else 1
        
        # Calculate t-statistic
        if std_ior > 0:
            t_stat = (sim_value - ior_value) / (std_ior / np.sqrt(1))
            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=1))
        else:
            p_value = 1.0
        
        # Determine if statistically significant
        is_significant = p_value < alpha
        
        return {
            'config': config,
            'iteration': iteration,
            'operation': operation,
            'metric': metric,
            'ior_value': ior_value,
            'sim_value': sim_value,
            'difference_pct': diff_pct,
            'p_value': p_value,
            'significant': is_significant,
            'alpha': alpha,
            'passed': not is_significant or abs(diff_pct) < 20  # Pass if not significant OR within 20%
        }
    
    def generate_summary(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from comparison results."""
        if comparison_df.empty:
            return {}
        
        total = len(comparison_df)
        passed = comparison_df['passed'].sum()
        failed = total - passed
        significant = comparison_df['significant'].sum()
        
        return {
            'total_comparisons': total,
            'passed': passed,
            'failed': failed,
            'significant_difference': significant,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'avg_difference_pct': comparison_df['difference_pct'].mean(),
            'config_summary': comparison_df.groupby('config').agg({
                'passed': 'sum',
                'difference_pct': 'mean'
            }).to_dict('index')
        }
    
    def export_comparison_csv(self, comparison_df: pd.DataFrame, output_file: str) -> str:
        """Export comparison results to CSV."""
        comparison_df.to_csv(output_file, index=False)
        return output_file


def validate_ior_results(sweep_db, ior_summary_file: str = "./traces/ior_summary.csv",
                       sweep_id: int = None, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Main function to validate IOR results.
    
    Args:
        sweep_db: Database connection
        ior_summary_file: Path to IOR summary CSV
        sweep_id: Optional sweep ID to filter results
        alpha: Significance level
        
    Returns:
        Dict with comparison results and summary
    """
    service = IORValidationService(ior_summary_file)
    
    # Load data
    ior_df = service.ior_data
    if ior_df is None or ior_df.empty:
        return {'error': 'No IOR data found'}
    
    sim_df = service.load_simulation_results(sweep_db, sweep_id)
    if sim_df.empty:
        return {'error': 'No simulation results found'}
    
    # Compare
    comparison_df = service.compare_results(ior_df, sim_df, alpha)
    
    if comparison_df.empty:
        return {'error': 'No matching configurations found between IOR and simulation'}
    
    # Generate summary
    summary = service.generate_summary(comparison_df)
    
    return {
        'comparison': comparison_df.to_dict('records'),
        'summary': summary
    }
