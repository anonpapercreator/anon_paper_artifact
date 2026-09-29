"""
Validation Module
=================
KS-test validation against your DMF 7 logs.

Statistical Methods:
- Kolmogorov-Smirnov test
- Confidence intervals
- Little's Law verification
- Warm-up period detection

Author: Simulation Team
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from dataclasses import dataclass, field

from validation.multiple_testing import summarise_holm


@dataclass
class ValidationResult:
    """Result of validation test."""
    test_name: str
    passed: bool
    statistic: float
    p_value: float
    significance_level: float = 0.05
    message: str = ""
    
    def __str__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"{self.test_name}: {status} (stat={self.statistic:.4f}, p={self.p_value:.4f})"


class KSValidator:
    """
    Kolmogorov-Smirnov test validator.
    
    Compares simulated latency distribution against observed DMF latencies.
    
    Mathematical relationship:
    D = max|F_sim(x) - F_obs(x)|
    
    H₀: F_sim = F_obs (distributions are identical)
    Reject H₀ if D > D_critical at α significance
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_latency_distributions(
        self,
        simulated_latencies: List[float],
        observed_latencies: List[float],
    ) -> ValidationResult:
        """
        Compare simulated vs observed latency distributions using KS test.
        
        Returns ValidationResult with test outcome.
        """
        if len(simulated_latencies) < 10 or len(observed_latencies) < 10:
            return ValidationResult(
                test_name="KS Test",
                passed=False,
                statistic=0.0,
                p_value=1.0,
                message="Insufficient data points for KS test"
            )
        
        # Perform KS test
        statistic, p_value = stats.ks_2samp(
            simulated_latencies,
            observed_latencies
        )
        
        passed = p_value > self.significance_level
        
        return ValidationResult(
            test_name="KS Test (Latency Distribution)",
            passed=passed,
            statistic=statistic,
            p_value=p_value,
            significance_level=self.significance_level,
            message=f"Simulated vs Observed latency comparison"
        )
    
    def compare_statistics(
        self,
        simulated: List[float],
        observed: List[float],
        metric_name: str = "mean"
    ) -> ValidationResult:
        """Compare summary statistics (mean, median, etc.)."""
        if not simulated or not observed:
            return ValidationResult(
                test_name=f"{metric_name} Comparison",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                message="No data"
            )
        
        sim_mean = np.mean(simulated)
        obs_mean = np.mean(observed)
        
        # Relative difference
        if obs_mean > 0:
            rel_diff = abs(sim_mean - obs_mean) / obs_mean
        else:
            rel_diff = abs(sim_mean - obs_mean)
        
        # Use 10% tolerance
        passed = rel_diff < 0.10
        
        return ValidationResult(
            test_name=f"{metric_name} Comparison",
            passed=passed,
            statistic=rel_diff,
            p_value=1.0 - rel_diff,
            message=f"Relative difference: {rel_diff:.2%}"
        )


class LittleLawValidator:
    """
    Validate Little's Law: L = λ × W
    
    Mathematical relationship:
    L = λ × W
    
    Where:
    - L = average number in system
    - λ = arrival rate
    - W = average time in system
    
    Validates simulation correctness.
    """
    
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance  # 5% tolerance
    
    def validate(
        self,
        arrival_rate: float,
        avg_time_in_system: float,
        avg_number_in_system: float,
    ) -> ValidationResult:
        """
        Validate Little's Law holds.
        
        L_observed = λ × W
        """
        expected_l = arrival_rate * avg_time_in_system
        
        if expected_l > 0:
            rel_error = abs(avg_number_in_system - expected_l) / expected_l
        else:
            rel_error = abs(avg_number_in_system - expected_l)
        
        passed = rel_error < self.tolerance
        
        return ValidationResult(
            test_name="Little's Law",
            passed=passed,
            statistic=rel_error,
            p_value=1.0 - rel_error,
            message=f"L={avg_number_in_system:.2f}, λ×W={expected_l:.2f}, error={rel_error:.2%}"
        )


class WarmUpDetector:
    """
    Detect warm-up period using MSER-5 method.
    
    Modified Standardized Error Series:
    MSER(d) = (1/(n-d)) × Σ(Y_i - Ȳ_d)² / Ȳ_n²
    
    Finds minimum d where variance stabilizes.
    
    Reference: White & G. (1997) "Simulation Output Analysis"
    """
    
    def __init__(self):
        pass
    
    def detect_warmup(
        self,
        observations: List[float],
        batch_size: int = 100,
    ) -> Tuple[int, float]:
        """
        Detect warm-up period.
        
        Returns:
            warmup_index: Index where warmup ends
            mser_value: MSER value at detected point
        """
        n = len(observations)
        
        if n < batch_size * 2:
            return 0, float('inf')
        
        # Calculate batch means
        n_batches = n // batch_size
        batch_means = []
        
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch_means.append(np.mean(observations[start:end]))
        
        if len(batch_means) < 2:
            return 0, float('inf')
        
        # Overall mean
        overall_mean = np.mean(observations)
        
        # Calculate MSER for different d values
        mser_values = []
        
        for d in range(1, len(batch_means) // 2):
            # First d batches (potential warmup)
            warmup_mean = np.mean(batch_means[:d]) if d > 0 else 0
            
            # Variance in warmup period
            if d > 1:
                warmup_var = np.var(batch_means[:d])
            else:
                warmup_var = 0
            
            # MSER formula
            if overall_mean > 0 and d < len(batch_means):
                mser = warmup_var / (overall_mean ** 2)
            else:
                mser = float('inf')
            
            mser_values.append((d, mser))
        
        # Find d with minimum MSER
        if mser_values:
            best_d, best_mser = min(mser_values, key=lambda x: x[1])
            return best_d * batch_size, best_mser
        
        return 0, float('inf')


class ConfidenceIntervalCalculator:
    """
    Calculate confidence intervals for simulation output.
    
    Mathematical relationship:
    
    95% CI: x̄ ± t_{n-1, 0.975} × s / √n
    
    Where:
    - x̄ = sample mean
    - s = sample standard deviation
    - n = number of observations
    - t = t-distribution critical value
    """
    
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
    
    def calculate_ci(
        self,
        observations: List[float],
    ) -> Dict[str, float]:
        """Calculate confidence interval for mean."""
        if len(observations) < 2:
            return {
                'mean': np.mean(observations) if observations else 0,
                'std': 0,
                'ci_lower': 0,
                'ci_upper': 0,
                'n': len(observations),
            }
        
        mean = np.mean(observations)
        std = np.std(observations, ddof=1)
        n = len(observations)
        
        # t critical value
        alpha = 1 - self.confidence
        t_crit = stats.t.ppf(1 - alpha/2, n - 1)
        
        # Margin of error
        me = t_crit * std / np.sqrt(n)
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': mean - me,
            'ci_upper': mean + me,
            'n': n,
            'relative_precision': me / mean if mean > 0 else 0,
        }
    
    def calculate_percentiles(
        self,
        observations: List[float],
        percentiles: List[float] = [0.5, 0.75, 0.95, 0.99]
    ) -> Dict[str, float]:
        """Calculate percentiles."""
        result = {}
        for p in percentiles:
            result[f'p{int(p*100)}'] = np.percentile(observations, p * 100)
        return result


def validate_simulation(
    simulated_latencies: List[float],
    observed_latencies: List[float],
    arrival_rate: float,
    avg_time_in_system: float,
    avg_queue_length: float,
    observations: List[float],
    extra_marginals: Optional[Dict[str, Tuple[List[float], List[float]]]] = None,
    family_alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Run complete validation suite.

    If ``extra_marginals`` is provided as ``{name: (simulated, observed)}``,
    one KS test is run per marginal in addition to the latency KS test, and
    the full family of p-values is corrected using the Holm (1979) step-down
    procedure at the family-wise significance level ``family_alpha``. The
    Holm family summary is returned under the ``holm`` key.

    Returns dict of validation results.
    """
    results: Dict[str, Any] = {}

    # Collect all KS p-values into one family for Holm correction.
    ks_family: List[Tuple[str, float]] = []

    # KS Test (primary latency comparison)
    ks_validator = KSValidator(significance_level=family_alpha)
    ks_latency = ks_validator.compare_latency_distributions(
        simulated_latencies,
        observed_latencies,
    )
    results['ks_test'] = ks_latency
    if len(simulated_latencies) >= 10 and len(observed_latencies) >= 10:
        ks_family.append(("ks_latency", float(ks_latency.p_value)))

    # Extra marginals — each contributes a KS test to the family.
    if extra_marginals:
        for name, (sim, obs) in extra_marginals.items():
            sim_arr = [s for s in sim if s is not None and s > 0]
            obs_arr = [o for o in obs if o is not None and o > 0]
            if len(sim_arr) < 10 or len(obs_arr) < 10:
                results[f"ks_{name}"] = ValidationResult(
                    test_name=f"KS Test ({name})",
                    passed=False,
                    statistic=0.0,
                    p_value=1.0,
                    significance_level=family_alpha,
                    message="Insufficient data points for KS test",
                )
                continue
            stat, p = stats.ks_2samp(sim_arr, obs_arr)
            vr = ValidationResult(
                test_name=f"KS Test ({name})",
                passed=bool(p > family_alpha),
                statistic=float(stat),
                p_value=float(p),
                significance_level=family_alpha,
                message=f"{name}: KS={stat:.4f}, p={p:.4f}",
            )
            results[f"ks_{name}"] = vr
            ks_family.append((f"ks_{name}", float(p)))

    # Apply Holm step-down if at least one KS p-value was collected.
    if ks_family:
        holm = summarise_holm(ks_family, alpha=family_alpha)
        results['holm'] = holm
        # Write the Holm verdict back onto each KS ValidationResult so
        # downstream reports reflect the family-corrected decision.
        by_name = {e['name']: e for e in holm['per_test']}
        if 'ks_latency' in by_name:
            entry = by_name['ks_latency']
            ks_latency.passed = not entry['rejected']
            ks_latency.message = (
                f"{ks_latency.message} | Holm-adj p={entry['adjusted_p']:.4f}, "
                f"{'rejected' if entry['rejected'] else 'not rejected'}"
            )
        if extra_marginals:
            for name in extra_marginals:
                key = f"ks_{name}"
                if key in by_name and key in results:
                    entry = by_name[key]
                    r = results[key]
                    r.passed = not entry['rejected']
                    r.message = (
                        f"{r.message} | Holm-adj p={entry['adjusted_p']:.4f}, "
                        f"{'rejected' if entry['rejected'] else 'not rejected'}"
                    )

    # Mean comparison (not a KS p-value; excluded from Holm family)
    results['mean_comparison'] = ks_validator.compare_statistics(
        simulated_latencies,
        observed_latencies,
        'mean'
    )

    # Little's Law
    ll_validator = LittleLawValidator(tolerance=0.05)
    results['littles_law'] = ll_validator.validate(
        arrival_rate,
        avg_time_in_system,
        avg_queue_length
    )

    # Warm-up detection
    warmup_detector = WarmUpDetector()
    warmup_index, mser = warmup_detector.detect_warmup(observations)
    results['warmup_detection'] = ValidationResult(
        test_name="Warm-up Detection",
        passed=warmup_index > 0,
        statistic=warmup_index,
        p_value=mser,
        message=f"Warmup ends at index {warmup_index}"
    )

    # Confidence intervals
    ci_calculator = ConfidenceIntervalCalculator(confidence=0.95)
    results['ci'] = ci_calculator.calculate_ci(observations)
    results['percentiles'] = ci_calculator.calculate_percentiles(
        simulated_latencies
    )

    return results
