"""
Validation Validators
===================
Statistical validators for simulation validation.

Provides:
- KS-test for latency distribution comparison
- Throughput validation
- Cache hit ratio validation
- Monotonic relationship validation for parameter sweeps
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass

from validation.multiple_testing import holm_bonferroni, summarise_holm


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    statistic: float
    p_value: float
    significance_level: float = 0.05
    message: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def __str__(self):
        status = "🟢 PASSED" if self.passed else "🔴 FAILED"
        return f"{self.test_name}: {status} (stat={self.statistic:.4f}, p={self.p_value:.4f})"


class LatencyValidator:
    """
    Validate latency distributions using Kolmogorov-Smirnov test.
    
    H0: Simulated and real latencies come from the same distribution
    H1: They come from different distributions
    
    Reject H0 if p-value < significance_level
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def validate(
        self,
        simulated_latencies: List[float],
        real_latencies: List[float],
    ) -> ValidationResult:
        """
        Compare simulated vs real latency distributions.
        
        Args:
            simulated_latencies: List of simulated latency values
            real_latencies: List of real observed latency values
            
        Returns:
            ValidationResult with KS-test results
        """
        if not simulated_latencies or not real_latencies:
            return ValidationResult(
                test_name="Latency KS-Test",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                message="Insufficient data for validation"
            )
        
        # Convert to numpy arrays
        sim = np.array(simulated_latencies)
        real = np.array(real_latencies)
        
        # Remove zeros and negatives
        sim = sim[sim > 0]
        real = real[real > 0]
        
        if len(sim) < 2 or len(real) < 2:
            return ValidationResult(
                test_name="Latency KS-Test",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                message="Insufficient valid data points"
            )
        
        # Perform KS test
        statistic, p_value = stats.ks_2samp(sim, real)
        
        # Pass if p-value > significance_level (cannot reject H0)
        passed = p_value > self.significance_level
        
        message = (
            f"Latency distributions match" if passed 
            else f"Latency distributions differ significantly"
        )
        
        return ValidationResult(
            test_name="Latency KS-Test",
            passed=passed,
            statistic=float(statistic),
            p_value=float(p_value),
            significance_level=self.significance_level,
            message=message,
            details={
                'sim_mean': float(np.mean(sim)),
                'real_mean': float(np.mean(real)),
                'sim_median': float(np.median(sim)),
                'real_median': float(np.median(real)),
                'sim_std': float(np.std(sim)),
                'real_std': float(np.std(real)),
                'n_sim': len(sim),
                'n_real': len(real),
            }
        )


class MultiMarginalKSValidator:
    """
    Run a family of two-sample KS tests across multiple marginal distributions
    (e.g., recall latency, migrate latency, file size, IAT) and apply the
    Holm (1979) step-down Bonferroni correction to control the family-wise
    error rate.

    Per-marginal significance thresholds grow stricter for the smaller
    (more significant) p-values, so the family-wise α is preserved while
    retaining more power than the classical Bonferroni correction.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def validate(
        self,
        marginals: Dict[str, Tuple[List[float], List[float]]],
        min_samples: int = 2,
    ) -> Dict[str, Any]:
        """
        Compare simulated vs observed distributions across several marginals.

        Args:
            marginals:    Dict ``{name: (simulated, observed)}``.
            min_samples:  Minimum samples per side to attempt a KS test;
                          under this, the marginal is flagged as SKIP and
                          excluded from the Holm family.

        Returns:
            Dict with keys:
                per_marginal:    {name: {KS_statistic, p_value, n_sim, n_obs, note?}}
                holm:            output of summarise_holm over the tested family
                family_passed:   True iff no null rejected after Holm correction
        """
        per_marginal: Dict[str, Dict[str, Any]] = {}
        named_p: List[Tuple[str, float]] = []
        for name, (sim, obs) in marginals.items():
            sim_arr = np.asarray([s for s in sim if s is not None and s > 0], dtype=float)
            obs_arr = np.asarray([o for o in obs if o is not None and o > 0], dtype=float)
            if len(sim_arr) < min_samples or len(obs_arr) < min_samples:
                per_marginal[name] = {
                    "status": "SKIP",
                    "note": f"Need ≥{min_samples} samples per side",
                    "n_sim": int(len(sim_arr)),
                    "n_obs": int(len(obs_arr)),
                }
                continue
            stat, p = stats.ks_2samp(sim_arr, obs_arr)
            per_marginal[name] = {
                "KS_statistic": float(stat),
                "p_value": float(p),
                "n_sim": int(len(sim_arr)),
                "n_obs": int(len(obs_arr)),
            }
            named_p.append((name, float(p)))

        holm_summary = summarise_holm(named_p, alpha=self.alpha) if named_p else {
            "procedure": "Holm-Bonferroni (step-down)",
            "alpha": self.alpha,
            "n_tests": 0,
            "family_rejected": False,
            "family_passed": True,
            "per_test": [],
            "note": "No marginals had sufficient data for KS testing",
        }

        # Attach adjusted-p and rejection flags back onto each per_marginal entry.
        for entry in holm_summary.get("per_test", []):
            name = entry["name"]
            if name in per_marginal:
                per_marginal[name]["adjusted_p"] = entry["adjusted_p"]
                per_marginal[name]["rejected_after_holm"] = entry["rejected"]
                per_marginal[name]["rank"] = entry["rank"]

        return {
            "per_marginal": per_marginal,
            "holm": holm_summary,
            "family_passed": bool(holm_summary.get("family_passed", True)),
        }


class ThroughputValidator:
    """
    Validate throughput/bandwidth against real system.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def validate(
        self,
        simulated_throughput_gbs: float,
        real_throughput_gbs: float,
        tolerance_pct: float = 20.0,
    ) -> ValidationResult:
        """
        Compare simulated vs real throughput.
        
        Args:
            simulated_throughput_gbs: Simulated throughput in GB/s
            real_throughput_gbs: Real observed throughput in GB/s
            tolerance_pct: Acceptable difference percentage
            
        Returns:
            ValidationResult
        """
        if real_throughput_gbs <= 0:
            return ValidationResult(
                test_name="Throughput Comparison",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                message="No real throughput data available"
            )
        
        # Calculate percentage difference
        diff_pct = abs(simulated_throughput_gbs - real_throughput_gbs) / real_throughput_gbs * 100
        
        passed = diff_pct <= tolerance_pct
        
        return ValidationResult(
            test_name="Throughput Comparison",
            passed=passed,
            statistic=diff_pct,
            p_value=1.0 - (diff_pct / 100),  # Approximate p-value
            message=f"Difference: {diff_pct:.1f}% (within {tolerance_pct}%)" if passed else f"Difference: {diff_pct:.1f}% (exceeds {tolerance_pct}%)",
            details={
                'simulated_gbs': simulated_throughput_gbs,
                'real_gbs': real_throughput_gbs,
                'difference_pct': diff_pct,
                'tolerance_pct': tolerance_pct,
            }
        )


class CacheHitValidator:
    """
    Validate cache hit ratios.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def validate(
        self,
        simulated_ratio: float,
        real_ratio: float,
        tolerance_pct: float = 20.0,
    ) -> ValidationResult:
        """
        Compare simulated vs real cache hit ratios.
        
        Args:
            simulated_ratio: Simulated cache hit ratio (0-1)
            real_ratio: Real observed cache hit ratio (0-1)
            tolerance_pct: Acceptable difference percentage
            
        Returns:
            ValidationResult
        """
        if real_ratio <= 0:
            return ValidationResult(
                test_name="Cache Hit Ratio",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                message="No real cache hit ratio available"
            )
        
        diff_pct = abs(simulated_ratio - real_ratio) / real_ratio * 100
        passed = diff_pct <= tolerance_pct
        
        return ValidationResult(
            test_name="Cache Hit Ratio",
            passed=passed,
            statistic=diff_pct,
            p_value=1.0 - (diff_pct / 100),
            message=f"Difference: {diff_pct:.1f}% (within {tolerance_pct}%)" if passed else f"Difference: {diff_pct:.1f}% (exceeds {tolerance_pct}%)",
            details={
                'simulated_ratio': simulated_ratio,
                'real_ratio': real_ratio,
                'difference_pct': diff_pct,
            }
        )


class MonotonicRelationshipValidator:
    """
    Validate monotonic relationships in parameter sweeps.
    
    Expected relationships:
    - age_threshold ↑ → archived_data ↓ (less aggressive = less archived)
    - age_threshold ↓ → bandwidth ↑ (more aggressive = more bandwidth)
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def validate_age_archived_relationship(
        self,
        age_thresholds: List[int],
        archived_data_tb: List[float],
    ) -> ValidationResult:
        """
        Validate: Higher age thresholds should result in LESS archived data.
        
        Args:
            age_thresholds: List of age threshold values (minutes)
            archived_data_tb: Corresponding archived data (TB)
            
        Returns:
            ValidationResult
        """
        if len(age_thresholds) < 2:
            return ValidationResult(
                test_name="Age→Archived Relationship",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                message="Need at least 2 data points"
            )
        
        # Calculate correlation (should be negative)
        correlation, p_value = stats.spearmanr(age_thresholds, archived_data_tb)
        
        # Pass if correlation is significantly negative
        passed = correlation < 0 and p_value < self.significance_level
        
        expected = "Negative correlation (higher age → less archived)"
        actual = f"Correlation: {correlation:.3f}"
        
        return ValidationResult(
            test_name="Age→Archived Relationship",
            passed=passed,
            statistic=abs(correlation),
            p_value=float(p_value),
            message=f"{expected}. Actual: {actual}",
            details={
                'correlation': float(correlation),
                'p_value': float(p_value),
                'expected': 'negative',
                'n_points': len(age_thresholds),
            }
        )
    
    def validate_age_bandwidth_relationship(
        self,
        age_thresholds: List[int],
        bandwidth_gbs: List[float],
    ) -> ValidationResult:
        """
        Validate: Higher age thresholds should result in LESS bandwidth.
        
        Args:
            age_thresholds: List of age threshold values (minutes)
            bandwidth_gbs: Corresponding bandwidth (GB/s)
            
        Returns:
            ValidationResult
        """
        if len(age_thresholds) < 2:
            return ValidationResult(
                test_name="Age→Bandwidth Relationship",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                message="Need at least 2 data points"
            )
        
        # Calculate correlation (should be negative)
        correlation, p_value = stats.spearmanr(age_thresholds, bandwidth_gbs)
        
        # Pass if correlation is significantly negative
        passed = correlation < 0 and p_value < self.significance_level
        
        expected = "Negative correlation (higher age → less bandwidth)"
        actual = f"Correlation: {correlation:.3f}"
        
        return ValidationResult(
            test_name="Age→Bandwidth Relationship",
            passed=passed,
            statistic=abs(correlation),
            p_value=float(p_value),
            message=f"{expected}. Actual: {actual}",
            details={
                'correlation': float(correlation),
                'p_value': float(p_value),
                'expected': 'negative',
                'n_points': len(age_thresholds),
            }
        )


def run_all_validations(
    simulated_results: Dict[str, Any],
    real_data: Dict[str, Any],
    significance_level: float = 0.05,
) -> List[ValidationResult]:
    """
    Run all validation tests. Any KS-based p-values are collected into a
    family and the Holm (1979) step-down procedure is applied to control
    the family-wise error rate; the adjusted outcome is stored back on each
    affected ValidationResult as ``details['holm_adjusted_p']`` and
    ``details['holm_rejected']``.

    Args:
        simulated_results: Dict with simulated metrics
        real_data: Dict with real system metrics
        significance_level: p-value threshold

    Returns:
        List of ValidationResult
    """
    results: List[ValidationResult] = []
    ks_family: List[Tuple[str, float, ValidationResult]] = []

    # Latency validation (KS-based)
    lat_validator = LatencyValidator(significance_level)
    if 'latencies' in simulated_results and 'latencies' in real_data:
        r = lat_validator.validate(
            simulated_results['latencies'],
            real_data['latencies']
        )
        results.append(r)
        if r.p_value is not None:
            ks_family.append((r.test_name, float(r.p_value), r))

    # Multi-marginal KS if the caller supplied per-marginal sim/obs pairs
    if ('marginals' in simulated_results and 'marginals' in real_data):
        sim_marg = simulated_results['marginals']
        obs_marg = real_data['marginals']
        paired = {
            name: (sim_marg[name], obs_marg[name])
            for name in sim_marg if name in obs_marg
        }
        if paired:
            mmks = MultiMarginalKSValidator(alpha=significance_level)
            family = mmks.validate(paired)
            # Materialise each per-marginal KS as a ValidationResult
            for name, entry in family["per_marginal"].items():
                if "KS_statistic" not in entry:
                    continue
                r = ValidationResult(
                    test_name=f"KS Test ({name})",
                    passed=(not entry.get("rejected_after_holm", True)),
                    statistic=entry["KS_statistic"],
                    p_value=entry["p_value"],
                    significance_level=significance_level,
                    message=(
                        f"{name}: KS={entry['KS_statistic']:.4f}, "
                        f"p={entry['p_value']:.4f}, "
                        f"adj_p={entry.get('adjusted_p', entry['p_value']):.4f}"
                    ),
                    details={
                        "holm_adjusted_p": entry.get("adjusted_p"),
                        "holm_rejected": entry.get("rejected_after_holm"),
                        "n_sim": entry["n_sim"],
                        "n_obs": entry["n_obs"],
                    },
                )
                results.append(r)

    # Throughput validation (tolerance-based; not part of KS family)
    bw_validator = ThroughputValidator(significance_level)
    if 'throughput_gbs' in simulated_results and 'throughput_gbs' in real_data:
        results.append(bw_validator.validate(
            simulated_results['throughput_gbs'],
            real_data['throughput_gbs']
        ))

    # Cache hit validation (tolerance-based; not part of KS family)
    ch_validator = CacheHitValidator(significance_level)
    if 'cache_hit_ratio' in simulated_results and 'cache_hit_ratio' in real_data:
        results.append(ch_validator.validate(
            simulated_results['cache_hit_ratio'],
            real_data['cache_hit_ratio']
        ))

    # Apply Holm-Bonferroni to any standalone KS p-values collected above
    # (the multi-marginal path already applied Holm within its own family).
    if len(ks_family) >= 2:
        holm_summary = summarise_holm(
            [(n, p) for (n, p, _r) in ks_family],
            alpha=significance_level,
        )
        by_name = {e["name"]: e for e in holm_summary["per_test"]}
        for (name, _p, r) in ks_family:
            entry = by_name.get(name)
            if entry is None:
                continue
            r.details = dict(r.details or {})
            r.details["holm_adjusted_p"] = entry["adjusted_p"]
            r.details["holm_rejected"] = entry["rejected"]
            r.passed = not entry["rejected"]

    return results
