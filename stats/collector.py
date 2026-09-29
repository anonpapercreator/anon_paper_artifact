"""
stats/collector.py — Statistics Collection, Warm-Up Detection, and CI Computation
==================================================================================
Implements:
    - Per-event sample collection
    - Welch's method for warm-up period detection
    - Little's Law validation at each queue
    - 95% Confidence Interval computation across replications
    - Time-averaged metrics (utilisation, queue depth)

Scientific basis:
    Law & Kelton (2000), Chapter 9: Output Analysis for a Single System.
    Welch, P.D. (1983). "The statistical analysis of simulation results."
    Little, J.D.C. (1961). "A proof for the queuing formula: L = λW."
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# Sample Series — tracks a time series of one metric
# ---------------------------------------------------------------------------

@dataclass
class SampleSeries:
    """Records a sequence of (time, value) samples for one metric."""
    name: str
    _times: List[float] = field(default_factory=list)
    _values: List[float] = field(default_factory=list)

    def record(self, time: float, value: float) -> None:
        self._times.append(time)
        self._values.append(value)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self._times), np.array(self._values)

    def __len__(self) -> int:
        return len(self._values)


# ---------------------------------------------------------------------------
# Welch's Method for Warm-Up Detection
# ---------------------------------------------------------------------------

def welch_warmup(values: np.ndarray, window_fraction: float = 0.1,
                 stability_threshold: float = 0.01) -> int:
    """
    Detect the warm-up period index using Welch's moving average method.

    Algorithm (Law & Kelton, 2000, §9.5.1):
        1. Compute moving average with window w = max(1, n * window_fraction)
        2. Find the first index i where the moving average has stabilised:
           |MA[i] - MA[i-1]| / |MA[i]| < stability_threshold

    Args:
        values:               1-D array of observations in time order.
        window_fraction:      Moving average window as fraction of series length.
        stability_threshold:  Relative change threshold for stability.

    Returns:
        Index of estimated warm-up end (discard observations before this index).
    """
    n = len(values)
    if n < 10:
        return 0

    w = max(1, int(n * window_fraction))
    ma = np.convolve(values, np.ones(2 * w + 1) / (2 * w + 1), mode='valid')
    # ma has length n - 2w; offset by w
    offset = w

    for i in range(1, len(ma)):
        if abs(ma[i]) > 1e-12:
            rel_change = abs(ma[i] - ma[i - 1]) / abs(ma[i])
            if rel_change < stability_threshold:
                return offset + i

    # If never stabilised, return 10% as conservative estimate
    return int(n * 0.10)


# ---------------------------------------------------------------------------
# Queue Statistics — tracks one queue's L, λ, W for Little's Law
# ---------------------------------------------------------------------------

class QueueStats:
    """
    Tracks statistics for a single queue resource (Little's Law validation).

    Little's Law: L = λ × W
        L = time-average number in system
        λ = long-run average arrival rate
        W = mean time spent in system (waiting + service)

    Warm-up handling:
        Events before `warmup_s` update the *live queue length* so L(t) is
        correct across the warmup boundary, but do NOT contribute to arrival
        counts, sojourn samples, or the area integral. The effective observation
        window is [warmup_s, sim_end_time].
    """

    def __init__(self, name: str, warmup_s: float = 0.0) -> None:
        self.name = name
        self._warmup_s = float(max(warmup_s, 0.0))
        self._arrival_times: List[float] = []
        self._sojourn_times: List[float] = []   # W samples (time in system)
        self._sojourn_records: List[Tuple[float, float]] = []  # (arrival_time, sojourn)
        self._queue_length_samples: List[Tuple[float, int]] = []  # (time, length)
        # Step-function record of L(t) within the observation window
        # [warmup_s, sim_end]: each entry is (time, length_immediately_after_event).
        # Used to compute batch-mean time-averages for the CI-based Little's Law test.
        self._length_events: List[Tuple[float, int]] = []
        self._n_arrivals: int = 0
        self._n_departures: int = 0
        self._current_length: int = 0
        self._last_sample_time: float = 0.0
        self._area_under_queue: float = 0.0   # ∫ L(t) dt over [warmup, now]
        self._sim_end_time: float = 0.0

    def _append_length_event(self, time: float, pre_event_length: int) -> None:
        """
        Append a (time, current_length) entry for reconstruction of the L(t)
        step function. If this is the first event in the observation window
        and the event fires strictly after warmup_s, backfill a baseline entry
        at (warmup_s, pre_event_length) so downstream integrators see the
        correct initial state.
        """
        if time < self._warmup_s:
            return
        if not self._length_events and time > self._warmup_s:
            self._length_events.append((self._warmup_s, pre_event_length))
        self._length_events.append((time, self._current_length))

    def arrival(self, time: float) -> None:
        """Record an arrival event."""
        self._update_area(time)
        pre = self._current_length
        self._current_length += 1
        if time >= self._warmup_s:
            self._n_arrivals += 1
            self._arrival_times.append(time)
            self._append_length_event(time, pre)

    def departure(self, time: float, arrival_time: float) -> None:
        """
        Record a departure event.

        Args:
            time:         Time of departure (simulation seconds).
            arrival_time: Time this item arrived (for sojourn calculation).
        """
        self._update_area(time)
        pre = self._current_length
        self._current_length = max(0, self._current_length - 1)
        # Keep the sojourn sample only when both arrival and departure fall
        # inside the observation window; otherwise the sojourn is censored.
        if arrival_time >= self._warmup_s and time >= self._warmup_s:
            sojourn = time - arrival_time
            if sojourn >= 0:
                self._sojourn_times.append(sojourn)
                self._sojourn_records.append((arrival_time, sojourn))
                self._n_departures += 1
        if time >= self._warmup_s:
            self._append_length_event(time, pre)

    def _update_area(self, current_time: float) -> None:
        """Accumulate area under L(t) only over [warmup_s, current_time]."""
        t0 = max(self._last_sample_time, self._warmup_s)
        t1 = max(current_time, self._warmup_s)
        if t1 > t0:
            self._area_under_queue += self._current_length * (t1 - t0)
        self._last_sample_time = current_time

    def snapshot(self, time: float) -> None:
        """Record a periodic snapshot of current queue length."""
        self._queue_length_samples.append((time, self._current_length))

    def finalise(self, sim_end_time: float) -> None:
        """Call at end of simulation to close the area integral."""
        self._update_area(sim_end_time)
        self._sim_end_time = sim_end_time

    @property
    def observation_window(self) -> float:
        """Effective observation duration after warm-up censoring."""
        return max(self._sim_end_time - self._warmup_s, 0.0)

    def mean_L(self) -> float:
        """Time-average queue length over [warmup_s, sim_end_time]."""
        T = self.observation_window
        if T <= 0:
            return float("nan")
        return self._area_under_queue / T

    def mean_W(self) -> float:
        """Mean sojourn time (waiting + service): W̄"""
        if not self._sojourn_times:
            return float("nan")
        return float(np.mean(self._sojourn_times))

    def lambda_hat(self) -> float:
        """Estimated arrival rate over the post-warmup window."""
        T = self.observation_window
        if T <= 0:
            return float("nan")
        return self._n_arrivals / T

    def _compute_batch_stats(self, n_batches: int,
                             min_samples_per_batch: int = 2) -> Optional[Dict[str, np.ndarray]]:
        """
        Partition the post-warmup observation window into ``n_batches`` equal-time
        batches and compute (L_b, λ_b, W_b) for each batch.

        L_b is the time-average of the step function L(t) reconstructed from
        ``_length_events``. λ_b is ``arrivals_in_batch / batch_duration``.
        W_b is the mean sojourn of items whose *arrival* time falls in the batch.

        Robustness fixes for short runs (Glynn & Whitt 2002 §4):
          - Batches with fewer than ``min_samples_per_batch`` arrival-side
            sojourn samples are *dropped* (not failure-inducing). The CI
            test then uses the surviving batches' between-batch variance.
          - Trailing batches whose mean sojourn exceeds the batch width are
            dropped: their items most likely depart after sim_end, so their
            sojourn is censored and W_b is biased low.

        Returns None only when fewer than ``min_batches=4`` batches survive
        these filters — at that point the batch-means CI estimator has too
        few degrees of freedom to be informative.
        """
        T = self.observation_window
        if T <= 0 or n_batches < 2:
            return None
        if not self._length_events or not self._sojourn_records:
            return None

        batch_dt = T / n_batches
        edges = self._warmup_s + batch_dt * np.arange(n_batches + 1)

        events = list(self._length_events)
        # Close the step function at the end of the observation window so the
        # final segment is integrated to sim_end rather than to the last event.
        if events[-1][0] < self._sim_end_time:
            events.append((self._sim_end_time, self._current_length))

        ev_times = np.asarray([e[0] for e in events], dtype=float)
        ev_lens = np.asarray([e[1] for e in events], dtype=float)

        # L_b: integrate step function L(t) over each batch
        L_b = np.zeros(n_batches)
        for b in range(n_batches):
            t_lo, t_hi = float(edges[b]), float(edges[b + 1])
            idx_lo = int(np.searchsorted(ev_times, t_lo, side="right") - 1)
            current_len = ev_lens[idx_lo] if idx_lo >= 0 else ev_lens[0]
            area = 0.0
            prev_t = t_lo
            idx = idx_lo + 1
            while idx < len(ev_times) and ev_times[idx] < t_hi:
                area += current_len * (ev_times[idx] - prev_t)
                prev_t = ev_times[idx]
                current_len = ev_lens[idx]
                idx += 1
            area += current_len * (t_hi - prev_t)
            L_b[b] = area / batch_dt

        # λ_b: arrivals per batch / batch duration
        arrivals = np.asarray(self._arrival_times, dtype=float)
        lambda_b = np.zeros(n_batches)
        for b in range(n_batches):
            n_arr = int(np.sum((arrivals >= edges[b]) & (arrivals < edges[b + 1])))
            lambda_b[b] = n_arr / batch_dt

        # W_b: mean sojourn for items whose arrival falls in the batch.
        # Track the sample count per batch so we can drop sparse batches
        # rather than failing the whole test.
        soj_arr_times = np.asarray([r[0] for r in self._sojourn_records], dtype=float)
        soj_values = np.asarray([r[1] for r in self._sojourn_records], dtype=float)
        W_b = np.full(n_batches, np.nan)
        n_samples_b = np.zeros(n_batches, dtype=int)
        for b in range(n_batches):
            mask = (soj_arr_times >= edges[b]) & (soj_arr_times < edges[b + 1])
            n_samples_b[b] = int(mask.sum())
            if n_samples_b[b] >= min_samples_per_batch:
                W_b[b] = float(soj_values[mask].mean())

        # Trailing-batch censoring filter: any batch whose mean sojourn is
        # greater than its width is treated as suspect (its items extend past
        # the batch, and likely past sim_end → W_b biased low). We drop the
        # contiguous suspect tail.
        valid = ~np.isnan(W_b)
        if valid.any():
            for b in range(n_batches - 1, -1, -1):
                if not valid[b]:
                    continue
                if W_b[b] > batch_dt:
                    valid[b] = False
                else:
                    break  # first non-censored from the right; stop
        keep = valid
        if int(keep.sum()) < 4:
            return None

        return {
            "L_b": L_b[keep],
            "lambda_b": lambda_b[keep],
            "W_b": W_b[keep],
            "n_dropped": int((~keep).sum()),
            "n_total": n_batches,
        }

    def verify_littles_law(self, n_batches: int = 20, alpha: float = 0.05,
                           rel_err_tol: float = 0.05) -> dict:
        """Two-tier robust Little's Law diagnostic (H₀: L − λW = 0).

        Tier 1 — batch-means CI test (Law & Kelton 2000 §9.4; Glynn & Whitt
        1989). Partition the post-warmup window into ``n_batches`` and
        construct a 100·(1−α)% delta-method CI for L̂ − λ̂Ŵ; pass if the CI
        covers 0. ``n_batches`` auto-shrinks to the largest B ∈ {n_batches,
        n_batches/2, …, 4} for which ``_compute_batch_stats`` returns at
        least four valid batches; below that threshold the tier-1 CI is
        not informative and Tier 2 is used.

        Tier 2 — relative-error fallback (Law & Kelton 2000 §9.6.2).
        For short runs / low-rate queues, the variance estimator from a
        few batches is so poor that even an exact L = λW relationship
        produces a degenerate CI. We additionally pass when the
        full-sample point estimate satisfies
            |L − λW| / max(|L|, ε)  ≤  rel_err_tol
        with default tolerance 5%. The combined test passes when *either*
        tier passes; the report records which path closed.

        Why this matters:
            The previous implementation reported ``passed=False`` whenever
            tier-1 was unavailable, even for queues where the point
            estimate satisfied L = λW exactly (delta = 0). That made the
            diagnostic useless on short smoke runs and on low-rate queues
            (robot, recall_queue) — which is exactly when a model-defect
            check is most useful.
        """
        L = self.mean_L()
        lam = self.lambda_hat()
        W = self.mean_W()

        # Sample-mean point estimates (always defined when L, λ, W defined).
        if not any(map(math.isnan, (L, lam, W))):
            LW_point = lam * W
            delta_point = L - LW_point
            rel_err_point = abs(delta_point) / max(abs(L), 1e-12)
        else:
            LW_point = delta_point = rel_err_point = float("nan")

        passed_tier2 = (
            not math.isnan(rel_err_point) and rel_err_point <= rel_err_tol
        )

        # Tier 1 — try to form valid batches, shrinking n_batches if needed.
        batch = None
        B_used = 0
        for B_try in (n_batches, n_batches // 2, max(4, n_batches // 4)):
            if B_try < 4:
                break
            batch = self._compute_batch_stats(B_try)
            if batch is not None:
                B_used = B_try
                break

        def _result(passed: bool, note: str = "",
                    delta=delta_point, se=float("nan"),
                    lo=float("nan"), hi=float("nan"),
                    n_batches_used: int = 0,
                    n_dropped: int = 0,
                    method: str = "rel_err_only") -> dict:
            return {
                "queue": self.name,
                "L": round(L, 6) if not math.isnan(L) else L,
                "lambda": round(lam, 6) if not math.isnan(lam) else lam,
                "W": round(W, 6) if not math.isnan(W) else W,
                "LW_product": round(LW_point, 6) if not math.isnan(LW_point) else LW_point,
                "delta": round(delta, 6) if not math.isnan(delta) else delta,
                "SE_delta": round(se, 6) if not math.isnan(se) else se,
                "t_critical": float("nan"),
                "ci_lower": round(lo, 6) if not math.isnan(lo) else lo,
                "ci_upper": round(hi, 6) if not math.isnan(hi) else hi,
                "relative_error": (
                    round(rel_err_point, 6) if not math.isnan(rel_err_point) else rel_err_point
                ),
                "rel_err_tol": rel_err_tol,
                "n_batches": n_batches_used,
                "n_batches_dropped": n_dropped,
                "alpha": alpha,
                "df": max(0, n_batches_used - 1),
                "method": method,
                "passed": passed,
                "note": note,
            }

        if any(map(math.isnan, (L, lam, W))):
            return _result(False, note="Insufficient data")

        if batch is None:
            # Tier 1 unavailable; rely on the relative-error tier.
            return _result(
                passed_tier2,
                note=("Batch CI unavailable; passed via relative-error tier."
                      if passed_tier2 else
                      "Batch CI unavailable and relative-error exceeds tolerance."),
                method="rel_err_only",
            )

        L_b = batch["L_b"]; lambda_b = batch["lambda_b"]; W_b = batch["W_b"]
        B = len(L_b)
        L_hat = float(np.mean(L_b))
        lam_hat = float(np.mean(lambda_b))
        W_hat = float(np.mean(W_b))

        var_L = float(np.var(L_b, ddof=1) / B)
        var_lam = float(np.var(lambda_b, ddof=1) / B)
        var_W = float(np.var(W_b, ddof=1) / B)
        var_delta = var_L + (lam_hat ** 2) * var_W + (W_hat ** 2) * var_lam
        se_delta = math.sqrt(max(var_delta, 0.0))

        df = B - 1
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        delta_hat = L_hat - lam_hat * W_hat
        ci_lower = delta_hat - t_crit * se_delta
        ci_upper = delta_hat + t_crit * se_delta
        passed_tier1 = bool(ci_lower <= 0.0 <= ci_upper)

        passed = passed_tier1 or passed_tier2
        method = (
            "batch_ci" if passed_tier1
            else ("rel_err_fallback" if passed_tier2 else "neither")
        )

        return _result(
            passed,
            note=("Pass via batch-means CI." if passed_tier1
                  else ("Pass via relative-error fallback (rel_err ≤ tol)."
                        if passed_tier2 else
                        "Both tiers rejected H₀ at given tolerances.")),
            delta=delta_hat,
            se=se_delta,
            lo=ci_lower, hi=ci_upper,
            n_batches_used=B,
            n_dropped=batch.get("n_dropped", 0),
            method=method,
        )


# ---------------------------------------------------------------------------
# Metric Accumulator — scalar metrics across replications
# ---------------------------------------------------------------------------

def percentile_bootstrap_ci(
    samples: np.ndarray,
    statistic: str = "mean",
    percentile: float = 50.0,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Percentile-bootstrap confidence interval for a summary statistic.

    Non-parametric: resamples the observed values with replacement
    ``n_boot`` times, computes the statistic on each resample, and returns
    the empirical α/2 and 1−α/2 quantiles of the bootstrap distribution.

    Args:
        samples:    1-D array of observations.
        statistic:  Which statistic to bootstrap; one of
                      - "mean"       — sample mean
                      - "percentile" — the ``percentile`` order statistic.
        percentile: Percentile level (0–100) used when ``statistic="percentile"``.
        n_boot:     Number of bootstrap replicates (default 1000 — Efron &
                    Tibshirani 1993 recommend ≥1000 for CI endpoints).
        alpha:      Significance level (default 0.05 for a 95% CI).
        rng_seed:   Optional seed for reproducible resampling.

    Returns:
        (lower, upper) bounds of the (1−α)·100% CI.
    """
    arr = np.asarray(samples, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        # Degenerate — no variability. Return the point as a zero-width CI.
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = arr[idx]
    if statistic == "mean":
        boot_stats = resamples.mean(axis=1)
    elif statistic == "percentile":
        boot_stats = np.percentile(resamples, percentile, axis=1)
    else:
        raise ValueError(f"Unsupported statistic {statistic!r}; "
                         f"use 'mean' or 'percentile'.")
    lo = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


class MetricAccumulator:
    """
    Accumulates per-replication scalar metric values and computes
    mean ± 95% confidence interval across all replications.

    The CI is a 1000-replicate percentile bootstrap (Efron & Tibshirani
    1993): it does not assume Gaussianity of replication-level values,
    which is important for ratios and latency percentiles whose sampling
    distribution is typically skewed.
    """

    def __init__(self, name: str, n_boot: int = 1000,
                 rng_seed: Optional[int] = None) -> None:
        self.name = name
        self._rep_values: List[float] = []
        self._n_boot = int(n_boot)
        self._rng_seed = rng_seed

    def add_replication(self, value: float) -> None:
        """Record one replication's value for this metric."""
        self._rep_values.append(value)

    @property
    def n(self) -> int:
        return len(self._rep_values)

    def mean(self) -> float:
        if not self._rep_values:
            return float("nan")
        return float(np.mean(self._rep_values))

    def std(self) -> float:
        if len(self._rep_values) < 2:
            return float("nan")
        return float(np.std(self._rep_values, ddof=1))

    def confidence_interval_95(self) -> Tuple[float, float]:
        """
        Percentile-bootstrap 95% CI (B=1000 resamples by default).
        With fewer than 2 replications the CI is undefined and returned
        as (nan, nan); callers should either run more replications or
        compute a within-replication CI via ``percentile_bootstrap_ci``
        on the raw samples (see ``LatencyHistogram.summary``).
        """
        if self.n < 2:
            return (float("nan"), float("nan"))
        return percentile_bootstrap_ci(
            np.asarray(self._rep_values, dtype=float),
            statistic="mean",
            n_boot=self._n_boot,
            alpha=0.05,
            rng_seed=self._rng_seed,
        )

    def half_width(self) -> float:
        """Half-width of the 95% CI."""
        lo, hi = self.confidence_interval_95()
        if math.isnan(lo):
            return float("nan")
        return (hi - lo) / 2

    def summary(self) -> dict:
        lo, hi = self.confidence_interval_95()
        return {
            "metric": self.name,
            "n_replications": self.n,
            "mean": round(self.mean(), 6),
            "std": round(self.std(), 6),
            "ci95_lower": round(lo, 6),
            "ci95_upper": round(hi, 6),
            "half_width": round(self.half_width(), 6),
            "ci_method": f"percentile_bootstrap_B={self._n_boot}",
        }


# ---------------------------------------------------------------------------
# Latency Histogram — for P50/P95/P99 reporting
# ---------------------------------------------------------------------------

class LatencyHistogram:
    """
    Collects latency samples and computes percentile statistics.
    Stored per replication then aggregated.

    If ``warmup_s`` is set and the caller passes an event time to ``record``,
    samples whose event time precedes the warm-up end are silently dropped.

    When only one replication is available the usual cross-replication
    bootstrap cannot be formed; instead we fall back to a within-replication
    (Woodruff-style) bootstrap over the raw latency sample, which is a
    well-defined CI for the quantile estimator from that single replication.
    """

    def __init__(self, name: str, warmup_s: float = 0.0,
                 n_boot: int = 1000,
                 rng_seed: Optional[int] = None) -> None:
        self.name = name
        self._warmup_s = float(max(warmup_s, 0.0))
        self._samples: List[float] = []
        self._rep_p50: List[float] = []
        self._rep_p95: List[float] = []
        self._rep_p99: List[float] = []
        # Last replication's raw samples (retained for within-rep bootstrap
        # when only one replication is run).
        self._last_rep_samples: List[float] = []
        self._n_boot = int(n_boot)
        self._rng_seed = rng_seed

    def record(self, latency_s: float, time: Optional[float] = None) -> None:
        if latency_s < 0:
            return
        if time is not None and time < self._warmup_s:
            return
        self._samples.append(latency_s)

    def flush_replication(self) -> None:
        """Call at end of each replication to store percentiles."""
        if self._samples:
            arr = np.array(self._samples)
            self._rep_p50.append(float(np.percentile(arr, 50)))
            self._rep_p95.append(float(np.percentile(arr, 95)))
            self._rep_p99.append(float(np.percentile(arr, 99)))
            self._last_rep_samples = list(self._samples)
        self._samples = []

    def _woodruff_ci(self, percentile: float) -> Tuple[float, float, dict]:
        """
        Within-replication percentile-bootstrap CI for a given order
        statistic. Used when n_replications < 2.

        Returns ``(lower, upper, meta)`` where ``meta`` carries the
        fallback flag for downstream reporting.
        """
        if not self._last_rep_samples:
            return (float("nan"), float("nan"),
                    {"note": "no raw samples retained"})
        lo, hi = percentile_bootstrap_ci(
            np.asarray(self._last_rep_samples, dtype=float),
            statistic="percentile",
            percentile=percentile,
            n_boot=self._n_boot,
            alpha=0.05,
            rng_seed=self._rng_seed,
        )
        return (lo, hi, {
            "ci_method": f"woodruff_within_rep_bootstrap_B={self._n_boot}",
            "n_samples": len(self._last_rep_samples),
        })

    def summary(self) -> dict:
        """Compute mean ± CI of each percentile across replications."""
        results = {}
        rep_values_by_name = {
            "p50": (self._rep_p50, 50.0),
            "p95": (self._rep_p95, 95.0),
            "p99": (self._rep_p99, 99.0),
        }
        for pct_name, (values, pct_level) in rep_values_by_name.items():
            if len(values) >= 2:
                acc = MetricAccumulator(
                    f"{self.name}_{pct_name}",
                    n_boot=self._n_boot,
                    rng_seed=self._rng_seed,
                )
                for v in values:
                    acc.add_replication(v)
                results[pct_name] = acc.summary()
            elif len(values) == 1:
                lo, hi, meta = self._woodruff_ci(pct_level)
                point = values[0]
                entry = {
                    "metric": f"{self.name}_{pct_name}",
                    "n_replications": 1,
                    "mean": round(point, 6),
                    "std": float("nan"),
                    "ci95_lower": round(lo, 6) if not math.isnan(lo) else lo,
                    "ci95_upper": round(hi, 6) if not math.isnan(hi) else hi,
                    "half_width": (round((hi - lo) / 2, 6)
                                   if not (math.isnan(lo) or math.isnan(hi))
                                   else float("nan")),
                }
                entry.update(meta)
                entry["note"] = "single replication — within-rep bootstrap fallback"
                results[pct_name] = entry
            else:
                results[pct_name] = {"mean": float("nan"),
                                     "note": "insufficient replications"}
        return results


# ---------------------------------------------------------------------------
# Per-Replication Statistics Collector
# ---------------------------------------------------------------------------

class ReplicationStats:
    """
    Collects all statistics for a single simulation replication.
    At the end of the replication, statistics are extracted and passed to
    the cross-replication accumulators.
    """

    def __init__(self, replication_id: int, warmup_s: float) -> None:
        self.replication_id = replication_id
        self.warmup_s = float(max(warmup_s, 0.0))

        # Queue statistics (Little's Law)
        self.queues: Dict[str, QueueStats] = {
            name: QueueStats(name, warmup_s=self.warmup_s)
            for name in (
                "disk_cache", "interconnect", "mover_pool", "robot",
                "drive_pool", "ls_accrual", "recall_queue",
            )
        }

        # Latency histograms
        self.recall_latency   = LatencyHistogram("recall_latency",  warmup_s=self.warmup_s)
        self.migrate_latency  = LatencyHistogram("migrate_latency", warmup_s=self.warmup_s)
        self.mount_latency    = LatencyHistogram("mount_latency",   warmup_s=self.warmup_s)

        # Scalar time series
        self.disk_usage_series   = SampleSeries("disk_usage_pct")
        self.cache_hit_series    = SampleSeries("cache_hit")
        self.drive_util_series: Dict[int, SampleSeries] = {}

        # Counters. Only events observed after warm-up contribute.
        self.n_migrations      = 0
        self.n_recalls         = 0
        self.n_space_releases  = 0   # DUL → OFL space-release transitions
        self.n_cache_hits      = 0
        self.n_cache_misses    = 0
        self.n_policy_cycles   = 0
        self.n_files_created   = 0
        self.n_mounts          = 0
        self.n_seeks           = 0

        # Interval-model observables (post-warm-up): tape write volume C,
        # time-integral of unprotected bytes V, and deletion outcomes.
        self.lt_bytes_created = 0            # bytes written by users (post warm-up)
        self.lt_bytes_archived = 0           # bytes that completed migration
        self.lt_bytes_x_pipeline_s = 0.0     # sum of bytes * (completion - selection)
        self.lt_bytes_tape_garbage = 0       # archived bytes later deleted
        self.lt_bytes_deleted_unarchived = 0 # bytes deleted before a tape copy
        self._unprot_bytes = 0.0             # current REG+MIG (no tape copy) bytes
        self._unprot_last_t = 0.0
        self._unprot_integral = 0.0          # integral over [warmup, now]

        # Tape / archive byte counters
        self.bytes_archived  = 0
        self.bytes_recalled  = 0
        self.bytes_written_to_cache = 0
        self.bytes_read_from_cache  = 0

        # IOR-style timing (first/last operation timestamps)
        self.first_write_time: Optional[float] = None
        self.last_write_time:  Optional[float] = None
        self.first_read_time:  Optional[float] = None
        self.last_read_time:   Optional[float] = None
        self.write_operations = 0
        self.read_operations  = 0

        # Multi-tenant per-tenant tallies (created lazily on first record).
        # ``tenant_id`` keys are set by the workload generator; for a
        # single-tenant run these dicts stay empty.
        self.tenant_recall_latency: Dict[str, LatencyHistogram] = {}
        self.tenant_n_writes:  Dict[str, int] = {}
        self.tenant_n_reads:   Dict[str, int] = {}
        self.tenant_bytes_written: Dict[str, int] = {}
        self.tenant_bytes_read:    Dict[str, int] = {}

    def record_recall_latency(self, latency_s: float, time: Optional[float] = None,
                              tenant_id: Optional[str] = None) -> None:
        if time is not None and time < self.warmup_s:
            return
        self.recall_latency.record(latency_s, time=time)
        self.n_recalls += 1
        if tenant_id is not None:
            hist = self.tenant_recall_latency.get(tenant_id)
            if hist is None:
                hist = LatencyHistogram(f"recall_latency[{tenant_id}]",
                                        warmup_s=self.warmup_s)
                self.tenant_recall_latency[tenant_id] = hist
            hist.record(latency_s, time=time)

    def record_migrate_latency(self, latency_s: float, time: Optional[float] = None) -> None:
        if time is not None and time < self.warmup_s:
            return
        self.migrate_latency.record(latency_s, time=time)
        self.n_migrations += 1

    def record_cache_event(self, hit: bool, time: Optional[float] = None) -> None:
        if time is not None and time < self.warmup_s:
            return
        if hit:
            self.n_cache_hits += 1
        else:
            self.n_cache_misses += 1

    def record_disk_usage(self, time: float, fraction: float) -> None:
        self.disk_usage_series.record(time, fraction * 100.0)

    def record_drive_job(self, kind: str, t_arrive: float, t_start: float,
                         t_end: float, nbytes: int, vsn=None) -> None:
        """Per-job drive-pool record (post warm-up arrivals only): wait and
        hold time for recalls and write zones, used to validate the queue model."""
        if t_arrive < self.warmup_s:
            return
        if not hasattr(self, "drive_jobs"):
            self.drive_jobs = {"recall": [], "zone": []}
        self.drive_jobs[kind].append((t_arrive, t_start - t_arrive, t_end - t_start, nbytes, vsn))

    def unprot_change(self, t: float, delta_bytes: float) -> None:
        """Update the unprotected-bytes level and its post-warm-up integral."""
        lo = max(self._unprot_last_t, self.warmup_s)
        if t > lo:
            self._unprot_integral += self._unprot_bytes * (t - lo)
        self._unprot_last_t = max(self._unprot_last_t, t)
        self._unprot_bytes += delta_bytes

    def unprotected_time_average(self, t_end: float) -> float:
        """Time-average unprotected bytes over [warmup, t_end]."""
        self.unprot_change(t_end, 0.0)
        span = t_end - self.warmup_s
        return self._unprot_integral / span if span > 0 else float("nan")

    def record_tape_write(self, bytes_count: int) -> None:
        """Record bytes written to tape (migration)."""
        self.bytes_archived += bytes_count

    def record_tape_read(self, bytes_count: int) -> None:
        """Record bytes read from tape (recall)."""
        self.bytes_recalled += bytes_count

    def record_cache_write(self, bytes_count: int, time: Optional[float] = None,
                           tenant_id: Optional[str] = None) -> None:
        """Record bytes written to disk cache."""
        self.bytes_written_to_cache += bytes_count
        self.write_operations += 1
        if time is not None:
            if self.first_write_time is None:
                self.first_write_time = time
            self.last_write_time = time
        if tenant_id is not None:
            self.tenant_n_writes[tenant_id] = self.tenant_n_writes.get(tenant_id, 0) + 1
            self.tenant_bytes_written[tenant_id] = (
                self.tenant_bytes_written.get(tenant_id, 0) + int(bytes_count)
            )

    def record_cache_read(self, bytes_count: int, time: Optional[float] = None,
                          tenant_id: Optional[str] = None) -> None:
        """Record bytes read from disk cache."""
        self.bytes_read_from_cache += bytes_count
        self.read_operations += 1
        if time is not None:
            if self.first_read_time is None:
                self.first_read_time = time
            self.last_read_time = time
        if tenant_id is not None:
            self.tenant_n_reads[tenant_id] = self.tenant_n_reads.get(tenant_id, 0) + 1
            self.tenant_bytes_read[tenant_id] = (
                self.tenant_bytes_read.get(tenant_id, 0) + int(bytes_count)
            )

    def record_seek(self) -> None:
        self.n_seeks += 1

    def summary_dict(self) -> dict:
        """Return a flat dict of all scalar metrics for this replication."""
        return {
            "replication_id": self.replication_id,
            "n_migrations": self.n_migrations,
            "n_recalls": self.n_recalls,
            "n_cache_hits": self.n_cache_hits,
            "n_cache_misses": self.n_cache_misses,
            "n_policy_cycles": self.n_policy_cycles,
            "n_files_created": self.n_files_created,
            "n_mounts": self.n_mounts,
            "n_seeks": self.n_seeks,
            "bytes_archived": self.bytes_archived,
            "bytes_recalled": self.bytes_recalled,
            "bytes_written_to_cache": self.bytes_written_to_cache,
            "bytes_read_from_cache": self.bytes_read_from_cache,
            "cache_hit_rate": self.cache_hit_rate,
            "write_operations": self.write_operations,
            "read_operations": self.read_operations,
        }

    def finalise(self, sim_end_time: float) -> None:
        """Finalise all queue statistics at end of replication."""
        for q in self.queues.values():
            q.finalise(sim_end_time)
        # Flush latency histograms so cross-replication aggregation can occur
        self.recall_latency.flush_replication()
        self.migrate_latency.flush_replication()

    @property
    def cache_hit_rate(self) -> float:
        total = self.n_cache_hits + self.n_cache_misses
        if total == 0:
            return float("nan")
        return self.n_cache_hits / total

    def littles_law_report(self, n_batches: int = 20, alpha: float = 0.05) -> List[dict]:
        """Run Little's Law verification (batch-means CI test) on all queues."""
        return [q.verify_littles_law(n_batches=n_batches, alpha=alpha)
                for q in self.queues.values()]


# ---------------------------------------------------------------------------
# Cross-Replication Aggregator
# ---------------------------------------------------------------------------

class SimulationStats:
    """
    Aggregates statistics across all replications.
    Produces final output with mean ± 95% CI for all KPIs.
    """

    def __init__(self) -> None:
        # Primary KPI accumulators
        self.recall_latency_p50  = MetricAccumulator("recall_latency_p50_s")
        self.recall_latency_p95  = MetricAccumulator("recall_latency_p95_s")
        self.recall_latency_p99  = MetricAccumulator("recall_latency_p99_s")
        self.migrate_latency_p50 = MetricAccumulator("migrate_latency_p50_s")
        self.migrate_latency_p95 = MetricAccumulator("migrate_latency_p95_s")
        self.cache_hit_rate      = MetricAccumulator("cache_hit_rate")
        self.drive_utilisation: Dict[int, MetricAccumulator] = {}
        self.robot_utilisation   = MetricAccumulator("robot_utilisation")
        self.mover_utilisation   = MetricAccumulator("mover_utilisation")
        self.n_migrations        = MetricAccumulator("n_migrations")
        self.n_recalls           = MetricAccumulator("n_recalls")
        self.mean_disk_usage     = MetricAccumulator("mean_disk_usage_pct")
        self.n_mounts            = MetricAccumulator("n_mounts")
        self.n_seeks             = MetricAccumulator("n_seeks")
        self.bytes_archived      = MetricAccumulator("bytes_archived")
        self.bytes_recalled      = MetricAccumulator("bytes_recalled")
        self.bytes_written_to_cache = MetricAccumulator("bytes_written_to_cache")
        self.bytes_read_from_cache  = MetricAccumulator("bytes_read_from_cache")
        self.write_operations    = MetricAccumulator("write_operations")
        self.read_operations     = MetricAccumulator("read_operations")

        # Per-tenant accumulators (populated as tenants are observed across
        # replications). For multi-tenant runs, each tenant gets a recall-
        # latency P95 series so we can compute fairness metrics (Gini) on
        # tail latency at end of run.
        self.tenant_p95_recall: Dict[str, MetricAccumulator] = {}
        self.tenant_n_writes: Dict[str, MetricAccumulator] = {}
        self.tenant_n_reads:  Dict[str, MetricAccumulator] = {}

        self._replication_results: List[dict] = []
        self._littles_law_results: List[List[dict]] = []

    def ingest_replication(self, rep_stats: ReplicationStats) -> None:
        """Ingest results from one completed replication."""
        # Flush latency histograms
        rep_stats.recall_latency.flush_replication()
        rep_stats.migrate_latency.flush_replication()

        # Latency percentiles
        if rep_stats.recall_latency._rep_p50:
            self.recall_latency_p50.add_replication(rep_stats.recall_latency._rep_p50[-1])
            self.recall_latency_p95.add_replication(rep_stats.recall_latency._rep_p95[-1])
            self.recall_latency_p99.add_replication(rep_stats.recall_latency._rep_p99[-1])

        if rep_stats.migrate_latency._rep_p50:
            self.migrate_latency_p50.add_replication(rep_stats.migrate_latency._rep_p50[-1])
            self.migrate_latency_p95.add_replication(rep_stats.migrate_latency._rep_p95[-1])

        # Scalar metrics
        self.cache_hit_rate.add_replication(rep_stats.cache_hit_rate)
        self.n_migrations.add_replication(rep_stats.n_migrations)
        self.n_recalls.add_replication(rep_stats.n_recalls)
        self.n_mounts.add_replication(rep_stats.n_mounts)
        self.n_seeks.add_replication(rep_stats.n_seeks)
        self.bytes_archived.add_replication(rep_stats.bytes_archived)
        self.bytes_recalled.add_replication(rep_stats.bytes_recalled)
        self.bytes_written_to_cache.add_replication(rep_stats.bytes_written_to_cache)
        self.bytes_read_from_cache.add_replication(rep_stats.bytes_read_from_cache)
        self.write_operations.add_replication(rep_stats.write_operations)
        self.read_operations.add_replication(rep_stats.read_operations)

        # Disk usage: average only over the post-warmup window.
        times, values = rep_stats.disk_usage_series.as_arrays()
        if len(values) > 0:
            mask = times >= rep_stats.warmup_s
            if mask.sum() > 0:
                self.mean_disk_usage.add_replication(float(np.mean(values[mask])))

        # Per-tenant ingest. For each tenant observed in this replication,
        # flush the histogram, then push the per-replication P95 / counts
        # into the cross-replication accumulator (lazily created).
        for tid, hist in rep_stats.tenant_recall_latency.items():
            hist.flush_replication()
            if hist._rep_p95:
                acc = self.tenant_p95_recall.get(tid)
                if acc is None:
                    acc = MetricAccumulator(f"recall_p95_{tid}")
                    self.tenant_p95_recall[tid] = acc
                acc.add_replication(hist._rep_p95[-1])
        for tid, n in rep_stats.tenant_n_writes.items():
            acc = self.tenant_n_writes.get(tid)
            if acc is None:
                acc = MetricAccumulator(f"n_writes_{tid}")
                self.tenant_n_writes[tid] = acc
            acc.add_replication(float(n))
        for tid, n in rep_stats.tenant_n_reads.items():
            acc = self.tenant_n_reads.get(tid)
            if acc is None:
                acc = MetricAccumulator(f"n_reads_{tid}")
                self.tenant_n_reads[tid] = acc
            acc.add_replication(float(n))

        # Little's Law
        self._littles_law_results.append(rep_stats.littles_law_report())

    def produce_report(self) -> dict:
        """
        Produce the final output report with all KPIs.

        Returns:
            Dictionary suitable for JSON/YAML output.
        """
        report = {
            "kpis": {
                "recall_latency_s": {
                    "p50": self.recall_latency_p50.summary(),
                    "p95": self.recall_latency_p95.summary(),
                    "p99": self.recall_latency_p99.summary(),
                },
                "migrate_latency_s": {
                    "p50": self.migrate_latency_p50.summary(),
                    "p95": self.migrate_latency_p95.summary(),
                },
                "cache_hit_rate":    self.cache_hit_rate.summary(),
                "n_migrations":      self.n_migrations.summary(),
                "n_recalls":         self.n_recalls.summary(),
                "n_tape_mounts":     self.n_mounts.summary(),
                "n_seeks":           self.n_seeks.summary(),
                "bytes_archived":    self.bytes_archived.summary(),
                "bytes_recalled":    self.bytes_recalled.summary(),
                "bytes_written_to_cache": self.bytes_written_to_cache.summary(),
                "bytes_read_from_cache":  self.bytes_read_from_cache.summary(),
                "write_operations":  self.write_operations.summary(),
                "read_operations":   self.read_operations.summary(),
                "mean_disk_usage_pct": self.mean_disk_usage.summary(),
            },
            "littles_law_validation": self._aggregate_littles_law(),
        }
        if self.tenant_p95_recall:
            report["multi_tenant"] = self._tenant_report()
        return report

    def _tenant_report(self) -> dict:
        """Per-tenant KPI block plus a fairness summary (Gini coefficient
        on per-tenant P95 recall latency). Only emitted when at least one
        tenant has been observed across replications."""
        # Avoid hard-importing multi_tenant here so single-tenant runs
        # don't pull in scipy paths they don't need.
        try:
            from workload.multi_tenant import gini_coefficient
        except Exception:
            def gini_coefficient(values):
                arr = np.asarray(values, dtype=float)
                arr = arr[~np.isnan(arr)]
                n = arr.size
                if n == 0 or arr.min() < 0 or arr.sum() == 0:
                    return float("nan") if n == 0 else 0.0
                arr = np.sort(arr)
                idx = np.arange(1, n + 1)
                return float(((2 * idx - n - 1) * arr).sum() / (n * arr.sum()))

        tenant_summaries = {}
        means_p95 = []
        for tid, acc in sorted(self.tenant_p95_recall.items()):
            s = acc.summary()
            s["n_writes"] = (
                self.tenant_n_writes[tid].summary()
                if tid in self.tenant_n_writes else None
            )
            s["n_reads"] = (
                self.tenant_n_reads[tid].summary()
                if tid in self.tenant_n_reads else None
            )
            tenant_summaries[tid] = s
            if not math.isnan(s["mean"]):
                means_p95.append(s["mean"])

        return {
            "n_tenants_observed": len(tenant_summaries),
            "p95_recall_gini": (
                round(gini_coefficient(np.asarray(means_p95)), 6)
                if means_p95 else float("nan")
            ),
            "p95_recall_max": (
                round(float(np.max(means_p95)), 6) if means_p95 else float("nan")
            ),
            "p95_recall_min": (
                round(float(np.min(means_p95)), 6) if means_p95 else float("nan")
            ),
            "tenants": tenant_summaries,
        }

    def _aggregate_littles_law(self) -> dict:
        """Summarise Little's Law pass/fail across all replications."""
        if not self._littles_law_results:
            return {}

        # Group by queue name
        queue_names = [r["queue"] for r in self._littles_law_results[0]
                       if "queue" in r]
        summary = {}
        for qname in queue_names:
            pass_count = sum(
                1 for rep in self._littles_law_results
                for r in rep
                if r.get("queue") == qname and r.get("passed", False)
            )
            total = len(self._littles_law_results)
            summary[qname] = {
                "pass_rate": round(pass_count / total, 3) if total > 0 else float("nan"),
                "n_replications": total,
            }
        return summary
