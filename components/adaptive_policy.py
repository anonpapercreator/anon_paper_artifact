"""
components/adaptive_policy.py — Adaptive Policy Advisor
========================================================
Closes the loop between the MMFQ/IOM/CTMC computational models and the DES.

Runs as an independent SimPy process that periodically:
    1. Samples telemetry from ReplicationStats
    2. Classifies workload phase via PhaseClassifier
    3. Fits CTMC generator matrix Q and drift rates r
    4. Solves MMFQ for cache-fill distribution
    5. Optimises archive interval tau* via IOM
    6. Predicts upcoming phase transitions via CTMCPredictor
    7. Dynamically sets PolicyEngine and LibraryServer parameters

Architecture (per design doc):
    WorkloadGen → PhaseClassifier → AdaptivePolicyAdvisor
                                          ↓
                              ┌──────────┴──────────┐
                              ↓                     ↓
                         PolicyEngine         LibraryServer
                         (age_threshold)      (batch thresholds)
"""
from __future__ import annotations
import logging
import simpy
import numpy as np
from typing import Iterator, Optional, Dict, Any

logger = logging.getLogger(__name__)

from stats.collector import ReplicationStats
from components.filesystem import PolicyEngine
from components.tape_subsystem import LibraryServer

from models.phase_classifier import (
    PhaseClassifier,
    TelemetrySample,
    build_telemetry_from_simulator_state,
    AbsorbingPhaseWarning,
    DisconnectedChainError,
)
from models.iom import IOM, IOMParameters, iom_params_from_config
from models.ctmc_predictor import CTMCPredictor
from models.mmfq import MMFQSolver


class AdaptivePolicyAdvisor:
    """
    Independent SimPy process that dynamically adjusts DMF policy parameters
    based on the unified MMFQ/IOM/CTMC probabilistic model.
    """

    def __init__(
        self,
        env: simpy.Environment,
        cfg: dict,
        policy_engine: PolicyEngine,
        library_server: LibraryServer,
        stats: ReplicationStats,
        disk: Any,
    ) -> None:
        self._env = env
        self._cfg = cfg
        self._policy = policy_engine
        self._ls = library_server
        self._stats = stats
        self._disk = disk

        ap_cfg = cfg.get("adaptive_policy", {})
        self._interval_s = ap_cfg.get("advisor_interval_s", 60.0)
        self._tau_min_s = ap_cfg.get("tau_min_s", 300.0)
        self._tau_max_s = ap_cfg.get("tau_max_s", 86400.0)
        self._baseline_age_threshold = ap_cfg.get(
            "baseline_age_threshold_s",
            cfg["policy_engine"].get("candidate_age_threshold_s", 1800.0),
        )
        self._clamp_min = ap_cfg.get("clamp_factor_min", 0.25)
        self._clamp_max = ap_cfg.get("clamp_factor_max", 4.0)
        self._checkpoint_horizon_s = ap_cfg.get("checkpoint_horizon_s", 600.0)
        self._checkpoint_threshold = ap_cfg.get("checkpoint_threshold_prob", 0.3)
        self._emergency_age_s = ap_cfg.get("emergency_age_threshold_s", 300.0)
        self._batch_adaptive = ap_cfg.get("batch_size_adaptive", True)
        self._batch_min_bytes = ap_cfg.get("batch_size_min_bytes", 1073741824)
        self._batch_max_bytes = ap_cfg.get("batch_size_max_bytes", 10737418240)

        # Phase classifier state
        cache_cap = cfg["disk_cache"]["capacity_bytes"]
        ess_peak_gbs = cfg["disk_cache"].get("aggregate_throughput_gib_s", 65.0)
        tape_cfg = cfg["tape_drives"]
        tape_peak_gbs = (
            tape_cfg["n_drives"] * tape_cfg["native_rate_bytes_per_s"] / (1024**3)
        )
        self._phase_classifier = PhaseClassifier(
            cache_capacity_bytes=cache_cap,
            ess_peak_gbs=ess_peak_gbs,
            tape_peak_gbs=tape_peak_gbs,
        )
        self._ctmc_predictor: Optional[CTMCPredictor] = None
        self._fitted_Q: Optional[np.ndarray] = None
        self._fitted_r: Optional[np.ndarray] = None

        # Telemetry deltas (previous cumulative counters)
        self._prev_ess_written = 0.0
        self._prev_ess_read = 0.0
        self._prev_tape_written = 0.0
        self._prev_tape_read = 0.0

        # History for IOM parameter estimation
        self._telemetry_log: list = []

    def run(self) -> Iterator:
        """SimPy process: periodic advisor loop."""
        # Initial delay before first advisory cycle
        yield self._env.timeout(self._interval_s)

        while True:
            self._advise()
            yield self._env.timeout(self._interval_s)

    def _advise(self) -> None:
        """Single advisory cycle: sample, model, decide, act."""
        now = self._env.now

        # ── 1. Build telemetry sample ────────────────────────────────────
        telemetry = self._build_telemetry_sample(now)
        phase = self._phase_classifier.observe(telemetry)
        self._telemetry_log.append(
            {
                "sim_time_s": now,
                "phase": phase,
                "cache_fill_frac": telemetry.cache_fill_fraction,
                "write_rate_gbs": telemetry.write_rate_bps / (1024**3),
                "archival_rate_gbs": telemetry.archival_rate_bps / (1024**3),
                "recall_rate_gbs": telemetry.recall_rate_bps / (1024**3),
            }
        )

        # ── 2. Fit CTMC if enough data ────────────────────────────────────
        # Only narrow, well-typed exceptions are swallowed: a degenerate-fit
        # signal (DisconnectedChainError / AbsorbingPhaseWarning) is the
        # advisor's "not enough data yet" condition and is expected to clear
        # on later cycles. Anything else escapes so a real bug isn't masked.
        if len(self._telemetry_log) >= 10:
            try:
                self._fitted_Q = self._phase_classifier.fit_generator_matrix()
                self._fitted_r = self._phase_classifier.net_drift_rates_bps()
            except (DisconnectedChainError, AbsorbingPhaseWarning) as exc:
                logger.debug("CTMC fit not yet ready: %s", exc)
                self._fitted_Q = None
                self._fitted_r = None

        # ── 3. Default tau = baseline ──────────────────────────────────
        tau_star = self._baseline_age_threshold
        prob_checkpoint = 0.0
        expected_drift = 0.0

        # ── 4. CTMC prediction & IOM optimisation ────────────────────────
        if self._fitted_Q is not None and self._fitted_r is not None:
            # Initialise/update CTMC predictor
            if self._ctmc_predictor is None:
                self._ctmc_predictor = CTMCPredictor(
                    self._fitted_Q,
                    names=["IDLE", "COMPUTE", "CHECKPOINT", "RECALL"],
                )
            else:
                self._ctmc_predictor.Q = self._fitted_Q

            self._ctmc_predictor.observe_phase(int(phase))

            # Predict probability of entering CHECKPOINT phase
            prob_checkpoint = self._ctmc_predictor.hitting_probability(
                target_phases=[2],  # CHECKPOINT
                horizon_s=self._checkpoint_horizon_s,
            )

            expected_drift = self._ctmc_predictor.predicted_drift_rate(
                self._fitted_r,
                horizon_s=self._checkpoint_horizon_s,
            )

            # MMFQ: compute prob_above_hwm for safety margin.
            # The solver may legitimately fail when the CTMC fit is degenerate
            # (e.g. all drifts the same sign for a transient observation
            # window). Catch only numerical / linear-algebra problems; raise
            # programming bugs as before.
            cache_cap = self._cfg["disk_cache"]["capacity_bytes"]
            hwm = self._cfg["disk_cache"]["hwm_fraction"]
            lwm = self._cfg["disk_cache"]["lwm_fraction"]
            try:
                mmfq = MMFQSolver(
                    self._fitted_Q, self._fitted_r, cache_cap, hwm=hwm, lwm=lwm
                )
                mmfq_result = mmfq.solve()
                prob_above_hwm = mmfq_result.prob_above_hwm
            except (np.linalg.LinAlgError, ValueError) as exc:
                logger.debug("MMFQ solve failed (%s); using safe default.", exc)
                prob_above_hwm = 0.0

            # IOM optimise can fail if telemetry is degenerate (e.g. zero
            # rates produce ρ=0 and an ill-conditioned cost). Fall back to
            # the configured baseline age threshold and log loudly.
            try:
                iom_params = self._build_iom_parameters(telemetry)
                iom = IOM(iom_params)
                tau_opt, _ = iom.optimise(
                    tau_min_s=self._tau_min_s, tau_max_s=self._tau_max_s
                )
                tau_star = float(tau_opt)
            except (ValueError, RuntimeError) as exc:
                logger.warning("IOM optimisation failed (%s); using baseline τ.", exc)
                tau_star = self._baseline_age_threshold

            # Safety margin: if MMFQ predicts high HWM crossing probability,
            # bias tau downward (more aggressive migration)
            if prob_above_hwm > 0.5:
                tau_star *= 0.8

        # ── 5. Clamp tau_star ───────────────────────────────────────────
        clamp_low = self._baseline_age_threshold * self._clamp_min
        clamp_high = self._baseline_age_threshold * self._clamp_max
        tau_star = max(clamp_low, min(tau_star, clamp_high))

        # ── 6. Emergency override for predicted CHECKPOINT ───────────────
        if prob_checkpoint > self._checkpoint_threshold:
            tau_star = min(tau_star, self._emergency_age_s)
            # Also flush tape batches faster
            self._ls.set_max_wait_s(60.0)
        else:
            # Restore default max wait
            default_max_wait = self._cfg["library_server"].get("ls_max_wait_s", 300.0)
            self._ls.set_max_wait_s(default_max_wait)

        # ── 7. Apply to PolicyEngine ─────────────────────────────────────
        self._policy.set_age_threshold(tau_star)
        cycle_period = min(tau_star / 4.0, 300.0)
        self._policy.set_cycle_period(cycle_period)

        # ── 8. Adaptive batch sizing ────────────────────────────────────
        if self._batch_adaptive:
            # Expected bytes written in tau_star/2 seconds at current drift
            expected_bytes = max(0.0, expected_drift) * (tau_star / 2.0)
            batch_bytes = int(
                max(
                    self._batch_min_bytes,
                    min(expected_bytes, self._batch_max_bytes),
                )
            )
            self._ls.set_min_batch_bytes(batch_bytes)

    def _build_telemetry_sample(self, now: float) -> TelemetrySample:
        """Construct TelemetrySample from current ReplicationStats deltas."""
        interval_s = self._interval_s

        # Cumulative counters from stats
        curr_ess_w = float(self._stats.bytes_written_to_cache)
        curr_ess_r = float(self._stats.bytes_read_from_cache)
        curr_tape_w = float(self._stats.bytes_archived)
        curr_tape_r = float(self._stats.bytes_recalled)

        dt = max(interval_s, 1e-9)
        write_rate = max(0.0, (curr_ess_w - self._prev_ess_written) / dt)
        read_rate = max(0.0, (curr_ess_r - self._prev_ess_read) / dt)
        archival_rate = max(0.0, (curr_tape_w - self._prev_tape_written) / dt)
        recall_rate = max(0.0, (curr_tape_r - self._prev_tape_read) / dt)

        # Update previous snapshots
        self._prev_ess_written = curr_ess_w
        self._prev_ess_read = curr_ess_r
        self._prev_tape_written = curr_tape_w
        self._prev_tape_read = curr_tape_r

        cache_cap = self._cfg["disk_cache"]["capacity_bytes"]
        cache_bytes = self._disk.used_bytes
        files_in_cache = sum(
            1
            for f in getattr(self._policy, "_file_registry", {}).values()
            if getattr(f, "state", None) is not None
            and f.state.value in ("REG", "DUL")
        )
        files_on_tape = sum(
            1
            for f in getattr(self._policy, "_file_registry", {}).values()
            if getattr(f, "state", None) is not None
            and f.state.value in ("OFL", "UNM")
        )

        return TelemetrySample(
            sim_time_s=now,
            cache_bytes=float(cache_bytes),
            cache_capacity=float(cache_cap),
            write_rate_bps=write_rate,
            read_rate_bps=read_rate,
            archival_rate_bps=archival_rate,
            recall_rate_bps=recall_rate,
            files_in_cache=files_in_cache,
            files_on_tape=files_on_tape,
            archiver_active=archival_rate > 0,
            interval_s=interval_s,
        )

    def _build_iom_parameters(self, telemetry: TelemetrySample) -> IOMParameters:
        """Build IOMParameters from telemetry and config."""
        cfg = self._cfg
        disk_cfg = cfg["disk_cache"]
        tape_cfg = cfg["tape_drives"]
        pol_cfg = cfg["policy_engine"]

        # Estimate mean file size from working set config
        wl_cfg = cfg["workload"]
        # LogNormal mean: exp(mu + sigma^2/2)
        mu = wl_cfg["file_size_lognormal_mu"]
        sigma = wl_cfg["file_size_lognormal_sigma"]
        mean_file_bytes = float(np.exp(mu + sigma**2 / 2))

        # Disk bandwidth (aggregate interconnect + cache)
        disk_bw = cfg["interconnect"]["bandwidth_bytes_per_s"]

        # Scratch write rate from telemetry
        lambda_scratch = telemetry.write_rate_bps

        # Archival data rate (from telemetry or estimate)
        rho_data = telemetry.archival_rate_bps
        if rho_data <= 0:
            # Fallback: estimate from policy cycle and candidate count
            rho_data = 1e6  # conservative 1 MB/s default

        # Tape bandwidth capacity
        bmax = tape_cfg["n_drives"] * tape_cfg["native_rate_bytes_per_s"]

        # File lifetime estimate (not directly tracked; use 1 hour default)
        mean_file_life_s = wl_cfg.get("initial_file_age_s_mu", 3600.0)
        file_life_cv = 1.0

        # Tape-side parameters used by the τ-dependent P_loss model
        # (mount-amortisation back-pressure + M[X]/G/1 burst surcharge).
        mount_mu_log = float(tape_cfg.get("mount_time_lognormal_mu_s", 3.689))
        mount_sigma_log = float(tape_cfg.get("mount_time_lognormal_sigma_s", 0.3))
        mount_mean_s = float(np.exp(mount_mu_log + 0.5 * mount_sigma_log ** 2))

        return IOMParameters(
            rho_data_bps=rho_data,
            bmax_bps=bmax,
            disk_bw_bps=disk_bw,
            lambda_scratch_bps=lambda_scratch,
            mean_file_bytes=mean_file_bytes,
            cv_iat=1.5,  # from Pareto alpha=1.8
            cv_service=1.0,
            mean_file_life_s=mean_file_life_s,
            file_life_cv=file_life_cv,
            cache_capacity=disk_cfg["capacity_bytes"],
            hwm=pol_cfg.get("hwm_fraction", 0.90),
            lwm=pol_cfg.get("lwm_fraction", 0.80),
            wp=0.4,
            wm=0.3,
            wr=0.3,
            mount_time_s=mount_mean_s,
            n_tape_drives=int(tape_cfg["n_drives"]),
            native_rate_bps=float(tape_cfg["native_rate_bytes_per_s"]),
        )
