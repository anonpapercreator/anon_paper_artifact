"""
components/hwm_advisor.py — Ablation arm B: reactive watermark advisor.

Subclasses the production AdaptivePolicyAdvisor and overrides ONLY _advise.
It uses the SAME actuators (age threshold, cycle period, LS max-wait) as the
full advisor, driven purely by the OBSERVED disk cache-fill fraction against
the high-water mark — with NO CTMC fit, NO MMFQ solve, NO IOM optimisation,
and NO phase prediction. This isolates the marginal value of the MMFQ/IOM/CTMC
machinery: the full arm (C) minus this reactive arm (B) is exactly that value.

Because __init__, telemetry construction, and the SimPy run-loop are inherited
unchanged, arm B stays automatically in sync with the production advisor; the
only behavioural difference is the decision rule in _advise.

Selected via config: adaptive_policy.enabled = true, adaptive_policy.advisor =
"hwm" (the default "full" runs the production MMFQ/IOM/CTMC advisor).
"""
from __future__ import annotations
from components.adaptive_policy import AdaptivePolicyAdvisor


class HWMOnlyAdvisor(AdaptivePolicyAdvisor):
    """Reactive bang-bang watermark controller (no predictive model).

    Standard two-watermark hysteresis, the same HWM/LWM pair the free-space
    path itself is specified with: ENGAGE (drain hard) when the cache hits its
    high-water mark; DISENGAGE (relax to baseline) only when fill falls below
    the low-water mark. Two details make the trigger observable in practice:

    * A watermark hit is detected as new space-release events since the last
      look (the signal an operator reads from the dmfsfree log), not as
      "instantaneous fill >= HWM at the sampling instant" — the release path
      is event-driven and knocks fill back toward the LWM the moment the HWM
      is crossed, so a sampled instantaneous fill almost never catches it.
    * The instantaneous test is kept as a fallback for the saturated case
      where releases fire but cannot free enough (no dual-state bytes),
      pinning fill above the mark.

    Without the hysteresis the controller is edge-triggered and drains for a
    single advisor interval per release burst — indistinguishable from static
    under sustained pressure, where fill oscillates between the marks
    indefinitely.
    """

    _prev_space_releases: int = 0
    _engaged: bool = False

    def _advise(self) -> None:
        telemetry = self._build_telemetry_sample(self._env.now)
        hwm = self._cfg["disk_cache"]["hwm_fraction"]
        lwm = self._cfg["disk_cache"]["lwm_fraction"]
        default_max_wait = self._cfg["library_server"].get("ls_max_wait_s", 300.0)

        releases = int(getattr(self._stats, "n_space_releases", 0))
        crossed_since_last = releases > self._prev_space_releases
        self._prev_space_releases = releases

        if crossed_since_last or telemetry.cache_fill_fraction >= hwm:
            self._engaged = True
        elif telemetry.cache_fill_fraction < lwm:
            self._engaged = False

        if self._engaged:
            tau_star = self._emergency_age_s          # observed pressure -> drain hard
            self._ls.set_max_wait_s(60.0)
        else:
            tau_star = self._baseline_age_threshold   # relax to baseline
            self._ls.set_max_wait_s(default_max_wait)

        self._policy.set_age_threshold(tau_star)
        self._policy.set_cycle_period(min(tau_star / 4.0, 300.0))
