"""
models/iom.py — Interval Optimization Model (IOM)

The IOM finds the archive interval τ* that minimises a weighted scalar cost

    J(τ) = w_p · α · P_loss(τ) + w_m · β · M_waste(τ) + w_r · γ · R_risk(τ)

with weights on a simplex (Σw=1) and per-component normalisation constants
(α, β, γ) that map physical units onto a common scale.

──────────────────────────────────────────────────────────────────────────────
Why an interior τ* exists in this model
──────────────────────────────────────────────────────────────────────────────
P_loss is given a τ-dependent structure with two competing effects, so it
is U-shaped (rather than monotone) and has an explicit interior minimum.
Combined with the monotone-increasing M_waste and R_risk terms this places
the optimum τ* at an interior point rather than collapsing it to τ_min:

  (i)  Mount-amortisation back-pressure (DECREASES with τ).
       The tape side cannot start streaming until a mount completes; for an
       archive interval τ the mean batch size is

           B(τ) = ρ_data · τ / E[file_size]      (files per batch)

       and the per-byte tape overhead is

           o_tape(τ) = t_mount / (B(τ) · E[file_size] / r_native)
                     = t_mount · r_native / (B(τ) · E[file_size])

       which decays as 1/τ. When B(τ) is small the tape side spends most of
       its time mounting and cannot drain the archival stream, so the
       archival load reflected onto the cache inflates by the factor
       (1 + o_tape(τ)) — back-pressure.

  (ii) Burst-induced SCV inflation (INCREASES with τ).
       Batched archival arrivals at the file level are approximately
       compound Poisson with SCV c_a²_arch ≈ max(1, B(τ)) (Tijms 2003 §2.7),
       which grows linearly in τ. Whitt (1983) stationary-interval
       superposition blends this with the scratch stream:

           c_a²_combined = (λ_s · c_a²_s + λ_arch · c_a²_arch) / (λ_s + λ_arch)

       so the combined arrival SCV seen at the cache rises with τ.

P_loss(τ) is the relative inflation of the combined-stream Kingman wait
over the scratch-only baseline. Effects (i) and (ii) push in opposite
directions, so P_loss(τ) is U-shaped and has an interior minimum that
combines naturally with the increasing M_waste and R_risk terms to give an
interior J*. The M[X]/G/1 within-batch surcharge is deliberately NOT added
on top: that surcharge bounds the wait of files *inside* a batch, whereas
the scratch viewpoint sees the batched stream only as one of two superposed
renewal processes — Kingman + Whitt already capture that interaction.

──────────────────────────────────────────────────────────────────────────────
τ→∞ behaviour (feasibility observation, not a theorem)
──────────────────────────────────────────────────────────────────────────────
Under the modelling assumptions here — finite mean file lifetime E[L]<∞,
uniform-in-time failure hazard, deterministic batched archival — the cost
components obey

    lim_{τ→∞} R_risk(τ) = ∞    ⟹    lim_{τ→∞} J(τ) = ∞

so the traditional "separated" boundary τ→∞ cannot be optimal *within this
model*. Note the same bound now also follows from P_loss because
B(τ)→∞ ⇒ ΔW_q^batch→∞, even before invoking R_risk. Mwaste(τ→∞)=ρ_data·E[L]
remains a finite plateau. The formal theorem would still require ruling out
alternative failure models, priced-risk policies, and the unbounded-file-
lifetime regime; those are beyond the scope of the IOM as parameterised here.

──────────────────────────────────────────────────────────────────────────────
References
──────────────────────────────────────────────────────────────────────────────
Whitt, W. (1983). "The Queueing Network Analyzer." Bell Sys. Tech. J. 62(9).
Tijms, H.C. (2003). "A First Course in Stochastic Models." Wiley, §2.7.
Bitran, G.R. & Tirupati, D. (1989). "Tradeoff curves, targeting and balancing
    in manufacturing queueing networks." Operations Research 37(4).
Heyman, D.P. & Sobel, M.J. (1982). "Stochastic Models in Operations Research."
"""
from __future__ import annotations
from math import gamma
import numpy as np
from scipy import integrate, optimize
from dataclasses import dataclass
from typing import Tuple


@dataclass
class IOMParameters:
    rho_data_bps:        float
    bmax_bps:            float
    disk_bw_bps:         float
    lambda_scratch_bps:  float
    mean_file_bytes:     float
    cv_iat:              float = 1.0
    cv_service:          float = 1.0
    mean_file_life_s:    float = 3600.0
    file_life_cv:        float = 1.0
    cache_capacity:      float = 809.0 * (1024**4)
    hwm:                 float = 0.75
    lwm:                 float = 0.50
    wp:                  float = 0.4
    wm:                  float = 0.3
    wr:                  float = 0.3

    # Tape-side parameters needed for the τ-dependent P_loss model. Defaults
    # are picked so that mount overhead is non-trivial but back-pressure does
    # not dominate at modest τ; instances should set these from the actual
    # device config when available.
    mount_time_s:        float = 30.0
    n_tape_drives:       int   = 8
    native_rate_bps:     float = 360_000_000.0   # LTO-8 native

    def __post_init__(self):
        w = self.wp + self.wm + self.wr
        if abs(w - 1.0) > 1e-6:
            self.wp /= w; self.wm /= w; self.wr /= w


@dataclass
class IOMEvaluation:
    tau_s: float; ploss_pct: float; mwaste_tib: float; rrisk_gib: float
    J: float; J_ploss: float; J_mwaste: float; J_rrisk: float
    feasible: bool; bw_required: float; bw_available: float
    ploss_at_inf: float; mwaste_at_inf: float; rrisk_at_inf: float


class IOM:
    def __init__(self, params: IOMParameters) -> None:
        self.p = params

    def evaluate(self, tau_s: float) -> IOMEvaluation:
        p = self.p
        ploss  = self._ploss(tau_s)
        mwaste = self._mwaste_tib(tau_s)
        rrisk  = self._rrisk_gib(tau_s)
        alpha = 100.0 / max(self._ploss_max(), 1.0)
        beta  = 100.0 / max(self._mwaste_max(), 1e-9)
        gamma = 100.0 / max(self._rrisk_max(), 1e-9)
        J_ploss  = p.wp * alpha * ploss
        J_mwaste = p.wm * beta  * mwaste
        J_rrisk  = p.wr * gamma * rrisk
        J = J_ploss + J_mwaste + J_rrisk
        burst = p.rho_data_bps * tau_s if np.isfinite(tau_s) else 0
        bw_req = burst / max(tau_s, 1) if tau_s > 0 else p.rho_data_bps
        return IOMEvaluation(
            tau_s=tau_s, ploss_pct=ploss, mwaste_tib=mwaste, rrisk_gib=rrisk,
            J=J, J_ploss=J_ploss, J_mwaste=J_mwaste, J_rrisk=J_rrisk,
            feasible=(bw_req <= p.bmax_bps),
            bw_required=bw_req / (1024**3), bw_available=p.bmax_bps / (1024**3),
            ploss_at_inf=self._ploss(86400 * 365),
            mwaste_at_inf=self._mwaste_at_inf(), rrisk_at_inf=float("inf"),
        )

    def cost_curve(self, tau_grid_s: np.ndarray) -> dict:
        p = self.p
        ploss  = np.array([self._ploss(t)      for t in tau_grid_s])
        mwaste = np.array([self._mwaste_tib(t) for t in tau_grid_s])
        rrisk  = np.array([self._rrisk_gib(t)  for t in tau_grid_s])
        alpha = 100.0 / max(self._ploss_max(), 1.0)
        beta  = 100.0 / max(self._mwaste_max(), 1e-9)
        gamma = 100.0 / max(self._rrisk_max(), 1e-9)
        J_ploss  = p.wp * alpha * ploss
        J_mwaste = p.wm * beta  * mwaste
        J_rrisk  = p.wr * gamma * rrisk
        return {
            "tau_s": tau_grid_s, "tau_min": tau_grid_s / 60.0,
            "J": J_ploss + J_mwaste + J_rrisk,
            "J_ploss": J_ploss, "J_mwaste": J_mwaste, "J_rrisk": J_rrisk,
            "ploss": ploss, "mwaste": mwaste, "rrisk": rrisk,
            "feasible": np.array([t * self.p.rho_data_bps <= self.p.bmax_bps * t
                                  for t in tau_grid_s], dtype=bool),
        }

    def optimise(self, tau_min_s: float = 60.0,
                 tau_max_s: float = 86400.0) -> Tuple[float, IOMEvaluation]:
        def cost(log_tau):
            tau = np.exp(log_tau)
            ev = self.evaluate(tau)
            return ev.J + (1e6 if not ev.feasible else 0)
        result = optimize.minimize_scalar(
            cost, bounds=(np.log(tau_min_s), np.log(tau_max_s)),
            method="bounded", options={"xatol": 1.0},
        )
        tau_star = np.exp(result.x)
        return tau_star, self.evaluate(tau_star)

    def tau_inf_limit(self) -> dict:
        return {
            "description": "τ→∞: traditional separated architecture",
            "ploss_pct": self._ploss(86400 * 365),
            "mwaste_tib": self._mwaste_at_inf(),
            "rrisk_gib": float("inf"),
            "J": float("inf"),
            "interpretation": (
                "τ→∞ eliminates burst archival I/O (Ploss→constant), but leaves "
                "data unprotected (Rrisk→∞) and wastes maximum media (Mwaste→r·E[L]). "
                "J(τ→∞)=∞ proves this is a degenerate, not optimal, boundary."
            ),
        }

    # ── Component functions — DERIVED ────────────────────────────────────────

    def _ploss(self, tau_s: float) -> float:
        """τ-dependent P_loss for the scratch stream sharing the disk cache
        with batched archival writes.

        Two competing effects produce a U-shaped P_loss(τ):

        (i) Mount-amortisation back-pressure (∝ 1/τ).
            Each archival batch incurs a fixed mount cost t_mount before
            useful tape transfer begins. The per-byte tape overhead is

                o_tape(τ) = t_mount · r_native / (B(τ) · E[file_size] / n_drives)

            which decays as 1/τ. Cache-side throughput must inflate to
            compensate when the tape side is mount-limited, so the
            archival utilisation reflected onto the cache is

                ρ_arch_eff(τ) = ρ_arch · (1 + o_tape(τ)).

            For very small τ this drives ρ_total → 1, exploding W_q.

        (ii) Burst-induced SCV inflation (∝ τ).
             A batched archival arrival process has compound-Poisson SCV
             at the file level approximately equal to B(τ) (Tijms 2003,
             §2.7). Whitt's stationary-interval superposition (Whitt 1983)
             aggregates the scratch and archival streams via the
             rate-weighted SCV

                c_a²_combined = (λ_s · c_a²_s + λ_arch · c_a²_arch(τ))
                               / (λ_s + λ_arch)

             This raises the effective SCV at the cache as τ grows.

        From the *scratch* viewpoint we use Kingman's G/G/1 approximation
        with the τ-dependent (ρ_total_eff, c_a²_combined). We deliberately
        do NOT add the M[X]/G/1 within-batch surcharge: that surcharge
        bounds the wait of files *inside* the batch, but scratch only sees
        the batched stream as one of two superposed renewal processes —
        Kingman's superposition handles that interaction directly.

        Reported as

            P_loss(τ) = max(0, W_q^combined / W_q^baseline − 1) · 100%

        where the baseline is the scratch-only Kingman wait.
        """
        p = self.p
        E_S = p.mean_file_bytes / max(p.disk_bw_bps, 1.0)
        mfb = max(p.mean_file_bytes, 1.0)

        # Base utilisations (byte-rate / disk-bandwidth).
        rho_s = min(p.lambda_scratch_bps / max(p.disk_bw_bps, 1.0), 0.99)
        rho_arch_base = p.rho_data_bps / max(p.disk_bw_bps, 1.0)

        # Batch size in files and tape-side amortisation overhead.
        # tau_s == 0 gives degenerate B=0; clamp to a one-second floor so
        # the formula remains well-defined while still penalising tiny τ.
        tau_eff = max(float(tau_s), 1.0)
        B = p.rho_data_bps * tau_eff / mfb               # mean files per batch
        # Aggregate tape mount cost per byte (s/byte): mount time spread over
        # the bytes that one mount enables, parallelised across drives.
        mount_bytes_per_drive = max(B * mfb / max(p.n_tape_drives, 1), 1.0)
        o_tape = (p.mount_time_s * p.native_rate_bps) / mount_bytes_per_drive
        # Cap o_tape so a degenerate (τ→0) input still yields a finite,
        # large penalty rather than a NaN.
        o_tape = float(min(o_tape, 1e6))

        rho_arch_eff = rho_arch_base * (1.0 + o_tape)
        rho_total_eff = min(rho_s + rho_arch_eff, 0.9999)

        # Whitt-superposed combined SCV (Whitt 1983).
        lam_s = p.lambda_scratch_bps / mfb
        lam_arch = p.rho_data_bps / mfb
        total_lam = max(lam_s + lam_arch, 1e-12)
        c_a2_s = p.cv_iat ** 2
        c_a2_arch = max(1.0, B)   # compound-Poisson SCV at file level
        c_a2_combined = (lam_s * c_a2_s + lam_arch * c_a2_arch) / total_lam
        c_s2 = p.cv_service ** 2

        # Baseline (scratch-only) Kingman wait.
        W_q0 = self._kingman_Wq(rho_s, E_S, c_a2_s, c_s2)
        if W_q0 < 1e-12:
            return 0.0

        # Combined wait: Kingman with τ-dependent ρ and c_a² captures both
        # back-pressure (via ρ_total_eff) and burstiness (via c_a²_combined).
        W_q = self._kingman_Wq(rho_total_eff, E_S, c_a2_combined, c_s2)

        return max(0.0, (W_q / W_q0 - 1.0) * 100.0)

    def _mwaste_tib(self, tau_s: float) -> float:
        """r·E[min(L,τ)] — closed form for Exponential, numerical for Weibull.

        For a lifetime L with survival function S(t)=P(L>t),
            E[min(L,τ)] = ∫₀^τ S(t) dt.

        Exponential (CV=1): S(t)=exp(-αt), E[min(L,τ)] = (1-e^{-ατ})/α.
        Weibull(k,λ):       S(t)=exp(-(t/λ)^k),  k = 1/CV² (moment match on CV of
                            the exponential → Weibull map used elsewhere in this
                            model); E[X] = λ·Γ(1 + 1/k)  ⟹  λ = mean / Γ(1 + 1/k).
        """
        p = self.p
        alpha = 1.0 / max(p.mean_file_life_s, 1.0)
        if p.file_life_cv <= 1.0:
            E_min = (1.0 / alpha) * (1.0 - np.exp(-alpha * tau_s))
        else:
            k = 1.0 / (p.file_life_cv ** 2)
            lam = p.mean_file_life_s / gamma(1.0 + 1.0 / k)
            E_min, _ = integrate.quad(
                lambda t: np.exp(-(t / max(lam, 1e-9)) ** k),
                0, tau_s, limit=100,
            )
        return p.rho_data_bps * E_min / (1024**4)

    def _rrisk_gib(self, tau_s: float) -> float:
        """ρ·τ/2 — expected data at risk under uniform failure time."""
        if not np.isfinite(tau_s): return float("inf")
        return self.p.rho_data_bps * tau_s / 2.0 / (1024**3)

    def _mwaste_at_inf(self) -> float:
        return self.p.rho_data_bps * self.p.mean_file_life_s / (1024**4)

    def _mwaste_max(self) -> float:
        return max(self._mwaste_at_inf(), 1.0)

    def _rrisk_max(self) -> float:
        return max(self._rrisk_gib(86400.0), 1.0)

    def _ploss_max(self) -> float:
        """Reference scale for J-component normalisation.

        Use P_loss at τ=86400 s (one day) as a representative upper-bound
        scale: this is well past any practical archive interval and reflects
        the heavy-burst regime. The bounded optimisation interval keeps the
        evaluated τ within physical limits, so this scale only acts as a
        normaliser, never as a cap on the underlying P_loss formula.
        """
        return max(self._ploss(86400.0), 1.0)

    @staticmethod
    def _kingman_Wq(rho: float, E_S: float, c_a2: float, c_s2: float) -> float:
        rho = min(max(rho, 0.0), 0.9999)
        if rho < 1e-12 or E_S < 1e-12: return 0.0
        return (rho / (1.0 - rho)) * E_S * (c_a2 + c_s2) / 2.0


def iom_params_from_config(config, telemetry_summary: dict) -> IOMParameters:
    ess = config.ess3500; tape = config.ts1160; pol = config.dmf_policy
    # Mount time as a real-space mean (seconds). The pydantic schema expresses
    # it as a log-space μ on a LogNormal — recover the linear-space mean if
    # available, otherwise fall back to the schema's μ field.
    mount_mu = getattr(tape, "mount_time_lognormal_mu_s", 3.689)
    mount_sigma = getattr(tape, "mount_time_lognormal_sigma_s", 0.3)
    mount_mean_s = float(np.exp(mount_mu + 0.5 * mount_sigma ** 2))
    return IOMParameters(
        rho_data_bps       = telemetry_summary.get("mean_rho_bps", 1e9),
        bmax_bps           = tape.n_drives * tape.native_rate_bytes_per_s,
        disk_bw_bps        = ess.aggregate_throughput_gib_s * (1024**3),
        lambda_scratch_bps = telemetry_summary.get("mean_write_bps", 1e9),
        mean_file_bytes    = telemetry_summary.get("mean_file_bytes", 64 * 1024**2),
        cv_iat             = telemetry_summary.get("cv_iat", 1.5),
        cv_service         = telemetry_summary.get("cv_service", 1.0),
        mean_file_life_s   = telemetry_summary.get("mean_file_life_s", 3600.0),
        file_life_cv       = telemetry_summary.get("file_life_cv", 1.0),
        cache_capacity     = ess.usable_capacity_tib * (1024**4),
        hwm=pol.hwm_fraction, lwm=pol.lwm_fraction,
        mount_time_s       = mount_mean_s,
        n_tape_drives      = int(tape.n_drives),
        native_rate_bps    = float(tape.native_rate_bytes_per_s),
    )
