"""
cli/pages/7_probabilistic_model.py
Two-Tier Probabilistic Model — DESCASSI

Argues (under the explicit modelling assumptions of the IOM): the
traditional separation of scratch and archive is a degenerate boundary
condition τ→∞ of a continuous policy optimisation problem whose
dynamics are governed by a Markov-Modulated Fluid Queue.

Tab 1: MMFQ Steady-State  — stationary cache fill CDF, P(>HWM), E[fill]
Tab 2: CTMC Transient     — P(phase=j at t+h), expected drift rate
Tab 3: IOM Optimisation — J(τ) curve, τ*, component breakdown
Tab 4: τ→∞ Limit (feasibility observation) — numerical + analytical
Tab 5: Multi-Tenant Noisy-Neighbour — per-tenant Gini fairness summary
"""
import streamlit as st
import numpy as np
import sys, os
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Probabilistic Model — DESCASSI", page_icon="🔬", layout="wide")

# ── Lazy imports ─────────────────────────────────────────────────────────────
try:
    from models.phase_classifier import PHASE_NAMES, Phase
    from models.mmfq import MMFQSolver
    from models.ctmc_predictor import CTMCPredictor, transient_trajectory, stationary_from_Q
    from models.iom import IOM, IOMParameters
    MODELS_OK = True
except ImportError as e:
    st.error(f"Model import error: {e}")
    MODELS_OK = False
    st.stop()

PHASE_COLORS = {"IDLE": "#6baed6", "COMPUTE": "#74c476",
                "CHECKPOINT": "#fd8d3c", "RECALL": "#9e9ac8"}


def _download_png(fig: go.Figure, label: str, filename: str,
                  width: int = 1400, height: int = 600, scale: int = 3) -> None:
    """Render fig server-side via Kaleido at print resolution and offer a download button.

    scale=3 → effective ~288 DPI (3× the base pixel dimensions).
    A deep copy is used so the interactive chart displayed in the browser is untouched.
    """
    try:
        ef = copy.deepcopy(fig)
        ef.update_layout(
            font=dict(size=13, family="Arial"),
            title_font_size=16,
            width=width,
            height=height,
        )
        img_bytes = ef.to_image(format="png", width=width, height=height, scale=scale)
        st.download_button(
            label=f"⬇ {label}",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
            help=f"High-resolution PNG — {width * scale} × {height * scale} px (~{scale * 96} DPI)",
        )
    except Exception as e:
        st.warning(
            f"PNG export unavailable. Install kaleido: `pip install kaleido`\n\n"
            f"Error: {e}"
        )

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🔬 Two-Tier Probabilistic Model")
st.markdown("""
**Claim** *(under the explicit modelling assumptions of the IOM below):*
the traditional separation of scratch and archive is a degenerate boundary
condition τ→∞ of a continuous policy optimisation problem whose dynamics
are governed by a Markov-Modulated Fluid Queue (MMFQ).

The claim is a feasibility observation, not a theorem in the strict
sense — a formal theorem would require ruling out priced-risk failure
models, infinite-lifetime regimes, and mount-amortisation regimes that
fall outside this IOM's assumptions. Within those assumptions, both
the U-shape of P_loss (mount-amortisation back-pressure ∝ 1/τ; burst-
SCV inflation ∝ τ) and the unboundedness of R_risk drive J(τ) → ∞ as
τ → ∞, so τ* is interior.
""")
st.markdown("---")

# ── Sidebar: parameters ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Parameters")

    st.subheader("System")
    ess_bw    = st.number_input("ESS bandwidth (GiB/s)",   value=65.0, min_value=1.0,   step=1.0)
    tape_bw   = st.number_input("Tape bandwidth (GiB/s)",  value=6.4,  min_value=0.1,   step=0.1)
    cache_tib = st.number_input("Cache capacity (TiB)",    value=809.0, min_value=1.0,  step=10.0)
    hwm = st.slider("HWM (%)", 50, 95, 75) / 100
    lwm = st.slider("LWM (%)", 20, 80, 50) / 100

    st.subheader("Workload")
    rho_gbs    = st.number_input("Data generation (GiB/s)", value=0.5,  min_value=0.01, step=0.1)
    lambda_gbs = st.number_input("Scratch rate (GiB/s)",    value=30.0, min_value=0.1,  step=1.0)
    mean_gb    = st.number_input("Mean file size (GiB)",     value=0.064, min_value=0.001, step=0.001, format="%.3f")
    cv_iat     = st.slider("CV of scratch IAT", 0.5, 3.0, 1.5, 0.1)
    life_h     = st.number_input("Mean file lifetime (h)", value=1.0, min_value=0.1, step=0.1)
    life_cv    = st.slider("CV of file lifetime", 0.5, 3.0, 1.2, 0.1)

    st.subheader("Phase Transition Rates (1/s)")
    q01 = st.number_input("IDLE→COMPUTE",       value=0.005, format="%.4f", step=0.001)
    q12 = st.number_input("COMPUTE→CHECKPOINT", value=0.003, format="%.4f", step=0.001)
    q21 = st.number_input("CHECKPOINT→COMPUTE", value=0.020, format="%.4f", step=0.001)
    q13 = st.number_input("COMPUTE→RECALL",     value=0.001, format="%.4f", step=0.001)
    q31 = st.number_input("RECALL→COMPUTE",     value=0.010, format="%.4f", step=0.001)
    q10 = st.number_input("COMPUTE→IDLE",       value=0.002, format="%.4f", step=0.001)

    st.subheader("IOM Weights")
    wp_r = st.slider("Performance (wp)", 0.0, 1.0, 0.4, 0.05)
    wm_r = st.slider("Media (wm)",       0.0, 1.0, 0.3, 0.05)
    wr_r = st.slider("Risk (wr)",         0.0, 1.0, 0.3, 0.05)
    ws   = wp_r + wm_r + wr_r
    wp, wm, wr = (wp_r/ws, wm_r/ws, wr_r/ws) if ws > 0 else (1/3, 1/3, 1/3)
    st.caption(f"Normalised: wp={wp:.2f} wm={wm:.2f} wr={wr:.2f}")

    st.subheader("IOM — Tape-side parameters")
    mount_s   = st.number_input("Mean tape mount time (s)", value=30.0, min_value=1.0, step=1.0,
                                help="Real-space mean of LogNormal mount time. Drives the "
                                     "mount-amortisation back-pressure component of P_loss.")
    n_drives  = st.number_input("# tape drives",            value=8,    min_value=1,   step=1,
                                help="Aggregate tape concurrency for batch streaming.")
    nat_mbps  = st.number_input("Native tape rate (MB/s)",  value=360.0, min_value=10.0, step=10.0,
                                help="Per-drive native bytes/s — TS1160 = 400, LTO-8 = 360.")

# ── Build Q and r ─────────────────────────────────────────────────────────────
def build_Q():
    Q = np.zeros((4, 4))
    Q[0,1]=q01; Q[1,2]=q12; Q[2,1]=q21; Q[1,3]=q13; Q[3,1]=q31; Q[1,0]=q10
    for i in range(4): Q[i,i] = -Q[i,:].sum()
    return Q

def build_r():
    gb = 1024**3
    return np.array([
        -rho_gbs*gb*0.5,
         lambda_gbs*gb*0.1 - rho_gbs*gb,
         lambda_gbs*gb*0.6,
         tape_bw*gb*0.5,
    ])

Q = build_Q()
r = build_r()
C = cache_tib * 1024**4

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tier 1 — MMFQ Steady-State",
    "⏱️ Tier 2 — CTMC Transient",
    "⚙️ IOM Optimisation",
    "∞  τ→∞ Limit (feasibility)",
    "👥 Multi-Tenant Noisy-Neighbour",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MMFQ
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Tier 1 — MMFQ: Stationary Cache Fill Distribution")
    st.markdown(r"""
    The cache fill X(t)∈[0,C] is a fluid driven by CTMC phase φ(t). The stationary
    density satisfies **f′(x) = f(x) Q R⁻¹** (Ahn & Ramaswami 2005).
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Generator matrix Q**")
        names = list(PHASE_NAMES.values())
        st.dataframe(pd.DataFrame(Q.round(5), index=names, columns=names))
    with c2:
        pi = stationary_from_Q(Q)
        st.markdown("**Per-phase drift rates & stationary distribution**")
        st.dataframe(pd.DataFrame({
            "Phase": names,
            "r (GiB/s)": (r / 1024**3).round(3),
            "π": pi.round(3),
            "Direction": ["Fill ↑" if ri > 0 else "Drain ↓" if ri < 0 else "=" for ri in r],
        }))

    with st.spinner("Solving MMFQ…"):
        solver = MMFQSolver(Q, r, C, hwm=hwm, lwm=lwm)
        res = solver.solve(n_grid=200)

    if res.solver_ok:
        st.success(f"Solver: {res.solver_message}")
    else:
        st.warning(f"Solver: {res.solver_message}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.x_grid*100, y=res.cdf,
                             line=dict(color="#2c7bb6", width=2), name="F(x)"))
    fig.add_vline(x=hwm*100, line_color="red",    line_dash="dash",
                  annotation_text=f"HWM {hwm*100:.0f}%")
    fig.add_vline(x=lwm*100, line_color="orange", line_dash="dash",
                  annotation_text=f"LWM {lwm*100:.0f}%")
    fig.update_layout(title="Stationary CDF: F(x) = P(cache fill ≤ x)",
                      xaxis_title="Cache fill (%)", yaxis_title="Probability", height=340)
    st.plotly_chart(fig, use_container_width=True)
    _download_png(fig, "Download high-res PNG", "mmfq_stationary_cdf.png",
                  width=1400, height=600)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(cache > HWM)", f"{res.prob_above_hwm:.1%}")
    m2.metric("P(cache > LWM)", f"{res.prob_above_lwm:.1%}")
    m3.metric("E[cache fill]",  f"{res.expected_fill:.1%}")
    m4.metric("E[T→HWM]",
              f"{res.mfpt_to_hwm/60:.1f} min" if np.isfinite(res.mfpt_to_hwm) else "∞")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CTMC Transient
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Tier 2 — CTMC Transient: P(phase=j at t+h | current state)")
    st.markdown("**P(h) = e_i · exp(Q·h)** — enables dynamic τ*(t) updates each archiver cycle.")

    col1, col2 = st.columns(2)
    with col1:
        init = st.selectbox("Starting phase", range(4),
                            format_func=lambda i: PHASE_NAMES[Phase(i)], index=1)
    with col2:
        horizon_h = st.slider("Prediction horizon (hours)", 0.5, 8.0, 2.0, 0.5)

    traj = transient_trajectory(Q, init, t_max_s=horizon_h*3600, n_points=120)

    fig2 = go.Figure()
    for i, name in PHASE_NAMES.items():
        fig2.add_trace(go.Scatter(
            x=traj["t_grid_min"], y=traj["P_t"][:, int(i)],
            name=name, stackgroup="one", fillcolor=PHASE_COLORS[name],
            line=dict(color=PHASE_COLORS[name]),
        ))
    fig2.update_layout(
        title=f"P(phase=j at t+h | start in {PHASE_NAMES[Phase(init)]})",
        xaxis_title="Horizon (minutes)", yaxis_title="Probability",
        yaxis=dict(range=[0,1]), height=350, legend=dict(orientation="h"),
    )
    st.plotly_chart(fig2, use_container_width=True)
    _download_png(fig2, "Download high-res PNG", "ctmc_phase_probabilities.png",
                  width=1400, height=650)

    pred = CTMCPredictor(Q, list(PHASE_NAMES.values()))
    pred.observe_phase(init)
    h_grid = np.linspace(0, horizon_h*3600, 80)
    drifts = np.array([pred.predicted_drift_rate(r, h) for h in h_grid])

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=h_grid/60, y=drifts/1024**3,
                              line=dict(color="#e6550d", width=2),
                              fill="tozeroy", fillcolor="rgba(230,85,13,0.12)"))
    fig3.add_hline(y=0, line_color="black", line_dash="dot")
    fig3.update_layout(title="E[r(t+h)] — predicted net cache drift (GiB/s)",
                       xaxis_title="Horizon (minutes)",
                       yaxis_title="GiB/s (+ = filling)", height=280)
    st.plotly_chart(fig3, use_container_width=True)
    _download_png(fig3, "Download high-res PNG", "ctmc_drift_rate.png",
                  width=1400, height=500)

    summary = pred.phase_summary_at_horizon(r, horizon_h*3600)
    st.info(f"At t+{horizon_h:.1f}h: **{summary['most_likely_phase']}** most likely "
            f"| drift = **{summary['expected_drift_gbs']:.3f} GiB/s** "
            f"| P(CHECKPOINT) = **{summary['prob_checkpoint']:.1%}**")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — IOM
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### IOM — Interval Optimization Model with τ-dependent P_loss")
    st.markdown("""
| Term | Form |
|---|---|
| Ploss(τ) | **U-shaped**: Whitt-superposed Kingman with mount-amortisation back-pressure ρ_arch_eff(τ) ∝ 1/τ + compound-Poisson burst SCV c_a²_arch(τ) ≈ B(τ) (Tijms 2003 §2.7) |
| Mwaste(τ) | r·E[min(L,τ)] — closed form for Exp(L), numerical for Weibull |
| Rrisk(τ) | ρτ/2 — uniform failure-time hazard |
| BW constraint | Peak burst ≤ Bmax |
| Weights | Simplex-normalised |
| τ→∞ behaviour | **Feasibility observation** — both Ploss and Rrisk drive J→∞; framed as observation, not theorem |
| Interior τ* | Genuinely interior — U-shape of Ploss + monotone Mwaste / Rrisk |

**Why the optimum is interior.** P_loss is given two competing
τ-dependent effects, so it is U-shaped rather than monotone; combined with
the monotone-increasing M_waste and R_risk this yields a genuine interior
τ* rather than collapsing to τ_min. The two effects are:

- **Mount-amortisation back-pressure ∝ 1/τ.** For batch size B(τ) =
  ρ_data·τ / E[file], the per-byte tape mount overhead is
  o_tape(τ) = t_mount · r_native / (B(τ) · E[file] / n_drives).
  For tiny τ the tape side is mount-limited, so the archival utilisation
  reflected onto the cache inflates as ρ_arch_eff(τ) = ρ_arch · (1 + o_tape).
- **Compound-Poisson burst SCV ∝ τ.** Batched archival arrivals at the
  file level have c_a²_arch ≈ B(τ) (Tijms 2003 §2.7). Whitt's stationary-
  interval superposition (Whitt 1983) blends this with the scratch SCV
  c_a²_s on the rate-weighted scale.

Sum is U-shaped, with an interior minimum that combines with the linear
R_risk and saturating M_waste to give a genuine interior τ*. The
M[X]/G/1 within-batch surcharge is **not** added on top: that surcharge
governs files inside the batch waiting on each other, but the scratch
viewpoint sees the batched stream only as one of two superposed
renewals — Kingman + Whitt already handle that interaction directly.

**References.** Whitt (1983) BSTJ 62(9); Tijms (2003) *A First Course in
Stochastic Models* §2.7; Bitran & Tirupati (1989) *Operations Research*
37(4); Heyman & Sobel (1982) *Stochastic Models in OR*.
    """)

    params = IOMParameters(
        rho_data_bps=rho_gbs*1024**3, bmax_bps=tape_bw*1024**3,
        disk_bw_bps=ess_bw*1024**3, lambda_scratch_bps=lambda_gbs*1024**3,
        mean_file_bytes=mean_gb*1024**3, cv_iat=cv_iat, cv_service=1.0,
        mean_file_life_s=life_h*3600, file_life_cv=life_cv,
        cache_capacity=C, hwm=hwm, lwm=lwm, wp=wp, wm=wm, wr=wr,
        mount_time_s=float(mount_s), n_tape_drives=int(n_drives),
        native_rate_bps=float(nat_mbps) * 1e6,
    )
    iom = IOM(params)
    tau_grid = np.logspace(np.log10(60), np.log10(86400), 200)

    with st.spinner("Computing J(τ)…"):
        curve = iom.cost_curve(tau_grid)
        tau_star, ev_star = iom.optimise(60, 86400)

    fig4 = make_subplots(rows=2, cols=2, subplot_titles=[
        "Total Cost J(τ)", "Ploss(τ) — Performance Loss (%)",
        "Mwaste(τ) — Excess Media (TiB)", "Rrisk(τ) — Data at Risk (GiB)",
    ])
    tmin = curve["tau_s"] / 60
    colors = {"J": "#2c7bb6", "ploss": "#d7191c", "mwaste": "#fdae61", "rrisk": "#9e9ac8"}
    fig4.add_trace(go.Scatter(x=tmin, y=curve["J"],      line=dict(color=colors["J"],     width=2)), row=1, col=1)
    fig4.add_trace(go.Scatter(x=tmin, y=curve["ploss"],  line=dict(color=colors["ploss"], width=2)), row=1, col=2)
    fig4.add_trace(go.Scatter(x=tmin, y=curve["mwaste"], line=dict(color=colors["mwaste"],width=2)), row=2, col=1)
    fig4.add_trace(go.Scatter(x=tmin, y=curve["rrisk"],  line=dict(color=colors["rrisk"], width=2)), row=2, col=2)
    for row, col in [(1,1),(1,2),(2,1),(2,2)]:
        fig4.add_vline(x=tau_star/60, line_color="green", line_dash="dash", row=row, col=col)
    fig4.update_xaxes(type="log", title_text="τ (minutes)")
    fig4.update_layout(height=520, showlegend=False,
                       title=f"IOM — τ* = {tau_star/60:.0f} min (green line)")
    st.plotly_chart(fig4, use_container_width=True)
    _download_png(fig4, "Download high-res PNG", "iom_cost_curves.png",
                  width=1400, height=900)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("τ*",          f"{tau_star/60:.0f} min")
    m2.metric("J(τ*)",       f"{ev_star.J:.2f}")
    m3.metric("Ploss(τ*)",   f"{ev_star.ploss_pct:.1f}%")
    m4.metric("Rrisk(τ*)",   f"{ev_star.rrisk_gib:.1f} GiB")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — τ→∞ Limit (feasibility observation)
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### τ→∞ Limit — Traditional Separation as a Degenerate Boundary")

    st.markdown(r"""
#### Statement (feasibility observation, not a theorem)

*Under the modelling assumptions of the IOM — finite mean file lifetime
E[L] < ∞, uniform-in-time failure hazard, deterministic batched archival,
Whitt-superposed two-stream Kingman approximation at the cache — the cost
function*

$$J(\tau) = w_p\,\alpha\,P_{loss}(\tau) + w_m\,\beta\,M_{waste}(\tau) + w_r\,\gamma\,R_{risk}(\tau)$$

*satisfies*

$$\lim_{\tau \to \infty} J(\tau) = \infty$$

*driven by **two** divergent components:*

1. **R_risk(τ) = ρ·τ/2 → ∞** — cumulative unprotected data grows linearly.
2. **P_loss(τ) → ∞** — batch size B(τ) = ρ·τ / E[file] grows linearly in
   τ; the compound-Poisson archival SCV c_a²_arch ≈ B(τ) grows with it; the
   Whitt-superposed Kingman wait at the cache inflates without bound. This
   τ-dependent SCV is what gives P_loss its U-shape and the interior τ*.

*M_waste(τ) → ρ·E[L] saturates and does not contribute to divergence;
it is included for completeness.*

*The degenerate boundary τ → ∞ corresponds to the traditional separated
scratch / archive architecture. The observation says: that architecture
is not interior-optimal **within this model**. The qualifier matters —
a formal theorem would require excluding priced-risk failure models,
unbounded-lifetime regimes, and alternative back-pressure formulations
that fall outside the IOM's assumptions.*
    """)

    # Numerical verification over wide range
    tau_proof = np.logspace(1, 6, 300)
    params_p = IOMParameters(
        rho_data_bps=rho_gbs*1024**3, bmax_bps=tape_bw*1024**3,
        disk_bw_bps=ess_bw*1024**3, lambda_scratch_bps=lambda_gbs*1024**3,
        mean_file_bytes=mean_gb*1024**3, cv_iat=cv_iat, cv_service=1.0,
        mean_file_life_s=life_h*3600, file_life_cv=life_cv,
        cache_capacity=C, hwm=hwm, lwm=lwm, wp=wp, wm=wm, wr=wr,
        mount_time_s=float(mount_s), n_tape_drives=int(n_drives),
        native_rate_bps=float(nat_mbps) * 1e6,
    )
    iom_p = IOM(params_p)
    curve_p = iom_p.cost_curve(tau_proof)
    tau_star_p, ev_star_p = iom_p.optimise(60, 86400)
    lim = iom_p.tau_inf_limit()

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=tau_proof/3600, y=np.minimum(curve_p["J"], curve_p["J"].max()*2),
                              name="J(τ) → ∞", line=dict(color="#2c7bb6", width=3)))
    fig5.add_trace(go.Scatter(x=tau_proof/3600, y=curve_p["J_ploss"],
                              name="Ploss term → ∞ (burst-SCV path)", line=dict(color="#d7191c", width=1.5, dash="dash")))
    fig5.add_trace(go.Scatter(x=tau_proof/3600, y=curve_p["J_mwaste"],
                              name="Mwaste term → r·E[L] (saturates)", line=dict(color="#fdae61", width=1.5, dash="dash")))
    fig5.add_trace(go.Scatter(x=tau_proof/3600, y=np.minimum(curve_p["J_rrisk"], curve_p["J"].max()*1.5),
                              name="Rrisk term → ∞ (linear)", line=dict(color="#9e9ac8", width=1.5, dash="dash")))
    fig5.add_vline(x=tau_star_p/3600, line_color="green", line_width=2,
                   annotation_text=f"τ*={tau_star_p/60:.0f}min", annotation_position="top")
    fig5.add_vrect(x0=12, x1=tau_proof[-1]/3600, fillcolor="red", opacity=0.04,
                   annotation_text="Traditional separation regime (τ→∞)", annotation_position="top left")
    fig5.update_layout(
        title="J(τ)→∞ as τ→∞: driven by both P_loss (burst-SCV) AND R_risk (linear)",
        xaxis_title="τ (hours)", xaxis_type="log",
        yaxis_title="Weighted cost (dimensionless)", height=430,
        legend=dict(orientation="h", yanchor="top", y=-0.15),
    )
    st.plotly_chart(fig5, use_container_width=True)
    _download_png(fig5, "Download high-res PNG", "tau_inf_limit_observation.png",
                  width=1400, height=700)

    st.info(
        f"**τ → ∞ degenerate boundary** *(numerical evaluation at τ = 1 year)*:  "
        f"Ploss → **{lim['ploss_pct']:.1f}%** (diverges via the burst-SCV path),  "
        f"Mwaste → **{lim['mwaste_tib']:.2f} TiB** (saturating maximum),  "
        f"Rrisk → **∞** (linear),  J → **∞**.\n\n"
        f"_{lim['interpretation']}_"
    )

    st.markdown("""
#### Three-component sketch (under the model's assumptions)

1. **Analytically:** R_risk(τ) = ρτ/2 → ∞, so J(τ) → ∞ for any w_r > 0.

2. **Also analytically:** B(τ) = ρτ/E[file] → ∞, so c_a²_arch(τ) → ∞,
   so the Whitt-superposed Kingman wait at the cache → ∞, so P_loss(τ) → ∞
   for any w_p > 0.

3. **Via MMFQ:** as τ → ∞, the archiver fires less frequently, the cache
   fill distribution shifts right (P(X > HWM) increases), and the system
   enters a regime of chronic emergency migration — more disruptive than
   the controlled periodic policy that minimises J(τ*).

The interior τ* therefore exists not because of a single dominant term
but because three out of three τ → ∞ cost components either diverge
(P_loss, R_risk) or hit a non-trivial finite floor (M_waste). *That's a
feasibility observation about this model, not a theorem about HSM
systems in general.* The framing is deliberately conservative because
the thesis intends to defend the IOM as a useful operating-point
selector, not as a universal optimality theorem.
    """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Multi-Tenant Noisy-Neighbour
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    from workload.multi_tenant import (
        MultiTenantWorkloadGenerator,
        gini_coefficient,
    )
    import simpy

    st.markdown("### Multi-Tenant Noisy-Neighbour Workload")
    st.markdown("""
A real DMF-fronted Lustre / SpectrumScale cache serves hundreds–thousands
of concurrent tenants. The single-stream generator on Tabs 1–4 cannot
reproduce the two effects that dominate operational HSM tail latency:

1. **Aggregation** — superposing many independent renewal streams converges
   slowly toward Poisson; the leftover heavy-tail / autocorrelation is
   exactly what a single Pareto stream over- or under-states depending
   on its tuning.
2. **Noisy-neighbour bursts** — a tenant running checkpoint-restart at
   the end of every compute step produces *clustered* bursts unrelated
   to the long-run Pareto background. These bursts inflate P99 / P99.9
   recall latency for *every other* tenant sharing the cache.

Below: spawn N concurrent tenants, dial the noise level β, and observe
how the per-tenant P95 distribution and Gini fairness coefficient
respond. *This panel does not run a full DES — it samples the tenant
specs that the multi-tenant generator would produce, plots the spec
distribution, and lets you see the heterogeneity directly.*
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_t = st.number_input("# tenants", value=128, min_value=4, max_value=4096, step=4)
    with c2:
        β = st.slider("Noise level β", 0.0, 1.0, 0.5, 0.05)
    with c3:
        agg_λ = st.number_input("Aggregate file rate (files/s)", value=4.0, min_value=0.01, step=0.5)

    c4, c5, c6 = st.columns(3)
    with c4:
        burst_period = st.number_input("Noisy burst period (s)", value=600.0, min_value=30.0, step=30.0)
    with c5:
        burst_size = st.number_input("Noisy burst size (files)", value=50, min_value=5, step=5)
    with c6:
        noisy_max = st.slider("Max noisy fraction at β=1", 0.0, 0.5, 0.10, 0.01)

    # Build the spec distribution with the requested config (no DES run).
    cfg_mt = {
        "simulation": {"seed": 42},
        "workload": {
            "iat_pareto_alpha": 1.8,
            "iat_pareto_xmin_s": 0.5,
            "file_size_lognormal_mu": 16.118,
            "file_size_lognormal_sigma": 2.5,
            "working_set_size": 5000,
            "zipf_s": 1.0,
            "read_fraction": 0.7,
            "diurnal_enabled": False,
            "multi_tenant": {
                "enabled": True,
                "n_tenants": int(n_t),
                "noise_level": float(β),
                "aggregate_lambda_files_per_s": float(agg_λ),
                "noisy_burst_period_s": float(burst_period),
                "noisy_burst_size": int(burst_size),
                "noisy_fraction_max": float(noisy_max),
                "power_user_fraction": 0.05,
                "rate_lognormal_sigma_max": 1.5,
                "size_perturbation_sigma_max": 0.5,
            },
        },
    }
    env = simpy.Environment()

    class _NullFS:
        class _P:
            def register_file(self, *a, **k): pass
        policy = _P()
        def handle_write(self, *a, **k):
            if False: yield
        def handle_access(self, *a, **k):
            if False: yield

    rng_mgr = __import__("core.rng", fromlist=["create_rng_manager"]).create_rng_manager(
        master_seed=cfg_mt["simulation"]["seed"]
    )
    gen = MultiTenantWorkloadGenerator(env, cfg_mt, rng_mgr, _NullFS())
    specs = gen.tenant_specs

    df_specs = pd.DataFrame([{
        "tenant": s.tenant_id,
        "role": s.role,
        "rate (files/s)": s.lambda_files_per_s,
        "xmin (s)": s.iat_pareto_xmin_s,
        "size_mu": s.file_size_lognormal_mu,
        "ws_size": s.working_set_size,
        "diurnal": s.diurnal_enabled,
    } for s in specs])

    role_counts = df_specs["role"].value_counts().to_dict()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tenants", len(specs))
    m2.metric("Background", role_counts.get("background", 0))
    m3.metric("Power-user", role_counts.get("power", 0))
    m4.metric("Noisy", role_counts.get("noisy", 0))

    # Per-tenant rate distribution across roles.
    fig6 = go.Figure()
    for role, color in (("background", "#6baed6"),
                        ("power",       "#fd8d3c"),
                        ("noisy",       "#d7191c")):
        sub = df_specs[df_specs["role"] == role]
        if len(sub) > 0:
            fig6.add_trace(go.Histogram(
                x=sub["rate (files/s)"], name=role.title(),
                marker_color=color, opacity=0.7,
                xbins=dict(size=max(0.005, float(df_specs["rate (files/s)"].max())/40)),
            ))
    fig6.update_layout(
        barmode="overlay", height=350,
        title=f"Per-tenant arrival rate distribution at β={β:.2f}",
        xaxis_title="rate (files/s)", yaxis_title="# tenants",
    )
    st.plotly_chart(fig6, use_container_width=True)
    _download_png(fig6, "Download high-res PNG", "tenant_rate_distribution.png",
                  width=1400, height=600)

    # Aggregate-rate-invariance check.
    bg_rate_sum = df_specs[df_specs["role"] != "noisy"]["rate (files/s)"].sum()
    n_non_noisy = (df_specs["role"] != "noisy").sum()
    expected = agg_λ * (n_non_noisy / max(len(df_specs), 1))
    st.info(
        f"**Aggregate-rate-invariance check.** Non-noisy tenants summed rate "
        f"= **{bg_rate_sum:.3f} files/s**;  expected (= aggregate × non-noisy/n) "
        f"= **{expected:.3f} files/s**;  error = **{abs(bg_rate_sum-expected)/max(expected,1e-9)*100:.2f}%**.  "
        f"Aggregate background load is held fixed across β; the noise dial "
        f"controls *heterogeneity*, not load growth."
    )

    # Synthetic per-tenant P95 distribution illustrative model: the larger
    # the tenant rate, the more likely it is to be a noisy-neighbour
    # *victim* (sharing the cache with bursts). We simulate this by
    # sampling per-tenant P95 from a heavy-tailed distribution conditioned
    # on tenant role, just for illustration of how Gini responds to β —
    # it is *not* a substitute for running the actual DES end-to-end.
    rng_demo = np.random.default_rng(42)
    p95_synth = np.zeros(len(specs))
    for i, s in enumerate(specs):
        base_p95 = 1.5 / max(s.lambda_files_per_s if s.role != "noisy" else 0.1, 1e-3)
        if s.role == "noisy":
            # A noisy tenant inflates own P95 *and* a few victims; here
            # we model only the self-inflicted part.
            base_p95 *= rng_demo.lognormal(2.0, 1.0)
        else:
            # Probability of being a victim grows with β.
            p_victim = β * (role_counts.get("noisy", 0) /
                            max(len(specs) - role_counts.get("noisy", 0), 1))
            if rng_demo.uniform() < p_victim:
                base_p95 *= rng_demo.lognormal(1.5, 0.8)
        p95_synth[i] = base_p95

    g_synth = gini_coefficient(p95_synth)
    fig7 = go.Figure()
    fig7.add_trace(go.Histogram(x=p95_synth, nbinsx=40, marker_color="#9e9ac8"))
    fig7.update_layout(
        title=f"Illustrative per-tenant P95 distribution — Gini ≈ {g_synth:.3f}",
        xaxis_title="P95 recall latency (s, illustrative)",
        yaxis_title="# tenants",
        xaxis_type="log", height=320,
    )
    st.plotly_chart(fig7, use_container_width=True)
    st.caption(
        "*This panel is a schematic — to compute the real Gini and per-tenant "
        "P95 distribution, run a DES with `multi_tenant.enabled: true` in your "
        "config and inspect the `multi_tenant` block of the generated "
        "`simulation_report.json`. The CLI also prints the summary in its "
        "stdout report.*"
    )

    st.markdown("""
#### What to take away from this panel

- The role mix at β=0 is fully background; at β=1 the noisy fraction
  reaches `noisy_fraction_max` of the population.
- The non-noisy aggregate rate is invariant in β (the dial does not
  silently inflate load).
- Heterogeneity grows with β through three independent channels: rate
  σ on the LogNormal, file-size σ, and noisy-neighbour fraction.
- The full noisy-neighbour effect on tail latency is a *running-system*
  property, observable only after a DES run; this panel exposes the
  tenant-spec distribution that drives that running-system behaviour.
""")

st.markdown("---")
st.caption(
    "DESCASSI — Discrete Event Simulator for Combined Archival and Scratch Storage Infrastructure. "
    "MMFQ: Ahn & Ramaswami (2005). CTMC MLE: Albert (1962); Billingsley (1961). "
    "Kingman (1961). Whitt (1983). Tijms (2003) §2.7. "
    "IOM — τ-dependent P_loss (mount-amortisation back-pressure + burst-SCV inflation). "
    "Multi-tenant noisy-neighbour: workload/multi_tenant.py."
)
