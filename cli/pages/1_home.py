"""
DESCASSI - System Dashboard
===========================
Visual overview of HSM system configuration with default vs configured values.
Includes workflow diagram showing simulation architecture.
"""

import streamlit as st
from config.models import (
    SimulationConfig, ESS3500Config, TS1160Config, 
    DMFPolicyConfig, DeclarativePolicyConfig, CostConfig
)
from db.database import SweepDB
from pathlib import Path

st.set_page_config(
    page_title="DESCASSI - Dashboard",
    page_icon="💾",
    layout="wide"
)

DEFAULT_CONFIG = SimulationConfig()

# Diagram width constants for consistent alignment
DIAGRAM_WIDTH = 87   # Total width of diagram box borders
CONTENT_WIDTH = 83   # Inner content width (DIAGRAM_WIDTH - 4 for "│ " prefix and " │" suffix)

# Tooltip CSS for workflow diagrams
TOOLTIP_CSS = """
<style>
.stTooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted #888;
    cursor: help;
}
.stTooltip .stTooltip-content {
    visibility: hidden;
    width: 480px;
    background-color: #1e1e1e;
    color: #d4d4d4;
    text-align: left;
    border-radius: 8px;
    padding: 12px;
    position: absolute;
    z-index: 100;
    bottom: 125%;
    left: 50%;
    margin-left: -240px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    font-family: monospace;
}
.stTooltip:hover .stTooltip-content {
    visibility: visible;
    opacity: 1;
}
.stTooltip-content code {
    background-color: #2d2d2d;
    padding: 4px 6px;
    border-radius: 4px;
    display: block;
    white-space: pre-wrap;
    word-break: break-all;
}
.stTooltip-content .label {
    color: #4ec9b0;
    font-weight: bold;
    margin-bottom: 4px;
}
</style>
"""

CODE_SNIPPETS = {
    'simpy_env': """env = simpy.Environment()
env.process(self._run_archiver(env, duration))
env.process(self._generate_workload(env, duration))
env.run(until=duration)""",

    'archiver_logic': """# Check cache state
if self.cache_bytes > hwm_bytes:
    should_migrate = True
elif self.cache_bytes < lwm_bytes:
    should_migrate = False
else:
    file_age = env.now - last_access
    should_migrate = file_age > age_threshold""",

    'file_tracking': """self.files[file_id] = {
    'size': bytes,
    'last_access': timestamp,
    'created_at': timestamp,
    'on_tape': bool
}""",

    'cache_miss': """with tape_drive_resource.request() as req:
    recall_latency = ts.calculate_recall_latency(size)
    yield env.timeout(recall_latency)
    ts.record_mount()
    ts.record_seek()
    ts.record_read(size)""",

    'migration_time': """migration_time = ts.calculate_service_time(size)
yield env.timeout(migration_time)
ts.record_write(size)
archived_bytes += size
files[id].on_tape = True""",

    'compare_metric': """# Calculate difference
diff_pct = ((sim - ior) / ior) * 100
# t-statistic
t_stat = (sim - ior) / (std / sqrt(n))
p_value = 2 * (1 - t.cdf(abs(t_stat), df=1))
# Pass if NOT significant OR within 20%
passed = p_value >= alpha or abs(diff_pct) < 20""",

    'bandwidth_calc': """def _calculate_simulated_bw(bytes_count, duration_s):
    if duration_s <= 0:
        return 0.0
    return bytes_count / duration_s"""
}

def verify_diagram(lines: list, name: str, width: int = 80) -> list:
    """Verify all lines in a diagram are exactly 'width' characters.
    
    Args:
        lines: List of string lines to verify
        name: Name of diagram for error messages
        width: Expected width (default 80)
    
    Returns:
        List of error messages (empty if all pass)
    """
    errors = []
    for i, line in enumerate(lines, 1):
        actual_len = len(line)
        if actual_len != width:
            errors.append(f"  Line {i:2d}: {actual_len:2d}/{width} chars - {repr(line[:50])}")
    if errors:
        return [f"Diagram '{name}' alignment errors:"] + errors
    return []

def verify_all_diagrams(diagrams: dict) -> bool:
    """Verify all diagrams and print results.
    
    Args:
        diagrams: Dict of {name: [lines]}
    
    Returns:
        True if all pass, False if any failures
    """
    all_errors = []
    for name, lines in diagrams.items():
        errors = verify_diagram(lines, name)
        all_errors.extend(errors)
    
    if all_errors:
        for err in all_errors:
            print(f"WARNING: {err}")
        return False
    return True

def render_tooltip(title: str, code_key: str) -> str:
    """Render an HTML tooltip with code snippet."""
    code = CODE_SNIPPETS.get(code_key, "")
    return f"""
    <div class="stTooltip">
        ℹ️
        <span class="stTooltip-content">
            <span class="label">{title}</span>
            <code>{code}</code>
        </span>
    </div>
    """

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"

def format_bytes_per_s(bytes_val: int) -> str:
    """Format bytes/s to MB/s or GB/s."""
    mbps = bytes_val / (1024 * 1024)
    if mbps >= 1024:
        return f"{mbps/1024:.1f} GB/s"
    return f"{mbps:.0f} MB/s"

def get_delta_label(configured, default):
    """Get delta label for metric."""
    if configured == default:
        return None
    diff = configured - default
    if isinstance(diff, float):
        return f"{diff:+.1f}"
    if isinstance(diff, (int,)):
        return f"{diff:+d}"
    return f"{diff:+}"

st.title("💾 DESCASSI Dashboard")
st.markdown("### Hierarchical Storage Management System Configuration")
st.markdown("---")

# Inject tooltip CSS
st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

cfg = st.session_state.get('config', DEFAULT_CONFIG)

# Create tabs
tab_overview, tab_workflow = st.tabs(["📖 Overview", "⚙️ Technical Details"])

with tab_overview:
    # ========== SECTION 1: WHAT IS DESCASSI? ==========
    st.markdown("## What is DESCASSI?")

    st.markdown("""
**DESCASSI** (Discrete Event Simulator for Combined Archival and Scratch Storage Infrastructure)
models hierarchical storage management (HSM) systems where fast cache storage (NVMe/disk) works
together with slower archive storage (tape).

The simulator helps answer a critical question: **How often should we move data from cache to tape?**

- Move too **frequently** and you spend most of the tape's wall-clock time mounting cartridges
  rather than streaming data — the per-byte mount/seek overhead amortises poorly over tiny batches,
  back-pressuring the cache. You also waste media on files deleted before they're ever recalled.
- Move too **rarely** and large batched archival bursts inflate the cache's queueing variance,
  hurting tail latency for scratch I/O — *and* you risk losing unprotected data if the cache fails.

DESCASSI uses discrete event simulation (SimPy) to model real HSM hardware configurations
(IBM ESS 3500 cache, TS1160 tape drives) and policy engines (HPE DMF 7), enabling rigorous
exploration of the performance-cost-risk trade-off space — at realistic multi-tenant load,
not just single-user dedicated configurations.
    """)

    st.markdown("---")

    # ========== SECTION 2: THE THESIS ==========
    with st.expander("**The Two-Tier Probabilistic Model**", expanded=True):
        st.markdown("""
### The Core Problem

Traditional HSM systems treat cache and archive as **separate systems** with static,
rule-based policies (e.g., "migrate files older than 30 minutes when cache exceeds 85%").
This approach fails to account for:

1. **Workload variability**: HPC workloads have distinct phases (computation, checkpointing, data recall)
   with dramatically different I/O patterns
2. **Competing costs**: Any archive interval choice involves trade-offs between three costs
3. **Dynamic optimality**: The "best" archive interval changes as workload conditions change

### The Three Competing Costs

Every choice of archive interval **τ** (tau) incurs three types of cost:
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**Performance Loss** *(P_loss)* — **U-shaped in τ**

The cache shares disk bandwidth with archival
writes. Two effects compete:

- *Mount-amortisation back-pressure* (∝ 1/τ).
  Tiny batches spend most of the tape's time
  mounting; the cache backs up.
- *Burst-SCV inflation* (∝ τ). Big batches
  arrive in clusters, raising the combined
  arrival SCV (Whitt 1983) and inflating
  the Kingman G/G/1 wait at the cache.

Sum has an interior minimum.
            """)
        with col2:
            st.markdown("""
**Media Waste** *(M_waste)* — non-decreasing in τ

Tape written for files that get deleted before
recall. For exponential lifetime *L*:

M_waste(τ) = ρ · (1 − e^{−τ/E[L]}) / α

Saturates at ρ·E[L] as τ → ∞.
            """)
        with col3:
            st.markdown("""
**Data Risk** *(R_risk)* — linear in τ

Bytes in cache that haven't yet reached tape are
exposed to a cache-side failure. Under uniform
failure-time hazard:

R_risk(τ) = ρ · τ / 2

Grows without bound.
            """)

        st.markdown("### The Cost Function")
        st.markdown("""
The total cost is a weighted sum of these three components, with each
component normalised onto a comparable scale (α, β, γ) and weights on a
simplex (Σw = 1):
        """)
        st.latex(r"J(\tau) = w_p \cdot \alpha \cdot P_{loss}(\tau) + w_m \cdot \beta \cdot M_{waste}(\tau) + w_r \cdot \gamma \cdot R_{risk}(\tau)")

        st.markdown("""
*w_p*, *w_m*, *w_r* reflect organisational priorities (performance-
focused vs. cost-focused vs. risk-averse).

### The Key Observation (not a theorem — a feasibility statement)

**The traditional "never archive" boundary (τ → ∞) is degenerate
*within this model*: J(τ) → ∞ there, so τ* must be interior.**

As the archive interval grows without bound, *under the modelling
assumptions of this IOM* (finite mean file lifetime, uniform-in-time
failure hazard, Whitt-superposed batched archival arrivals):

- *Performance loss* — diverges. Batch size B(τ) = ρ·τ / E[file] grows
  linearly; the compound-Poisson arrival SCV grows with it; the
  Kingman G/G/1 wait at the cache inflates without bound.
- *Media waste* — saturates at ρ·E[L], the **maximum** possible waste.
- *Data risk* — grows linearly in τ without bound.

Both P_loss and R_risk drive J(τ) → ∞, so τ* is finite. This is a
feasibility observation, not a theorem in the strict sense — a formal
theorem would require ruling out alternative failure-cost models,
priced-risk policies, and the unbounded-lifetime regime, which are
outside the scope of this model.

A *fully* rigorous proof of an interior optimum would also need
existence and lower-semicontinuity arguments at τ = 0, which the
mount-amortisation back-pressure term in P_loss provides. The interior
τ* is therefore not a special case of the model; it is a structural
consequence of the two-effect P_loss formulation.
        """)

    # ========== SECTION 3: THE HYPOTHESIS ==========
    with st.expander("**What We're Testing**", expanded=False):
        st.markdown("""
### Research Hypothesis

**Dynamic, workload-aware archive scheduling outperforms static policies —
*and the gap widens as the tenant population becomes more heterogeneous.***

The hypothesis rests on four observations:

1. **Workload has structure**: HPC workloads exhibit distinct phases:
   - **IDLE**: Minimal I/O (< 5% of peak)
   - **COMPUTE**: Moderate, steady I/O
   - **CHECKPOINT**: Burst writes (> 40% of peak write rate)
   - **RECALL**: Burst reads from tape (> 20% of peak tape rate)

2. **Phases are predictable**: A Continuous-Time Markov Chain (CTMC) can model phase transitions,
   enabling probabilistic prediction of near-future workload states.

3. **Optimal τ* varies by phase**: During CHECKPOINT, aggressive archiving may be beneficial;
   during COMPUTE, less frequent archiving reduces interference.

4. **Real HSMs serve hundreds–thousands of concurrent tenants** — not one. The interesting
   tail-latency dynamics (noisy-neighbour, checkpoint-storm victims) only appear when
   the workload has *multi-stream* structure with a fraction of bursty tenants. A static
   policy that performs adequately at homogeneous load will degrade non-uniformly across
   tenants as the noise dial β rises.

### What Success Looks Like

If the hypothesis is correct, an adaptive controller that adjusts τ*(t) based on
CTMC phase predictions should achieve **lower total cost J** than any fixed τ policy
*and* a **lower Gini coefficient on per-tenant P95 recall latency** (i.e., distribute
the noisy-neighbour pain more fairly), across a range of realistic workload scenarios
and noise levels.

### Validation Approach

- **Synthetic workloads** (single-stream Pareto with diurnal time-rescaling): controlled
  experiments with known phase transition rates.
- **Multi-tenant noisy-neighbour workloads**: N concurrent tenant streams (background,
  power-user, noisy-neighbour) with adjustable noise dial β ∈ [0, 1]. β = 0 reduces
  to the homogeneous-farm baseline; β = 1 maximises heterogeneity. Aggregate file rate
  held fixed across β, so the dial measures heterogeneity-at-fixed-load.
- **Trace replay**: real IOR benchmark traces from production HPC systems.
- **Statistical tests**: Kolmogorov–Smirnov for distribution agreement; Holm step-down
  Bonferroni for family-wise control across multiple marginals.
- **Two-tier robust Little's Law**: per-queue batch-means CI test (Glynn & Whitt 1989)
  with a relative-error fallback (Law & Kelton 2000 §9.6.2) so low-rate queues with
  exact L = λW relations don't spuriously fail on short observation windows.
- **Analytical verification**: degenerate cases (M/M/1 queue) match closed-form solutions
  to within 5% at 30 replications.
        """)

    st.markdown("---")

    # ========== SECTION 3.5: MULTI-TENANT NOISY-NEIGHBOUR ==========
    with st.expander("**Multi-tenant noisy-neighbour workloads**", expanded=False):
        st.markdown("""
### Why a single workload generator isn't enough

Real DMF-fronted Lustre/SpectrumScale caches don't serve one user. ORNL Atlas/Spider, NCSA Blue Waters — every site DMF 7 ships into —
support hundreds–thousands of concurrent tenants. A single-stream
generator cannot reproduce two effects that dominate operational HSM
performance:

1. **Aggregation**: Superposing many independent renewal streams converges
   (slowly) toward Poisson, but the convergence is incomplete on the
   timescales that matter for cache and tape; the leftover heavy-tail /
   autocorrelation is exactly what a single-stream Pareto over- or
   under-states depending on its tuning.

2. **Noisy-neighbour bursts**: A tenant running checkpoint-restart at
   the end of every compute step submits a *clustered* burst of writes
   that has nothing to do with the long-run Pareto tail of background
   traffic. These bursts inflate the P99 / P99.9 recall latency for
   every other tenant sharing the cache.

### The model

DESCASSI's multi-tenant generator runs **N concurrent tenant streams**,
each with its own:

- Renewal IAT distribution (Pareto with diurnal time-rescaling).
- LogNormal file-size distribution.
- Zipf working set (each tenant has its *own* hot files — not a shared
  global popularity ranking).
- A `tenant_id` tag carried on every produced file, so per-tenant KPIs
  (recall P95, byte counts) can be computed at end of run.

Three tenant classes:

- **Background** — smooth Pareto, low rate, large working set. Most tenants.
- **Power-user** — ~10× background rate, larger files, smaller working set.
- **Noisy-neighbour** — clustered bursts every `burst_period_s` seconds.
  Bursts are *not* diurnally modulated — checkpoint cadence is set by
  the application's compute loop, not time of day.

### A single dial: noise level β ∈ [0, 1]

| β    | Behaviour                                                                |
|------|--------------------------------------------------------------------------|
| 0.0  | Homogeneous farm. Same aggregate behaviour as the single-stream baseline. |
| 0.5  | Moderate σ on per-tenant rates; a small fraction of noisy tenants.        |
| 1.0  | Maximum heterogeneity, max noisy-neighbour fraction (config-controlled).  |

**Aggregate file-arrival rate is held fixed across β.** Noisy slots
*replace* their Pareto share with bursts rather than adding load on top,
so the dial measures heterogeneity-at-fixed-load rather than load growth.
This is the right comparison for tail-latency studies.

### Fairness summary: Gini on per-tenant P95 recall latency

The simulator emits a per-tenant breakdown plus a **Gini coefficient on
per-tenant P95 recall latency**:

- 0  = perfectly fair (all tenants see the same tail latency).
- 1  = maximally unequal (a single tenant absorbs all the pain).

A 64-tenant smoke run at β = 0.7 produces:

```
Tenants observed       : 63
P95 recall Gini        : 0.4893
Per-tenant P95 range   : 0.636s — 2543.602s (4001× ratio)
```

That 4000× ratio is the noisy-neighbour signal — a small set of victim
tenants absorb most of the tail-latency cost. The single-stream model
could not produce this at all.

### Where to enable it

`config/default_config.yaml`, section `workload.multi_tenant`:

```yaml
workload:
  multi_tenant:
    enabled: true
    n_tenants: 256
    noise_level: 0.5     # the β dial
    noisy_burst_period_s: 600
    noisy_burst_size: 50
```

Or load a multi-tenant YAML directly in the **Configure** page and run
from the command line.
        """)

    st.markdown("---")

    # ========== SECTION 4: HOW IT WORKS ==========
    st.markdown("## How It Works")
    st.markdown("*Click each step to expand for details*")

    with st.expander("**Step 1: Configure the System**", expanded=False):
        st.markdown("""
Define the hardware and policy parameters for your HSM simulation:

**Cache Tier (ESS 3500)**
- Number of NSD nodes and aggregate throughput (default: 65 GB/s)
- Usable capacity (default: 809 TiB)
- Network type and striping configuration

**Archive Tier (TS1160 Tape)**
- Number of tape drives (default: 16)
- Native transfer rate per drive (default: 400 MB/s)
- Mount, seek, and positioning times (LogNormal distributions)

**Migration Policy (DMF)**
- High/Low water marks (HWM/LWM) for cache utilization triggers
- Age threshold for file eligibility
- Archiver interval (how often the archiver runs)

*Configuration can be done via the sidebar or loaded from YAML files.*
        """)

    with st.expander("**Step 2: Generate or Load Workload**", expanded=False):
        st.markdown("""
Choose how files arrive and are accessed in the simulation:

**Synthetic Workload** *(for controlled experiments)*
- File sizes: LogNormal distribution (default median ~10 MB)
- Inter-arrival times: Pareto distribution (self-similar HPC traffic)
- Access patterns: Zipf distribution (popular files accessed more)
- Time modulation: Diurnal pattern via **time rescaling** (Çinlar 1975 §6.5),
  not thinning — so the heavy-tail / autocorrelation structure of the
  Pareto base is preserved when the diurnal envelope is applied. Lewis–Shedler
  thinning (1979) is a Poisson-process result and would not produce a
  process with rate λ_base · φ(t) when applied to a non-Poisson candidate
  stream.

**Multi-tenant Noisy-Neighbour Workload** *(for credibility on real shared HSMs)*
- N concurrent tenant streams (background, power-user, noisy-neighbour).
- Single noise dial β ∈ [0, 1] controls heterogeneity.
- Aggregate file-arrival rate held fixed across β (heterogeneity-at-fixed-load).
- Per-tenant `tenant_id` tagged on every file → per-tenant KPIs + Gini fairness on tail latency.
- See the **Multi-tenant noisy-neighbour workloads** section above.

**Trace Replay** *(for validation against real data)*
- Load JSONL trace files from IOR benchmarks or production logs
- Events: WRITE, READ, MIGRATE, RECALL with timestamps and file sizes
- Time scaling to compress/expand trace duration

*Trace files can be generated from IOR benchmark output using the included parser.*
        """)

    with st.expander("**Step 3: Run the Simulation**", expanded=False):
        st.markdown("""
Execute the discrete event simulation using SimPy:

**Concurrent Processes**
- **Workload Generator**: Creates file arrivals according to configured distribution
- **Archiver Process**: Runs at configured interval, migrates eligible files to tape

**Event Flow**
1. Files arrive → written to cache → tracked in file registry
2. Cache fills toward HWM → archiver activates
3. Eligible files (age > threshold OR HWM exceeded) → migrate to tape
4. Read requests → cache hit (fast) or cache miss (tape recall)
5. Metrics collected: latencies, queue lengths, cache fill, tape utilization

**Simulation Controls**
- Duration: How long to simulate (hours)
- Warmup: Initial period excluded from statistics
- Replications: Multiple runs with different random seeds
        """)

    with st.expander("**Step 4: Analyze Results**", expanded=False):
        st.markdown("""
Examine the simulation outputs to understand system behavior:

**Performance Metrics**
- Latency distributions (read, write, recall) at P50/P95/P99
- Throughput over time
- Queue lengths and waiting times
- **Cumulative write block time** — total seconds writes were blocked on
  the disk-cache capacity gate; a direct write back-pressure / SLA
  signal that exposes when migration cannot keep pace with arrivals.

**Cost Metrics (IOM)**
- P_loss(τ): two-effect model — mount-amortisation back-pressure (∝ 1/τ)
  + Whitt-superposed burst-SCV inflation (∝ τ). U-shaped in τ.
- M_waste(τ): r·E[min(L, τ)] from file-lifetime distribution (closed form
  for exponential, numerical for Weibull).
- R_risk(τ): ρ·τ/2 under uniform failure-time hazard.

**Optimization Output**
- Optimal τ* from the Interval Optimization Model — interior to (τ_min, τ_max)
  by construction (P_loss U-shape + R_risk linearity).
- Cost function J(τ) curve showing the trade-off landscape.
- Phase distribution from CTMC classification.

**Multi-tenant Output** *(when enabled)*
- Per-tenant recall-latency P95 across replications, with bootstrap CIs.
- Gini coefficient on per-tenant P95 — fairness summary; high Gini means
  a small set of victim tenants absorbs most of the noisy-neighbour pain.
- Per-tenant byte counts (writes / reads) for load attribution.

*Results are stored in SQLite database for comparison across runs.*
        """)

    with st.expander("**Step 5: Validate**", expanded=False):
        st.markdown("""
Ensure simulation results are trustworthy:

**Analytical Verification**
- Configure as degenerate M/M/1 queue.
- Verify mean waiting time matches W = 1/(μ−λ) within 5%.
- Two-tier robust **Little's Law** test on every queue:
  - *Tier 1* — batch-means delta-method CI (Glynn & Whitt 1989) with
    adaptive batch count, sparse-batch handling, and trailing-batch
    censoring (boundary items with W̄ > batch width are dropped because
    their sojourn is truncated by sim_end).
  - *Tier 2* — relative-error fallback (Law & Kelton 2000 §9.6.2): if
    the CI is non-informative due to too few sojourn samples (typical
    on low-rate queues like `robot` or `recall_queue`), pass when
    |L − λW| / L ≤ 5%. The report records which tier closed.

**Statistical Validation**
- Kolmogorov–Smirnov test: simulated vs. observed latency distributions.
- Holm step-down Bonferroni correction across multiple marginals
  (recall latency, migrate latency, file size, IAT) so family-wise
  α = 0.05 is preserved.

**Trace Replay Validation**
- Feed actual production traces through the simulator.
- Compare predicted vs. actual throughput and latencies.
- Identify model calibration needs.

**Disk-cache invariants**
- Disk usage cannot exceed capacity. The simulator enforces this with a
  `simpy.Container` capacity gate; writes block on the gate until space
  is freed. The cumulative write-block time is exposed as a back-pressure
  KPI.
- Files transition through DMF 7's REG → MIG → DUL → OFL state machine,
  with `dmfsfree`-style space release (DUL → OFL on the oldest-touched
  files) firing whenever the cache crosses HWM.

*Validation results are logged and can be reviewed in the Analysis tab.*
        """)

# ========== WORKFLOW TAB ==========
with tab_workflow:
    st.markdown("### ⚙️ Simulator Workflow Diagram")
    st.markdown("*Interactive visualization of the DESCASSI simulation engine*")
    st.markdown("---")

    # Pre-calculate all dynamic values for consistent formatting (all fixed-width strings)
    hwm_pct = int(cfg.dmf_policy.hwm_fraction * 100)
    lwm_pct = int(cfg.dmf_policy.lwm_fraction * 100)
    age_thresh = cfg.dmf_policy.age_threshold_minutes
    arch_int = cfg.dmf_policy.archiver_interval_minutes
    duration_hrs = cfg.sim_duration_hours
    
    # Fixed-width formatted values for diagrams (all exactly X chars wide)
    hwm_s = f"HWM={hwm_pct:>2}%"     # e.g., "HWM=85%" (8 chars)
    lwm_s = f"LWM={lwm_pct:>2}%"     # e.g., "LWM=60%" (8 chars)
    age_s = f"Age={age_thresh:>2}min"  # e.g., "Age=30min" (10 chars)
    int_s = f"Int={arch_int:>2}min"   # e.g., "Int=15min" (9 chars)
    dur_h = f"{duration_hrs:.1f}"     # e.g., "1.0" (3 chars)
    age_m = f"{age_thresh:>2}min"     # e.g., "30min" (5 chars)
    arc_m = f"{arch_int:>2}min"       # e.g., "15min" (5 chars)
    arc2_m = f"{arch_int*2:>2}min"     # e.g., "30min" (5 chars)
    arc3_m = f"{arch_int*3:>2}min"     # e.g., "45min" (5 chars)
    
    # Trace info with padding
    trace_info = f"Trace: `{cfg.trace_file.name if cfg.trace_file else 'None'}`" if cfg.trace_file else "Trace: None"
    trace_pad = trace_info.ljust(CONTENT_WIDTH)
    
    # Mode info with padding
    mode = cfg.workload_mode
    trace_name = cfg.trace_file.name if cfg.trace_file else "None"
    mode_info = f"WORKLOAD MODE: {mode}"
    mode_pad = mode_info.ljust(CONTENT_WIDTH)

    # QUICK VIEW - Always visible
    st.markdown("#### 📊 Complete System Overview")

    # Build policy params line with proper padding for inner box
    # Inner box format: │ │  {content} │ │ = 5 + content + 4 = 87, so content = 78
    policy_params = f"{hwm_s}  {lwm_s}  {age_s}  {int_s}".ljust(78)

    # 87 chars per line: │ + 85 content + │
    st.code(f"""\
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          DESCASSI SIMULATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│ ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌────────────┐        │
│ │   IOR    │──▶│   IOR    │──▶│   TRACE   │──▶│ SIMULATOR │──▶│ VALIDATION │        │
│ │   FILE   │   │  PARSER  │   │ GENERATOR │   │           │   │            │        │
│ └──────────┘   └──────────┘   └───────────┘   └─────┬─────┘   └──────┬─────┘        │
│                                                     │                │              │
│                                                     ▼                ▼              │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│ │                            SIMULATION ENGINE                                    │ │
│ │                                                                                 │ │
│ │  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐                │ │
│ │  │     ESS     │         │   ARCHIVER  │         │   TS1160    │                │ │
│ │  │   (Cache)   │◀────────│    (DMF)    │────────▶│   (Tape)    │                │ │
│ │  │   65 GB/s   │         │   Policy    │         │  16 drives  │                │ │
│ │  └─────────────┘         └─────────────┘         └─────────────┘                │ │
│ │                                                                                 │ │
│ │  {policy_params} │ │
│ └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
│ {trace_pad} │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘""", language=None)

    st.markdown("---")

    # SECTION A: SimPy Architecture
    with st.expander("🔄 SimPy Event-Driven Architecture", expanded=False):
        dur_pad = f"(Duration: {dur_h} hours)".ljust(82)
        st.code(f"""\
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SimPy EVENT-DRIVEN MODEL                                  │
│  {dur_pad} │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  env = simpy.Environment()                                                          │
│  env.process(self._run_archiver(env, duration))    <-- Archiver process             │
│  env.process(self._generate_workload(env, duration)) <-- Workload process           │
│  env.run(until=duration)                           <-- Run simulation               │
│                                                                                     │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐            │
│  │  _run_archiver  │       │   _generate_    │       │    env.now      │            │
│  │                 │       │   workload()    │       │                 │            │
│  │  env.process()  │<----->│                 │<----->│   Simulation    │            │
│  │                 │       │  env.process()  │       │     Clock       │            │
│  │  ┌───────────┐  │       │  ┌───────────┐  │       │                 │            │
│  │  │   yield   │  │       │  │   yield   │  │       │  Advances via   │            │
│  │  │ timeout() │  │       │  │ timeout() │  │       │      yield      │            │
│  │  └───────────┘  │       │  └───────────┘  │       │                 │            │
│  └─────────────────┘       └─────────────────┘       └─────────────────┘            │
│                                                                                     │
│  Key: env.now advances to next event. Processes run CONCURRENTLY, don't block.     │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘""", language=None)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(render_tooltip("SimPy Environment", "simpy_env"), unsafe_allow_html=True)
        with col2:
            st.caption("Hover over ℹ️ for code snippet")

    # SECTION B: Workload Mode Selection
    with st.expander("📁 Workload Mode Selection", expanded=False):
        mode = cfg.workload_mode
        trace_name = cfg.trace_file.name if cfg.trace_file else "None"

        st.markdown(f"**Current Mode:** `{mode}` {f'({trace_name})' if trace_name != 'None' else ''}")

        st.code(f"""\
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ {mode_pad} │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌───────────────────────────┐             ┌───────────────────────────┐            │
│  │      MODE: synthetic      │             │    MODE: trace_replay     │            │
│  └─────────────┬─────────────┘             └─────────────┬─────────────┘            │
│                │                                         │                          │
│                ▼                                         ▼                          │
│  ┌───────────────────────────┐             ┌───────────────────────────┐            │
│  │  _generate_synthetic()    │             │     _replay_trace()       │            │
│  │                           │             │                           │            │
│  │  1. Pre-populate cache    │             │  1. Load JSONL file       │            │
│  │  2. Create initial        │             │  2. Parse events          │            │
│  │     files                 │             │  3. Pre-populate cache    │            │
│  │  3. Generate arrivals     │             │  4. Scale time            │            │
│  │     (exponential)         │             │                           │            │
│  └───────────────────────────┘             └───────────────────────────┘            │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘""", language=None)

    # SECTION C: Trace File Processing
    with st.expander("📄 Trace File Processing Pipeline", expanded=False):
        st.code("""\
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            TRACE FILE PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                        │
│   │  IOR BENCH-  │     │     IOR      │     │     IOR      │                        │
│   │  MARK OUTPUT │────▶│    PARSER    │────▶│    TRACE     │                        │
│   │              │     │              │     │  GENERATOR   │                        │
│   │  .txt file   │     │  Parses:     │     │              │                        │
│   │              │     │  - Config    │     │  Outputs:    │                        │
│   │  Example:    │     │  - Results   │     │  - .jsonl    │                        │
│   │  ior -t 4k   │     │  - Summary   │     │    files     │                        │
│   └──────────────┘     └──────────────┘     └──────┬───────┘                        │
│                                                    │                                │
│                                                    ▼                                │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         JSONL EVENT FORMAT                                  │   │
│   │                                                                             │   │
│   │   {{                                                                        │   │
│   │     "timestamp_s": 0.0,          <-- When event occurs                      │   │
│   │     "event_type": "WRITE",       <-- WRITE | READ | MIGRATE | RECALL        │   │
│   │     "file_id": "task0",                                                     │   │
│   │     "file_size_bytes": 67108864  <-- 64 MB default                          │   │
│   │   }}                                                                        │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘""", language=None)
        
        st.markdown(render_tooltip("Trace Events", "file_tracking"), unsafe_allow_html=True)
        st.caption("Hover over ℹ️ for file tracking code")

    # SECTION D: Policy Engine (DMF) - Expanded by default
    with st.expander("📋 Policy Engine (DMF)", expanded=True):
        # Show current values
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("HWM", f"{cfg.dmf_policy.hwm_fraction:.0%}")
        with col2:
            st.metric("LWM", f"{cfg.dmf_policy.lwm_fraction:.0%}")
        with col3:
            st.metric("Age Threshold", f"{cfg.dmf_policy.age_threshold_minutes} min")

        # Pre-format Policy Engine values with fixed widths
        lwm_s = f"{lwm_pct}%"
        hwm_s = f"{hwm_pct}%"
        age_s = f"{age_thresh}min"
        int_s = f"{arch_int} min"
        arc60 = f"{arch_int * 60}s"
        cycle_title = f"ARCHIVER CYCLE (every {int_s})".center(82)

        st.code(f"""\
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              POLICY ENGINE (DMF)                                    │
│                           Cache Capacity: 809 TiB                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Cache State:   cache < {lwm_s:<4}        {lwm_s:<4} < cache < {hwm_s:<4}       cache > {hwm_s:<4}          │
│  ──────────────────────────────────────────────────────────────────────────────     │
│  ┌──────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐     │
│  │  NO MIGRATION    │   │  AGE-BASED MIGRATION   │   │  AGGRESSIVE MIGRATION  │     │
│  │                  │   │                        │   │                        │     │
│  │  Archiver skips  │   │  LWM < cache < HWM     │   │  cache > HWM           │     │
│  │  this run        │   │  AND                   │   │                        │     │
│  │                  │   │  file_age > {age_s:<6}   │   │  Migrate ALL eligible  │       │
│  │                  │   │                        │   │    files               │     │
│  └──────────────────┘   └────────────────────────┘   └────────────────────────┘     │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  {cycle_title} │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  while env.now < duration:                                                          │
│      yield env.timeout({arc60:<5})    <-- Wait for interval                              │
│                                                                                     │
│      1. Calculate: cache_usage_pct = cache_bytes / capacity                         │
│      2. Determine mode:                                                             │
│         IF cache > {hwm_s:<4} (HWM): migrate_all()                                        │
│         ELIF cache > {lwm_s:<4} (LWM): check_age()                                        │
│      3. Find files: NOT on_tape AND (age > {age_s:<6} OR HWM exceeded)                  │
│      4. Sort by last_access (oldest first)                                          │
│      5. Migrate: yield env.timeout(migration_time)                                  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘""", language=None)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(render_tooltip("Archiver Decision Logic", "archiver_logic"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_tooltip("Migration Process", "migration_time"), unsafe_allow_html=True)
        st.caption("Hover over ℹ️ for code snippets")

    # SECTION E: Request Processing
    with st.expander("📖 Request Processing (READ/RECALL)", expanded=False):
        st.code("""\
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          READ REQUEST PROCESSING                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│                          READ REQUEST RECEIVED                                      │
│                                 │                                                   │
│                                 ▼                                                   │
│               ┌───────────────────────────┐                                         │
│               │     file_id IN files?     │                                         │
│               └─────────────┬─────────────┘                                         │
│                             │                                                       │
│               ┌─────────────┴─────────────┐                                         │
│               ▼                           ▼                                         │
│   ┌─────────────────────┐    ┌───────────────────────────────────────┐              │
│   │     CACHE HIT       │    │            CACHE MISS                 │              │
│   │   (on_tape=False)   │    │        (file is on tape)              │              │
│   └──────────┬──────────┘    └──────────────────┬────────────────────┘              │
│              │                                  │                                   │
│              ▼                                  ▼                                   │
│   ┌─────────────────────┐    ┌───────────────────────────────────────┐              │
│   │   ESS READ (fast)   │    │          RECALL SEQUENCE              │              │
│   │                     │    │                                       │              │
│   │ service_time =      │    │  1. Request tape drive                │              │
│   │   ess.calculate_    │    │     (queues if busy)                  │              │
│   │   service_time()    │    │                                       │              │
│   │                     │    │  2. Namespace lookup (~5s)            │              │
│   │ yield env.timeout(  │    │                                       │              │
│   │   service_time)     │    │  3. Mount time                        │              │
│   │                     │    │     (LogNormal ~20-30s)               │              │
│   │ ess.record_read()   │    │                                       │              │
│   │ files[id].last_     │    │  4. Seek time                         │              │
│   │   access = env.now  │    │     (LogNormal ~35s)                  │              │
│   └─────────────────────┘    │                                       │              │
│                              │  5. Transfer time (size/rate)         │              │
│                              │                                       │              │
│                              │  6. yield env.timeout(total)          │              │
│                              │                                       │              │
│                              │  7. ts.record_mount/seek/read()       │              │
│                              └──────────────────┬────────────────────┘              │
│                                                 │                                   │
│                                                 ▼                                   │
│                              ┌───────────────────────────────────────┐              │
│                              │       WRITE BACK TO CACHE             │              │
│                              │                                       │              │
│                              │ cache_write_time =                    │              │
│                              │   ess.calculate_service_time()        │              │
│                              │                                       │              │
│                              │ yield env.timeout(cache_write_time)   │              │
│                              │                                       │              │
│                              │ cache_bytes += size                   │              │
│                              │ files[id].on_tape = False             │              │
│                              └───────────────────────────────────────┘              │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘""", language=None)
        
        st.markdown(render_tooltip("Cache Miss (Recall)", "cache_miss"), unsafe_allow_html=True)
        st.caption("Hover over ℹ️ for recall code snippet")

    # SECTION F: Timing Relationships
    with st.expander("⏱️ Timing Relationships", expanded=False):
        # Pre-format timing values
        t1 = f"{arch_int:>2}min"
        t2 = f"{arch_int*2:>2}min"
        t3 = f"{arch_int*3:>2}min"
        dur_pad = f"(Simulation: {dur_h} hours)".ljust(82)

        st.code(f"""\
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             TIMING RELATIONSHIPS                                    │
│  {dur_pad} │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  SimPy processes run CONCURRENTLY - time advances via yield                         │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │ Sim Time ────────────────────────────────────────────────────────────────>    │  │
│  │ 0           {t1}         {t2}         {t3}                                 │  │
│  │ │            │              │              │                                 │  │
│  │ │         Archiver       Archiver       Archiver                             │  │
│  │ │          runs           runs           runs                                │  │
│  │ │            │              │              │                                 │  │
│  │ │            ▼              ▼              ▼                                 │  │
│  │ │  ┌──────────────────────────────────────────────────────────────────┐     │  │
│  │ │  │  File aging:                                                     │     │  │
│  │ │  │  t=0: files created, age=0                                       │     │  │
│  │ │  │  t={age_m}: files become ELIGIBLE                                    │     │  │
│  │ │  │  t={arc_m}: first archiver can migrate them                          │     │  │
│  │ │  └──────────────────────────────────────────────────────────────────┘     │  │
│  │ │                                                                           │  │
│  │ │  NOTE: Files created at t=0 are age={age_m} at first archiver run,            │  │
│  │ │        making them eligible immediately                                   │  │
│  │ └───────────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │ CONCURRENT PROCESSES (don't block each other)                                 │  │
│  │                                                                               │  │
│  │ Archiver:   ****----********----****----****----****----****----****----      │  │
│  │ Process     **** = working  ---- = waiting for timeout                        │  │
│  │                                                                               │  │
│  │ Workload:   ----********----****----********----****----********----****      │  │
│  │ Generator   ---- = idle     **** = processing events                          │  │
│  │                                                                               │  │
│  │ Both run simultaneously! Archiver doesn't block I/O!                          │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘""", language=None)

    # SECTION G: IOR Validation Process
    with st.expander("🔬 IOR Validation Process", expanded=False):
        st.code("""\
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           IOR VALIDATION WORKFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                      │
│   │     IOR      │      │  SIMULATION  │      │  COMPARISON  │                      │
│   │  BENCHMARK   │      │   RESULTS    │      │    ENGINE    │                      │
│   │  REAL DATA   │      │     (DB)     │      │              │                      │
│   └──────┬───────┘      └──────┬───────┘      └──────┬───────┘                      │
│          │                     │                     │                              │
│          │                     │                     ▼                              │
│          │                     │      ┌─────────────────────────────────────┐       │
│          │                     │      │   MATCH BY CONFIG + ITERATION       │       │
│          │                     │      │                                     │       │
│          │                     │      │   ior_config_name = config          │       │
│          │                     │      │   ior_iteration = iteration         │       │
│          │                     │      │                                     │       │
│          │                     │      │   Match IOR row <--> Sim row        │       │
│          │                     │      └─────────────┬───────────────────────┘       │
│          │                     │                    │                               │
│          ▼                     ▼                    ▼                               │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          METRIC COMPARISON                                  │   │
│   │                                                                             │   │
│   │   WRITE:                                                                    │   │
│   │     IOR BW:   ior_row['bw_mib_s']                                           │   │
│   │     Sim BW:   sim_row['total_bytes_written'] /                              │   │
│   │               sim_row['write_duration_s']                                   │   │
│   │                                                                             │   │
│   │   READ:                                                                     │   │
│   │     IOR BW:   ior_row['bw_mib_s']                                           │   │
│   │     Sim BW:   sim_row['total_bytes_read'] /                                 │   │
│   │               sim_row['read_duration_s']                                    │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                               │                                                     │
│                               ▼                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                    STATISTICAL SIGNIFICANCE TEST                            │   │
│   │                                                                             │   │
│   │   t-statistic = (sim_value - ior_value) / (std / sqrt(n))                   │   │
│   │   p-value = 2 * (1 - t.cdf(abs(t_stat), df=1))                              │   │
│   │                                                                             │   │
│   │   Decision:                                                                 │   │
│   │     IF p-value < alpha (0.05): statistically significant difference         │   │
│   │     IF p-value >= alpha: cannot reject null hypothesis (similar)            │   │
│   │                                                                             │   │
│   │   Pass Criteria:                                                            │   │
│   │     NOT significant OR |difference| < 20%                                   │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                               │                                                     │
│                               ▼                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                           RESULTS SUMMARY                                   │   │
│   │                                                                             │   │
│   │   total_comparisons, passed, failed                                         │   │
│   │   pass_rate = passed / total                                                │   │
│   │   avg_difference_pct                                                        │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘""", language=None)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(render_tooltip("Statistical Comparison", "compare_metric"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_tooltip("Bandwidth Calculation", "bandwidth_calc"), unsafe_allow_html=True)
        st.caption("Hover over ℹ️ for code snippets")

st.markdown("---")
st.caption("💾 DESCASSI - Discrete Event Simulator for Combined Archival and Scratch Storage Infrastructure")
