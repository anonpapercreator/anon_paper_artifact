# DMF 7 Discrete Event Simulator — Formal Mathematical Model Specification

**Version:** 1.0  
**Target System:** HPE Data Management Framework 7  
**Method:** Discrete Event Simulation (DES) with statistical output analysis  
**Language:** Python / SimPy 4.x  

---

## 1. Scope and Objectives

This document constitutes the formal mathematical specification for a DES of HPE DMF 7.
It is written to satisfy the requirements of academic reproducibility and scientific
defensibility per the standards of:

- Shannon, R.E. (1998). *Introduction to the Art and Science of Simulation.*
- Law, A.M. & Kelton, W.D. (2000). *Simulation Modeling and Analysis.* 3rd ed.
- ANSI/AIAA G-077-1998 Guide for the Verification and Validation of Computational
  Fluid Dynamics Simulations (used as a general V&V framework analogue).

---

## 2. System Boundary and Abstraction Level

### 2.1 In-Scope Components

| Physical Component | Model Abstraction |
|---|---|
| Lustre/HPC filesystem disk cache | G/G/1 queue with finite buffer |
| SAN interconnect (IB/100GbE) | M/D/1 queue (bandwidth-limited) |
| DMF data movers (N nodes) | N parallel M/G/1 servers |
| OpenVault library controller | M/G/1 queue (scheduling overhead) |
| Tape robot (single or dual arm) | M/G/1 queue with geometry-aware service time |
| Tape drives (N drives) | N parallel M/G/1 servers |
| Tape volumes (VSNs in VGs) | Finite resource pool with position state |
| DMF policy engine | Deterministic cyclic scheduler (D/D/1 trigger) |
| DMF BFID database (Cassandra) | Fixed-latency lookup (parameterised) |

### 2.2 Out-of-Scope (Explicitly)

- Network packet-level simulation (modelled as aggregate bandwidth)
- Cassandra cluster internals (modelled as fixed lookup latency)
- Mesos/Kafka/Spark internals (modelled as fixed scheduling overhead)
- Filesystem metadata operations (modelled as fixed latency)
- Physical tape wear / error rates (available as optional extension)

---

## 3. File State Machine (DMF 7 Faithful)

DMF 7 assigns each managed file one of the following states, derived from the
official DMF architecture documentation and `dmls` command output semantics.

```
States S = {REG, MIG, DUL, OFL, UNM, PAR}

REG  — Regular (fully online, no tape copy)
MIG  — Migrating (copy-to-tape in progress, online copy intact)
DUL  — Dual-state (tape copy exists, online copy intact; space not yet released)
OFL  — Offline (tape copy exists, online space released; stub in inode)
UNM  — Unmigrating (recall from tape in progress)
PAR  — Partial (some regions online, some offline; e.g., partial recall)
```

### 3.1 Transition Function δ: S × E → S

| From | Event E | To | Guard Condition |
|---|---|---|---|
| REG | policy_select | MIG | file_age ≥ t_age AND disk_usage ≥ HWM |
| REG | policy_select | MIG | disk_usage ≥ HWM (size-priority) |
| MIG | migrate_complete | DUL | tape write acknowledged |
| DUL | space_release | OFL | dmfsfree OR dmput -r OR HWM exceeded |
| OFL | read_access | UNM | open() or read() syscall on stub |
| OFL | dmget | UNM | explicit user recall request |
| UNM | recall_complete | DUL | tape read complete, data on disk |
| DUL | modify | REG | file modified (BFID soft-deleted) |
| OFL | dmunput | REG | explicit unput (removes tape copy) |
| PAR | recall_complete | DUL | all regions recalled |

**Note on MIG → DUL accrual:** Per DMF 7 architecture, the LS (library server) 
**accrues** migration requests until the data volume justifies a tape mount. 
This is modelled as a batching queue with configurable minimum batch size 
`LS_MIN_BATCH_MB` and maximum wait time `LS_MAX_WAIT_S`.

---

## 4. Queuing-Theoretic Models

### 4.1 Disk Cache — G/G/1 with Finite Buffer

The disk cache is modelled as a G/G/1 queue.

**Arrival process A(t):** General, determined by workload generator.  
**Service time S:** LogNormal(μ_disk, σ_disk), parameterised from disk I/O benchmarks.

**Performance metrics (Kingman's approximation for G/G/1):**

```
ρ = λ · E[S]                          (utilisation)

W_q ≈ (ρ / (1 - ρ)) · E[S] · (c_a² + c_s²) / 2    (mean waiting time)

where:
  c_a = σ_A / E[A]   (coefficient of variation of inter-arrival times)
  c_s = σ_S / E[S]   (coefficient of variation of service times)
```

**Buffer overflow model:** When buffer reaches capacity `DISK_BUFFER_MAX_FILES`,
new arrivals are rejected (EAGAIN) with probability p_block, modelled as:

```
p_block = P(N = K) in M/M/1/K (used as conservative upper bound)
```

### 4.2 SAN Interconnect — M/D/1

The interconnect between disk and tape movers is modelled as M/D/1 because
bandwidth-limited transfers have deterministic service times for a given file size.

**Service time:**
```
S_interconnect(f) = f.size_bytes / BW_interconnect_bytes_per_s
                  + latency_fixed_s
```

For InfiniBand (typical DMF 7 deployment): BW ≈ 10–100 GB/s configurable.

**M/D/1 mean queue length:**
```
L_q = ρ² / (2(1 - ρ))     [Pollaczek-Khinchine formula, c_s = 0]
```

### 4.3 Data Movers — N Parallel M/G/1 Servers

DMF 7 uses parallel data mover nodes. Each mover is an M/G/1 server.

**Service time per mover for file f:**
```
S_mover(f) = f.size_bytes / mover_bandwidth_bytes_per_s
           + mover_overhead_s
```

**For N movers (M/G/N queue), utilisation per server:**
```
ρ_mover = λ / (N · μ_mover)
```

**HBA bandwidth constraint (per DMF 7 config: HBA_BANDWIDTH):**
```
effective_bandwidth = min(NODE_BANDWIDTH, 
                          n_active_drives × drive_rate × BANDWIDTH_MULTIPLIER)
```

### 4.4 OpenVault Library Controller — M/G/1

The OpenVault layer adds scheduling overhead above the hardware robot.

**Service time:**
```
S_OV = OV_SCHEDULING_LATENCY_S   (fixed, configurable, typ. 0.5–2s)
```

### 4.5 Tape Robot — M/G/1 with Geometry-Aware Service Time

The robot service time depends on the physical geometry of the library.

**Single-arm robot service time (mount):**
```
S_robot_mount(src_slot, dst_drive) = 
    t_arm_travel(d(src_slot, dst_drive)) + t_cartridge_load

t_arm_travel(distance) = t_base + distance / v_arm

d(src_slot, dst_drive) = library_geometry.distance(src_slot, dst_drive)
```

**Spectra Logic TFinity dual-robot model:**
For dual-robot libraries (zoning support), two independent M/G/1 servers
serve disjoint zones. Cross-zone requests incur handoff latency.

**Typical parameters (LTO library):**
```
v_arm       ≈ 1.5 m/s  (horizontal)
t_load      ≈ 10–20s   (cartridge insertion to drive ready)
t_unload    ≈ 8–15s    (drive to slot)
t_base      ≈ 2s       (arm acceleration/deceleration)
```

### 4.6 Tape Drives — N Parallel M/G/1 Servers (per VG)

**Total service time for file f on tape volume v, current tape position p:**
```
S_drive(f, v, p) = S_mount(v) + S_seek(f, v, p) + S_transfer(f) + S_unload(v)
```

**Mount time** (if volume not loaded):
```
S_mount(v) = S_robot_mount + t_thread_tape + t_bot_seek
           ~ LogNormal(μ_mount, σ_mount)   [typ. μ ≈ 40s, σ ≈ 8s for LTO-8]
```

**Seek time** (block ID positioning, per LTO spec):
LTO drives support **Locate** command (block ID positioning). Seek time is
approximately linear in positional distance for LTO-8:

```
S_seek(f, v, p) = |block_id(f, v) - p| / tape_speed_blocks_per_s
               × seek_overhead_factor

For LTO-8:
  tape_speed   ≈ 360 MB/s native (720 MB/s compressed)
  seek_factor  ≈ 1.0–2.5× depending on direction and tension
```

**Transfer time:**
```
S_transfer(f) = f.size_bytes / drive_native_rate_bytes_per_s
              × (1 / compression_ratio)   [compression_ratio ~ LogNormal]
```

**DMF 7 recall optimisation:** Recall requests on a tape are sorted by chunk 
position (ascending). This is modelled as a **Shortest Seek Time First (SSTF)** 
variant on tape position, implemented in the recall queue scheduler.

---

## 5. Policy Engine Model (DMF 7 Faithful)

### 5.1 Policy Cycle

The policy engine runs as a periodic deterministic process:

```
T_next_cycle = T_current + T_cycle_period

At each cycle:
  candidates = {f ∈ filesystem : 
                  (f.age ≥ t_age_threshold) AND 
                  (f.state ∈ {REG, DUL})}
                  
  if disk_usage ≥ HWM:
      order candidates by (size DESC, age DESC)   [size-priority migration]
      migrate until disk_usage ≤ LWM
  elif disk_usage ≥ LWM:
      order candidates by (age DESC)              [age-priority migration]
      migrate oldest candidates
```

### 5.2 LS Accrual Buffer (DMF 7-specific)

```
accrual_buffer = []
accrual_pending_bytes = 0

On migration_request(f):
    accrual_buffer.append(f)
    accrual_pending_bytes += f.size

    if accrual_pending_bytes >= LS_MIN_BATCH_MB × 2^20:
        flush_buffer_to_drive()
    elif time_since_first_request >= LS_MAX_WAIT_S:
        flush_buffer_to_drive()
```

### 5.3 Trickle Migration (DMMIGRATE_TRICKLE)

When trickle mode is enabled:
```
max_concurrent_migrations = DMMIGRATE_UNACK   [default: configurable]
```
Implemented as a SimPy semaphore with capacity = DMMIGRATE_UNACK.

### 5.4 Space Watermark Feedback Control

The watermark system implements a **hysteresis controller**:

```
if disk_usage(t) > HWM:
    migration_mode = AGGRESSIVE   (all eligible files)
elif disk_usage(t) > LWM:
    migration_mode = NORMAL       (age-threshold only)
else:
    migration_mode = IDLE

Migration stops when disk_usage(t) ≤ LWM.
```

This prevents thrashing (rapid migration/recall cycles) near the watermark boundary.

---

## 6. Workload Generator

### 6.1 File Size Distribution

Empirical storage workloads in HPC environments follow a log-normal distribution
(Smirni & Reed, 1998; Schroeder & Gibson, 2007):

```
F ~ LogNormal(μ_f, σ_f)

Recommended initial parameters (HPC workloads):
  μ_f = ln(10 × 2^20)   [~10 MB median]
  σ_f = 2.5              [heavy right tail]

P(F ≤ x) = Φ((ln x - μ_f) / σ_f)
```

### 6.2 File Inter-Arrival Time (IAT)

HPC filesystem traffic exhibits **self-similarity** (long-range dependence).
The Hurst parameter H > 0.5 for real storage workloads (Leland et al., 1994).

For tractable simulation, we use a **Pareto (Type II / Lomax) distribution**
which produces heavy-tailed IATs consistent with H ≈ 0.75–0.85:

```
IAT ~ Pareto(α, x_min)

E[IAT]  = x_min / (α - 1)   [Lomax/Pareto Type II mean; requires α > 1]
Var[IAT] = x_min² × α / ((α-1)²(α-2))  for α > 2

Recommended: α = 1.8, x_min = configurable (controls load intensity)
```

For simpler (Poisson) validation runs:
```
IAT ~ Exponential(λ)   [enables analytical M/M/1 comparison]
```

### 6.3 File Access Frequency Distribution

File access follows **Zipf's law** (a small fraction of files account for most I/O):

```
P(file rank r accessed) ∝ 1/r^s

s ≈ 0.8–1.2 for HPC workloads
```

Implemented as a discrete Zipf distribution over the file working set.

### 6.4 Read/Write Ratio

```
P(operation = READ) = r_w   [configurable, default 0.7]
P(operation = WRITE) = 1 - r_w
```

### 6.5 Diurnal Access Pattern

A multiplicative time-of-day modulator:

```
λ(t) = λ_base × Φ_diurnal(t mod T_day)

Φ_diurnal(τ) = 1 + A_diurnal × sin(2π(τ - τ_peak) / T_day)

Parameters:
  A_diurnal = 0.6   [peak:trough ratio ≈ 2.5:1]
  τ_peak    = 10h   [10am peak, configurable]
  T_day     = 86400s
```

---

## 7. Random Number Generation

Per Law & Kelton (2000), each stochastic input must use an **independent RNG stream**
to avoid correlation artifacts between components.

**Generator:** numpy.random.PCG64 with seed splitting.

```python
# Stream assignment
STREAM_ASSIGNMENTS = {
    'file_size':        0,
    'iat':              1,
    'access_freq':      2,
    'rw_ratio':         3,
    'mount_time':       4,
    'seek_time':        5,
    'transfer_time':    6,
    'compression':      7,
    'disk_service':     8,
    'robot_travel':     9,
    'replication_seed': 10,   # per-replication base seed
}
```

Each stream is created as an independent numpy Generator from a seeded SeedSequence:
```python
ss = np.random.SeedSequence(master_seed)
child_seeds = ss.spawn(N_STREAMS)
streams = [np.random.default_rng(s) for s in child_seeds]
```

---

## 8. Statistical Output Analysis Framework

### 8.1 Warm-Up Period Detection (Welch's Method)

To remove initialisation bias (Law & Kelton, 2000, §9.5):

```
Given time-series {Y_1, Y_2, ..., Y_n} of observation i:

Moving average: Ȳ_i(w) = (1/(2w+1)) Σ_{j=i-w}^{i+w} Ȳ_j(w)

Warm-up period L = first i such that Ȳ_i(w) has stabilised
(stabilisation criterion: |Ȳ_i(w) - Ȳ_{i-1}(w)| / Ȳ_i(w) < ε, ε = 0.01)
```

### 8.2 Replication Methodology

```
N_rep independent replications, each with a different seed.

For each KPI metric M:
  Sample mean:    X̄ = (1/N_rep) Σ X_i
  Sample std dev: S  = sqrt((1/(N_rep-1)) Σ (X_i - X̄)²)
  95% CI:         X̄ ± t_{N_rep-1, 0.025} × S / sqrt(N_rep)

Minimum N_rep = 30 for CLT applicability.
```

### 8.3 Little's Law Validation

At every queue Q, verify:

```
L_Q = λ_Q × W_Q

Test: |L̂_Q - λ̂_Q × Ŵ_Q| / L̂_Q < δ_tolerance

where δ_tolerance = 0.05 (5% relative error tolerance)

Failure of Little's Law indicates a model defect or insufficient warm-up.
```

### 8.4 Analytical Baseline Comparisons

For the degenerate M/M/1 case (Poisson arrivals, exponential service):

```
Expected: W = 1 / (μ - λ)     [mean sojourn time]
Expected: L = λ / (μ - λ)     [mean queue length]
Expected: W_q = ρ / (μ - λ)   [mean waiting time]
Expected: L_q = ρ² / (1 - ρ)  [mean queue length waiting]

Simulator must reproduce these within ± 5% at 95% CI when configured
for M/M/1 parameters.
```

---

## 9. Key Performance Indicators (KPIs)

| KPI | Symbol | Definition | Unit |
|---|---|---|---|
| Recall latency (P50/P95/P99) | L_recall | Stub access → first byte delivered | seconds |
| Cache hit rate | H_cache | Requests served from disk / total requests | dimensionless |
| Tape drive utilisation | U_drive_i | Fraction of time drive i is busy | dimensionless |
| Robot utilisation | U_robot | Fraction of time robot is busy | dimensionless |
| Mover utilisation | U_mover_i | Fraction of time mover i is busy | dimensionless |
| Migration lag | Δ_mig | Age of oldest un-migrated eligible file | seconds |
| Policy cycle efficiency | η_policy | Files migrated / eligible files per cycle | dimensionless |
| Queue depth | Q_i(t) | Queue length at resource i as time series | files |
| Throughput | Γ_i | Data rate at tier i | bytes/s |
| LS accrual wait | T_accrue | Mean time in LS accrual buffer | seconds |
| Mount frequency | f_mount | Volume mounts per unit time | mounts/hour |
| Disk usage | U_disk(t) | Fraction of disk capacity used | dimensionless |

---

## 10. Trace File Format Specification

DMF 7 produces observable data via `dmstat`, `dmqview`, `dmls`, and system logs.
The simulator accepts a **normalised trace format** (CSV or JSON Lines) with the
following required fields:

```
timestamp_s         float   Seconds since simulation epoch
event_type          str     One of: MIGRATE, RECALL, POLICY_CYCLE, SPACE_RELEASE,
                                     FILE_CREATE, FILE_DELETE, FILE_MODIFY
file_id             str     DMF BFID or path hash (anonymised acceptable)
file_size_bytes     int     File size in bytes
file_state_before   str     One of: REG, MIG, DUL, OFL, UNM, PAR
file_state_after    str     (same enum)
vsn                 str     Volume serial number (optional, for tape events)
drive_id            int     Drive index (optional, for tape events)
latency_s           float   Observed latency for this event (optional)
queue_depth         int     Queue depth at time of event (optional)
disk_usage_pct      float   Disk tier usage % at time of event (optional)
```

**Trace replay mode:** In replay mode, the workload generator is replaced by the
trace reader. All stochastic service times still apply (the trace provides arrival
patterns only, not internal resource service times, unless `latency_s` is present
for validation comparison).

---

## 11. Verification & Validation Plan

### Phase 1 — Unit Verification
- Each component tested in isolation against known-answer analytical solutions
- M/M/1 degenerate case: configure disk cache as M/M/1, verify W = 1/(μ-λ) ± 5%
- Robot: verify mean service time matches configured parameters ± 1%

### Phase 2 — Integration Verification
- Little's Law holds at all queues simultaneously
- File state machine: verify no invalid state transitions occur (assert-based)
- Conservation check: every file created is either on disk or tape (no data loss events)

### Phase 3 — Validation Against DMF 7
- Feed DMF 7 trace files; compare simulated recall latency distribution against observed
- KS test: H₀ = simulated and observed latency distributions are from same population
  - Accept H₀ at α = 0.05 significance level
- Calibrate drive and robot timing parameters until KS test passes

---

## 12. Assumptions and Limitations

1. **Independence assumption:** File arrivals are modelled as independent. In reality,
   HPC workloads exhibit job-level correlations (burst arrivals). The Pareto IAT
   distribution partially captures burstiness but not job structure.

2. **Homogeneous tape drives:** All drives within a VG are assumed identical.
   Heterogeneous drive types require extension.

3. **Perfect tape reliability:** Tape errors and retries are not modelled in v1.0.

4. **Single filesystem:** v1.0 models a single Lustre filesystem. Multi-filesystem
   DMF 7 configurations require extension.

5. **Cassandra latency:** Modelled as fixed. In practice, Cassandra latency under
   high load is non-trivial.

6. **Network contention:** SAN modelled as aggregate bandwidth. Per-flow contention
   and HoL blocking not modelled.
