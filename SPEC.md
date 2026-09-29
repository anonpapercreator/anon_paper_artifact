# Hierarchical Storage Management System DES Specification
## DMF 7 Simulator with NVMe Cache + IBM TS1160 Tape

**Version:** 2.0  
**Date:** 2026-03-14  
**Target System:** HPE DMF 7 with NVMe Cache Tier + IBM TS1160  
**Language:** Python / SimPy 4.x  

---

## 1. Executive Summary

This specification extends the existing DMF 7 DES (v1.0) to model a two-tier hierarchical storage system consisting of:
- **Tier 1 (Cache):** NVMe scratch disk, 1 PiB usable capacity
- **Tier 2 (Archive):** IBM TS1160 tape drives, up to 16 concurrent drives

The simulator supports both synthetic workload generation and trace-driven replay using real DMF 7 logs. Validation against your production DMF system is a primary requirement.

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT SOURCES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. DMF 7 Trace File (YOUR LOG)                                           │
│     - 88,250 lines, 16.6 hours                                             │
│     - 11,635 GetFile (RECALL), 9,596 PutFile (MIGRATE)                    │
│     - 5 volume groups: vg1_j02, vg2_j02, vg3_j02, vgb_j02, vg4_j02       │
│                                                                             │
│  2. Synthetic Generator (parameterizable)                                    │
│     - File size: LogNormal(μ=16.1, σ=2.5) → ~10 MB median                 │
│     - Inter-arrival: Pareto (self-similar HPC workload)                    │
│     - Access frequency: Zipf(α=0.85)                                        │
│     - Diurnal modulation                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVENT SCHEDULER                                       │
│                    (SimPy Priority Queue)                                    │
│    - Timestamp-ordered events                                               │
│    - Event types: REQUEST, CACHE_HIT, CACHE_MISS, RECALL, MIGRATE, etc.   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌──────────────────────────┴──────────────────────────┐
          ▼                                                     ▼
┌─────────────────────────┐                       ┌─────────────────────────┐
│    NVMe CACHE TIER     │                       │      TAPE TIER         │
│   (1 PiB Scratch)      │                       │   (IBM TS1160 x 16)   │
├─────────────────────────┤                       ├─────────────────────────┤
│ M/G/1 Queue            │                       │ M/G/k Queue (k=16)    │
│ - IOPS limited         │                       │ - Robot arm queue      │
│ - LRU cache manager    │                       │ - Drive queues         │
│ - Age tracking         │                       │ - Volume groups        │
│ - Prefetch predictor   │                       │ - Mount/unmount        │
└─────────────────────────┘                       └─────────────────────────┘
          │                                                     │
          ▼                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      METRICS COLLECTOR                                      │
│  - Latency percentiles (p50, p95, p99)                                     │
│  - Cache hit ratio                                                          │
│  - Drive/robot utilization                                                  │
│  - Cost analytics (TCO, $/operation)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Abstraction | Key Parameters |
|-----------|-------------|----------------|
| NVMe Cache | M/G/1 with IOPS limits | 1 PiB, 750K IOPS, 5 GB/s |
| Migration Policy | Time-based + Prefetch | Age threshold, evaluation interval |
| Tape Robot | M/G/1 with geometry | 1 arm, 500 slots |
| Tape Drives | N×M/G/1 (k=16) | IBM TS1160, 400 MB/s |
| Volume Groups | Resource pools | 5 VGs from your log |

---

## 3. Mathematical Models

### 3.1 NVMe Cache Tier (M/G/1 with IOPS Limits)

**Service Time Model:**
```
S_cache(f) = min(Size / R_sequential, 1/IOPS_random)

Where:
- R_sequential = 5,000 MB/s (configurable: 3,000-7,000 MB/s)
- IOPS_random = 750,000 IOPS (configurable: 500K-1M)
- Service time CV: c_s ≈ 0.35 (mixed workload)
```

**Queueing Delay (Kingman approximation):**
```
W_q ≈ (ρ / (1 - ρ)) × E[S] × (c_a² + c_s²) / 2

Where:
- ρ = λ × E[S] (utilization)
- c_a = σ_A / E[A] (arrival CV)
- c_s = σ_S / E[S] (service CV)
```

**Cache Hit Probability (Zipf popularity):**
```
P(hit) = Σ_{i=1}^{k} (1/i^α) / Σ_{i=1}^{N} (1/i^α)

Where:
- α = 0.85 (typical HPC workload)
- k = cache capacity / average file size
- N = total files in working set
```

### 3.2 Migration Policy (Time-Based + Prefetch)

**Trigger Condition:**
```
∀f ∈ Cache: if (now - f.last_access_time) > T_age_threshold
             then migrate(f)

Parameters:
- T_age_threshold = 30 days (configurable: 1-365 days)
- Evaluation interval τ = 3,600 seconds (configurable)
```

**Migration Selection (LRU within age threshold):**
```
f* = argmin_{f: age(f) > T_age} (age(f))
```

**Prefetch Model (Access Pattern-Based):**
```
If access pattern detected (periodic job):
   // Detect periodicity via autocorrelation
   If period detected with confidence > 0.75:
      Pre-migrate files likely to be accessed
      Prefetch benefit = E[future_access_latency] - prefetch_cost
```

**Space-Based Eviction (Hybrid):**
```
When cache_utilization > HWM (90%):
   Migrate oldest files until LWM (80%) reached
   Files remain in DUL state until space needed → then DUL→OFL
```

### 3.3 Tape Tier (IBM TS1160)

**IBM TS1160 Specifications:**

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Native capacity | C_tape | 20 TB | bytes (JE media) |
| Sustained read rate | R_read | 400 MB/s | bytes/s |
| Sustained write rate | R_write | 400 MB/s | bytes/s |
| Search speed | v_search | 12.4 | m/s |
| Load time (median) | t_load | ~15 | seconds |
| Unload time (median) | t_unload | ~3 | seconds |
| MTBF | MTBF | 250,000 | hours |

**Total Recall Service Time:**
```
S_recall(f) = t_mount + t_seek + t_read + t_unmount

Where:
- t_mount ~ LogNormal(μ=3.0, σ=0.4) ≈ 20-30s median
- t_seek = position / v_search (if repositioning)
- t_read = Size / R_read
- t_unmount ~ LogNormal(μ=1.1, σ=0.3) ≈ 3s median
```

### 3.4 Declarative Policy Derivation

**From user goal to system parameters:**
```
cache_priority (0.0-1.0) → age_threshold:
    age_threshold = base_age × cache_priority^0.5
    
cache_priority → batch_size:
    batch_size = max_batch / (1 + cache_priority × 2)
    
cache_priority → prefetch_aggression:
    prefetch_aggression = cache_priority
```

### 3.5 Cost Model

**Total Cost:**
```
C_total = C_capex + C_opex + C_wear

Where:
- C_capex = (Cost_NVMe × Cache_TB) / Amortization_years
- C_opex = Power_W × Hours/year × $/kWh
- C_wear = (TBW_written / TBW_rated) × Cost_replacement
```

---

## 4. Input Data: Your DMF 7 Log

### 4.1 Log File Summary

| Property | Value |
|----------|-------|
| **Source** | `ls.43a50397ada24f999bca01a2eefa72c5.log` |
| **Format** | ls-agent (Library Server) log |
| **Duration** | 16.6 hours (2026-03-12 00:01:32 → 16:36:48) |
| **Total Events** | 88,250 lines |
| **Recall Requests** | 11,635 GetFile operations |
| **Migration Requests** | 9,596 PutFile operations |
| **Completions** | 44,002 |

### 4.2 Log Field Mapping

| Raw Field | Normalized Field | Notes |
|-----------|-----------------|-------|
| ISO timestamp | timestamp_s | Convert to seconds since epoch |
| "operation":"GetFile" | event_type: RECALL | |
| "operation":"PutFile" | event_type: MIGRATE | |
| "key" | file_id | DMF identifier |
| "length" | file_size_bytes | File size in bytes |
| "volGrp" | vsn | Volume group |

### 4.3 Observed Latency Distribution

| Percentile | Value |
|------------|-------|
| Mean | 95.56 seconds |
| p50 | 21.20 seconds |
| p95 | 308.85 seconds |
| p99 | 1334.51 seconds |
| Max | 7114.73 seconds |

---

## 5. Policy Presets

| Preset | Cache Priority | Age Threshold | Batch Size | Prefetch | Description |
|--------|---------------|---------------|------------|----------|-------------|
| `maximum_performance` | 1.0 | 30 days | 1 TB | 1.0 | Minimize recall latency |
| `balanced` | 0.7 | 25 days | 10 GB | 0.5 | Balance cost/performance |
| `archive_optimized` | 0.3 | 16 days | 20 GB | 0.0 | Minimize media costs |

---

## 6. Validation Framework

### 6.1 KS-Test Against Your DMF 7

```
Procedure:
1. Run simulation with your DMF trace as input
2. Extract simulated recall latency distribution
3. Extract observed latencies from your log
4. Kolmogorov-Smirnov test:
   - H₀: simulated and observed from same population
   - Accept at α = 0.05 significance level
```

### 6.2 Sanity Checks

| Check | Method | Tolerance |
|-------|--------|-----------|
| Little's Law | L = λW at each queue | < 5% error |
| Conservation | Files created = Files (cache + tape) | 0% error |
| State Machine | No invalid transitions | Assert-based |
| Utilization | ρ < 1.0 for all queues | Hard limit |

---

## 7. Output Metrics

### 7.1 Primary KPIs

| Metric | Symbol | Description | Unit |
|--------|--------|--------------|------|
| Cache hit ratio | H | Requests served from cache | dimensionless |
| Recall latency p50 | L₅₀ | 50th percentile recall latency | seconds |
| Recall latency p95 | L₉₅ | 95th percentile recall latency | seconds |
| Recall latency p99 | L₉₉ | 99th percentile recall latency | seconds |
| Drive utilization | Uᵈᵢ | Fraction drive i busy | dimensionless |
| Cost per recall | C_recall | Cost per recall operation | $ |

### 7.2 Cost Metrics

| Metric | Description |
|--------|-------------|
| $/recall | Cost per recall operation |
| $/TB/year | Annual cost per TB managed |
| Power cost | $/kW/month × kW used |

---

## 8. Implementation

### 8.1 New Files

| File | Purpose |
|------|---------|
| `workload/dmf_log_parser.py` | Parse your raw DMF log → normalized trace |
| `tiers/cache_tier.py` | NVMe cache model with IOPS limits |
| `tiers/ts1160_model.py` | IBM TS1160 parameters |
| `policies/migration_policy.py` | Age-based migration + prefetch |
| `analytics/cost_analytics.py` | TCO calculations |
| `validation/validate_against_dmf.py` | KS-test validation |

### 8.2 Configuration

All parameters are configurable in `default_config.yaml`:

- NVMe cache: capacity, IOPS, latency
- Policy: preset, age threshold, batching
- Tape: drive count, mount times, volume groups
- Cost: power ($427/kW/month), wear, maintenance

---

## 9. Research Contribution

This simulator demonstrates:

1. **Declarative Policy Control** - Users specify WHAT they want (performance vs cost), system derives HOW to achieve it
2. **Hybrid Storage** - Combines NVMe scratch + tape archive in unified model
3. **Policy Comparison** - Compare different HSM policies declaratively
4. **Validation** - KS-test against real DMF 7 production data

This is more valuable than static policies because:
- Users can express "I need fast recall for this project" → use more cache
- Users can express "I want to save costs" → use more tape, accept slower recalls
- The simulation shows the cost/performance tradeoffs for each configuration

---

## 10. References

- Law, A.M. & Kelton, W.D. (2000). Simulation Modeling and Analysis.
- Iliadis, I. et al. (2016). "Tape Storage Systems: Analytical Models and Performance Evaluation"
- Breslau, L. et al. (1999). "Web caching and Zipf-like distributions"
- IBM TS1160 Product Documentation
- HPE DMF 7 Architecture Documentation
