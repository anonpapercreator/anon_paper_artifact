# DESCASSI Experimental Protocol

This directory contains reproducible experiments for DESCASSI. E1 to E4 are simulator
experiments; the lifetime study (E7 to E10) is described in `../LIFETIME_STUDY.md`.

## Quick Start

```bash
# E1: M/M/1 analytical baseline (30 replications)
bash experiments/e01_mm1_baseline/run.sh

# E2: Trace calibration validation
bash experiments/e02_trace_calibration/run.sh

# E3: Policy sweep (12 values x 10 replications)
python experiments/e03_policy_sweep/run_sweep.py

# E4: Phase-adaptive vs static comparison
python experiments/e04_adaptive_vs_static/run_comparison.py
```

## Experiment Summaries

### E1 — M/M/1 Analytical Baseline
**Purpose:** Verify DESCASSI queueing mechanics against closed-form M/M/1.
**Config:** Poisson arrivals (λ=0.8), exponential service (μ=1.0), single server.
**Duration:** 3,600s per replication, 600s warmup, 30 replications.
**Validation:** Theoretical ρ=0.8, Wq=4.0s. Target: simulated Wq within ±5%.

### E2 — Trace Calibration Validation
**Purpose:** Validate calibrated parameters against production DMF 7 trace.
**Config:** Trace replay mode, 42,438 events, calibrated parameters.
**Duration:** Full trace replay (~60,000s), 1 replication (trace is deterministic).
**Validation:** KS distance between simulated and observed recall latency CDFs.

### E3 — Policy Sweep (τ*)
**Purpose:** Demonstrate existence of finite optimal archive interval.
**Config:** Synthetic workload, weights (wp,wm,wr)=(1.0,0.5,0.8).
**Parameters:** τ ∈ [100,500,1000,1800,3600,7200,14400,28800,43200,57600,72000,86400]s.
**Duration:** 3,600s per replication, 600s warmup, 10 replications per τ.
**Output:** J(τ) curve with 95% CI error bars.

### E4 — Phase-Adaptive vs Static
**Purpose:** Demonstrate CTMC-driven adaptive policy outperforms static threshold.
**Config:** Synthetic workload with 15-minute COMPUTE↔CHECKPOINT phase transitions.
**Duration:** 3,600s per replication, 600s warmup, 10 replications per policy.
**Metrics:** Mean queue wait, tape cartridge writes, peak data-at-risk.

## Data Management

All results are stored in per-experiment `results/` subdirectories:
- `simulation_report.json` — Full KPI report with 95% CIs
- `kpi_summary.csv` — Flat CSV for analysis
- `replication_results.jsonl` — Per-replication raw data

## Reproducibility

- All experiments use fixed RNG seeds (seed=42 + replication offset).
- Config files are self-contained YAML (no external dependencies beyond the trace).
- Exact regeneration commands are documented above.
- Result files include metadata: timestamp, config hash, simulator version.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py validate-config --config config/default_config.yaml
```
