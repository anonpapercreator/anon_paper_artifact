# DESCASSI - Discrete Event Simulator for Combined Archival and Scratch Storage Infrastructure

A scientifically rigorous Discrete Event Simulation (DES) of Hierarchical Storage
Management (HSM) systems, built with Python and SimPy. Designed for academic-quality
HSM performance research with full statistical output analysis. Supports both
DMF 7 and MSP (P4) log formats.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
3b. [Web Interface (Streamlit)](#3b-web-interface-streamlit)
4. [CLI Commands Reference](#4-cli-commands-reference)
5. [Configuration Reference](#5-configuration-reference)
6. [Understanding the Progress Display](#6-understanding-the-progress-display)
7. [Understanding Simulation Outputs](#7-understanding-simulation-outputs)
8. [Trace Replay Mode](#8-trace-replay-mode)
9. [Validation and Scientific Rigour](#9-validation-and-scientific-rigour)
10. [Running the Test Suite](#10-running-the-test-suite)
11. [Experiment Design Guide](#11-experiment-design-guide)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Structure

```
hsm_sim/
│
├── cli/
│   ├── web_app.py                 Streamlit web interface (recommended)
│   └── simple_cli.py              Command-line interface
│
├── core/
│   ├── simulator.py               SimPy-based discrete event simulator
│   └── sweeper.py                 Parameter sweep runner
│
├── config/
│   ├── models.py                  Pydantic configuration models
│   └── default_config.yaml        Default configuration
│
├── db/
│   └── database.py                SQLite database for results
│
├── tiers/
│   ├── cache_tier.py              ESS 3500 cache model
│   └── ts1160_model.py            TS1160 tape drive model
│
├── policies/
│   └── migration_policy.py        Declarative migration policies
│
├── workload/
│   └── dmf_log_parser.py          DMF 7 log file parser
│
├── analytics/
│   └── cost_analytics.py           TCO and cost calculations
│
├── validation/
│   └── validate_against_dmf.py    KS-test validation framework
│
├── requirements.txt               Python dependencies
├── QUICKSTART.md                  Quick start guide
├── SPEC.md                        Technical specification
└── README.md                      This file
```

---

## 2. Installation

### Requirements
- Python 3.10 or later
- macOS, Linux, or Windows

### Step-by-step

```bash
# 1. Navigate into the project folder
cd files_DES_20260312        # or whatever you named it

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify everything is working
python main.py validate-config --config config/default_config.yaml
```

Expected output from step 5:
```
  Configuration valid: config/default_config.yaml
  Replications  : 30
  Duration      : 14400s
  Seed          : 42
  ...
```

If you see this, you are ready to run.

---

## 3. Quick Start

### Fastest possible test (5 replications, 1 simulated hour)
```bash
python main.py run \
  --config config/default_config.yaml \
  --replications 5 \
  --duration 3600 \
  --output-dir ./results_test
```

### Full production run (30 replications, 4 simulated hours)
```bash
python main.py run --config config/default_config.yaml
```
Results go to `./results/` by default (set in `config/default_config.yaml`).

### Check M/M/1 analytical values before running
```bash
python main.py analytical-baseline --arrival-rate 1.0 --service-rate 2.0
```

---

## 3b. Web Interface (Streamlit)

The simulator includes a modern web interface built with Streamlit.

### Starting the Server

```bash
cd <path-to-repository>
streamlit run cli/web_app.py --server.port=8501
```

### Accessing the Interface

**Local access:**
- Open http://localhost:8501 in your browser

**Remote access via SSH tunnel:**
```bash
# From your local machine
ssh -L 8501:localhost:8501 user@your-server
```
Then open http://localhost:8501 in your browser.

### Stopping the Server

Press `Ctrl+C` in the terminal running streamlit, or:
```bash
ps aux | grep streamlit
kill <PID>
```

### Web Interface Pages

| Page | Description |
|------|-------------|
| 🏠 Home | System overview and quick stats |
| 📂 Parse Logs | Parse DMF log files into trace format |
| ⚙️ Configure | Set simulation parameters |
| 🔬 Parameter Sweeps | Run multi-parameter experiments |
| 📊 Reports | View charts and export results |
| ✅ Validate | Compare against real DMF data |

### Features

- **Parameter Sweeps**: Test multiple age thresholds (5, 10, 20, 30, 40, 50, 60+ min) in one run
- **Charts**: Visualize media consumption and bandwidth impact
- **Export**: Download results as CSV
- **Database**: All results stored in SQLite for later analysis

---

## 4. CLI Commands Reference

All commands are run as `python main.py <command> [options]`.

---

### `run` — Execute the simulation

```bash
python main.py run [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--config`, `-c` | `config/default_config.yaml` | Path to YAML config file |
| `--replications`, `-r` | (from config) | Number of independent replications |
| `--seed`, `-s` | (from config) | Master RNG seed for reproducibility |
| `--duration`, `-d` | (from config) | Simulated duration in seconds |
| `--output-dir`, `-o` | (from config) | Directory to write result files |
| `--chunk` | `60` | Simulated seconds between progress display updates |

**Examples:**

```bash
# Quick 1-hour test, results in ./test_out/
python main.py run -c config/default_config.yaml -r 5 -d 3600 -o ./test_out

# Reproduce an exact previous run
python main.py run -c config/default_config.yaml --seed 42

# Run with faster progress bar updates
python main.py run -c config/default_config.yaml --chunk 30
```

---

### `validate-config` — Check a config file without running

```bash
python main.py validate-config --config config/default_config.yaml
```

Checks that:
- All required sections are present
- Watermarks satisfy 0 < LWM < HWM < 1
- Drive counts are consistent
- Pareto alpha > 1 (required for finite mean inter-arrival time)
- read_fraction is in [0, 1]

Run this every time you edit `default_config.yaml` before launching a simulation.

---

### `analytical-baseline` — Compute M/M/1 ground truth

```bash
python main.py analytical-baseline --arrival-rate 1.0 --service-rate 2.0
```

Computes and prints the exact analytical solution for an M/M/1 queue:
- λ = arrival rate (items/second)
- μ = service rate (items/second)
- ρ = utilisation = λ/μ  (must be < 1 for stable queue)

**Use this to validate the simulator:** configure the disk cache as a pure M/M/1
(exponential IAT, exponential service times) and confirm the simulator reproduces
these values within the 95% CI. This is Phase 1 of the V&V plan.

---

### `trace-template` — Generate a trace file template

```bash
python main.py trace-template --output my_trace.csv --n-events 500
```

Generates a CSV file with the correct column structure for DMF trace replay.
Fill this with real DMF 7 log data to run in trace replay mode.

| Option | Default | Description |
|---|---|---|
| `--output`, `-o` | `trace_template.csv` | Output file path |
| `--n-events`, `-n` | `200` | Number of example rows to generate |
| `--format` | `csv` | `csv` or `jsonl` |

---

## 5. Configuration Reference

All parameters are in `config/default_config.yaml`. The file is self-documenting
with inline comments. Key sections:

---

### `simulation` — Top-level run parameters

```yaml
simulation:
  seed: 42                  # Master RNG seed. Change this to get a different
                            # but reproducible run. Same seed = same results.
  n_replications: 30        # Number of independent simulation runs.
                            # Must be >= 30 for statistically valid 95% CIs.
  sim_duration_s: 14400     # Simulated duration per replication (seconds).
                            # 14400 = 4 hours. Start with 3600 for testing.
  warmup_s: 1800            # Warm-up period in simulated seconds.
                            # Statistics collected AFTER this point only.
                            # Rule of thumb: ~10-20% of sim_duration_s.
  output_dir: ./results     # Where to write result files.
  trace_interval_s: 60      # How often (sim-seconds) to snapshot queue depths.
```

---

### `disk_cache` — Disk tier (Lustre / HPE XFS)

```yaml
disk_cache:
  capacity_bytes: 107374182400   # Total disk tier capacity (bytes). Default 100 GiB.
  hwm_fraction: 0.90             # HIGH-WATER MARK: when disk reaches 90% full,
                                 # migration switches to AGGRESSIVE mode — all
                                 # eligible files are migrated until LWM is reached.
  lwm_fraction: 0.80             # LOW-WATER MARK: migration stops when disk drops
                                 # to 80% full. The gap between HWM and LWM is the
                                 # hysteresis band that prevents thrashing.
  initial_fill_fraction: 0.50    # How full the disk starts at simulation begin.
```

**Experiment tip:** Narrowing the HWM-LWM gap (e.g., 0.88/0.86) causes more
frequent but smaller migration bursts. Widening it (e.g., 0.95/0.70) causes
less frequent but very large bursts. This directly affects recall latency
variance and robot utilisation.

---

### `policy_engine` — DMF 7 Migration Policy

```yaml
policy_engine:
  cycle_period_s: 300              # Policy engine runs every 5 minutes.
                                   # This is the primary policy timing knob.
  candidate_age_threshold_s: 1800  # Files must be at least 30 minutes old
                                   # (since last access) to be migration candidates.
  migration_priority: size_descending  # How to order candidates:
                                   # 'size_descending': largest files first
                                   # 'age_descending':  oldest files first
                                   # 'combined':        weighted mix of both
  space_release_on_migration: false  # false = REG -> MIG -> DUL (tape copy made,
                                   # online copy kept until space needed)
                                   # true  = REG -> MIG -> OFL (online copy
                                   # released immediately after tape write)
  dual_copy: true                  # Write to both VGs for data protection.
                                   # Doubles tape write load.
  recall_sort: tape_order          # Sort pending recalls by tape position
                                   # before mounting (minimises seek time).
                                   # DMF 7 default behaviour.
```

**The two most important policy levers for experiments are:**
1. `cycle_period_s` — longer cycles mean less frequent policy runs but
   larger backlogs can build up before migration starts.
2. `candidate_age_threshold_s` — lower values migrate files sooner,
   keeping the disk cleaner but increasing tape mount frequency.

---

### `tape_drives` — Tape Drive Characteristics

```yaml
tape_drives:
  n_drives: 8                          # Number of tape drives. Must equal
                                       # tape_robot.n_drives.
  native_rate_bytes_per_s: 360000000   # LTO-8 native rate: 360 MB/s
  compressed_rate_bytes_per_s: 720000000  # LTO-8 compressed: 720 MB/s
  mount_time_lognormal_mu_s: 40.0      # Mean mount time ~40 seconds (LTO-8)
  seek_speed_factor: 1.5               # Seek overhead above streaming speed
  tape_capacity_bytes: 12000000000000  # 12 TB native per cartridge (LTO-8)
```

---

### `library_server` — LS Accrual Buffer (DMF 7-specific)

```yaml
library_server:
  ls_min_batch_bytes: 10737418240   # LS accrues migration requests until 10 GB
                                    # of data is pending before mounting a tape.
                                    # Smaller = more mounts, less wait.
                                    # Larger = fewer mounts, longer wait.
  ls_max_wait_s: 300                # Even if batch threshold not reached, flush
                                    # the accrual buffer after 5 minutes max.
```

This models a real DMF 7 behaviour: the Library Server batches migration requests
to amortise the mount cost across multiple files. Tuning these two parameters
directly controls the trade-off between mount frequency and migration latency.

---

### `workload` — Workload Generator

```yaml
workload:
  mode: synthetic            # 'synthetic': generate workload from distributions
                             # 'trace_replay': replay a real DMF log file

  # File size distribution (LogNormal — empirically validated for HPC)
  file_size_lognormal_mu: 16.118     # ln(10 MiB) — median file size ~10 MB
  file_size_lognormal_sigma: 2.5     # Heavy right tail (many large files)

  # Inter-arrival time (Pareto = self-similar, like real HPC storage traffic)
  iat_distribution: pareto           # 'pareto' for realistic bursty traffic
                                     # 'exponential' for M/M/1 validation runs
  iat_pareto_alpha: 1.8              # Shape parameter. Lower = burstier.
                                     # Must be > 1.0 for finite mean.
  iat_pareto_xmin_s: 0.5             # Minimum IAT. Controls overall load.
                                     # Smaller = more arrivals = heavier load.

  # Access frequency (Zipf — a few files are accessed most)
  working_set_size: 10000            # Number of distinct files in the system
  zipf_s: 1.0                        # Zipf exponent. Higher = more skewed.

  # Diurnal pattern (time-of-day load variation)
  diurnal_enabled: true
  diurnal_amplitude: 0.6             # 0 = flat load. 0.6 = peak is ~2.5x trough.
  diurnal_peak_hour: 10              # Peak activity at 10am.
```

---

## 6. Understanding the Progress Display

When you run `python main.py run`, you will see two lines that update in place:

```
  Reps [######--------------]  6/30  |  SimTime [##########----------]  50.0%  elapsed=1m23s  ETA=1m21s
  files=3,847  migrated=312  recalled=47  cache=74.3%  disk=87.1%
```

### Line 1 — Progress bars

| Field | Meaning |
|---|---|
| `Reps [######----] 6/30` | How many replications have completed out of the total. Each replication is a full independent run of the simulation. |
| `SimTime [##########--] 50.0%` | How far through the current replication's simulated time period. |
| `elapsed=1m23s` | Wall-clock time elapsed since this replication started. |
| `ETA=1m21s` | Estimated wall-clock time remaining for this replication, extrapolated from current pace. |

### Line 2 — Live counters

| Field | Meaning |
|---|---|
| `files=3,847` | Total number of files created/written to the filesystem so far in this replication. |
| `migrated=312` | Files successfully migrated to tape in this replication. |
| `recalled=47` | Files recalled from tape back to disk (triggered by a read access on an offline file). |
| `cache=74.3%` | Cache hit rate: fraction of file accesses served directly from disk without a tape recall. Higher is better. A healthy DMF system typically runs 85-95%+. |
| `disk=87.1%` | Current disk tier fill level as a percentage of capacity. You should see this oscillate around the LWM/HWM band as the policy engine manages space. |

### After each replication completes

```
  Replication 6/30 done  (1m44s wall-time,  migrations=312,  recalls=47)
```

This pins a summary line for the completed replication before starting the next.

### Tuning the display refresh rate

`--chunk` controls how many simulated seconds pass between display updates.
Default is 60 simulated seconds. If your simulation is slow and you want
more frequent updates, use `--chunk 10`. There is negligible performance cost.

```bash
python main.py run --config config/default_config.yaml --chunk 10
```

---

## 7. Understanding Simulation Outputs

Three files are written to your `output_dir` after each run.

---

### `simulation_report.json` — Primary result file

Full JSON report with all KPIs and confidence intervals. Structure:

```json
{
  "kpis": {
    "recall_latency_s": {
      "p50": { "mean": 48.2, "ci95_lower": 45.1, "ci95_upper": 51.3, "n_replications": 30 },
      "p95": { "mean": 142.7, ... },
      "p99": { "mean": 198.4, ... }
    },
    "migrate_latency_s": {
      "p50": { "mean": 312.4, ... },
      "p95": { "mean": 847.2, ... }
    },
    "cache_hit_rate":      { "mean": 0.874, "ci95_lower": 0.861, "ci95_upper": 0.887, ... },
    "n_migrations":        { "mean": 1247, ... },
    "n_recalls":           { "mean": 89, ... },
    "n_tape_mounts":       { "mean": 34, ... },
    "mean_disk_usage_pct": { "mean": 84.3, ... }
  },
  "littles_law_validation": {
    "disk_cache":   { "pass_rate": 1.0, "n_replications": 30 },
    "interconnect": { "pass_rate": 0.97, "n_replications": 30 },
    ...
  },
  "metadata": {
    "config_file": "config/default_config.yaml",
    "n_replications": 30,
    "master_seed": 42,
    "sim_duration_s": 14400,
    "wall_time_s": 847.2,
    "simulator_version": "1.0.0",
    "target_system": "HPE DMF 7"
  }
}
```

#### KPI Interpretation Guide

**Recall Latency** — Time in seconds from when an offline file is accessed
(triggering a recall) to when the first byte is delivered to the application.
This is the most user-visible HSM performance metric.

| Percentile | Meaning |
|---|---|
| P50 | Half of recalls complete faster than this. The "typical" user experience. |
| P95 | 95% of recalls complete within this time. Captures the tail. |
| P99 | 99% of recalls complete within this time. Captures rare worst cases (e.g., robot contention, tape seek to far end of tape). |

**Typical DMF 7 values (LTO-8, well-configured):**
- P50: 40–80 seconds (one mount + mid-tape seek + transfer)
- P95: 120–200 seconds (mount queue wait + full tape traverse)
- P99: 200–400 seconds (robot contention, cleaning events, near-full tape)

If your P99 is orders of magnitude higher than P95, that indicates a bottleneck
(likely robot or drive contention) creating a long tail.

---

**Migrate Latency** — Time in seconds from when the policy engine selects a file
for migration to when the tape write is acknowledged. Users do not see this
directly (the file stays online during migration), but high migrate latency
means the disk does not free up space promptly after policy selection.

- P50 > 600s typically indicates LS accrual buffer is batching too aggressively
  (increase `ls_min_batch_bytes` or decrease `ls_max_wait_s`)
- P95 >> P50 indicates drive contention (too few drives for the migration rate)

---

**Cache Hit Rate** — Fraction of file accesses served from disk (no tape involved).

| Value | Interpretation |
|---|---|
| > 0.95 | Excellent. Policy is keeping hot files on disk. |
| 0.80–0.95 | Good. Some recalls occurring — expected and normal. |
| 0.60–0.80 | Moderate. Policy may be too aggressive or disk too small. |
| < 0.60 | Poor. Files being recalled are immediately re-migrated. Consider raising `candidate_age_threshold_s` or disk capacity. |

---

**Mean Disk Usage (%)** — Time-averaged disk fill level (post-warmup).

A well-tuned HSM should show this oscillating between LWM and HWM. If mean
disk usage is consistently near HWM, the policy is struggling to keep up with
ingest rate. If it's consistently near LWM, the policy is over-migrating.

---

**Total Migrations / Recalls / Tape Mounts** — Raw event counts per replication.

A high recalls/migrations ratio (> 0.3) suggests files are being recalled
frequently after migration — a sign the `candidate_age_threshold_s` is too low
(files are being migrated before they are truly cold).

Tape mounts / migrations ratio: a ratio much less than 1 means the LS accrual
buffer is working well (many files per mount). A ratio near 1 means almost every
file requires its own mount, which is very inefficient.

---

#### Confidence Intervals — What they mean

Every KPI is reported as:
```
mean ± half-width  (ci95_lower to ci95_upper)
```

This is a 95% confidence interval computed using the Student t-distribution
across `n_replications` independent runs. It means: if you repeated this
entire experiment many times, 95% of the resulting intervals would contain
the true steady-state mean.

**Narrow CI** (half-width < 5% of mean) — good, your estimate is precise.
Achieved by running more replications or longer simulations.

**Wide CI** (half-width > 20% of mean) — run more replications. The default
of 30 replications is the CLT minimum; 50–100 replications narrows CIs further.

---

### `kpi_summary.csv` — Flat CSV for Excel / R / Python

Same data as `simulation_report.json` but in flat key=value format:

```
metric,value
kpis.recall_latency_s.p50.mean,48.2
kpis.recall_latency_s.p50.ci95_lower,45.1
...
```

Import directly into Excel, R (`read.csv`), or pandas (`pd.read_csv`).

---

### `replication_results.jsonl` — Per-replication raw data

One JSON record per line, one line per replication. Useful for:
- Plotting the distribution of outcomes across replications
- Checking for non-stationarity (if values drift across replications,
  your warm-up period may be too short)
- Identifying outlier replications

```json
{"replication": 0, "n_migrations": 1247, "n_recalls": 89, "cache_hit_rate": 0.874, "n_mounts": 34}
{"replication": 1, "n_migrations": 1253, "n_recalls": 91, "cache_hit_rate": 0.871, "n_mounts": 35}
...
```

---

### Console output — Summary tables

Two tables are printed at the end of every run:

**Table 1 — Simulation Results:** All KPIs with mean and 95% CI.

**Table 2 — Little's Law Validation:** Internal consistency check.
Little's Law states L = λW at every queue (mean items in system =
arrival rate × mean sojourn time). A pass rate of 100% means the
simulator's internal accounting is consistent. A fail indicates a
model defect or insufficient warm-up period.

A pass rate below 90% on any queue should be investigated before
trusting the results — increase `warmup_s` first, then examine the
queue in question.

---

## 8. Trace Replay Mode

Trace replay lets you feed real DMF 7 log data into the simulator to see
how the modelled system responds to your actual workload patterns.

### Step 1 — Extract data from your DMF 7 system

The simulator accepts events from DMF logs. Relevant DMF 7 commands:
- `dmstat -a` — file state information
- `dmqview` — migration/recall queue status
- `dmls -l <path>` — per-file state and BFID

### Step 2 — Generate the template to understand the format

```bash
python main.py trace-template --output my_trace.csv --n-events 10
```

This creates a file showing the required column structure:

| Column | Required | Description |
|---|---|---|
| `timestamp_s` | YES | Seconds since your chosen epoch (e.g., start of log) |
| `event_type` | YES | One of: `FILE_CREATE`, `RECALL`, `MIGRATE`, `FILE_MODIFY`, `FILE_DELETE`, `POLICY_CYCLE`, `SPACE_RELEASE` |
| `file_id` | YES | DMF BFID or any unique file identifier (can be anonymised) |
| `file_size_bytes` | YES | File size in bytes |
| `file_state_before` | optional | DMF state before event: `REG`, `MIG`, `DUL`, `OFL`, `UNM`, `PAR` |
| `file_state_after` | optional | DMF state after event |
| `vsn` | optional | Volume serial number (tape cartridge ID) |
| `drive_id` | optional | Tape drive index (0-based) |
| `latency_s` | optional | Observed latency for this event (used for KS-test validation) |
| `queue_depth` | optional | Queue depth at time of event |
| `disk_usage_pct` | optional | Disk fill % at time of event |

### Step 3 — Populate with real data

Map your DMF log fields to the columns above. The `timestamp_s` column
should be seconds since the first event in the log (set that event to 0.0).

### Step 4 — Configure trace replay mode

Edit `config/default_config.yaml`:

```yaml
workload:
  mode: trace_replay
  trace_file: ./my_trace.csv
```

### Step 5 — Run

```bash
python main.py run --config config/default_config.yaml
```

In trace replay mode, the arrival pattern (timestamps + file sizes) comes
from your real DMF data. Service times (tape seek, mount latency, etc.) are
still drawn from the configured probability distributions — this is correct
because the trace records *what happened*, not the underlying hardware times.

### Validation: comparing simulated vs observed latency

If your trace includes `latency_s` values for RECALL events, the simulator
will automatically run a two-sample Kolmogorov-Smirnov test comparing the
simulated recall latency distribution against the observed one. This is the
primary validation test.

The KS test result in the output means:
- **Accept H₀** (p ≥ 0.05): simulated and observed latency distributions are
  statistically indistinguishable. The model is validated for this workload.
- **Reject H₀** (p < 0.05): the distributions differ. Calibrate the tape
  timing parameters (`mount_time_lognormal_mu_s`, `seek_speed_factor`,
  `native_rate_bytes_per_s`) and re-run until the test passes.

---

## 9. Validation and Scientific Rigour

The simulator implements a three-phase V&V plan:

### Phase 1 — Unit Verification (automated)

Run `python -m pytest tests/ -v` to execute 33 unit tests covering:
- All DMF 7 file state machine transitions
- RNG stream independence (Pearson correlation test)
- Welch warm-up detection
- Little's Law queue statistics
- 95% CI coverage (1000-experiment binomial test)
- LogNormal and Pareto distribution moments
- Zipf distribution ordering
- Tape seek monotonicity
- Trace file reader schema validation

### Phase 2 — Degenerate Case Validation (manual)

Configure the simulator as a pure M/M/1 queue and verify it reproduces
the analytical solution:

1. Get analytical values:
   ```bash
   python main.py analytical-baseline --arrival-rate 0.5 --service-rate 2.0
   ```

2. Edit `config/default_config.yaml`:
   ```yaml
   workload:
     iat_distribution: exponential
     iat_exponential_rate: 0.5
   disk_cache:
     io_service_time_lognormal_mu_s: -0.693   # ln(0.5) => mean 0.5s
     io_service_time_lognormal_sigma_s: 0.001  # near-zero sigma => ~deterministic
   ```

3. Run with 30 replications and compare simulated W (mean disk sojourn)
   against the analytical W = 1/(μ-λ). They should agree within the 95% CI.

### Phase 3 — Real-World Validation (trace replay)

Feed real DMF 7 trace data and check the KS test passes (p ≥ 0.05).
If it fails, calibrate tape timing parameters systematically.

---

## 10. Running the Test Suite

```bash
# Run all 33 tests with verbose output
python -m pytest tests/ -v

# Run a specific test class
python -m pytest tests/test_core.py::TestFileStateMachine -v

# Run with short traceback on failure
python -m pytest tests/ --tb=short
```

All 33 tests should pass. A failure indicates either a dependency version
mismatch or an inadvertent edit to a core module. Do not run the simulation
with failing tests.

---

## 11. Experiment Design Guide

### Changing one parameter at a time

The correct scientific approach is to vary one parameter at a time while
holding all others constant. Use `--seed` to hold the random seed fixed
across comparisons so that any difference in output is attributable only
to the parameter change.

```bash
# Baseline
python main.py run -c config/default_config.yaml --seed 42 -o ./exp_baseline

# Effect of halving the policy cycle period
# (edit config: cycle_period_s: 150, then run)
python main.py run -c config/exp_policy150.yaml --seed 42 -o ./exp_cycle150

# Compare the two simulation_report.json files
```

### Suggested experiment matrix

| Experiment | Parameter | Values to test |
|---|---|---|
| Policy cycle timing | `cycle_period_s` | 60, 150, 300, 600, 1200 |
| Candidate age threshold | `candidate_age_threshold_s` | 300, 900, 1800, 3600, 7200 |
| Watermark gap | HWM-LWM | 0.95/0.85, 0.90/0.80, 0.85/0.70 |
| LS batch size | `ls_min_batch_bytes` | 1 GB, 5 GB, 10 GB, 20 GB |
| Drive count | `n_drives` (+ `tape_robot.n_drives`) | 2, 4, 8, 16 |
| Workload intensity | `iat_pareto_xmin_s` | 0.1, 0.5, 1.0, 5.0 |
| Workload burstiness | `iat_pareto_alpha` | 1.2, 1.5, 1.8, 2.5 |

### Minimum replications for valid CIs

For 95% CI validity (Central Limit Theorem): **n ≥ 30**.
For tight CIs (half-width < 5% of mean): typically **n = 50–100**.

Start with n=5 for exploratory runs to check the simulation behaves
sensibly, then use n=30 for publishable results.

### Warm-up period

The default `warmup_s: 1800` (30 simulated minutes) discards the
initialisation transient. If you change `initial_fill_fraction` or
`working_set_size` significantly, check the `replication_results.jsonl`
to confirm metrics are stable across replications (not trending).

---

## 12. Troubleshooting

### "No module named 'core'"

You are using an old version of `main.py`. The current version has a path
bootstrap at the very top. Download the latest `main.py` from this package
and replace yours.

### Progress display not updating

If you see nothing or only a spinner, your terminal may be line-buffering.
Try `--chunk 10` for more frequent writes. The display uses `\r` (carriage
return) which works in all standard terminals — it will not work if you
redirect output to a file (`> log.txt`). For file logging, output is
still written but each update appears as a new line.

### Simulation completes instantly with zero migrations/recalls

This happens with a very short `--duration`. Migrations require files to
age past `candidate_age_threshold_s` (default 1800s) AND a policy cycle
to fire (default every 300s). Use `--duration 3600` or longer.
For quick testing, reduce `candidate_age_threshold_s` to 60 and
`cycle_period_s` to 30 in your config.

### "Watermarks must satisfy 0 < LWM < HWM < 1.0"

Your `lwm_fraction` is >= your `hwm_fraction`. The low-water mark must
be strictly below the high-water mark. Typical values: HWM=0.90, LWM=0.80.

### "Pareto alpha must be > 1.0"

The `iat_pareto_alpha` parameter controls the tail of the inter-arrival
time distribution. Values ≤ 1.0 produce infinite-mean distributions
(the queue is always unstable). Set it to at least 1.1; typical values
for HPC workloads are 1.5–2.0.

### Little's Law failing for a queue

A pass rate below 90% means the internal accounting for that queue is
inconsistent. Most commonly caused by `warmup_s` being too short — the
queue statistics include the non-stationary initialisation period.
Double `warmup_s` and re-run. If it still fails, please report it as
a model defect.

### Tests failing

Run `pip install -r requirements.txt` again to ensure correct dependency
versions. The tests are sensitive to numpy version (≥1.24 required).

---

## Citation

If you use this simulator in academic work, please cite:

> Law, A.M. & Kelton, W.D. (2000). *Simulation Modeling and Analysis*, 3rd ed.
> McGraw-Hill. [Foundational DES methodology]

> Leland, W.E. et al. (1994). On the self-similar nature of Ethernet traffic.
> *IEEE/ACM Transactions on Networking*, 2(1), 1–15.
> [Pareto IAT distribution basis]

> Smirni, E. & Reed, D.A. (1998). Workload characterization of input/output
> intensive parallel applications. *Proc. IFIP WG 7.3*. [LogNormal file size basis]

The lifetime study (model, file-lifetime measurement and policy log) is described in `LIFETIME_STUDY.md`.
