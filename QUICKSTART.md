# DESCASSI - Discrete Event Simulator for Combined Archival and Scratch Storage Infrastructure

## Quick Start Guide

## Starting the Web Interface (Recommended)

The web interface provides a modern GUI for running simulations and viewing results.

### Start the Server

```bash
cd <path-to-repository>
streamlit run cli/web_app.py --server.port=8501
```

### Access the Interface

**Local access:**
- Open browser to: http://localhost:8501

**Remote access (SSH tunnel):**
```bash
# From your local machine
ssh -L 8501:localhost:8501 user@your-server
```
Then open http://localhost:8501 in your browser.

### Stop the Server

```bash
# Press Ctrl+C in the terminal running streamlit
# Or find and kill the process:
ps aux | grep streamlit
kill <PID>
```

---

## Running the Simulator (CLI Alternative)

If you prefer command-line interface:

### Simple CLI

```bash
cd <path-to-repository>
python -m cli.simple_cli --help
```

**Common Commands:**

```bash
# Parse a DMF log file
python -m cli.simple_cli parse /path/to/log.log --stats

# Show configuration  
python -m cli.simple_cli config

# List sweeps
python -m cli.simple_cli sweeps

# Show status
python -m cli.simple_cli status
```

---

## Step 1: Parse Your DMF Log

### Using Simple CLI:
```bash
python -m cli.simple_cli parse <dmf-ls-log> --stats
```

Expected output:
```
Parsing: <dmf-ls-log>
Parsed 42438 events
Found 21207 completed requests with latency

=== Statistics ===
Total events: 42438
Recalls: 11635
Migrations: 9596
Mean latency: 95.56s
p50 latency: 21.20s
p95 latency: 308.85s
```

---

## Step 2: Configure Simulation

Default configuration matches your system:

| Parameter | Value |
|-----------|-------|
| **ESS 3500** | |
| NSD Nodes | 4 |
| Throughput | 65 GiB/s |
| Capacity | 809 TiB |
| **TS1160** | |
| Drives | 16 |
| **DMF Policy** | |
| Age Threshold | 30 min |
| HWM | 80% |

View config:
```bash
python -m cli.simple_cli config
```

---

## Step 3: Run Parameter Sweep

Use the **web interface** for full sweep functionality:

1. Go to "🔬 Parameter Sweeps" page
2. Select parameters to sweep (e.g., age threshold: 5, 10, 20, 30, 40, 50, 60+ minutes)
3. Set number of replications
4. Click "Start Parameter Sweep"

Or check status via CLI:
```bash
python -m cli.simple_cli status
python -m cli.simple_cli sweeps
```

---

## Web Interface Pages

| Page | Description |
|------|-------------|
| 🏠 Home | System overview and quick stats |
| 📂 Parse Logs | Parse DMF log files |
| ⚙️ Configure | Set simulation parameters |
| 🔬 Parameter Sweeps | Run multi-parameter sweeps |
| 📊 Reports | View charts and export results |
| ✅ Validate | Compare against real DMF data |

---

## Trace Replay Mode

The simulator supports two workload modes:

1. **Synthetic** (default): Generates random I/O pattern
2. **Trace Replay**: Replays from parsed DMF log

To use trace replay via the web:
1. Go to "⚙️ Configure" page
2. Set "Workload Mode" to "trace_replay"
3. Enter path to trace file (e.g., `trace.jsonl`)

Or set in config:
```yaml
simulation:
  workload_mode: trace_replay
  trace_file: ./trace.jsonl
```

---

## Database

Results stored in: `~/.hsm_sim/hsm_sim.db`

```bash
# View saved configurations
sqlite3 ~/.hsm_sim/hsm_sim.db "SELECT name FROM configs;"

# View sweep results
sqlite3 ~/.hsm_sim/hsm_sim.db "SELECT * FROM sweep_results LIMIT 5;"

# List sweeps
sqlite3 ~/.hsm_sim/hsm_sim.db "SELECT id, name, status FROM sweeps;"
```

---

## Need Help?

- Full README: `README.md`
- Specification: `SPEC.md`
- Configuration: `default_config.yaml`
