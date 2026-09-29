#!/bin/bash
#SBATCH --job-name=descassi
#SBATCH --array=0-29              # 30 array tasks; total reps = (array size) * BLOCK
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         # each rep is single-threaded; pin to avoid oversubscription
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Scaled DESCASSI run as a SLURM job array.
#
# Each array task runs a DISJOINT block of replication IDs under ONE master
# seed; per-replication seeding is independent and reproducible, so the union
# of all blocks is a single coherent campaign of (array_size * BLOCK) reps.
#
# Submit TWICE (or via submit.sh) -- once per policy -- with the SAME SEED and
# the SAME replication-ID layout so the two arms are paired (common random
# numbers) for paired_compare.py.
#
# Override at submit time, e.g.:
#   sbatch --export=ALL,POLICY=adaptive,OUTDIR=runs/adaptive,TRACE=traces/new_trace.jsonl \
#          slurm/descassi_array.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- locate the code + scaffolding (works with a separate runner dir) ------
export DESCASSI_ROOT=${DESCASSI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # dir with main.py
RUNNER=${RUNNER:-slurm}                                           # dir with run_block.py (can be absolute)
cd "${DESCASSI_ROOT}"

# ---- knobs (env-overridable) ----------------------------------------------
CONFIG=${CONFIG:-config/default_config.yaml}
BLOCK=${BLOCK:-10}                       # replications per array task
SEED=${SEED:-42}                         # ONE master seed for the whole campaign
DURATION=${DURATION:-14400}              # sim seconds per replication (4 h)
WARMUP=${WARMUP:-1800}                   # warm-up seconds (MSER-5 / Welch)
MODE=${MODE:-trace_replay}               # synthetic | trace_replay | multi_tenant
TRACE=${TRACE:-}                          # used only when MODE=trace_replay
BETA=${BETA:-}                            # multi_tenant heterogeneity dial in [0,1]
N_TENANTS=${N_TENANTS:-}                  # multi_tenant tenant count
CYCLE_PERIOD=${CYCLE_PERIOD:-}            # migration cycle override (s); MT stress test
CACHE_BYTES=${CACHE_BYTES:-}              # disk cache capacity override (bytes); MT stress test
WORKING_SET=${WORKING_SET:-}              # base working-set size; MT stress test
AGG_LAMBDA=${AGG_LAMBDA:-}                # aggregate MT arrival rate (files/s); contention dial
POLICY=${POLICY:-static}                 # static | adaptive | hwm
OUTDIR=${OUTDIR:-runs/${POLICY}}

# ---- environment ----------------------------------------------------------
# Single-thread the math libraries: one rep per core, no oversubscription.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs "${OUTDIR}"

# ---- Python environment (compute nodes) -----------------------------------
# Load the cluster Anaconda module. It resolves to the SAME Python 3.11.5
# installation on every node (login and compute), with simpy/scipy/numpy
# already present, removing the interpreter ambiguity that previously made
# compute nodes miss the packages. Override ANACONDA_MODULE if the site name
# changes.
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0}"
# Initialise the module system in non-interactive batch shells (it is a shell
# function that may not be defined until lmod/environment-modules is sourced).
if ! command -v module >/dev/null 2>&1; then
  for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
              /etc/profile.d/z00_lmod.sh /usr/share/lmod/lmod/init/bash; do
    [ -f "$init" ] && source "$init" && break
  done
fi
module load "${ANACONDA_MODULE}" 2>/dev/null || module load anaconda3 2>/dev/null || true
PYBIN=python
# Hard guard: fail loudly (not silently) if the environment is wrong.
if ! "${PYBIN}" -c "import simpy, scipy, numpy" 2>/dev/null; then
  echo "FATAL: ${PYBIN} on $(hostname) cannot import simpy/scipy/numpy after 'module load ${ANACONDA_MODULE}'." >&2
  echo "Check the module name with: module avail anaconda" >&2
  exit 1
fi

REP_START=$(( SLURM_ARRAY_TASK_ID * BLOCK ))

echo "task ${SLURM_ARRAY_TASK_ID}: reps ${REP_START}..$((REP_START+BLOCK-1)) policy=${POLICY} mode=${MODE} seed=${SEED}"

EXTRA_ARGS=()
[ "${MODE}" = "trace_replay" ] && [ -n "${TRACE}" ] && EXTRA_ARGS+=(--trace-file "${TRACE}")
[ -n "${BETA}" ]         && EXTRA_ARGS+=(--beta "${BETA}")
[ -n "${N_TENANTS}" ]    && EXTRA_ARGS+=(--n-tenants "${N_TENANTS}")
[ -n "${CYCLE_PERIOD}" ] && EXTRA_ARGS+=(--cycle-period "${CYCLE_PERIOD}")
[ -n "${CACHE_BYTES}" ]  && EXTRA_ARGS+=(--cache-bytes "${CACHE_BYTES}")
[ -n "${WORKING_SET}" ]  && EXTRA_ARGS+=(--working-set-size "${WORKING_SET}")
[ -n "${AGG_LAMBDA}" ]   && EXTRA_ARGS+=(--agg-lambda "${AGG_LAMBDA}")

srun "${PYBIN}" "${RUNNER}/run_block.py" \
    --config   "${CONFIG}" \
    --rep-start "${REP_START}" \
    --rep-count "${BLOCK}" \
    --out-dir  "${OUTDIR}" \
    --policy   "${POLICY}" \
    --mode     "${MODE}" \
    "${EXTRA_ARGS[@]}" \
    --seed     "${SEED}" \
    --duration "${DURATION}" \
    --warmup   "${WARMUP}"
