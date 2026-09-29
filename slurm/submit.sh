#!/bin/bash
# slurm/submit.sh -- multi-tenant beta-sweep paired campaign for the adaptive-vs-
# static result (paper section VI.D). For each heterogeneity level beta in
# {0.0, 0.5, 1.0} it submits a paired pair of arrays:
#     static   (adaptive advisor OFF)
#     adaptive (adaptive advisor ON)
# both under the SAME master seed and the SAME replication-ID layout, so the two
# arms are paired (common random numbers). After both arms of a beta finish, a
# dependent job aggregates each arm and runs paired_compare.py on the decisive
# metrics. The trace-replay path is deliberately NOT used here: its offered tape
# load is rho >> 1 (unbounded recall tail), so recall_p95 is not a meaningful
# paired metric there. The multi-tenant synthetic workload is bounded and is
# where the policy choice is decisive.
#
# Usage:  bash slurm/submit.sh
# Override any knob from the environment, e.g.:  REPS_PER_ARM=100 bash slurm/submit.sh
set -euo pipefail

# ---- campaign knobs --------------------------------------------------------
CONFIG=${CONFIG:-config/default_config.yaml}
SEED=${SEED:-42}                       # ONE master seed for the whole campaign
DURATION=${DURATION:-14400}            # 4 h per replication (matches paper VI.D)
WARMUP=${WARMUP:-1800}                  # 30 min warm-up (MSER-5 / Welch)
MODE=multi_tenant                       # fixed: this campaign is multi-tenant
ANACONDA_MODULE=${ANACONDA_MODULE:-anaconda3/2023.09-0}  # cluster Python module (3.11.5 + simpy/scipy)
N_TENANTS=${N_TENANTS:-128}             # tenant count; recall load now driven by WORKING_SET
CYCLE_PERIOD=${CYCLE_PERIOD:-300}       # MT stress test: 300s migration cycle so the
                                         # policy is exercised within the 4h window
                                         # (production trace replay keeps the real 6h=21600s)
CACHE_BYTES=${CACHE_BYTES:-1000000000000} # 1 TB: below the aggregate working-set footprint
                                         # so a large offline fraction forms and reads hit tape
WORKING_SET=${WORKING_SET:-64000}        # base working-set size; large so aggregate WS >> cache
AGG_LAMBDA=${AGG_LAMBDA:-}               # aggregate MT arrival rate (files/s); empty = single-stream default
                                         # (this is the lever that creates real recall load)
BETAS=${BETAS:-"0.0 0.5 1.0"}           # heterogeneity dial sweep
BLOCK=${BLOCK:-10}                       # replications per array task
NTASKS=${NTASKS:-20}                     # array tasks per arm -> reps/arm = NTASKS*BLOCK
                                         # default 20*10 = 200 reps per arm
METRICS=${METRICS:-"tenant_p95_max_s recall_p95_s tenant_p95_gini"}

ARRAY="0-$(( NTASKS - 1 ))"
export DESCASSI_ROOT=${DESCASSI_ROOT:-$PWD}
RUNNER=${RUNNER:-slurm}                  # dir holding run_block.py / aggregate.py / paired_compare.py

mkdir -p logs runs

common="DESCASSI_ROOT=${DESCASSI_ROOT},RUNNER=${RUNNER},CONFIG=${CONFIG},SEED=${SEED},DURATION=${DURATION},WARMUP=${WARMUP},MODE=${MODE},N_TENANTS=${N_TENANTS},CYCLE_PERIOD=${CYCLE_PERIOD},CACHE_BYTES=${CACHE_BYTES},WORKING_SET=${WORKING_SET},AGG_LAMBDA=${AGG_LAMBDA},ANACONDA_MODULE=${ANACONDA_MODULE},BLOCK=${BLOCK}"

echo "campaign: beta in {${BETAS}}  n_tenants=${N_TENANTS}  reps/arm=$(( NTASKS*BLOCK ))  seed=${SEED}"

for BETA in ${BETAS}; do
  tag="b$(echo "${BETA}" | tr -d '.')"          # 0.0 -> b00, 0.5 -> b05, 1.0 -> b10
  out_s="runs/mt_${tag}_static"
  out_a="runs/mt_${tag}_adaptive"

  jid_s=$(sbatch --parsable --array="${ARRAY}" \
          --export=ALL,${common},BETA=${BETA},POLICY=static,OUTDIR=${out_s} \
          slurm/descassi_array.sbatch)
  echo "  beta=${BETA}  static   array ${jid_s} -> ${out_s}"

  jid_a=$(sbatch --parsable --array="${ARRAY}" \
          --export=ALL,${common},BETA=${BETA},POLICY=adaptive,OUTDIR=${out_a} \
          slurm/descassi_array.sbatch)
  echo "  beta=${BETA}  adaptive array ${jid_a} -> ${out_a}"

  # Build the per-metric paired_compare calls for the dependent job.
  cmp_cmds=""
  for M in ${METRICS}; do
    cmp_cmds="${cmp_cmds} python ${RUNNER}/paired_compare.py --static ${out_s} --adaptive ${out_a} --metric ${M} --out runs/paired_${tag}_${M}.json;"
  done

  sbatch --dependency=afterok:${jid_s}:${jid_a} \
         --job-name=descassi_agg_${tag} --time=00:30:00 --mem=4G \
         --cpus-per-task=1 \
         --output=logs/agg_${tag}_%j.out --error=logs/agg_${tag}_%j.err \
         --wrap "cd ${DESCASSI_ROOT}; module load ${ANACONDA_MODULE:-anaconda3/2023.09-0} 2>/dev/null || module load anaconda3 2>/dev/null || true; \
   python ${RUNNER}/aggregate.py --in-dir ${out_s} --policy static; \
   python ${RUNNER}/aggregate.py --in-dir ${out_a} --policy adaptive; \
   ${cmp_cmds}"
  echo "  beta=${BETA}  aggregate+compare gated on ${jid_s} & ${jid_a}"
done

echo "submitted. results: runs/mt_b*/aggregate_report.json and runs/paired_b*_*.json"
