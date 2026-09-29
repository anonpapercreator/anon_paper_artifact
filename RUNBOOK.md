# DESCASSI fresh-runs runbook — regenerating every paper number on the corrected sim

Prereqs (once):
  cd /QRISdata/Q6680/phd/DESCASSI_2026_CODE
  # install the corrected code from descassi_FINAL.tar.gz, then:
  find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  # sanity that the corrected set is live:
  grep -c no_outstanding main.py                      # 2
  grep -c "def write_batch" components/tape_subsystem.py   # 1
  grep -c "def free_space_check" components/filesystem.py  # 1
  grep -c 977465837092864 config/default_config.yaml  # 1  (889 TiB)
  python3 -c "from main import load_config; c=load_config('config/default_config.yaml'); print(c['disk_cache']['hwm_fraction'], c['disk_cache']['lwm_fraction'], c['tape_drives']['n_drives'], c['tape_drives']['n_write_drives'])"
  # expect: 0.8 0.6 9 7

Define once per shell:
  export R=/QRISdata/Q6680/phd/DESCASSI_2026_CODE
  export SL=/QRISdata/Q6680/phd/DESCASSI_SLURM_RUNNER/slurm
  cd $R; export DESCASSI_ROOT=$PWD PYTHONPATH=$PWD
  GREP='grep -v -e AbsorbingPhase -e fitted_Q'   # noise filter

=====================================================================
STEP 1 — M/M/1 baseline (confirms abstract "within five percent"; should be unchanged)
=====================================================================
  python main.py analytical-baseline --arrival-rate 0.8 --service-rate 1.0 2>&1 | $GREP
  # then the simulated check (30 reps, 3600s, 600s warmup) the paper cites:
  python $SL/run_block.py --config config/default_config.yaml \
    --rep-start 0 --rep-count 30 --out-dir runs/mm1 --policy static \
    --mode synthetic --seed 42 --duration 3600 --warmup 600 2>&1 | $GREP
  python $SL/aggregate.py --in-dir runs/mm1 2>&1 | $GREP
  # writes runs/mm1/aggregate_report.json (no --out flag; output goes into --in-dir)
  # READ: Wq mean + 95% CI, Little's-law check. Expect ~4.17s, +~4.2%. -> abstract/§VI.A intro stays.

=====================================================================
STEP 2 — Calibration on the 903k trace (fills tab:calibration TBDs) ** the headline regen **
=====================================================================
Important: recall latency must be read from a FULLY DRAINED run. The migration
backlog drains after the arrival window; max_drain_s=5000000 is already in the
config so the run drains to quiescence. Migrations completing slowly during drain
is expected and correct (background work behind the recall backlog).

  python $SL/run_block.py --config config/default_config.yaml \
    --rep-start 0 --rep-count 1 --out-dir runs/calib --policy static \
    --mode trace_replay --trace-file largest_trace_20260604.jsonl \
    --seed 42 --duration 90000 --warmup 0 2>&1 | $GREP
  cat runs/calib/rep_000000.json

Then produce the exact tab:calibration rows (buckets, P50/P95/P99, KS) with the
self-contained calibration script (uses the repo's own trace loader for the
observed side and the stats object for the simulated side):

  python calibrate_903k.py --trace largest_trace_20260604.jsonl \
    --duration 90000 --seeds 42,43,44,45,46 2>&1 | $GREP

This prints, ready to paste:
  - observed fast/warm/cold %% and P50/P95/P99 (from the trace's latency_s field)
  - simulated fast/warm/cold %% and P50/P95/P99 as MEAN +/- 95%% CI across the 5 seeds
  - the pp differences, %% errors, and the two-sample KS distance (+ reject/accept)

Requirements: the real trace MUST carry a per-recall `latency_s` field for the
observed side (the loader reads rec["latency_s"]); the 903k trace does. scipy is
used for KS if present, with a no-scipy empirical-CDF fallback built in.

FILL INTO tab:calibration (6.results.tex): the script's "TABLE ROWS" block maps
1:1 onto the table's Simulated / Observed / Error columns and the KS row. Replace
every TBD. Expected qualitative outcome: median and bucket structure agree; the
simulated tail (P95/P99) falls SHORT of observed (model lacks RAO deferral) -
report as-is; the §VI.A prose already explains this and must not be tuned away.

=====================================================================
STEP 3 — Policy sweep tau* (confirms §VI.B + Fig sweep; analytic, should hold)
=====================================================================
  python regen/regen_sweep.py 2>&1 | $GREP        # regenerates data/sweep_data.csv
  # If regen_sweep.py reads production params from the config, it will pick up the
  # corrected drive rate/topology automatically. Verify tau* still ~100s and J(tau)
  # shape unchanged. If the script hardcodes old params (50TB/8 drives/360MB/s),
  # update those constants in regen_sweep.py to match config first.
  # READ: confirm monotone-with-interior-min; tau* ~ 100s. -> §VI.B numbers stay or
  # get minor updates; conclusion unchanged.

=====================================================================
STEP 4 — Single-tenant adaptive vs static (confirms §VI.C; synthetic)
=====================================================================
  python regen/regen_adaptive_static.py 2>&1 | $GREP   # regenerates adaptive_static_results.json
  # READ: P50/P95 static vs adaptive should remain statistically indistinguishable
  # under the restructured policy engine. Update the in-text numbers if they shift.

=====================================================================
STEP 5 — Multi-tenant beta-sweep (confirms §VI.D, tab:multitenant; synthetic)
=====================================================================
  python regen/regen_multi_tenant.py 2>&1 | $GREP      # regenerates multi_tenant_beta_sweep.json
  # READ: re-extract agg P95 / Gini / worst-tenant P95 / mounts for the 6 cells.
  # The policy restructure (independent 6h migration + HWM trigger) can shift the
  # magnitudes; the qualitative asymmetry (adaptive wins at beta>0, neutral/worse at
  # beta=0) should hold. Update tab:multitenant numbers; keep narrative if direction
  # holds. If direction changes, tell me and we revisit the interpretation.

=====================================================================
STEP 6 — IOR negative control (confirms §VI.E; mechanically insensitive)
=====================================================================
  # Optional re-run; IOR never crosses the tier boundary so seek/topology/capacity
  # cannot move it. If you re-run, use the IOR trace container as before. Expect
  # unchanged: 100% hits, 0 mounts, static==adaptive. tab:ior-negctl stays.

=====================================================================
STEP 7 — Scaled paired campaign (retires the "n=5, no CIs" reviewer weakness)
=====================================================================
  # Edit knobs in $SL/submit.sh (TRACE, SEED base, DURATION, BLOCK size, ARRAY range),
  # then:
  bash $SL/submit.sh
  # After the array completes:
  python $SL/aggregate.py --in-dir runs/static   2>&1 | $GREP
  python $SL/aggregate.py --in-dir runs/adaptive 2>&1 | $GREP
  # each writes aggregate_report.json into its --in-dir
  python $SL/paired_compare.py --static runs/static --adaptive runs/adaptive \
    --out runs/paired.json
  # READ: per-arm mean +/- 95% CI, plus the paired Wilcoxon + effect-size CI on the
  # static-vs-adaptive median ratio. Use these to report the headline adaptive claim
  # with CIs and effect size rather than n=5 point estimates.

=====================================================================
WHAT TO HAND BACK FOR THE PAPER
=====================================================================
Paste to me, and I will drop them into the .tex precisely:
  1. runs/calib sidecar(s) + the fast/warm/cold/KS buckets  -> fills tab:calibration
  2. regen_sweep tau* confirmation (or new sweep_data.csv)   -> §VI.B
  3. regen_adaptive_static numbers                            -> §VI.C
  4. regen_multi_tenant 6-cell table                          -> tab:multitenant
  5. paired_compare.json summary                              -> headline CIs/effect size

=====================================================================
APPENDIX A — superseded
=====================================================================
The earlier hand-rolled bucket/KS snippet is replaced by calibrate_903k.py
(Step 2), which uses the repo's own load_trace_latencies and the post-flush
recall sample array (stats.recall_latency._last_rep_samples). No manual snippet
needed.

