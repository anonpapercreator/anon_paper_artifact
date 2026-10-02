# Lifetime study: where each result is produced

| Paper section | Directory | Main scripts |
|---|---|---|
| Model (closed forms, hazard rule) | `models/` | `interval_model.py` |
| Comparison of model and simulator | `experiments/e07_interval_model_validation/` | `run_vc.py`, `run_pool.py` |
| Production file lifetimes, policy alternatives (file system A) | `experiments/e08_production_lifetimes/` | see its README |
| Twelve file systems, second month | `experiments/e10_multi_fs/` | `pipeline.py`, `pooled.py`, `pred_fs.py` |
| Run intervals and the policy engine | `experiments/e09_policy_log/` | `plog.py`, `poll_runs.py` |
| Audit-log reduction | `tools/production_extract/` | `extract_audit.py`, `run_multi_fs.sh` |

Released data: `data/policy_log/` (policy log and run polls, UTC, file systems labelled as in the
paper). The reduced audit records are available subject to the operating facility's approval.
