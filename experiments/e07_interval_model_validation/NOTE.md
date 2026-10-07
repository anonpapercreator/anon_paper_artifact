# e07: superseded

This earlier comparison of the model with the simulator is kept for the record. The paper
reports e11 (`../e11_model_sim_sweep/`) instead.

- `vc_results_bw.json`: 30 replications at six settings, exponential lifetimes. It uses the
  flow estimator, which divides the data copied in a window by the model evaluated at the data
  created in the same window. With heavy-tailed file sizes this overstates the archive write
  rate C as a + p grows (see `../e11_model_sim_sweep/PLAN.md`, Addendum 2, and `RESULTS.md`).
- `vc_results.json`: produced by an older version of `run_vc.py` that weighted the
  selection-to-tape delay by file count instead of volume. Not used.
