# E10: lifetime measurement on twelve file systems

`pipeline.py` applies the e08 method (file system A) to one file system's reduced audit extract.
Extracts are produced on a Storage Scale NSD server by `tools/production_extract/run_multi_fs.sh`,
which calls `extract_audit.py` per day with a keyed hash. The reduced audit data are available
subject to the operating facility's approval; result files for all file systems are included.

## Validation against e08

File system A was extracted a second time with the same `extract_audit.py` and window.

1. `repro_check.py` (`repro_a.json`): every one of the 12,748,929 records of the first extract
   appears, in order, in the second, with all non-hash fields equal. The mapping between the two
   hash keys is one-to-one for file keys (4,935,698) and directory-tree keys (319,773). The second
   extract has 149,866 further records, all after the end of the window.
2. `pipeline.py` (`validation_a_results.json`): 5,245,883 lifecycles, identical to e08 in every
   non-hash field, with point estimates equal to e08.

## Observation period (August 28 to September 26, 2026)

1. `plog_window.py ../../data/policy_log/policy_log.txt 2026-08-28 2026-09-27 plog_window.json 0`
   gives runs and mean intervals per file system.
2. `pipeline.py --in <extract dir> --fs <label> --out results --peff <mean interval, h>` per file
   system (400 bootstrap replicates over directory trees).
3. `table.py results` (per-file-system table), `pred_fs.py results ../../data/policy_log/policy_log.txt`
   (model against policy log; the ten file systems with submissions give 29.1 million files logged
   against 28.7 million predicted), `pooled.py results` (all 12) and
   `pooled.py results --exclude J K L --tag nine` (the nine-file-system result), `pooled_extra.py`.

J has no policy submissions in the period, K almost no deletions, and L wrote 44 files; they are
reported but excluded from the pooled result. Row A of the paper's table uses the e08 intervals.

## Second month (July 29 to August 27, 2026)

`run_multi_fs.sh` with START=2026-07-29 and END=2026-08-27; `pipeline.py --t0 2026-07-29 --t1 2026-08-28`
with intervals from `plog_window_w2.json`; outputs in `results_w2/`. C is excluded:
`surge_profile.py` on its extract counted 813,693,439 closes after writing, 12,190,576 creates,
5,532,128 unlinks and 1,245,843 renames, a rewrite-dominated month that assumption (A6) excludes.
The eight file systems A, B and D to I are pooled in each month in `results/pooled_eight_w1.json`
and `results_w2/pooled_eight_w2.json`.
