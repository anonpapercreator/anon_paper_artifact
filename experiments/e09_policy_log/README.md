# E09: migration-policy log, July 29 to September 29, 2026

Input: `data/policy_log/policy_log.txt`, one line per policy run that submitted files (or none) for
copying, with file systems labelled as in the paper (A to L; M to Q are file systems outside the
lifetime study) and all times in UTC. Host names, file-system identifiers and job identifiers were
removed before release. Checksums: `data/policy_log/SHA256SUMS`.

- `plog.py`: run intervals per file system and over all file systems (4,012 intervals; median
  6.24 h), passes of the policy engine over all file systems, and the runs on file system A in the
  observation period (115 runs, 4,707,802 files submitted, mean interval 6.26 h). Run from a
  directory holding the log as `policylog.txt`. Output: `plog.json`.
- `pred.py`: model prediction of files copied on A (run from the e08 working directory).
- `poll_runs.py`: run durations and the gap to the next run from a poll of running policy jobs.
  On `data/policy_log/policy_runs_poll_a.txt` (about once a minute, September 27 21:47 to
  September 28 06:22 UTC) the 12 file systems with two recorded migration runs have runs of
  0.7 to 17.9 minutes and gaps of 6.00 to 6.10 hours. `policy_runs_poll_b.txt` continues the poll.

The per-file-system comparison of the model with the log is in `e10_multi_fs` (`plog_window.py`,
`pred_fs.py`).
