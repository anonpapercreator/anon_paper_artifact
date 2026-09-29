# E08: production file lifetimes, one production file system, 28 Aug to 26 Sep 2026

Input: the reduced audit records produced by `tools/production_extract/extract_audit.py`,
extracted into `audit/` in the working directory as `*.tsv.gz`. The reduced data are
available subject to the operating facility's approval.

Run order: load.py, a1.py, a2.py, a3.py (diagnostics), life.py (lifecycles),
boot.py (attach prefix), km.py, haz.py, boot2.py (prefix-cluster bootstrap, 400 reps, seed 1), eqv.py.

Definitions
- Window [2026-08-28 00:00 UTC, 2026-09-27 00:00 UTC). Only files whose CREATE lies in the window are used.
- Death: UNLINK with linkCount = 1 (count is reported before the decrement).
- Inode reuse (CREATE followed by CREATE, no UNLINK): death interval-censored in (last event, next CREATE];
  T_hi uses the upper bound, T_lo the lower bound.
- Otherwise right-censored at the window end. Bytes: maximum size seen at WCLOSE or UNLINK.
- mtime is not used as a birth time: for window-born files it precedes CREATE by more than 37 days in over 25% of cases.
