"""Per-file-system policy runs inside the audit window (e09 parsing, all file systems).
usage: plog_window.py LOG [DAY0 DAY1 OUT OFFSET_H]. LOG is the released UTC policy log (OFFSET_H 0).
Output: plog_window.json."""
import re, json, sys, numpy as np, pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else 'policylog.txt'
day0 = sys.argv[2] if len(sys.argv) > 2 else '2026-08-28'   # window start, UTC date
day1 = sys.argv[3] if len(sys.argv) > 3 else '2026-09-27'   # window end, UTC date (exclusive)
out_name = sys.argv[4] if len(sys.argv) > 4 else 'plog_window.json'
off_h = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0   # log clock minus UTC, hours (raw site log: 10)
rows = []
for l in open(src):
    m = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*?[Pp]olicy \'?([A-Za-z0-9]+)_migrate_policy\'? submitted (\d+|no) records', l)
    if not m:
        m2 = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*"policy_name":"([A-Za-z0-9]+)_migrate_policy".*No records submitted', l)
        if m2: rows.append((m2.group(1), m2.group(2).upper(), 0))
        continue
    rows.append((m.group(1), m.group(2).upper(), 0 if m.group(3) == 'no' else int(m.group(3))))
d = pd.DataFrame(rows, columns=['t', 'fs', 'n']); d['t'] = pd.to_datetime(d['t'])
d = d.drop_duplicates().sort_values(['fs', 't'])
# the audit window is in UTC; shift it onto the log clock (0 for the released UTC log)
w0, w1 = pd.Timestamp(day0) + pd.Timedelta(hours=off_h), pd.Timestamp(day1) + pd.Timedelta(hours=off_h)
out = {}
for fs, g in d.groupby('fs'):
    w = g[(g.t >= w0) & (g.t < w1)]; P = w.t.diff().dt.total_seconds().dropna().to_numpy() / 3600
    out[fs] = {'runs': int(len(w)), 'files_submitted': int(w.n.sum()),
               'mean_h': float(P.mean()) if len(P) else None, 'median_h': float(np.median(P)) if len(P) else None,
               'max_h': float(P.max()) if len(P) else None}
json.dump(out, open(out_name, 'w'), indent=1)
for fs, s in out.items(): print(fs, s)
