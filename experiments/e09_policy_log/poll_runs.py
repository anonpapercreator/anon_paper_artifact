"""Duration of migration-policy runs and the gap to the next run, from a policy-run poll.
Input: a released poll file (UTC; one line per running job per poll, about once a minute).
A run is identified by its run number; its duration is the time from its start to the last
poll that saw it. For file systems with two or more recorded runs, the gap is the time from the
last sighting of one run to the start of the next.
usage: python3 poll_runs.py policy_runs_poll_a.txt [KIND]   (KIND default 'migrate')"""
import sys, collections
from datetime import datetime
kind = sys.argv[2] if len(sys.argv) > 2 else 'migrate'
runs = {}
for l in open(sys.argv[1]):
    x = l.split()
    if x[1] != 'RUN' or not x[6].endswith(f'-{kind}-policy'): continue
    seen = datetime.strptime(x[0], '%Y-%m-%dT%H:%M:%S')
    start = datetime.strptime(f'{x[2]} {x[3]}', '%Y-%m-%d %H:%M:%S.%f')
    r = runs.setdefault(x[4], {'fs': x[6].split('-')[0], 'start': start, 'last': seen})
    r['last'] = max(r['last'], seen)
by = collections.defaultdict(list)
for r in runs.values(): by[r['fs']].append(r)
polls = sorted({l.split()[0] for l in open(sys.argv[1])})
print('poll span', polls[0], polls[-1], 'runs', len(runs))
for fs in sorted(by):
    rs = sorted(by[fs], key=lambda r: r['start'])
    d = [round((r['last'] - r['start']).total_seconds() / 60, 1) for r in rs]
    g = [round((b['start'] - a['last']).total_seconds() / 3600, 2) for a, b in zip(rs, rs[1:])]
    print(fs, 'runs', len(rs), 'duration_min', d, 'gap_h', g)
