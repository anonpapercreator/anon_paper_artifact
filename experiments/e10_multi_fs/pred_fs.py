"""Per-file-system check of the model against the policy log: predicted number of files copied
(count-weighted C at the measured mean interval, all files and non-empty files only) versus files
submitted by the policy runs in the window.
usage: pred_fs.py RESDIR LOG [DAY0 DAY1 OFFSET_H]"""
import json, os, re, sys, numpy as np, pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location('pl', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline.py'))
pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
res, logf = sys.argv[1], sys.argv[2]
day0 = sys.argv[3] if len(sys.argv) > 3 else '2026-08-28'   # window start, UTC date
day1 = sys.argv[4] if len(sys.argv) > 4 else '2026-09-27'   # window end, UTC date (exclusive)
off_h = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0   # log clock minus UTC, hours (raw site log: 10)
rows = []
for l in open(logf):
    m = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*?[Pp]olicy \'?([A-Za-z0-9]+)_migrate_policy\'? submitted (\d+|no) records', l)
    if m: rows.append((m.group(1), m.group(2).upper(), 0 if m.group(3) == 'no' else int(m.group(3))))
d = pd.DataFrame(rows, columns=['t', 'fs', 'n']); d['t'] = pd.to_datetime(d.t); d = d.drop_duplicates()
def sub(fs, off_h):
    w0 = pd.Timestamp(day0) + pd.Timedelta(hours=off_h); w1 = pd.Timestamp(day1) + pd.Timedelta(hours=off_h)
    g = d[(d.fs == fs) & (d.t >= w0) & (d.t < w1)]; return int(g.n.sum())
out = {}
for f in sorted(os.listdir(res)):
    if not f.endswith('_lifecycles.parquet'): continue
    fs = f.split('_')[0]; k = fs
    r = json.load(open(os.path.join(res, f'{fs}_results.json'))); peff = r['peff_h'] * 3600
    L = pd.read_parquet(os.path.join(res, f), columns=['T_hi', 'status', 'bytes'])
    E = (L.status != 'censored').to_numpy().astype(float); b = L.bytes.to_numpy()
    km = pl.KM(L.T_hi.to_numpy(), E)
    Ca, _ = pl.CV(km.S(np.ones(len(L))), 1800, peff); Cn, _ = pl.CV(km.S((b > 0).astype(float)), 1800, peff)
    out[fs] = {'N': int(len(L)), 'N_nonempty': int((b > 0).sum()), 'pred_all': Ca * len(L), 'pred_nonempty': Cn * (b > 0).sum(),
               'submitted': sub(k, off_h)}
    print(fs, {x: (round(y) if isinstance(y, float) else y) for x, y in out[fs].items()})
json.dump(out, open(os.path.join(res, 'pred_fs.json'), 'w'), indent=1, default=float)
