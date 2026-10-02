"""Point estimates quoted in the paper for the pooled file systems (no bootstrap):
exposure and copied fraction, survival at key ages, absolute rates, and the fraction of
files deleted within a day that a daily snapshot at a random time would record."""
import json, os, sys, glob, importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('pl', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline.py'))
pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
res = sys.argv[1]; excl = set(sys.argv[2:])
P = []
for f in sorted(glob.glob(os.path.join(res, '*_lifecycles.parquet'))):
    if os.path.basename(f).split('_')[0] in excl: continue
    P.append(pd.read_parquet(f, columns=['T_hi', 'T_lo', 'status', 'bytes']))
L = pd.concat(P, ignore_index=True); del P
E = (L.status != 'censored').to_numpy().astype(float); b = L.bytes.to_numpy()
kh = pl.KM(L.T_hi.to_numpy(), E); kl = pl.KM(L.T_lo.to_numpy(), E)
out = {'lifecycles': int(len(L)), 'TB': float(b.sum() / 1e12), 'deleted_recorded': int((L.status == 'death').sum()),
       'deleted_inferred': int((L.status == 'reuse').sum()), 'censored': int((L.status == 'censored').sum())}
D = 86400; ig = np.searchsorted(pl.GRID, D)
for wn, w in (('bytes', b), ('count', np.ones(len(L)))):
    S = kh.S(w); m = pl.metrics(S, 6.0)
    C1, V1 = pl.CV(S, 1800, 3600); C15, V15 = pl.CV(S, 1800, 900)
    snap = np.trapezoid(S[:ig + 1] - S[ig], pl.GRID[:ig + 1]) / D / (1 - S[ig])
    out[wn] = {'C6': m['C6'], 'V6_h': m['V6_h'], 'C1': C1, 'V1_h': V1, 'C15': C15, 'V15_h': V15,
               'S_30m': m['S_30m'], 'S_6h': m['S_6h'], 'S_1d': m['S_1d'], 'S_7d': m['S_7d'], 'S_28d': m['S_28d'],
               'waste_28d': m['waste_28d'], 'waste_28d_earliest': pl.metrics(kl.S(w), 6.0)['waste_28d'],
               'snapshot_records_frac_of_deleted_within_1d': float(snap)}
TBd = out['TB'] / 30
out['TB_per_day'] = TBd; out['MB_per_s'] = TBd * 1e12 / 86400 / 1e6
out['extra_TB_per_day_hourly'] = TBd * (out['bytes']['C1'] - out['bytes']['C6'])
out['unprotected_TB_6h'] = TBd / 24 * out['bytes']['V6_h']; out['unprotected_TB_1h'] = TBd / 24 * out['bytes']['V1_h']
tag = 'nine' if excl else 'all'
json.dump(out, open(os.path.join(res, f'pooled_extra_{tag}.json'), 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
