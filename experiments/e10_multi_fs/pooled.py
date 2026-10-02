"""Pooled lifetime analysis over all measured file systems: one Kaplan-Meier curve from all
lifecycles, intervals from a cluster bootstrap over (file system, directory tree) pairs.
usage: python3 pooled.py RESULTS_DIR [--reps 400] [--seed 11] [--exclude FS ...]"""
import argparse, json, os, glob, time, importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('pl', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline.py'))
pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
ap = argparse.ArgumentParser(); ap.add_argument('res'); ap.add_argument('--reps', type=int, default=400)
ap.add_argument('--seed', type=int, default=11); ap.add_argument('--exclude', nargs='*', default=[])
ap.add_argument('--peff', type=float, default=6.26); ap.add_argument('--tag', default='all')
a = ap.parse_args(); t0 = time.time()
parts = []; fsl = []
for f in sorted(glob.glob(os.path.join(a.res, '*_lifecycles.parquet'))):
    fs = os.path.basename(f).split('_')[0]
    if fs in a.exclude: continue
    L = pd.read_parquet(f, columns=['T_hi', 'T_lo', 'status', 'bytes', 'prefix'])
    L['fs'] = np.int16(len(fsl)); fsl.append(fs); parts.append(L)
L = pd.concat(parts, ignore_index=True); del parts
E = (L.status != 'censored').to_numpy().astype(float); b = L.bytes.to_numpy()
cl = L.fs.to_numpy().astype(np.uint64) << np.uint64(48) ^ L.prefix.to_numpy()   # (fs, tree) cluster
pid = np.unique(cl, return_inverse=True)[1].ravel(); P = pid.max() + 1; del cl
kh = pl.KM(L.T_hi.to_numpy(), E); kl = pl.KM(L.T_lo.to_numpy(), E)
out = {'file_systems': fsl, 'lifecycles': int(len(L)), 'TB_written': float(b.sum() / 1e12), 'clusters': int(P), 'peff_h': a.peff}
one = np.ones(len(L))
for wn, w in (('bytes', b), ('count', one)):
    out[wn] = pl.metrics(kh.S(w), a.peff); out[wn]['waste_28d_earliest'] = pl.metrics(kl.S(w), a.peff)['waste_28d']
rng = np.random.default_rng(a.seed); R = {'bytes': [], 'count': []}
keys = ['Vratio_1h', 'Cinc_1h_pct', 'Vratio_15m', 'Cinc_15m_pct', 'waste_28d', 'S_28d', 'V6_h', 'C6']
for r in range(a.reps):
    mm = rng.multinomial(P, np.full(P, 1 / P))[pid].astype(float)
    for wn, w in (('bytes', mm * b), ('count', mm)):
        m = pl.metrics(kh.S(w), a.peff); R[wn].append([m[k] for k in keys])
for wn in R:
    A = np.array(R[wn]); out[wn]['ci95'] = {k: [float(np.percentile(A[:, j], 2.5)), float(np.percentile(A[:, j], 97.5))] for j, k in enumerate(keys)}
out['reps'] = a.reps; out['seed'] = a.seed; out['elapsed_s'] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(a.res, f'pooled_{a.tag}.json'), 'w'), indent=1, default=float)
print(json.dumps({wn: {k: round(out[wn][k], 4) for k in keys} for wn in ('bytes', 'count')}))
print({wn: {k: [round(x, 4) for x in v] for k, v in out[wn]['ci95'].items()} for wn in ('bytes', 'count')})
