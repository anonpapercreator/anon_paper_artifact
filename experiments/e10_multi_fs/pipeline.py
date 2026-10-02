"""Per-file-system lifetime analysis (IPDPS 2027, Sec. V replication across file systems).

Same method as e08 (file system A): load.py -> life.py -> boot.py -> paper_numbers.py/waste_ci.py,
applied to one file system's reduced audit extract. Point estimates use exactly the e08
estimators; intervals use one prefix-cluster bootstrap (default 400 reps, as in e08) for all metrics.

usage: python3 pipeline.py --in DIR_WITH_TSV_GZ --fs NAME --out OUTDIR [--reps 400] [--seed 7] [--peff HOURS]
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd

T0 = pd.Timestamp('2026-08-28').timestamp(); T1 = pd.Timestamp('2026-09-27').timestamp()
COLS = "t ev key size nlink ext prefix node excl atime mtime".split()
HOR = 29 * 86400; GRID = np.linspace(0, HOR, 29 * 24 * 60 + 1); DX = GRID[1] - GRID[0]
H, M = 3600, 60

EVC = {'CREATE': 0, 'WCLOSE': 1, 'UNLINK': 2, 'RENAME': 3}

def _hex64(a):   # 16-digit hex strings (pyarrow array) -> uint64; null or '-' -> 0
    b = np.array(a.fill_null('0').to_pylist(), dtype=object)
    return np.fromiter((int(k, 16) if k != '-' else 0 for k in b), dtype=np.uint64, count=len(b))

def load(d):
    """Memory-lean loader (arrays, in-window rows only). Equivalent to the e08 loader followed by
    exact-duplicate removal: duplicates are found by a stable sort on all loaded fields and the
    first occurrence is kept in the original file order. Prefix hashes are held as uint64
    (0 = missing) and converted back to the 16-digit hex strings for output."""
    import pyarrow.csv as pc, pyarrow as pa
    fs = sorted(glob.glob(os.path.join(d, '*.tsv.gz')))
    if not fs: raise SystemExit(f'no .tsv.gz in {d}')
    ro = pc.ReadOptions(column_names=COLS); po = pc.ParseOptions(delimiter='\t', quote_char=False)
    co = pc.ConvertOptions(include_columns=['t', 'ev', 'key', 'size', 'nlink', 'prefix'],
                           column_types={'t': pa.float64(), 'ev': pa.string(), 'key': pa.string(),
                                         'size': pa.float64(), 'nlink': pa.float32(), 'prefix': pa.string()},
                           null_values=['', '-'], strings_can_be_null=True)
    A = {k: [] for k in ('t', 'ev', 'key', 'size', 'nlink', 'prefix')}; nrec = 0
    for f in fs:
        if os.path.getsize(f) <= 64:   # empty day file (gzip header only)
            with __import__('gzip').open(f, 'rb') as g:
                if not g.read(1): continue
        x = pc.read_csv(f, read_options=ro, parse_options=po, convert_options=co); nrec += x.num_rows
        t = x['t'].to_numpy(zero_copy_only=False); w = (t >= T0) & (t < T1)
        if not w.any(): continue
        x = x.filter(pa.array(w))
        A['t'].append(x['t'].to_numpy(zero_copy_only=False))
        evs = np.array(x['ev'].fill_null('').to_pylist(), dtype=object)
        A['ev'].append(np.fromiter((EVC.get(e, -1) for e in evs), dtype=np.int8, count=len(evs)))
        A['key'].append(_hex64(x['key'])); A['prefix'].append(_hex64(x['prefix']))
        A['size'].append(x['size'].to_numpy(zero_copy_only=False).astype(np.float64))
        A['nlink'].append(x['nlink'].to_numpy(zero_copy_only=False).astype(np.float32))
    A = {k: np.concatenate(v) if v else np.array([]) for k, v in A.items()}
    n0 = len(A['t'])
    # exact duplicates: stable sort on all fields, flag rows equal to their predecessor (NaN == NaN)
    o = np.lexsort((A['prefix'], np.nan_to_num(A['nlink'], nan=-1), np.nan_to_num(A['size'], nan=-1),
                    A['ev'], A['t'], A['key']))
    same = np.ones(n0 - 1, dtype=bool) if n0 > 1 else np.zeros(0, dtype=bool)
    for k in ('key', 't', 'ev', 'size', 'nlink', 'prefix'):
        a = A[k][o]
        eq = a[1:] == a[:-1]
        if a.dtype.kind == 'f': eq |= np.isnan(a[1:]) & np.isnan(a[:-1])
        same &= eq
    keep = np.ones(n0, dtype=bool); keep[o[1:][same]] = False; del o, same
    for k in list(A): A[k] = A[k][keep]   # one column at a time to limit peak memory
    return A, int(n0 - keep.sum()), len(fs), nrec

def lifecycles(A):   # identical logic to e08/life.py; A: dict of in-window arrays (consumed)
    t, key, ev, size, nl, pre = A.pop('t'), A.pop('key'), A.pop('ev'), A.pop('size'), A.pop('nlink'), A.pop('prefix')
    o = np.lexsort((t, key))
    t = t[o]; key = key[o]; ev = ev[o]; size = size[o]; nl = nl[o]; pre = pre[o]; del o
    newkey = np.r_[True, key[1:] != key[:-1]]; isc = (ev == 0)
    lcid = np.cumsum(isc)
    keystart_lc = np.maximum.accumulate(np.where(newkey, lcid - isc, 0)); del newkey
    sel = lcid > keystart_lc; del keystart_lc
    # prefix of each lifecycle: as the e08 merge on (key, tc) with the first CREATE per (key, t)
    ci = np.flatnonzero(isc); del isc
    gstart = np.r_[True, (key[ci][1:] != key[ci][:-1]) | (t[ci][1:] != t[ci][:-1])]
    first = ci[np.maximum.accumulate(np.where(gstart, np.arange(len(ci)), 0))]
    lcc, prc = lcid[ci], pre[first]; del ci, gstart, first, pre
    lc = lcid[sel]; tt = t[sel]; ee = ev[sel]; ss = size[sel]; nn = nl[sel]; kk = key[sel]
    del lcid, sel, t, key, ev, size, nl
    n = lc.max() + 1
    tc = np.full(n, np.nan); tc[lc[ee == 0]] = tt[ee == 0]
    pf = np.zeros(n, dtype=np.uint64); pf[lcc] = prc; del lcc, prc
    dm = (ee == 2) & (nn == 1); td = np.full(n, np.inf); np.minimum.at(td, lc[dm], tt[dm])
    tlast = np.full(n, -np.inf); np.maximum.at(tlast, lc, tt)
    b = np.zeros(n); mm = (ee == 1) | (ee == 2); np.maximum.at(b, lc[mm], ss[mm])
    ck = np.zeros(n, dtype=np.uint64); ck[lc[ee == 0]] = kk[ee == 0]
    ids = np.unique(lc); del lc, tt, ee, ss, nn, kk, dm, mm
    L = pd.DataFrame({'key': ck[ids], 'tc': tc[ids], 'td': td[ids], 'tlast': tlast[ids], 'bytes': b[ids], 'prefix': pf[ids]})
    del ck, tc, td, tlast, b, pf, ids
    L = L.sort_values(['key', 'tc'], kind='mergesort', ignore_index=True)
    L['tnext'] = L.groupby('key').tc.shift(-1)
    L['td'] = L.td.replace(np.inf, np.nan)
    st = np.where(L.td.notna(), 0, np.where(L.tnext.notna(), 1, 2))
    L['status'] = pd.Categorical.from_codes(st, ['death', 'reuse', 'censored'])
    L['T_hi'] = np.where(st == 0, L.td - L.tc, np.where(st == 1, L.tnext - L.tc, T1 - L.tc))
    L['T_lo'] = np.where(st == 1, L.tlast - L.tc, L.T_hi)
    return L[['key', 'tc', 'td', 'tlast', 'bytes', 'tnext', 'status', 'T_hi', 'T_lo', 'prefix']]

class KM:   # identical estimator to e08/paper_numbers.py
    def __init__(self, T, E):
        self.E = E
        ut, g = np.unique(T, return_inverse=True)
        self.g, self.ng = g, len(ut); self.gi = np.searchsorted(ut, GRID, side='right') - 1
    def S(self, w):
        wd = np.bincount(self.g, w * self.E, self.ng); wa = np.bincount(self.g, w, self.ng)
        ar = w.sum() - np.r_[0, np.cumsum(wa)[:-1]]
        S = np.cumprod(1 - np.divide(wd, ar, out=np.zeros_like(wd), where=ar > 0))
        return np.where(self.gi >= 0, S[np.maximum(self.gi, 0)], 1.0)

def CV(S, a, p):
    G = np.r_[0, np.cumsum((S[1:] + S[:-1]) / 2 * DX)]; Hh = np.r_[0, np.cumsum((G[1:] + G[:-1]) / 2 * DX)]
    f = lambda A, x: np.interp(x, GRID, A)
    if p == 0: return float(f(S, a)), float(f(G, a)) / 3600
    return float((f(G, a + p) - f(G, a)) / p), float((f(Hh, a + p) - f(Hh, a)) / p / 3600)

def metrics(S, peff):
    C6, V6 = CV(S, 30 * M, 6 * H); C1, V1 = CV(S, 30 * M, H); C15, V15 = CV(S, 30 * M, 15 * M)
    Ce, Ve = CV(S, 30 * M, peff * H)
    s = lambda x: float(np.interp(x, GRID, S))
    return {'C6': C6, 'V6_h': V6, 'C1': C1, 'V1_h': V1, 'Vratio_1h': V6 / V1, 'Cinc_1h_pct': 100 * (C1 / C6 - 1),
            'Vratio_15m': V6 / V15, 'Cinc_15m_pct': 100 * (C15 / C6 - 1), 'Veff_h': Ve, 'Ceff': Ce,
            'waste_28d': C6 - s(28 * 86400), 'S_30m': s(30 * M), 'S_6h': s(6 * H), 'S_1d': s(86400),
            'S_7d': s(7 * 86400), 'S_28d': s(28 * 86400), 'bound_ratio': (0.5 + 3) / (0.5 + 0.5)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--fs', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--reps', type=int, default=400)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--peff', type=float, default=6.25)
    ap.add_argument('--t0', default='2026-08-28', help='window start, UTC date (inclusive)')
    ap.add_argument('--t1', default='2026-09-27', help='window end, UTC date (exclusive)')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    global T0, T1
    T0 = pd.Timestamp(a.t0).timestamp(); T1 = pd.Timestamp(a.t1).timestamp()
    assert abs((T1 - T0) - 30 * 86400) < 1, 'HOR and the 28-day metrics assume a 30-day window'
    A, ndup, nfiles, nrec = load(a.inp)
    L = lifecycles(A); del A
    # prefix and key are the 64-bit values of the 16-digit hex hashes; prefix 0 = missing
    L.to_parquet(os.path.join(a.out, f'{a.fs}_lifecycles.parquet'))
    E = (L.status != 'censored').to_numpy().astype(float); b = L.bytes.to_numpy()
    pid = np.unique(L.prefix.to_numpy(), return_inverse=True)[1].ravel(); P = pid.max() + 1   # 0 sorts first, as '-' did
    kh = KM(L.T_hi.to_numpy(), E); kl = KM(L.T_lo.to_numpy(), E); one = np.ones(len(L))
    out = {'fs': a.fs, 'input_files': nfiles, 'records_in_files': int(nrec), 'duplicate_records_dropped_in_window': int(ndup),
           'lifecycles': int(len(L)), 'status': L.status.value_counts().to_dict(),
           'TB_written': float(b.sum() / 1e12), 'empty_file_share': float((b == 0).mean()),
           'directory_trees': int(P), 'prefix_missing_share': float((L.prefix == 0).mean()), 'peff_h': a.peff}
    bp = L.groupby('prefix').bytes.sum().sort_values(ascending=False)
    # as in the paper (Sec. VI): the five trees that wrote the most bytes, and their share of
    # all bytes deleted (death or inferred deletion by inode reuse)
    dp = L[L.status != 'censored'].groupby('prefix').bytes.sum()
    top5 = bp.index[:5]
    out['top5_share_written'] = float(bp.iloc[:5].sum() / bp.sum()) if bp.sum() > 0 else None
    out['top5_share_deleted'] = float(dp[dp.index.isin(top5)].sum() / dp.sum()) if len(dp) and dp.sum() > 0 else None
    for wn, w in (('bytes', b), ('count', one)):
        out[wn] = metrics(kh.S(w), a.peff)
        out[wn]['waste_28d_earliest'] = metrics(kl.S(w), a.peff)['waste_28d']
    rng = np.random.default_rng(a.seed); R = {'bytes': [], 'count': []}
    keys = ['Vratio_1h', 'Cinc_1h_pct', 'waste_28d', 'S_28d', 'V6_h']
    for r in range(a.reps):
        mm = rng.multinomial(P, np.full(P, 1 / P))[pid].astype(float)
        for wn, w in (('bytes', mm * b), ('count', mm)):
            if w.sum() == 0: continue
            m = metrics(kh.S(w), a.peff); R[wn].append([m[k] for k in keys])
    for wn in R:
        A = np.array(R[wn])
        out[wn]['ci95'] = {k: [float(np.percentile(A[:, j], 2.5)), float(np.percentile(A[:, j], 97.5))] for j, k in enumerate(keys)} if len(A) else None
    out['reps'] = a.reps; out['seed'] = a.seed; out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, f'{a.fs}_results.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('fs', 'lifecycles', 'TB_written', 'elapsed_s')}),
          {wn: {k: round(out[wn][k], 4) for k in ('Vratio_1h', 'Cinc_1h_pct', 'waste_28d', 'S_28d')} for wn in ('bytes', 'count')})

if __name__ == '__main__':
    main()
