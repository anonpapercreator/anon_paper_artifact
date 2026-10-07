#!/usr/bin/env python3
"""Export the byte-weighted Kaplan-Meier survival curve of file system A as an aggregate curve.

Input: A's lifecycle table from e10/pipeline.py (private; contains hashed keys and prefixes).
Output: survival_a.csv with columns age_s, S on a 1-minute grid to 29 days. The output holds
no keys, paths, identities or prefixes. Same estimator as e10/pipeline.py (T_hi placement).

usage: python3 export_survival_a.py LIFECYCLES_PARQUET OUT_CSV
"""
import sys
import numpy as np, pandas as pd

HOR = 29 * 86400
GRID = np.arange(0, HOR + 1, 60.0)

def km(T, E, w):
    o = np.argsort(T, kind='mergesort'); T, E, w = T[o], E[o], w[o]
    ut, idx = np.unique(T, return_index=True)
    wd = np.add.reduceat(w * E, idx); wa = np.add.reduceat(w, idx)
    atrisk = w.sum() - np.r_[0, np.cumsum(wa)[:-1]]
    Sv = np.cumprod(1 - np.divide(wd, atrisk, out=np.zeros_like(wd), where=atrisk > 0))
    return ut, Sv

def main():
    L = pd.read_parquet(sys.argv[1], columns=['T_hi', 'status', 'bytes'])
    E = (L.status != 'censored').to_numpy().astype(float)
    ut, Sv = km(L.T_hi.to_numpy(float), E, L.bytes.to_numpy(float))
    i = np.searchsorted(ut, GRID, side='right') - 1
    S = np.where(i >= 0, Sv[np.clip(i, 0, None)], 1.0)
    pd.DataFrame({'age_s': GRID.astype(int), 'S': S}).to_csv(sys.argv[2], index=False, float_format='%.8f')

if __name__ == '__main__':
    main()
