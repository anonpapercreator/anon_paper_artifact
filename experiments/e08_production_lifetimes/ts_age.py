import pandas as pd, numpy as np
T0=pd.Timestamp('2026-08-28').timestamp(); T1=pd.Timestamp('2026-09-27').timestamp()
L=pd.read_parquet('lifecycles.parquet',columns=['key','tc','bytes'])
w=pd.read_parquet('audit.parquet',columns=['t','ev','key','mtime','atime'])
w=w[(w.ev=='WCLOSE')&(w.t>=T0)&(w.t<T1)].copy()
w['key']=np.array([int(k,16) for k in w.key],dtype=np.uint64)
L=L.sort_values('tc'); w=w.sort_values('t')
# first write-close at or after creation, same inode key, within 1 day
m=pd.merge_asof(L,w.rename(columns={'t':'tw'}),left_on='tc',right_on='tw',by='key',direction='forward',tolerance=86400)
m=m[m.tw.notna()]
for col in ('mtime','atime'):
    age_at_close=m.tw-m[col]            # timestamp-based age when the file is first closed after writing
    old=(age_at_close-(m.tw-m.tc))>1800 # timestamp older than creation by more than 30 min
    print(f"{col}: files {len(m)}; timestamp predates creation by >30 min: count {old.mean():.3f}, bytes {(m.bytes*old).sum()/m.bytes.sum():.3f}")
    old1=(m.tc-m[col])>86400
    print(f"   predates creation by >1 day: count {old1.mean():.3f}, bytes {(m.bytes*old1).sum()/m.bytes.sum():.3f}")
print("lifecycles with a write-close within 1 day:",len(m),"of",len(L))
