import pandas as pd, numpy as np
T0=pd.Timestamp('2026-08-28').timestamp(); T1=pd.Timestamp('2026-09-27').timestamp()
L=pd.read_parquet('lifecycles.parquet',columns=['key','tc','td','status'])
L=L[L.status=='death'][['key','tc','td']]
u=pd.read_parquet('audit.parquet',columns=['t','ev','key','nlink','mtime'])
u=u[(u.ev=='UNLINK')&(u.nlink==1)&(u.t>=T0)&(u.t<T1)]
u['key']=np.array([int(k,16) for k in u.key],dtype=np.uint64)
m=L.merge(u[['key','t','mtime']].rename(columns={'t':'td'}),on=['key','td'],how='inner').drop_duplicates(['key','td'])
x=(m.mtime-m.tc)/86400
print("deaths matched",len(m),"of",len(L))
print("quantiles (days):",x.quantile([.01,.05,.1,.2,.25,.5]).round(2).to_dict())
for k in (1/24,1,7,37,365): print(f"share mtime >= {k} d before create: {(x<=-k).mean():.4f}")
