import pandas as pd, numpy as np
T0=pd.Timestamp('2026-08-28').timestamp(); T1=pd.Timestamp('2026-09-27').timestamp()
df=pd.read_parquet('audit.parquet',columns=['t','ev','key','nlink','mtime'])
df=df[(df.t>=T0)&(df.t<T1)&df.ev.isin(['CREATE','UNLINK'])]
df['ev']=df.ev.astype(str)
df=df.sort_values(['key','t'],kind='mergesort')
g=df.groupby('key',sort=False)
df['prev_ev']=g.ev.shift(1); df['prev_t']=g.t.shift(1)
# death = UNLINK nlink1 whose most recent CREATE/UNLINK predecessor is a CREATE (window-born, same lifecycle)
d=df[(df.ev=='UNLINK')&(df.nlink==1)&(df.prev_ev=='CREATE')]
x=(d.mtime-d.prev_t)/86400
print(len(d)); print("quantiles (days):",x.quantile([.01,.05,.1,.25,.5]).round(2).to_dict())
print("share with mtime >= 1 day before create:",(x<=-1).mean().round(4),"; >=37 d:",(x<=-37).mean().round(4))
