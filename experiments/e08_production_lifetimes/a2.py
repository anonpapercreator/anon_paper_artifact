import pandas as pd, numpy as np
df=pd.read_parquet('audit.parquet',columns=['t','ev','key','size','nlink','mtime','atime'])
df['ev']=df.ev.astype(str)
d=df.sort_values(['key','t']).reset_index(drop=True)
g=d.groupby('key')
d['prev']=g.ev.shift(1); d['tp']=g.t.shift(1); d['nxt']=g.ev.shift(-1); d['tn']=g.t.shift(-1)
cr=d[d.ev=='CREATE']
print("CREATE next event:",cr.nxt.fillna('none').value_counts().to_dict())
un=d[d.ev=='UNLINK']
print("UNLINK prev event:",un.prev.fillna('none').value_counts().to_dict())
print(pd.crosstab(un.prev.fillna('none'),un.nlink))
# collapse to CREATE/UNLINK only sequence
cu=d[d.ev.isin(['CREATE','UNLINK'])].copy(); h=cu.groupby('key')
cu['p2']=h.ev.shift(1); cu['tp2']=h.t.shift(1); cu['n2']=h.ev.shift(-1); cu['tn2']=h.t.shift(-1)
x=cu[(cu.ev=='CREATE')&(cu.n2=='CREATE')]
print("CREATE->CREATE (no UNLINK between):",len(x),"gap q:",(x.tn2-x.t).quantile([.1,.25,.5,.75,.9]).round(1).tolist())
print("  exact-zero gaps:",((x.tn2-x.t)<0.01).sum())
uu=cu[(cu.ev=='UNLINK')&(cu.p2=='UNLINK')]; print("UNLINK->UNLINK:",len(uu),"nlink:",uu.nlink.value_counts().to_dict())
pu=cu[(cu.ev=='UNLINK')&(cu.p2=='CREATE')]
print("paired",len(pu),"(mtime - tcreate) q:",(pu.mtime-pu.tp2).quantile([.01,.05,.25,.5,.75,.95,.99]).round(0).tolist())
print("lifetime q (s):",(pu.t-pu.tp2).quantile([.1,.25,.5,.75,.9,.99]).round(1).tolist())
