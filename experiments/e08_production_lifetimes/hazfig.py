import numpy as np, pandas as pd
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600
edges=np.unique(np.r_[60,np.logspace(np.log10(60),np.log10(28*24*h),22)])
idx=np.searchsorted(grid,edges); mid=np.sqrt(edges[1:]*edges[:-1])/h
def hz(S):
    s=S[idx]; return -np.diff(np.log(np.maximum(s,1e-12)))/(np.diff(edges)/h)
pt={wn:hz(S_grid(w)) for wn,w in [('count',np.ones(len(L))),('bytes',b)]}
rng=np.random.default_rng(5); R={'count':[],'bytes':[]}
for r in range(200):
    mm=rng.multinomial(P,np.full(P,1/P))[pid].astype(float)
    R['count'].append(hz(S_grid(mm))); R['bytes'].append(hz(S_grid(mm*b)))
df=pd.DataFrame({'age_h':mid})
for wn in pt:
    a=np.array(R[wn]); df[wn]=pt[wn]; df[wn+'_lo']=np.percentile(a,2.5,axis=0); df[wn+'_hi']=np.percentile(a,97.5,axis=0)
df=df.clip(lower=1e-7).round(8); df.to_csv('hazard.csv',index=False); print(df.to_string())
