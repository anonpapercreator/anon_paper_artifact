import pandas as pd, numpy as np
L=pd.read_parquet('lifecycles.parquet')
def km(T,E,w):
    o=np.argsort(T,kind='mergesort'); T,E,w=T[o],E[o],w[o]
    ut,idx=np.unique(T,return_index=True)
    wd=np.add.reduceat(w*E,idx); wa=np.add.reduceat(w,idx)
    atrisk=w.sum()-np.r_[0,np.cumsum(wa)[:-1]]
    S=np.cumprod(1-wd/np.where(atrisk>0,atrisk,1))
    return ut,S
def S_at(ut,S,x): i=np.searchsorted(ut,x,side='right')-1; return np.where(i>=0,S[np.clip(i,0,None)],1.0)
E=(L.status!='censored').to_numpy().astype(float)
pts=np.array([60,600,1800,3600,6*3600,6.5*3600,86400,7*86400,14*86400,28*86400])
lab=['1m','10m','30m','1h','6h','6.5h','1d','7d','14d','28d']
res={}
for wname,w in [('count',np.ones(len(L))),('bytes',L.bytes.to_numpy())]:
    for bname,T in [('hi',L.T_hi.to_numpy()),('lo',L.T_lo.to_numpy())]:
        ut,S=km(T,E,w); res[(wname,bname)]=(ut,S)
        print(f"{wname:5s} reuse@{bname}:", dict(zip(lab,np.round(S_at(ut,S,pts),4))))
import pickle; pickle.dump(res,open('km.pkl','wb'))
# size-class byte-weighted, >3MB (DMF freespace eligibility) and total
for lo_,hi_,nm in [(0,3*2**20,'<=3MB'),(3*2**20,1e9,'3MB-1GB'),(1e9,1e20,'>1GB')]:
    m=(L.bytes>lo_)&(L.bytes<=hi_)
    ut,S=km(L.T_hi.to_numpy()[m],E[m],L.bytes.to_numpy()[m])
    print(f"bytes {nm:8s} n={m.sum():8d} TB={L.bytes[m].sum()/1e12:6.1f}", dict(zip(lab,np.round(S_at(ut,S,pts),4))))
