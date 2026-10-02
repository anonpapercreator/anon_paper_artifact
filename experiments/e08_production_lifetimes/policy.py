import pandas as pd, numpy as np
L=pd.read_parquet('lifecycles_p.parquet',columns=['T_hi','T_lo','status','bytes','prefix'])
E=(L.status!='censored').to_numpy().astype(float)
pid=L.prefix.astype('category').cat.codes.to_numpy(); P=pid.max()+1
HOR=29*86400
grid=np.linspace(0,HOR,29*24*12+1)   # 5-min grid
def km_grid(T,E,w):
    o=np.argsort(T,kind='mergesort'); T,E,w=T[o],E[o],w[o]
    ut,idx=np.unique(T,return_index=True)
    wd=np.add.reduceat(w*E,idx); wa=np.add.reduceat(w,idx)
    atrisk=w.sum()-np.r_[0,np.cumsum(wa)[:-1]]
    S=np.cumprod(1-np.divide(wd,atrisk,out=np.zeros_like(wd),where=atrisk>0))
    i=np.searchsorted(ut,grid,side='right')-1
    return np.where(i>=0,S[np.clip(i,0,None)],1.0)
def CV(S,a,p):
    # C/rho = (G(a+p)-G(a))/p ; V/rho = (H(a+p)-H(a))/p, in hours; p=0 -> limits
    dx=grid[1]-grid[0]
    G=np.r_[0,np.cumsum((S[1:]+S[:-1])/2*dx)]; H=np.r_[0,np.cumsum((G[1:]+G[:-1])/2*dx)]
    f=lambda A,x: np.interp(x,grid,A)
    if p==0: return f(S,a), f(G,a)/3600
    return (f(G,a+p)-f(G,a))/p, (f(H,a+p)-f(H,a))/p/3600
pols=[(1800,6*3600),(0,0),(1800,3600),(1800,0),(6*3600,6*3600),(86400,6*3600),(7*86400,6*3600)]
names=['prod a=30m p=6h','a=0 p=0','a=30m p=1h','a=30m p=0','a=6h p=6h','a=1d p=6h','a=7d p=6h']
T=L.T_hi.to_numpy(); Tl=L.T_lo.to_numpy(); b=L.bytes.to_numpy()
def report(S,tag):
    print(tag, {n:tuple(np.round(CV(S,a,p),4)) for n,(a,p) in zip(names,pols)})
for wn,w in [('count',np.ones(len(L))),('bytes',b)]:
    for bn,TT in [('hi',T),('lo',Tl)]:
        report(km_grid(TT,E,w),f"{wn}/{bn} (C/rho, V/rho h):")
rng=np.random.default_rng(1)
B=200; out=[]
for r in range(B):
    m=rng.multinomial(P,np.full(P,1/P))[pid].astype(float)
    Sc=km_grid(T,E,m); Sb=km_grid(T,E,m*b)
    out.append([CV(Sc,*pols[0])[0],CV(Sb,*pols[0])[0],CV(Sb,*pols[0])[1],CV(Sb,*pols[1])[1],
                Sb[np.searchsorted(grid,28*86400)],Sc[np.searchsorted(grid,28*86400)]])
out=np.array(out); q=np.percentile(out,[2.5,50,97.5],axis=0)
for j,n in enumerate(['C/rho count prod','C/rho bytes prod','V/rho h bytes prod','V/rho h bytes a=0,p=0','S_bytes(28d)','S_count(28d)']):
    print(f"boot {n:24s} 2.5/50/97.5%:",np.round(q[:,j],4))
top=L.groupby('prefix').bytes.sum().sort_values(ascending=False).index
for k in (1,5,10):
    m=~L.prefix.isin(top[:k]).to_numpy()
    Sb=km_grid(T[m],E[m],b[m]); print(f"drop top-{k} prefixes:",'C/rho prod',round(CV(Sb,*pols[0])[0],4),'S(1d)',round(Sb[288],4),'S(28d)',round(Sb[np.searchsorted(grid,28*86400)],4),'TB',round(b[m].sum()/1e12,1))
