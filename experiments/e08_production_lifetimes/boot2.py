import pandas as pd, numpy as np
L=pd.read_parquet('lifecycles_p.parquet',columns=['T_hi','status','bytes','prefix'])
E=(L.status!='censored').to_numpy().astype(float)
pid=L.prefix.astype('category').cat.codes.to_numpy(); P=pid.max()+1
b=L.bytes.to_numpy(); T=L.T_hi.to_numpy()
grid=np.linspace(0,29*86400,29*24*12+1); dx=grid[1]-grid[0]
ut,g=np.unique(T,return_inverse=True); ng=len(ut)
gi=np.clip(np.searchsorted(ut,grid,side='right')-1,-1,None)
def S_grid(w):
    wd=np.bincount(g,w*E,ng); wa=np.bincount(g,w,ng)
    ar=w.sum()-np.r_[0,np.cumsum(wa)[:-1]]
    S=np.cumprod(1-np.divide(wd,ar,out=np.zeros_like(wd),where=ar>0))
    return np.where(gi>=0,S[np.maximum(gi,0)],1.0)
def CV(S,a,p):
    G=np.r_[0,np.cumsum((S[1:]+S[:-1])/2*dx)]; H=np.r_[0,np.cumsum((G[1:]+G[:-1])/2*dx)]
    f=lambda A,x: np.interp(x,grid,A)
    if p==0: return f(S,a), f(G,a)/3600
    return (f(G,a+p)-f(G,a))/p, (f(H,a+p)-f(H,a))/p/3600
i28=np.searchsorted(grid,28*86400); i1=288
rng=np.random.default_rng(1); out=[]
for r in range(400):
    m=rng.multinomial(P,np.full(P,1/P))[pid].astype(float)
    Sc=S_grid(m); Sb=S_grid(m*b)
    out.append([CV(Sc,1800,21600)[0],CV(Sb,1800,21600)[0],CV(Sb,1800,21600)[1],CV(Sb,1800,0)[1],Sb[i1],Sb[i28],Sc[i1],Sc[i28]])
out=np.array(out); np.save('boot.npy',out)
q=np.percentile(out,[2.5,50,97.5],axis=0)
for j,n in enumerate(['C/rho count prod','C/rho bytes prod','V/rho h bytes prod','V/rho h bytes a=30m p=0','S_bytes(1d)','S_bytes(28d)','S_count(1d)','S_count(28d)']):
    print(f"{n:26s}",np.round(q[:,j],4))
top=L.groupby('prefix').bytes.sum().sort_values(ascending=False).index
for k in (1,5,10):
    keep=(~L.prefix.isin(top[:k])).to_numpy().astype(float)
    Sb=S_grid(keep*b); print(f"drop top-{k}: C/rho prod {CV(Sb,1800,21600)[0]:.4f}  S(1d) {Sb[i1]:.4f}  S(28d) {Sb[i28]:.4f}  TB {(keep*b).sum()/1e12:.1f}")
