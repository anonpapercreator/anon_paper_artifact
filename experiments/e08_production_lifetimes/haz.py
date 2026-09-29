import pandas as pd, numpy as np
L=pd.read_parquet('lifecycles_p.parquet',columns=['T_hi','status','bytes'])
E=(L.status!='censored').to_numpy(); T=L.T_hi.to_numpy(); b=L.bytes.to_numpy()
edges=np.array([0,60,600,1800,3600,3*3600,6*3600,12*3600,86400,2*86400,4*86400,7*86400,14*86400,21*86400,29*86400])
print("age bin           hazard/h count    hazard/h bytes   (life-table: deaths / exposure)")
for lo,hi in zip(edges[:-1],edges[1:]):
    exp_t=np.clip(np.minimum(T,hi)-lo,0,None)   # time at risk in bin
    d=E&(T>lo)&(T<=hi)
    hc=d.sum()/exp_t.sum()*3600; hb=(b*d).sum()/(b*exp_t).sum()*3600
    print(f"{lo/3600:8.2f}-{hi/3600:8.2f} h  {hc:12.3e}   {hb:12.3e}")
dead=E&(T>1800)&(T<=28*86400)   # died after the production copy point
print("TB born in window:",b.sum()/1e12," TB dying at age in (30m,28d]:",(b*dead).sum()/1e12)
