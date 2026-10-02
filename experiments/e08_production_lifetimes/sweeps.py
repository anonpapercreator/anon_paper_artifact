import numpy as np, pandas as pd
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600; m=60; S=S_grid(b)
ps=np.r_[0,np.arange(5,60,5)*m,np.arange(1,12.01,0.25)*h]
rows=[(p/h,)+tuple(CV(S,30*m,p)) for p in ps]
pd.DataFrame(rows,columns=['p_h','C_frac','V_h']).assign(bound_h=lambda d:0.5+d.p_h/2).round(5).to_csv('sweep_p.csv',index=False)
As=np.r_[0,np.arange(5,60,5)*m,np.arange(1,24,1)*h,np.arange(24,24*14+1,6)*h]
rows=[(a/h,)+tuple(CV(S,a,6*h)) for a in As]
pd.DataFrame(rows,columns=['a_h','C_frac','V_h']).round(5).to_csv('sweep_a.csv',index=False)
print(pd.read_csv('sweep_p.csv').iloc[[0,6,12,15,23,-1]]); print(pd.read_csv('sweep_a.csv').iloc[[0,12,20,35,-1]])
