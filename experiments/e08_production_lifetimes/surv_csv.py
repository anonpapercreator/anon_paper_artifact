import numpy as np, pandas as pd
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
sa=np.unique(np.round(np.logspace(np.log10(60),np.log10(HOR),300)/60)*60); si=np.searchsorted(grid,sa)
rng=np.random.default_rng(2); cur={'count':[],'bytes':[]}
for r in range(200):
    mm=rng.multinomial(P,np.full(P,1/P))[pid].astype(float)
    cur['count'].append(S_grid(mm)[si]); cur['bytes'].append(S_grid(mm*b)[si])
df=pd.DataFrame({'age_h':sa/3600})
for wn,w in [('count',np.ones(len(L))),('bytes',b)]:
    c=np.array(cur[wn]); df[wn]=S_grid(w)[si]; df[wn+'_lo']=np.percentile(c,2.5,axis=0); df[wn+'_hi']=np.percentile(c,97.5,axis=0)
df.round(5).to_csv('lifetime_survival.csv',index=False)
# check the frontier-coincidence claim: max vertical gap between p curves at equal V (bytes and count)
f=pd.read_csv('policy_frontier.csv')
for w in ('bytes','count'):
    base=f[(f.weight==w)&(f.p_h==0)].sort_values('V_h')
    for p in (0.25,1,6):
        d=f[(f.weight==w)&((f.p_h-p).abs()<1e-9)]
        d=d[(d.V_h>=base.V_h.min())&(d.V_h<=base.V_h.max())]
        gap=d.C_frac-np.interp(d.V_h,base.V_h,base.C_frac)
        print(w,p,'max |gap| in C/rho at equal V: %.4f'%np.abs(gap).max())
print(len(df))
