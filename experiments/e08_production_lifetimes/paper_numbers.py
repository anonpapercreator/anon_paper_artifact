"""All numbers, curves and CIs for Sections IV-C and V-D/E/G (IPDPS 2027).
Inputs: lifecycles_p.parquet (life.py + boot.py). Seed 1, 400 prefix-cluster bootstrap reps."""
import numpy as np, pandas as pd, json
L=pd.read_parquet('lifecycles_p.parquet',columns=['T_hi','T_lo','status','bytes','prefix'])
E=(L.status!='censored').to_numpy().astype(float)
pid=L.prefix.astype('category').cat.codes.to_numpy(); P=pid.max()+1
b=L.bytes.to_numpy(); T=L.T_hi.to_numpy(); Tl=L.T_lo.to_numpy()
HOR=29*86400; grid=np.linspace(0,HOR,29*24*60+1); dx=grid[1]-grid[0]   # 1-min grid
def prep(TT):
    ut,g=np.unique(TT,return_inverse=True)
    gi=np.searchsorted(ut,grid,side='right')-1
    return g,len(ut),gi
PH=prep(T); PL=prep(Tl)
def S_grid(w,pp=PH):
    g,ng,gi=pp
    wd=np.bincount(g,w*E,ng); wa=np.bincount(g,w,ng)
    ar=w.sum()-np.r_[0,np.cumsum(wa)[:-1]]
    S=np.cumprod(1-np.divide(wd,ar,out=np.zeros_like(wd),where=ar>0))
    return np.where(gi>=0,S[np.maximum(gi,0)],1.0)
def GH(S):
    G=np.r_[0,np.cumsum((S[1:]+S[:-1])/2*dx)]; H=np.r_[0,np.cumsum((G[1:]+G[:-1])/2*dx)]; return G,H
def CV(S,a,p):
    G,H=GH(S); f=lambda A,x: np.interp(x,grid,A)
    if p==0: return float(f(S,a)), float(f(G,a))/3600
    return float((f(G,a+p)-f(G,a))/p), float((f(H,a+p)-f(H,a))/p/3600)
h=3600; m=60
POL=[('Production',30*m,6*h),('Copy at once',0,0),('Scan every 1 h',30*m,1*h),('Scan every 15 min',30*m,15*m),
     ('Fixed delay 30 min',30*m,0),('Age 1 d, scan 6 h',24*h,6*h),('Age 7 d, scan 6 h',7*24*h,6*h)]
ages=np.array([60,600,30*m,h,6*h,6.5*h,24*h,7*24*h,14*24*h,28*24*h])
w1=np.ones(len(L))
point={}
for wn,w in [('count',w1),('bytes',b)]:
    Sh=S_grid(w); Sl=S_grid(w,PL)
    point[wn]={'S_hi':Sh,'S_lo':Sl}
rng=np.random.default_rng(1); B=400
bootS={'count':[], 'bytes':[]}; bootP={'count':[], 'bytes':[]}
for r in range(B):
    mm=rng.multinomial(P,np.full(P,1/P))[pid].astype(float)
    for wn,w in [('count',mm),('bytes',mm*b)]:
        S=S_grid(w); bootS[wn].append(S[np.searchsorted(grid,ages)])
        bootP[wn].append([CV(S,a,p) for _,a,p in POL])
out={}
for wn in ('count','bytes'):
    bs=np.array(bootS[wn]); bp=np.array(bootP[wn])
    Sh=point[wn]['S_hi']; Sl=point[wn]['S_lo']; idx=np.searchsorted(grid,ages)
    out[wn]={'S':{str(int(a)):{'hi':Sh[i],'lo':Sl[i],'ci':list(np.percentile(bs[:,k],[2.5,97.5]))} for k,(a,i) in enumerate(zip(ages,idx))},
             'policy':{n:{'C':CV(Sh,a,p)[0],'V_h':CV(Sh,a,p)[1],'C_lo':CV(Sl,a,p)[0],'V_h_lo':CV(Sl,a,p)[1],
                          'C_ci':list(np.percentile(bp[:,j,0],[2.5,97.5])),'V_ci':list(np.percentile(bp[:,j,1],[2.5,97.5]))}
                       for j,(n,a,p) in enumerate(POL)}}
json.dump(out,open('paper_numbers.json','w'),indent=1,default=float)
# survival curve CSV (log-spaced ages, 1 min to 29 d) with pointwise bootstrap CI (separate pass on a sparse set)
sa=np.unique(np.round(np.logspace(np.log10(60),np.log10(HOR),60)/60)*60)
si=np.searchsorted(grid,sa)
rng=np.random.default_rng(2); cur={'count':[],'bytes':[]}
for r in range(200):
    mm=rng.multinomial(P,np.full(P,1/P))[pid].astype(float)
    cur['count'].append(S_grid(mm)[si]); cur['bytes'].append(S_grid(mm*b)[si])
df=pd.DataFrame({'age_h':sa/3600})
for wn in ('count','bytes'):
    c=np.array(cur[wn]); df[f'{wn}']=point[wn]['S_hi'][si]; df[f'{wn}_lo']=np.percentile(c,2.5,axis=0); df[f'{wn}_hi']=np.percentile(c,97.5,axis=0)
df.round(5).to_csv('lifetime_survival.csv',index=False)
# frontier CSV: for p in {0,15m,1h,6h}, a from 0 to 7 d
rows=[]
for wn in ('count','bytes'):
    S=point[wn]['S_hi']
    for p in (0,15*m,h,6*h):
        for a in np.r_[np.arange(0,2*h,5*m),np.arange(2*h,24*h,30*m),np.arange(24*h,7*24*h+1,6*h)]:
            C,V=CV(S,a,p); rows.append((wn,p/3600,a/3600,C,V))
pd.DataFrame(rows,columns=['weight','p_h','a_h','C_frac','V_h']).round(5).to_csv('policy_frontier.csv',index=False)
print(json.dumps({wn:{n:{k:(round(v,4) if not isinstance(v,list) else [round(x,4) for x in v]) for k,v in d.items()} for n,d in out[wn]['policy'].items()} for wn in out},indent=0))
print(json.dumps({wn:{k:{kk:(round(vv,4) if not isinstance(vv,list) else [round(x,4) for x in vv]) for kk,vv in d.items()} for k,d in out[wn]['S'].items()} for wn in out},indent=0))
