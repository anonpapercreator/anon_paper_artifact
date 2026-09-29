# Fraction of written data copied under the production policy (a=30 min, p=6 h)
# and then deleted within 28 days: W = C/rho - S(28 d). Kaplan-Meier point
# estimate (latest and earliest placement of inferred deletions) and a
# 400-replicate cluster bootstrap over second-level directory trees.
import numpy as np, json
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600; m=60; T=28*24*h
def W(S): return CV(S,30*m,6*h)[0]-S[np.searchsorted(grid,T)]
out={}
for wn,w in [('count',np.ones(len(L))),('bytes',b)]:
    out[wn]={'point_latest':W(S_grid(w)),'point_earliest':W(S_grid(w,PL))}
rng=np.random.default_rng(4); R={'count':[],'bytes':[]}
for r in range(400):
    mm=rng.multinomial(P,np.full(P,1/P))[pid].astype(float)
    R['count'].append(W(S_grid(mm))); R['bytes'].append(W(S_grid(mm*b)))
for wn in R: out[wn]['ci95']=list(np.percentile(R[wn],[2.5,97.5]))
json.dump(out,open('waste_ci.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
