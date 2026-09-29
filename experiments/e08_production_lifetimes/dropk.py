import numpy as np, pandas as pd
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600; m=60
top=L.groupby('prefix').bytes.sum().sort_values(ascending=False).index
for k in (0,1,5,10):
    keep=(~L.prefix.isin(top[:k])).to_numpy().astype(float)
    for wn,w in [('bytes',keep*b),('count',keep)]:
        S=S_grid(w); C6,V6=CV(S,30*m,6*h); C1,V1=CV(S,30*m,h)
        print(f"drop top-{k:2d} {wn:5s}: prod C {C6:.3f} V {V6:.2f} h | 1h C {C1:.3f} V {V1:.2f} | V ratio {V6/V1:.2f} C inc {100*(C1/C6-1):.2f}% | S(28d) {S[np.searchsorted(grid,28*86400)]:.3f} | TB {(keep*b).sum()/1e12:.1f}")
# equal-exposure check on 1-min grid
for wn,w in [('count',np.ones(len(L))),('bytes',b)]:
    S=S_grid(w); C0,V0=CV(S,30*m,6*h); G,_=GH(S); a_eq=np.interp(V0*3600,G,grid); C_eq=np.interp(a_eq,grid,S)
    print(wn,"equal-V fixed delay a=%.2f h, C %.4f vs sweep %.4f: %+.2f%%"%(a_eq/3600,C_eq,C0,100*(C_eq/C0-1)))
# deleted bytes after copy point (observed, lower bound)
dead=(L.status!='censored').to_numpy()&(T>1800)&(T<=28*86400)
print("TB dying in (30m,28d]:",(b*dead).sum()/1e12,"share",(b*dead).sum()/b.sum())
