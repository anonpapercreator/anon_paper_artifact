import numpy as np, pandas as pd
exec(open('boot2.py').read().split('rng=np.random')[0])
for wn,w in [('count',np.ones(len(L))),('bytes',b)]:
    S=S_grid(w)
    C0,V0=CV(S,1800,21600)
    # deterministic delay a' with same V
    G=np.r_[0,np.cumsum((S[1:]+S[:-1])/2*dx)]/3600
    a_eq=np.interp(V0,G,grid); C_eq=np.interp(a_eq,grid,S)
    # best sweep with p=6h and same V: vary a
    print(f"{wn}: prod C={C0:.4f} V={V0:.3f} h | deterministic delay with same V: a={a_eq/3600:.2f} h, C={C_eq:.4f} | change in copied bytes {100*(C_eq-C0)/C0:+.2f}%")
    # production at p=6h vs p=1h, 15 min with a=30m
    for p in (3600,900,0): print("   a=30m p=%5ds  C=%.4f V=%.3f h"%(p,*CV(S,1800,p)))
