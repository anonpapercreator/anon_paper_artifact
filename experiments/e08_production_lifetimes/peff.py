import numpy as np
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600; m=60
for wn,w in [('bytes',b),('count',np.ones(len(L)))]:
    S=S_grid(w)
    for d in (0, 0.25*h):
        C6,V6=CV(S,30*m,6*h+d); C1,V1=CV(S,30*m,h+d); C15,V15=CV(S,30*m,15*m+d)
        print(f"{wn} run={d/60:.0f}min: prod C {C6:.4f} V {V6:.2f}h | 1h C {C1:.4f} V {V1:.2f}h ratio {V6/V1:.2f} Cinc {100*(C1/C6-1):.2f}% | 15m V {V15:.2f} ratio {V6/V15:.2f} Cinc {100*(C15/C6-1):.2f}%")
