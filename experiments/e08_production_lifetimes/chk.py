import numpy as np
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600; m=60
for nm,S in [('latest',S_grid(b)),('earliest',S_grid(b,PL))]:
    i=np.searchsorted(grid,28*24*h)
    print(nm, 'C/rho',CV(S,30*m,6*h)[0], 'S(28d)',S[i], 'S(7d)',S[np.searchsorted(grid,7*24*h)],'S(6.5h)',S[np.searchsorted(grid,6.5*h)])
