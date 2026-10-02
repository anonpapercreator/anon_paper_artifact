import numpy as np, json
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600; m=60; peff=6.2686*h
out={'N':int(len(L)),'N_nonempty':int((b>0).sum())}
Sc=S_grid(np.ones(len(L))); Sn=S_grid((b>0).astype(float)); Sb=S_grid(b)
for nm,S,N in [('all',Sc,len(L)),('nonempty',Sn,int((b>0).sum()))]:
    for pp in (6*h,peff):
        C,V=CV(S,30*m,pp); out[f'{nm}_p{pp/h:.2f}']={'Cfrac':C,'pred_files':C*N}
Cb6,Vb6=CV(Sb,30*m,6*h); Cbe,Vbe=CV(Sb,30*m,peff); C1,V1=CV(Sb,30*m,h)
out['bytes']={'V6':Vb6,'Veff':Vbe,'C6':Cb6,'Ceff':Cbe,'ratio_eff_to_1h':Vbe/V1}
json.dump(out,open('pred.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
