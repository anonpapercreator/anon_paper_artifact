import numpy as np, json
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600; m=60; day=86400
S=S_grid(b)
A=np.r_[np.arange(0,2*h,60),np.arange(2*h,3*day,300),np.arange(3*day,20*day,3600)]
res={}
for p in (6*h,h):
    CVs=np.array([CV(S,a,p) for a in A])
    prev=None; rows=[]
    for th in np.round(np.arange(0.02,0.6,0.01),3):
        J=CVs[:,0]+th/24*CVs[:,1]; aopt=A[np.argmin(J)]/h
        rows.append((float(th),float(aopt),float(CVs[np.argmin(J),0]),float(CVs[np.argmin(J),1])))
    res[p//h]=rows
    for r in rows: print(p//h, r)
json.dump(res,open('rule2.json','w'))
