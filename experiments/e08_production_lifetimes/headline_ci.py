import numpy as np, json
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600; m=60
rng=np.random.default_rng(3); R={'count':[],'bytes':[]}
for r in range(400):
    mm=rng.multinomial(P,np.full(P,1/P))[pid].astype(float)
    for wn,w in [('count',mm),('bytes',mm*b)]:
        S=S_grid(w); C6,V6=CV(S,30*m,6*h); C1,V1=CV(S,30*m,h); C15,V15=CV(S,30*m,15*m)
        R[wn].append([V6/V1,100*(C1/C6-1),V6/V15,100*(C15/C6-1)])
out={wn:{k:list(np.percentile(np.array(R[wn])[:,j],[2.5,50,97.5])) for j,k in enumerate(['Vratio_1h','Cinc_1h_pct','Vratio_15m','Cinc_15m_pct'])} for wn in R}
json.dump(out,open('headline_ci.json','w'),indent=1); print(json.dumps(out,indent=1))
