import numpy as np, json
exec(open('paper_numbers.py').read().split("h=3600; m=60")[0])
h=3600; m=60; day=86400
rho_TBd = b.sum()/1e12/30.0
out={'rho_TB_per_day':rho_TBd,'rho_MBps':b.sum()/30/86400/1e6}
for wn,w in [('bytes',b),('count',np.ones(len(L)))]:
    S=S_grid(w)
    Gc=np.concatenate([[0],np.cumsum((S[1:]+S[:-1])/2*np.diff(grid))])
    Sa=lambda t: np.interp(t,grid,S); Ga=lambda t: np.interp(t,grid,Gc)
    hp=lambda a,p:(Sa(a)-Sa(a+p))/(Ga(a+p)-Ga(a))
    r={}
    for p in (6*h,h):
        r[f'hp_30m_p{p//h}h_per_day']=hp(30*m,p)*day
        # J/(rho c_w) = C/rho + theta*V/rho ; minimize over a grid for theta values
        A=np.r_[np.arange(0,2*h,60),np.arange(2*h,3*day,300),np.arange(3*day,20*day,3600)]
        CVs=np.array([CV(S,a,p) for a in A])  # C frac, V hours
        for th in (0.1,0.3,1,3,10,30):   # per day
            J=CVs[:,0]+th/24*CVs[:,1]
            r[f'p{p//h}h_theta{th}_per_day_opt_a_h']=A[np.argmin(J)]/h
        # range of theta for which a=30min is the global minimizer (scan)
        ths=np.logspace(-3,3,2000); ok=[]
        for th in ths:
            J=CVs[:,0]+th/24*CVs[:,1]
            if abs(A[np.argmin(J)]-30*m)<=300: ok.append(th)
        r[f'p{p//h}h_theta_range_a30min_optimal_per_day']=[min(ok),max(ok)] if ok else None
    out[wn]=r
    if wn=='bytes':
        for nm,(a,p) in {'prod':(30*m,6*h),'hourly':(30*m,h),'q15':(30*m,15*m)}.items():
            C,V=CV(S,a,p); out[wn][nm]={'C_frac':C,'V_h':V,'C_TBd':C*rho_TBd,'V_TB':V/24*rho_TBd}
json.dump(out,open('rule.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
