import pandas as pd, numpy as np
T0=pd.Timestamp('2026-08-28').timestamp(); T1=pd.Timestamp('2026-09-27').timestamp()
df=pd.read_parquet('audit.parquet',columns=['t','ev','key','size','nlink'])
df=df[(df.t>=T0)&(df.t<T1)]
t=df.t.to_numpy(); key=np.array([int(k,16) for k in df.key],dtype=np.uint64)
evmap={'CREATE':0,'WCLOSE':1,'UNLINK':2,'RENAME':3}
ev=df.ev.astype(str).map(evmap).to_numpy().astype(np.int8)
size=df['size'].to_numpy(); nl=df.nlink.to_numpy()
del df
o=np.lexsort((t,key)); t,key,ev,size,nl=t[o],key[o],ev[o],size[o],nl[o]
newkey=np.r_[True,key[1:]!=key[:-1]]
isc=(ev==0)
# lifecycle id global: increments at each CREATE; reset to -1 at new key before first CREATE
lcid=np.cumsum(isc)            # global counter
keystart_lc=np.maximum.accumulate(np.where(newkey,lcid-isc,0))  # lcid value just before key start... 
# events belong to a window-born lifecycle if a CREATE for this key has occurred at or before it
first_c_seen=lcid>keystart_lc
sel=first_c_seen
lc=lcid[sel]; tt=t[sel]; ee=ev[sel]; ss=size[sel]; nn=nl[sel]; kk=key[sel]
n=lc.max()+1
tc=np.full(n,np.nan); tc[lc[ee==0]]=tt[ee==0]
dm=(ee==2)&(nn==1); td=np.full(n,np.inf); np.minimum.at(td,lc[dm],tt[dm])
tlast=np.full(n,-np.inf); np.maximum.at(tlast,lc,tt)
b=np.zeros(n); m=(ee==1)|(ee==2); np.maximum.at(b,lc[m],ss[m])
lastw=np.full(n,np.nan); w=ee==1; lw=np.full(n,-np.inf); np.maximum.at(lw,lc[w],tt[w])
ck=np.zeros(n,dtype=np.uint64); ck[lc[ee==0]]=kk[ee==0]
ids=np.unique(lc)
L=pd.DataFrame({'key':ck[ids],'tc':tc[ids],'td':td[ids],'tlast':tlast[ids],'lastw':lw[ids],'bytes':b[ids]})
L=L.sort_values(['key','tc'],kind='mergesort')
L['tnext']=L.groupby('key').tc.shift(-1)
L['td']=L.td.replace(np.inf,np.nan); L['lastw']=L.lastw.replace(-np.inf,np.nan)
# a death after the next CREATE cannot belong to this lifecycle (guard)
L['status']=np.where(L.td.notna(),'death',np.where(L.tnext.notna(),'reuse','censored'))
L['T_hi']=np.where(L.status=='death',L.td-L.tc,np.where(L.status=='reuse',L.tnext-L.tc,T1-L.tc))
L['T_lo']=np.where(L.status=='reuse',L.tlast-L.tc,L.T_hi)
L.to_parquet('lifecycles.parquet')
print(len(L), L.status.value_counts().to_dict())
print("bytes by status TB:",(L.groupby('status').bytes.sum()/1e12).round(3).to_dict())
print("zero-byte share:",(L.bytes==0).mean().round(3),"total TB",round(L.bytes.sum()/1e12,2))
