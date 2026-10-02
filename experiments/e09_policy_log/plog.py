import re, numpy as np, pandas as pd, json
rows=[]
for l in open('policylog.txt'):
    m=re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*?[Pp]olicy \'?([A-Za-z0-9]+)_migrate_policy\'? submitted (\d+|no) records',l)
    if not m:
        m2=re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*"policy_name":"([A-Za-z0-9]+)_migrate_policy".*No records submitted',l)
        if m2: rows.append((m2.group(1),m2.group(2).upper(),0))
        continue
    n=0 if m.group(3)=='no' else int(m.group(3)); rows.append((m.group(1),m.group(2).upper(),n))
d=pd.DataFrame(rows,columns=['t','fs','n']); d['t']=pd.to_datetime(d['t']); d=d.drop_duplicates().sort_values(['fs','t'])
out={'span':[str(d.t.min()),str(d.t.max())],'runs_per_fs':d.groupby('fs').size().to_dict()}
# intervals between submissions per FS
iv=[]
for fs,g in d.groupby('fs'):
    x=g.t.diff().dt.total_seconds().dropna()/3600; iv.append(pd.DataFrame({'fs':fs,'P':x.values}))
iv=pd.concat(iv)
def stats(P):
    P=np.asarray(P); return {'n':len(P),'mean_h':P.mean(),'median_h':np.median(P),'p05':np.percentile(P,5),'p95':np.percentile(P,95),'max_h':P.max(),
      'frac_gt_7h':float((P>7).mean()),'mean_wait_h':float((P**2).mean()/(2*P.mean())),'half_mean_h':P.mean()/2}
out['all_fs']=stats(iv.P)
out['per_fs']={fs:stats(g.P) for fs,g in iv.groupby('fs')}
# file system A in the audit window (log times are UTC)
w=d[(d.fs=='A')&(d.t>='2026-08-28')&(d.t<'2026-09-27')]
out['A_window']={'runs':len(w),'files_submitted':int(w.n.sum()),'per_run_median':float(w.n.median()),'per_run_mean':float(w.n.mean())}
Pw=w.t.diff().dt.total_seconds().dropna()/3600; out['A_window_intervals']=stats(Pw)
# waves: cluster all FS submissions with gaps < 90 min
t=np.sort(d.t.values); gaps=np.diff(t).astype('timedelta64[s]').astype(float)/60
starts=np.r_[0,np.where(gaps>90)[0]+1]; ends=np.r_[starts[1:]-1,len(t)-1]
span=[(t[e]-t[s]).astype('timedelta64[s]').astype(float)/60 for s,e in zip(starts,ends)]
cnt=[e-s+1 for s,e in zip(starts,ends)]
ws=pd.DataFrame({'span_min':span,'nsub':cnt}); full=ws[ws.nsub>=12]
out['waves']={'n':len(ws),'n_with_12plus':len(full),'span_min_median':float(full.span_min.median()),'span_min_p90':float(full.span_min.quantile(.9)),'span_min_max':float(full.span_min.max())}
f16=ws[ws.nsub>=16]
out['waves_16plus']={'n':len(f16),'span_min_median':float(f16.span_min.median()),'span_min_p90':float(f16.span_min.quantile(.9))}
json.dump(out,open('plog.json','w'),indent=1,default=float)
print(json.dumps({k:v for k,v in out.items() if k!='per_fs'},indent=1,default=float))
for fs,s in out['per_fs'].items(): print(fs, round(s['mean_h'],2), round(s['median_h'],2), round(s['max_h'],1), round(s['mean_wait_h'],2), round(s['frac_gt_7h'],3))
