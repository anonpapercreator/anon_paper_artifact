# Reproduction check: original extract of file system A vs re-extract (same script, possibly different key).
# Tests: (1) original is an in-order subsequence of the new extract per day, on all non-hash fields;
# (2) the old->new hash map is a bijection over matched records (same grouping into files and prefixes);
# (3) where the unmatched (extra) new records fall in time.
import gzip, glob, os, sys, json
from datetime import datetime, timezone
import argparse; ap=argparse.ArgumentParser(); ap.add_argument('--orig',required=True); ap.add_argument('--new',required=True); ap.add_argument('--tag',required=True); ap.add_argument('--out',default='repro_a.json'); A=ap.parse_args()
O=A.orig; N=A.new
def rows(p):
    with gzip.open(p,'rt') as f:
        for l in f:
            x=l.rstrip('\n').split('\t')
            yield x
kmap={}; kinv={}; pmap={}; pinv={}; bad_k=bad_p=0
extra_t=[]; tot=dict(orig=0,new=0,matched=0)
def link(a,b,m,inv):
    if m.setdefault(a,b)!=b or inv.setdefault(b,a)!=a: return 1
    return 0
perday={}
for po in sorted(glob.glob(O+'/a_2026*.tsv.gz')):
    d=os.path.basename(po)[2:10]
    pn=glob.glob(N+'/a_%s_%s.tsv.gz'%(d,A.tag))[0]
    it=rows(po); cur=next(it,None); m=0; e=0; no=nn=0
    # count orig
    for x in rows(pn):
        nn+=1
        if cur is not None and x[0:2]+x[3:6]+x[7:]==cur[0:2]+cur[3:6]+cur[7:]:
            bad_k+=link(cur[2],x[2],kmap,kinv); bad_p+=link(cur[6],x[6],pmap,pinv)
            m+=1; cur=next(it,None)
        else:
            e+=1; extra_t.append(float(x[0]))
    left=0
    while cur is not None: left+=1; cur=next(it,None)
    no=m+left
    perday[d]=dict(orig=no,new=nn,matched=m,orig_unmatched=left,extra=e)
    tot['orig']+=no; tot['new']+=nn; tot['matched']+=m
    print(d,perday[d],flush=True)
import numpy as np
et=np.array(extra_t)
f=lambda s: datetime.fromtimestamp(s,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
res=dict(tot=tot,bad_key_links=bad_k,bad_prefix_links=bad_p,n_keys=len(kmap),n_prefixes=len(pmap),
         extra_n=int(et.size),extra_tmin=f(et.min()) if et.size else None,extra_tmax=f(et.max()) if et.size else None,
         extra_before_0927_0000Z=int((et<datetime(2026,9,27,tzinfo=timezone.utc).timestamp()).sum()))
print(json.dumps(res,indent=1))
json.dump(dict(res,perday=perday),open(A.out,'w'),indent=1)
