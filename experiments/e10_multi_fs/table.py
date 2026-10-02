"""Per-file-system summary table from <fs>_results.json (pipeline.py) and plog_window.json."""
import json, glob, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else '.'
pl = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plog_window.json')))
rows = []
for f in sorted(glob.glob(os.path.join(d, '*_results.json'))):
    r = json.load(open(f)); fs = r['fs']; k = fs
    B, C = r['bytes'], r['count']; ci = lambda x, m: B['ci95'][m] if B.get('ci95') else [None, None]
    rows.append(dict(fs=fs, lifecycles=r['lifecycles'], TB=r['TB_written'], trees=r['directory_trees'],
        deleted_frac_count=1 - r['status'].get('censored', 0) / r['lifecycles'],
        Vr=B['Vratio_1h'], Vr_ci=ci(B, 'Vratio_1h'), Ci=B['Cinc_1h_pct'], Ci_ci=ci(B, 'Cinc_1h_pct'),
        W=B['waste_28d'], W_ci=ci(B, 'waste_28d'), W_early=B['waste_28d_earliest'], S28=B['S_28d'],
        Vr_n=C['Vratio_1h'], Ci_n=C['Cinc_1h_pct'], W_n=C['waste_28d'], top5_del=r['top5_share_deleted'],
        runs=pl[k]['runs'], submitted=pl[k]['files_submitted'], peff=r['peff_h']))
rows.sort(key=lambda x: -x['TB'])
f2 = lambda x: '-' if x is None else f'{x:.2f}'
print(f"{'fs':10s} {'lifecycles':>10s} {'TB':>6s} {'trees':>6s} {'del%':>5s} | {'Vr_1h':>5s} {'CI':>11s} {'C+%':>5s} {'CI':>11s} {'waste%':>6s} {'CI':>11s} {'early%':>6s} {'S28':>5s} | count: {'Vr':>5s} {'C+%':>5s} {'W%':>5s} | top5del submitted")
for x in rows:
    print(f"{x['fs']:10s} {x['lifecycles']:10d} {x['TB']:6.1f} {x['trees']:6d} {100*x['deleted_frac_count']:5.1f} | {x['Vr']:5.2f} {f2(x['Vr_ci'][0])+'-'+f2(x['Vr_ci'][1]):>11s} {x['Ci']:5.2f} "
          f"{f2(x['Ci_ci'][0])+'-'+f2(x['Ci_ci'][1]):>11s} {100*x['W']:6.1f} {f2(100*x['W_ci'][0] if x['W_ci'][0] is not None else None)+'-'+f2(100*x['W_ci'][1] if x['W_ci'][1] is not None else None):>11s} {100*x['W_early']:6.1f} {x['S28']:5.3f} | "
          f"{x['Vr_n']:5.2f} {x['Ci_n']:5.2f} {100*x['W_n']:5.1f} | {f2(x['top5_del'])} {x['submitted']}")
json.dump(rows, open(os.path.join(d, 'summary_table.json'), 'w'), indent=1, default=float)
