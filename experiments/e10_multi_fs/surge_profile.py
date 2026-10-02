#!/usr/bin/env python3
"""Profile a reduced audit extract without loading it into memory.

Reads every <fs>_YYYYMMDD_<tag>.tsv.gz in a directory line by line and reports:
  - events per UTC day and event type;
  - the 20 directory trees (keyed prefix hashes) with the most CREATE events,
    with their creates, unlinks, and bytes at close;
  - for those trees, creates per UTC day.
Output is de-identified (prefix hashes only) and small. Standard library only, so it
runs on the NSD node next to the data.

usage: python3 surge_profile.py DIR [--out surge_profile.json]
"""
import argparse, collections, glob, gzip, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--out", default="surge_profile.json")
a = ap.parse_args()

by_day = collections.Counter()                 # (day, ev) -> n
tree = collections.defaultdict(lambda: [0, 0, 0])  # prefix -> [creates, unlinks, bytes at WCLOSE]
tree_day = collections.Counter()               # (prefix, day) -> creates
n = 0; t0 = time.time()
files = sorted(glob.glob(os.path.join(a.dir, "*.tsv.gz")))
for f in files:
    with gzip.open(f, "rt") as fh:
        for line in fh:
            x = line.rstrip("\n").split("\t")
            if len(x) < 7:
                continue
            n += 1
            try:
                day = time.strftime("%Y-%m-%d", time.gmtime(float(x[0])))
            except ValueError:
                continue
            ev, pre = x[1], x[6]
            by_day[(day, ev)] += 1
            s = tree[pre]
            if ev == "CREATE":
                s[0] += 1
                tree_day[(pre, day)] += 1
            elif ev == "UNLINK":
                s[1] += 1
            elif ev == "WCLOSE":
                try:
                    s[2] += int(x[3] or 0)
                except ValueError:
                    pass

top = sorted(tree.items(), key=lambda kv: -kv[1][0])[:20]
tot_c = sum(v[0] for v in tree.values()); tot_u = sum(v[1] for v in tree.values())
out = {
    "files": len(files), "records": n, "elapsed_s": round(time.time() - t0, 1),
    "trees": len(tree), "creates": tot_c, "unlinks": tot_u,
    "by_day": {f"{d} {e}": c for (d, e), c in sorted(by_day.items())},
    "top_trees_by_creates": [
        {"prefix": p, "creates": v[0], "unlinks": v[1], "bytes_at_close": v[2],
         "share_of_creates": v[0] / tot_c if tot_c else None,
         "creates_by_day": {d: c for (pp, d), c in sorted(tree_day.items()) if pp == p}}
        for p, v in top],
}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("files", "records", "trees", "creates", "unlinks", "elapsed_s")}))
for r in out["top_trees_by_creates"][:5]:
    print(r["prefix"], r["creates"], r["unlinks"], round(r["share_of_creates"], 3))
