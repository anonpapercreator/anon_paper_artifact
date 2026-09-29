#!/usr/bin/env python3
"""
extract_audit.py -- reduce IBM Storage Scale file-audit logs to file-lifetime events.

Runs on the DMF metadata server with the standard library only (Python >= 3.6).
Reads the audit JSON records under an audit directory for a date range and a
single file system, and writes one compact, de-identified TSV line per kept event:

    t_epoch  event  key  size  nlink  ext  prefix  node_class  excluded  atime  mtime

  event       CREATE | WCLOSE (close with bytesWritten > 0) | UNLINK |
              DESTROY | RENAME, and with --reads also RCLOSE (close with
              bytesRead > 0 and bytesWritten = 0; this resets the atime that
              DMF policies use). OPEN and other events are skipped before
              JSON parsing.
  key         keyed hash of (fsName, inode[, generation])   -- stable per file
  size        fileSize at the event, bytes
  nlink       linkCount at the event
  ext         lower-case file extension of the (new) path, max 12 chars, or '-'
  prefix      keyed hash of the first --prefix-depth path components below the
              file-system mount point (project/fileset level), or '-'
  node_class  D if the event came from a DMF node (name contains 'dmf'), else U
  excluded    1 if the path matches any --exclude-prefix (paths the migration
              policy excludes, e.g. NO_DMF directories), else 0
  atime,mtime the file's atime and mtime carried in the record (epoch s), or ''

Paths, user and group ids, and node names are never written. The hash key is
read from --key-file (keep it on the server); without it the output cannot be
linked back to paths.

Usage:
  python3 extract_audit.py --audit-dir <.../FSYS_fs01_audit> --fs fs01 \
      --start 2026-07-04 --end 2026-09-27 --key-file ~/.audit_hash_key \
      --out audit_fs01_20260704_20260927.tsv.gz
  python3 extract_audit.py --audit-dir ... --sample 5     # print field names only
"""
import argparse, gzip, hmac, hashlib, io, json, os, re, sys
from datetime import datetime, timedelta, timezone

KEEP = {"CREATE", "CLOSE", "UNLINK", "DESTROY", "RENAME"}
EVENT_RE = re.compile(r'"event"\s*:\s*"([A-Z]+)"')
TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[_ T](\d{2}:\d{2}:\d{2})(\.\d+)?([+-]\d{4})?")


def parse_time(s):
    m = TIME_RE.match(s)
    if not m:
        return None
    d, t, frac, tz = m.groups()
    dt = datetime.strptime(d + " " + t, "%Y-%m-%d %H:%M:%S")
    if tz:
        sign = 1 if tz[0] == "+" else -1
        off = timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])) * sign
        dt = dt.replace(tzinfo=timezone(off))
    else:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() + (float(frac) if frac else 0.0)


def open_any(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), errors="replace")
    return open(path, "r", errors="replace")


def resident(st):
    """True if the file's data is on disk. Audit files in a GPFS compressed
    fileset occupy about 10% of their size in blocks when resident; a file whose
    blocks DMF has released occupies almost none. The 2% threshold separates the
    two; st_blocks is metadata and reading it triggers no recall."""
    return st.st_size == 0 or st.st_blocks * 512 >= 0.02 * st.st_size


def audit_files(root, start, end):
    """Yield audit log files whose YYYY/MM/DD directory lies in [start, end]."""
    day = start
    while day <= end:
        d = os.path.join(root, "%04d" % day.year, "%02d" % day.month, "%02d" % day.day)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.startswith("auditLogFile"):
                    yield os.path.join(d, name)
        day += timedelta(days=1)


FS_RE = re.compile(r'"fsName"\s*:\s*"([^"]*)"')
BW_RE = re.compile(r'"bytesWritten"\s*:\s*"(\d+)"')


def census(a):
    """Per-file event census for --start..--end (resident files only)."""
    import time as _time
    start = datetime.strptime(a.start, "%Y-%m-%d")
    end = datetime.strptime(a.end, "%Y-%m-%d")
    cols = ["OPEN", "CLOSE", "CLOSE_W", "CREATE", "UNLINK", "DESTROY", "RENAME", "RMDIR", "OTHER"]
    print("file\tresident\trecords\tfsNames\t" + "\t".join(cols))
    tot = dict((c, 0) for c in cols)
    nf = 0
    t0 = _time.time(); nbytes = 0
    for path in audit_files(a.audit_dir, start, end):
        name = os.path.basename(path)
        if any(x in name for x in a.skip_name):
            continue
        st = os.stat(path)
        if not resident(st):
            print("%s\toffline" % name)
            continue
        nf += 1
        if a.max_files and nf > a.max_files:
            break
        c = dict((k, 0) for k in cols); n = 0; fss = {}
        for line in open_any(path):
            n += 1
            if a.max_mbps > 0:
                nbytes += len(line)
                if n % 20000 == 0:
                    ahead = nbytes / (a.max_mbps * 1e6) - (_time.time() - t0)
                    if ahead > 0:
                        _time.sleep(ahead)
            m = EVENT_RE.search(line)
            ev = m.group(1) if m else "OTHER"
            if ev not in c:
                ev = "OTHER"
            c[ev] += 1
            if ev == "CLOSE":
                w = BW_RE.search(line)
                if w and int(w.group(1)) > 0:
                    c["CLOSE_W"] += 1
            f = FS_RE.search(line)
            if f:
                fss[f.group(1)] = fss.get(f.group(1), 0) + 1
        for k in cols:
            tot[k] += c[k]
        print("%s\tyes\t%d\t%s\t%s" % (name, n, ",".join(sorted(fss)),
                                         "\t".join(str(c[k]) for k in cols)))
    print("TOTAL\t\t\t\t" + "\t".join(str(tot[k]) for k in cols))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--fs", help="fsName to keep (as it appears in the records)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--prefix-depth", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude-prefix", action="append", default=[],
                    help="path prefix below the mount point that the migration policy "
                         "excludes; matching files are flagged, not dropped (repeatable)")
    ap.add_argument("--skip-name", action="append", default=[],
                    help="skip audit files whose name contains this string, e.g. 'dmf' to "
                         "skip logs written by DMF nodes (repeatable)")
    ap.add_argument("--max-mbps", type=float, default=0.0,
                    help="cap the read rate in MB/s (0 = no cap); applied by sleeping")
    ap.add_argument("--max-files", type=int, default=0,
                    help="stop after this many audit files (0 = all); for trial runs")
    ap.add_argument("--reads", action="store_true",
                    help="also keep closes of read-only opens (RCLOSE); larger output")
    ap.add_argument("--census", action="store_true",
                    help="count events per audit file (by event type, fsName, and closes with "
                         "bytes written) without writing records; for diagnosis")
    ap.add_argument("--sample", type=int, default=0,
                    help="print the field names and event counts of N records and exit")
    a = ap.parse_args()
    if a.census:
        return census(a)
    if not a.sample and not (a.start and a.end and a.out):
        ap.error("--start, --end and --out are required unless --sample is given")

    if a.sample:
        # Look only in the most recent YYYY/MM/DD directory: no walk of the tree.
        d = a.audit_dir
        for _ in range(3):
            subs = sorted(x for x in os.listdir(d) if x.isdigit())
            d = os.path.join(d, subs[-1])
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("auditLogFile"))
        seen = 0
        for p in files[-3:]:
            for line in open_any(p):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                print({k: (type(v).__name__) for k, v in r.items()})
                print({k: r.get(k) for k in ("event", "subEvent", "fsName", "openFlags",
                                             "linkCount", "fileSize", "eventTime", "atime",
                                             "mtime", "bytesRead", "bytesWritten")})
                seen += 1
                if seen >= a.sample:
                    return 0
        return 0

    key = open(a.key_file, "rb").read().strip() if a.key_file else os.urandom(32)
    h = lambda s: hmac.new(key, s.encode("utf-8", "replace"), hashlib.sha256).hexdigest()[:16]
    start = datetime.strptime(a.start, "%Y-%m-%d")
    end = datetime.strptime(a.end, "%Y-%m-%d")
    out = gzip.open(a.out, "wt") if a.out.endswith(".gz") else open(a.out, "w")
    n_in = n_out = n_bad = 0
    counts = {}
    import time as _time
    t0 = _time.time(); nbytes = 0; nfiles = 0
    skipped = []
    for path in audit_files(a.audit_dir, start, end):
        # Never read a file whose data is not resident on disk: under DMF that
        # would trigger a transparent recall from tape. st_blocks is metadata
        # only and does not recall.
        if any(x in os.path.basename(path) for x in a.skip_name):
            continue
        st = os.stat(path)
        if not resident(st):
            skipped.append((path, st.st_size))
            continue
        nfiles += 1
        if a.max_files and nfiles > a.max_files:
            break
        for line in open_any(path):
            n_in += 1
            if a.max_mbps > 0:
                nbytes += len(line)
                if n_in % 20000 == 0:
                    ahead = nbytes / (a.max_mbps * 1e6) - (_time.time() - t0)
                    if ahead > 0:
                        _time.sleep(ahead)
            m = EVENT_RE.search(line)
            if m is None or m.group(1) not in KEEP:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                n_bad += 1
                continue
            ev = r.get("event")
            if ev not in KEEP:
                continue
            if a.fs and r.get("fsName") != a.fs:
                continue
            if ev == "CLOSE":
                def _int(x):
                    try:
                        return int(x or 0)
                    except ValueError:
                        return 0
                bw, br = _int(r.get("bytesWritten")), _int(r.get("bytesRead"))
                if bw > 0:
                    ev = "WCLOSE"
                elif br > 0 and a.reads:
                    ev = "RCLOSE"
                else:
                    continue
            t = parse_time(str(r.get("eventTime", "")))
            if t is None:
                n_bad += 1
                continue
            ino = str(r.get("inode", ""))
            gen = str(r.get("inodeGeneration", r.get("generation", "")))
            k = h("%s|%s|%s" % (r.get("fsName", ""), ino, gen))
            p = r.get("path") or ""
            base = p.rsplit("/", 1)[-1]
            ext = base.rsplit(".", 1)[-1].lower()[:12] if "." in base.strip(".") else "-"
            parts = [x for x in p.split("/") if x]
            # drop the mount-point components (e.g. gpfs/<fs>) before the project level
            try:
                i = parts.index(r.get("fsName", "")) + 1
            except ValueError:
                i = 1
            pre = parts[i:i + a.prefix_depth]
            prefix = h("/".join(pre)) if pre else "-"
            node = "D" if "dmf" in str(r.get("nodeName", "")).lower() else "U"
            rel = "/" + "/".join(parts[i:])
            excl = 1 if any(rel.startswith(x) for x in a.exclude_prefix) else 0
            at = parse_time(str(r.get("atime", "") or ""))
            mt = parse_time(str(r.get("mtime", "") or ""))
            out.write("%.3f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n" % (
                t, ev, k, r.get("fileSize", ""), r.get("linkCount", ""), ext, prefix, node, excl,
                "%.0f" % at if at is not None else "", "%.0f" % mt if mt is not None else ""))
            n_out += 1
            counts[ev] = counts.get(ev, 0) + 1
    out.close()
    if skipped:
        sys.stderr.write("SKIPPED %d non-resident files (%.1f GB); listed in %s.skipped\n"
                         % (len(skipped), sum(x[1] for x in skipped) / 1e9, a.out))
        with open(a.out + ".skipped", "w") as fh:
            for p, sz in skipped:
                fh.write("%s\t%d\n" % (os.path.basename(p), sz))
    sys.stderr.write("files %d, records read %d, kept %d, unparseable %d, by event %s, "
                     "elapsed %.0f s\n" % (min(nfiles, a.max_files or nfiles), n_in, n_out,
                                          n_bad, counts, _time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
