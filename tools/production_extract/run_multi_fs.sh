#!/bin/bash
# run_multi_fs.sh -- reduce 30 days of Storage Scale file-audit logs for several
# file systems, with the same method and parameters as the extract of file system A.
#
# Run on an NSD server as root, from the directory that holds
# extract_audit.py. Nothing here writes to /gpfs; output goes to $OUT.
#
#   bash run_multi_fs.sh plan              # stat only: audit dirs, day files, sizes, residency
#   bash run_multi_fs.sh probe             # read 1 record per file system: fsName as recorded
#   bash run_multi_fs.sh run [FS ...]      # extract (all listed file systems, or those given)
#   bash run_multi_fs.sh summary           # per-file-system totals from the logs
#   bash run_multi_fs.sh pack              # one tar per file system + manifest, for transfer
#
# The extraction reads one day per process, PAR processes at a time, each capped at
# MBPS/PAR MB/s, so the total read rate stays at or below MBPS. A day whose output
# and log already exist is skipped, so the run can be stopped and restarted.
# Audit files whose data is not resident on disk are never read (no DMF recall);
# they are listed per day in <out>.skipped, as for file system A.
set -u

FSLIST=${FSLIST:?set FSLIST to the file-system names, e.g. FSLIST="fs01 fs02"}
START=${START:-2026-08-28}          # same observation period as file system A
END=${END:-2026-09-26}
KEY=${KEY:-$HOME/.audit_hash_key}   # same key for all file systems; never leaves this host
OUT=${OUT:-./multi_fs}
MBPS=${MBPS:-500}                   # total read cap, MB/s
PAR=${PAR:-4}                       # concurrent day extractions
EXTRACT=${EXTRACT:-./extract_audit.py}
PY=${PY:-python3}

die() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$EXTRACT" ] || die "extract_audit.py not found at $EXTRACT (set EXTRACT=...)"
days() { local d=$START; while [ "$(date -d "$d" +%s)" -le "$(date -d "$END" +%s)" ]; do echo "$d"; d=$(date -d "$d +1 day" +%F); done; }
ROOT=${ROOT:-/gpfs}
AUDIT_SUBDIR=${AUDIT_SUBDIR:?set AUDIT_SUBDIR to the audit directory under each file system}
auditdirs() { ls -d "$ROOT/$1"/"$AUDIT_SUBDIR"/*_FSYS_*_audit 2>/dev/null; }

plan() {
  printf "%-10s %-4s %-6s %-9s %-9s %-8s %s\n" fs dirs days files GB_total offline missing_days
  for fs in $FSLIST; do
    dirs=$(auditdirs "$fs"); nd=$(echo "$dirs" | grep -c . )
    if [ "$nd" -eq 0 ]; then printf "%-10s %-4s  no audit directory under %s/%s/%s\n" "$fs" 0 "$ROOT" "$fs" "$AUDIT_SUBDIR"; continue; fi
    nf=0; bytes=0; off=0; miss=""; nday=0
    for d in $(days); do
      y=${d:0:4}; m=${d:5:2}; dd=${d:8:2}; found=0
      for ad in $dirs; do
        dir="$ad/$y/$m/$dd"; [ -d "$dir" ] || continue
        while read -r sz blk; do
          found=1; nf=$((nf+1)); bytes=$((bytes+sz))
          # resident test as in extract_audit.py: blocks >= 2% of size
          [ "$sz" -gt 0 ] && [ $((blk*512*50)) -lt "$sz" ] && off=$((off+1))
        done < <(find "$dir" -maxdepth 1 -name 'auditLogFile*' -printf '%s %b\n' 2>/dev/null)
      done
      [ $found -eq 1 ] && nday=$((nday+1)) || miss="$miss ${d:5}"
    done
    printf "%-10s %-4s %-6s %-9s %-9s %-8s %s\n" "$fs" "$nd" "$nday" "$nf" "$(awk "BEGIN{printf \"%.1f\", $bytes/1e9}")" "$off" "${miss:- -}"
  done
}

probe() {
  for fs in $FSLIST; do
    for ad in $(auditdirs "$fs"); do
      n=$(timeout 120 $PY "$EXTRACT" --audit-dir "$ad" --sample 1 2>/dev/null | grep -o "'fsName': '[^']*'" | grep -v "'str'" | head -1)
      echo "$fs  $(basename "$ad")  ${n:-no record read}"
    done
  done
}

one_day() {  # fs auditdir fsname day outdir cap
  local fs=$1 ad=$2 fsn=$3 d=$4 od=$5 cap=$6 tag
  tag=$(basename "$ad" | md5sum | cut -c1-6)
  local out="$od/${fs,,}_${d//-/}_$tag.tsv.gz" log="$od/${fs,,}_${d//-/}_$tag.log"
  if [ -s "$out" ] && [ -s "$log" ] && grep -q "^files " "$log"; then return 0; fi
  $PY "$EXTRACT" --audit-dir "$ad" --fs "$fsn" --start "$d" --end "$d" --key-file "$KEY" \
      --prefix-depth 2 --skip-name dmf --max-mbps "$cap" --out "$out" > "$log" 2>&1 \
    || echo "FAILED $fs $d (see $log)" >&2
}
export -f one_day
export PY EXTRACT KEY

run() {
  [ -s "$KEY" ] || die "hash key $KEY missing: use the key from the first extract"
  local sel=${*:-$FSLIST} cap
  cap=$(awk "BEGIN{printf \"%.0f\", $MBPS/$PAR}")
  for fs in $sel; do
    od="$OUT/$fs"
    [ -n "$(auditdirs "$fs")" ] || { echo "SKIP $fs: no audit directory" >&2; continue; }
    mkdir -p "$od"
    for ad in $(auditdirs "$fs"); do
      fsn=$(timeout 120 $PY "$EXTRACT" --audit-dir "$ad" --sample 1 2>/dev/null | grep -o "'fsName': '[^']*'" | grep -v "'str'" | head -1 | cut -d"'" -f4)
      [ -n "$fsn" ] || { echo "SKIP $fs $(basename "$ad"): no readable record to determine fsName" >&2; continue; }
      echo "$(date '+%F %T') $fs dir=$(basename "$ad") fsName=$fsn cap=${cap}MB/s x $PAR"
      days | xargs -P "$PAR" -I{} bash -c "one_day '$fs' '$ad' '$fsn' {} '$od' '$cap'"
    done
    grep -h "^files " "$od"/*.log 2>/dev/null | awk -v fs="$fs" '{gsub(",","");r+=$5;k+=$7} END{printf "%s done: %d records read, %d kept\n",fs,r,k}'
  done
}

summary() {
  printf "%-10s %-6s %-12s %-10s %-9s %s\n" fs days records_read kept offline failed
  for fs in $FSLIST; do
    od="$OUT/$fs"; [ -d "$od" ] || continue
    nd=$(ls "$od"/*.log 2>/dev/null | wc -l)
    rk=$(grep -h "^files " "$od"/*.log 2>/dev/null | awk '{gsub(",","");r+=$5;k+=$7} END{print r+0, k+0}')
    off=$(cat "$od"/*.skipped 2>/dev/null | wc -l)
    bad=$(grep -L "^files " "$od"/*.log 2>/dev/null | wc -l)
    printf "%-10s %-6s %-12s %-10s %-9s %s\n" "$fs" "$nd" ${rk% *} ${rk#* } "$off" "$bad"
  done
}

pack() {
  cd "$OUT" || die "no $OUT"
  for fs in $FSLIST; do
    ls "$fs"/*.tsv.gz >/dev/null 2>&1 || continue
    ( cd "$fs" && sha256sum *.tsv.gz > "${fs,,}_manifest.sha256" )
    # .skipped files contain audit-file paths; they stay on this host
    tar -cf "${fs,,}_audit_${START//-/}_${END//-/}.tar" -C "$fs" --exclude='*.skipped' .
    echo "$(ls -la "${fs,,}_audit_${START//-/}_${END//-/}.tar")"
  done
}

case "${1:-}" in
  plan) plan ;; probe) probe ;; run) shift; run "$@" ;; summary) summary ;; pack) pack ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
