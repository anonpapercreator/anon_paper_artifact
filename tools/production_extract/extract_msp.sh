#!/bin/bash
# extract_msp.sh -- reduce DMF MSP logs (ls1) to the records needed for the
# archived-file lifetime study. Run in the MSP spool directory, e.g.
#   cd <spool>/ls1 && bash extract_msp.sh 20260704 20260927 ~/msp_extract
# Keeps: chunk adds (tape copy written, incl. merge copies), zone update/complete
# markers (to pair merge copies), hard-delete chunk removals, merge removals, and
# the daemon's cumulative Put/Delete/Merge counters. Drops host and pid fields.
set -euo pipefail
start=$1; end=$2; out=$3
mkdir -p "$out"
pat='update_dbases: (Add|Update vsn|Complete vsn)|delete_chunks:|delete_merged_chunk:|t1a\.stats: +(Put_File|Delete_File|Merge) |t1a\.stats: data put'
for f in msplog.*; do
  d=${f#msplog.}; d=${d%.gz}
  [[ "$d" =~ ^[0-9]{8}$ ]] || continue
  [[ "$d" < "$start" || "$d" > "$end" ]] && continue
  if [[ $f == *.gz ]]; then cat_cmd="zcat"; else cat_cmd="cat"; fi
  $cat_cmd "$f" | grep -E "$pat" \
    | sed -E 's/^([0-9-]+ [0-9:.]+)-[A-Z] [^ ]+ +[^ ]+ /\1 /' \
    | gzip -6 > "$out/msp_$d.txt.gz"
  echo "$d $(zcat "$out/msp_$d.txt.gz" | wc -l) lines"
done
# The copy is not modified; sizes for a sanity check on this side:
du -sh "$out"
