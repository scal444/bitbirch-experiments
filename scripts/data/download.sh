#!/usr/bin/env bash
# Resumable parallel download of a URL list ("URL RELPATH" per line) into DEST/RELPATH.
# usage: download.sh URLS.txt DEST [PARALLEL=8]
# Each file goes to RELPATH.part (curl -C - resumes) and is renamed when curl succeeds; finished files are skipped,
# so rerunning after a failure or session loss only fetches what is missing. Failures are listed in DEST/download_failed.txt.
# URL lists in this directory: urls_zinc20ml_chunks_1_2.txt (ZINC20-ML, no login) and urls_zinc22_2d_h04_h24.txt
# (ZINC22 2D, 1,134 files, about 146 GB). ZINC22 needs the user name and password posted on
# https://files.docking.org/zinc22/ : export DOCKING_AUTH=user:password. Use PARALLEL=3 for ZINC22 (more streams
# got connection resets).
# For a multi-hour download, detach it:  setsid bash download.sh URLS DEST 3 < /dev/null >> LOG 2>&1 &
set -u
urls=$1; dest=$2; par=${3:-8}
mkdir -p "$dest"; : > "$dest/download_failed.txt"
fetch() {
  url=$1; rel=$2; out="$DEST/$rel"
  [ -s "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  if curl -sS -fL --retry 10 --retry-delay 20 --retry-all-errors -C - ${DOCKING_AUTH:+-u "$DOCKING_AUTH"} -o "$out.part" "$url"; then
    mv "$out.part" "$out"
  else
    echo "$url $rel" >> "$DEST/download_failed.txt"
  fi
}
export -f fetch; export DEST=$dest
echo "$(date -u +%FT%TZ) start $(wc -l < "$urls") files -> $dest (parallel $par)"
start=$(date +%s)
xargs -P "$par" -L 1 bash -c 'fetch "$0" "$1"' < "$urls"
echo "$(date -u +%FT%TZ) done in $(( $(date +%s) - start ))s, $(du -sb "$dest" | cut -f1) bytes, failed: $(wc -l < "$dest/download_failed.txt")"
