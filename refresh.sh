#!/usr/bin/env bash
# One-command weekly refresh: re-pull current-season data + regenerate the board.
# Usage:  bash ~/td-model/refresh.sh
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

CUR=${SEASON:-2026}          # current season
PREV=$((CUR-1))
BASE="https://github.com/nflverse/nflverse-data/releases/download"

echo "[1/3] downloading latest data for $PREV-$CUR ..."
for yr in $PREV $CUR; do
  curl -sL -m 180 -o "data/pbp_${yr}.parquet"     "$BASE/pbp/play_by_play_${yr}.parquet"      || true
  curl -sL -m 120 -o "data/roster_${yr}.parquet"  "$BASE/rosters/roster_${yr}.parquet"        || true
  curl -sL -m 120 -o "data/snaps_${yr}.parquet"   "$BASE/snap_counts/snap_counts_${yr}.parquet" || true
  curl -sL -m 120 -o "data/injuries_${yr}.parquet" "$BASE/injuries/injuries_${yr}.parquet"     || true
done
curl -sL -m 60 -o data/games.csv "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

echo "[2/3] projecting the next unplayed week ..."
# auto-detect the next unplayed week of the current season and export the slate
NEXTWK=$(python - <<PY
import duckdb
c=duckdb.connect()
r=c.sql("select min(week) from 'data/games.csv' where season=$CUR and result is null").fetchone()[0]
print(r if r else '')
PY
)
if [ -z "$NEXTWK" ]; then echo "No unplayed games left in $CUR season."; exit 0; fi
echo "    next unplayed week = $NEXTWK"
UPCOMING=$NEXTWK python - <<'PY'
import os,re
# patch UPCOMING/SEASON in run_proj at runtime via env is simplest: rewrite the two constants
src=open('run_proj.py').read()
src=re.sub(r'SEASON,UPCOMING=\d+,\d+', f"SEASON,UPCOMING={os.environ.get('SEASON','2026')},{os.environ['UPCOMING']}", src)
open('run_proj.py','w').write(src)
PY
SEASON=$CUR python run_proj.py

echo "[3/3] board regenerated -> out/td-board.html"
echo "Done. To publish the updated board, open this session and ask to republish out/td-board.html,"
echo "or open the file directly:  open out/td-board.html"
