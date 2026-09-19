#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
mkdir -p data out public
CUR="${SEASON:-2026}"
BASE="https://github.com/nflverse/nflverse-data/releases/download"
echo "downloading nflverse data (2021-$CUR)..."
for yr in 2021 2022 2023 2024 2025 "$CUR"; do
  curl -sL -m 240 -o "data/pbp_${yr}.parquet"      "$BASE/pbp/play_by_play_${yr}.parquet" || true
  curl -sL -m 120 -o "data/roster_${yr}.parquet"   "$BASE/rosters/roster_${yr}.parquet" || true
  curl -sL -m 120 -o "data/snaps_${yr}.parquet"    "$BASE/snap_counts/snap_counts_${yr}.parquet" || true
done
curl -sL -m 120 -o "data/injuries_${CUR}.parquet"  "$BASE/injuries/injuries_${CUR}.parquet" || true
curl -sL -m 60  -o data/games.csv "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
curl -sL -m 120 -o data/players.parquet "$BASE/players/players.parquet"
# auto-detect next unplayed week and point run_proj at it
NEXTWK=$(python3 - <<PY
import duckdb
r=duckdb.connect().sql("select min(week) from 'data/games.csv' where season=$CUR and result is null").fetchone()[0]
print(r if r else 1)
PY
)
echo "projecting $CUR week $NEXTWK"
python3 - <<PY
import re
s=open('run_proj.py').read()
s=re.sub(r'SEASON,UPCOMING=\d+,\d+', f"SEASON,UPCOMING=$CUR,$NEXTWK", s)
open('run_proj.py','w').write(s)
PY
python3 run_proj.py
python3 render_board.py
echo "build complete -> public/index.html"
