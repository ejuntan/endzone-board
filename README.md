# EndZone Board — NFL anytime-TD model

Calibrated model of each player's chance to score a touchdown in the upcoming week,
built from red-zone opportunity (incl. red-zone & goal-line carry share), snap share,
and Vegas implied totals. Injury-aware (Out/Doubtful excluded, red-zone work reallocated).
Backtested walk-forward across 2022–2025: Brier 0.152, log loss 0.475, AUC 0.688.

## Deploy on Render (static site, regenerated each deploy)
1. Push this repo to GitHub.
2. Render → New → Blueprint → pick the repo (uses `render.yaml`), or New → Static Site with:
   - Build command: `./build_site.sh`
   - Publish directory: `public`
3. Deploy. The build pulls fresh nflverse data, projects the next unplayed week, and
   renders `public/index.html`.

### Weekly auto-refresh
Render → the service → Settings → Deploy Hook (copy URL), then either:
- add a Render **Cron Job** that `curl`s the hook weekly, or
- a GitHub Actions scheduled workflow that `curl`s the hook.

## Local
- `bash refresh.sh` — pull data + regenerate `out/td-board.html` for the next unplayed week.
- Pipeline: `pipeline.py` (features), `run_backtest.py` (validation), `run_proj.py` (projection),
  `render_board.py` (template → HTML). Data (parquet) is downloaded, not committed.
