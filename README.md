# EndZone Board — NFL anytime-TD model

Calibrated model of each player's chance to score a touchdown in the upcoming week.
The displayed chance is a **two-stage pregame probability**: `P(plays) × P(TD | plays)`.

- **Conversion** — P(TD | plays): red-zone & goal-line carry share, red-zone target share,
  snap share, carry/target volume, recent-form usage & trend, Vegas implied team total, and
  **Next Gen Stats** tracking form (receiver separation & YAC-over-expected; rusher efficiency,
  yards-over-expected & box counts). Raw gradient-boosted classifier (no calibration wrapper).
- **Availability** — P(player takes the field): injury-report status, practice participation
  (Full/Limited/DNP), recent snap share & games played. Out/Doubtful are excluded outright and
  their red-zone work reallocated to available teammates.

Backtested walk-forward across 2018–2025 (~41.8k held-out player-games): conversion Brier 0.152,
log loss 0.475, AUC 0.703, calibration error 0.007. Availability model (dense pregame grid,
2022–2025): AUC 0.868; the two-stage cuts full-pipeline calibration error from 0.034 to 0.022.
See `out/model_report.md` and `out/eligibility_report.md`.

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
# endzone-board
