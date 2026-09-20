# Availability model — clean pregame two-stage backtest

The live board's anytime-TD chance is a **true pregame probability**:
`chance = P(player takes the field) × P(scores a TD | plays)`.

- **Stage 2 — conversion:** P(TD | ≥1 touch), the opportunity+NGS model (see `model_report.md`).
- **Stage 1 — availability:** P(≥1 offensive touch | eligible pregame), from final injury-report status, practice participation (Full/Limited/DNP), recent snap share, games actually played in the last 3 weeks, role, and workload.

Evaluated on a **dense player-week grid** (no touch-conditioning): every skill player who was eligible pregame — team playing, not ruled Out/Doubtful/IR, projected ≥3 touches/g — with players who were eligible but did not play counted as 0s. Models fit on seasons **≤2021**; tested **2022–2025**.

Universe: **19,841** player-weeks, **20.4%** scored, **15.5%** eligible no-show rate.

## Stage-1 availability model quality (held-out 2022–2025)

| Metric | Value |
|---|---|
| AUC (ranks who plays) | 0.868 |
| Brier | 0.0950 |
| Calibration err (ECE) | 0.011 |
| Mean predicted play rate | 83.6% |
| Actual play rate | 84.5% |

## Full-pipeline pregame P(TD) — does the availability multiplier help?

| Approach | Brier ↓ | Log loss ↓ | AUC ↑ | Calib. err ↓ |
|---|---|---|---|---|
| Current — assume every eligible player plays | 0.1420 | 0.4481 | 0.745 | 0.0340 |
| **Two-stage — P(plays) × P(TD·plays)** (shipped) | **0.1410** | **0.4417** | **0.750** | **0.0220** |

## Where it matters most — pregame-uncertain players

Players flagged Questionable or Limited/DNP in practice (1,777 weeks, 22.4% no-show):

| Approach | Brier ↓ | Log loss ↓ | AUC ↑ |
|---|---|---|---|
| Current | 0.1426 | 0.4510 | 0.738 |
| Two-stage | 0.1418 | 0.4471 | 0.729 |

---
*The two-stage lowers the displayed chance for players at real no-show risk, improving probability accuracy and calibration across the full pregame universe — the board no longer implicitly assumes every listed player suits up.*
