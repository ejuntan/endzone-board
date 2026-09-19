# EndZone Board — model performance report

Anytime-touchdown model, walk-forward out-of-sample backtest.

- Scoring rates fit only on seasons **before** each test year; the classifier is trained only on prior seasons; every prediction uses pre-kickoff info only.
- Test seasons: **2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025**. Evaluation universe: active, involved skill players (RB/WR/TE/QB with ≥1 touch).
- Pooled held-out sample: **41,763 player-games**, base rate **21.6%**.

## Pooled model comparison (2018–2025)

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | Calib. err (ECE) ↓ |
|---|---|---|---|---|
| Naive: count past TDs | 0.1658 | 0.6129 | 0.651 | 0.060 |
| Opportunity model | 0.1568 | 0.4865 | 0.680 | 0.005 |
| + carry-share + Vegas (shipped) | 0.1561 | 0.4848 | 0.683 | 0.004 |

## Per-season (shipped model)

| Season | n | Base rate | Brier | Log loss | AUC | ECE |
|---|---|---|---|---|---|---|
| 2018 | 4,955 | 22.4% | 0.1622 | 0.5025 | 0.664 | 0.028 |
| 2019 | 4,934 | 21.2% | 0.1570 | 0.4868 | 0.670 | 0.015 |
| 2020 | 5,111 | 23.8% | 0.1693 | 0.5173 | 0.667 | 0.017 |
| 2021 | 5,365 | 21.6% | 0.1568 | 0.4864 | 0.680 | 0.012 |
| 2022 | 5,333 | 20.5% | 0.1514 | 0.4736 | 0.680 | 0.015 |
| 2023 | 5,379 | 20.2% | 0.1462 | 0.4597 | 0.707 | 0.020 |
| 2024 | 5,309 | 21.8% | 0.1545 | 0.4792 | 0.703 | 0.014 |
| 2025 | 5,377 | 21.2% | 0.1529 | 0.4756 | 0.697 | 0.017 |
| **Pooled** | **41,763** | **21.6%** | **0.1561** | **0.4848** | **0.683** | **0.004** |

## Per-season Brier / log loss, all models

| Season | Naive Brier | Naive LogLoss | Opp Brier | Opp LogLoss | v3 Brier | v3 LogLoss |
|---|---|---|---|---|---|---|
| 2018 | 0.1704 | 0.6208 | 0.1633 | 0.5051 | 0.1622 | 0.5025 |
| 2019 | 0.1652 | 0.6087 | 0.1573 | 0.4877 | 0.1570 | 0.4868 |
| 2020 | 0.1777 | 0.6554 | 0.1699 | 0.5188 | 0.1693 | 0.5173 |
| 2021 | 0.1681 | 0.6244 | 0.1575 | 0.4884 | 0.1568 | 0.4864 |
| 2022 | 0.1629 | 0.6030 | 0.1516 | 0.4746 | 0.1514 | 0.4736 |
| 2023 | 0.1551 | 0.5676 | 0.1472 | 0.4621 | 0.1462 | 0.4597 |
| 2024 | 0.1628 | 0.6138 | 0.1550 | 0.4803 | 0.1545 | 0.4792 |
| 2025 | 0.1651 | 0.6121 | 0.1539 | 0.4781 | 0.1529 | 0.4756 |

## Calibration table — shipped model, pooled (2018–2025)

Predicted-probability band vs. the rate players in that band actually scored. Close = well calibrated.

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 78 | 3.9% | 7.7% |
| 5–10% | 4,001 | 8.6% | 8.8% |
| 10–15% | 11,446 | 12.5% | 12.2% |
| 15–20% | 7,943 | 17.3% | 17.6% |
| 20–25% | 5,633 | 22.4% | 22.9% |
| 25–30% | 4,254 | 27.3% | 27.6% |
| 30–40% | 4,855 | 34.3% | 34.9% |
| 40–50% | 2,395 | 44.5% | 44.2% |
| 50–100% | 1,158 | 55.3% | 55.0% |

## Tier hit-rate — shipped model, pooled

| Tier | Players | Model avg | Actually scored |
|---|---|---|---|
| Elite (45%+) | 2,168 | 51.6% | 50.6% |
| Strong (33-45%) | 4,319 | 38.2% | 38.5% |
| Live (22-33%) | 9,446 | 26.8% | 27.5% |
| Longshot (<22%) | 25,830 | 14.1% | 14.1% |

## Per-season calibration tables (shipped model)


### 2018

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 75 | 3.9% | 8.0% |
| 5–10% | 746 | 8.1% | 11.8% |
| 10–15% | 1,200 | 12.4% | 15.2% |
| 15–20% | 807 | 17.4% | 19.1% |
| 20–25% | 597 | 22.3% | 23.1% |
| 25–30% | 422 | 27.4% | 25.8% |
| 30–40% | 559 | 34.5% | 30.2% |
| 40–50% | 295 | 44.5% | 39.7% |
| 50–100% | 254 | 58.2% | 57.5% |

### 2019

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 277 | 9.0% | 7.9% |
| 10–15% | 1,546 | 12.4% | 12.2% |
| 15–20% | 981 | 17.2% | 18.1% |
| 20–25% | 588 | 22.4% | 22.8% |
| 25–30% | 574 | 27.5% | 26.8% |
| 30–40% | 605 | 34.2% | 34.4% |
| 40–50% | 276 | 44.8% | 43.5% |
| 50–100% | 87 | 53.3% | 47.1% |

### 2020

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 454 | 8.5% | 10.1% |
| 10–15% | 1,208 | 12.5% | 14.7% |
| 15–20% | 1,000 | 17.5% | 19.0% |
| 20–25% | 786 | 22.4% | 23.8% |
| 25–30% | 598 | 27.4% | 28.4% |
| 30–40% | 619 | 34.0% | 36.7% |
| 40–50% | 288 | 44.4% | 46.2% |
| 50–100% | 158 | 56.6% | 55.7% |

### 2021

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 401 | 8.7% | 7.5% |
| 10–15% | 1,324 | 12.5% | 11.9% |
| 15–20% | 1,130 | 17.4% | 17.3% |
| 20–25% | 796 | 22.4% | 23.4% |
| 25–30% | 554 | 27.3% | 23.6% |
| 30–40% | 662 | 34.5% | 33.1% |
| 40–50% | 332 | 44.4% | 45.8% |
| 50–100% | 166 | 53.6% | 52.4% |

### 2022

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 2 | 4.5% | 0.0% |
| 5–10% | 407 | 8.6% | 9.8% |
| 10–15% | 1,527 | 12.6% | 11.1% |
| 15–20% | 1,076 | 17.3% | 16.3% |
| 20–25% | 740 | 22.5% | 21.2% |
| 25–30% | 554 | 27.3% | 26.9% |
| 30–40% | 605 | 34.5% | 35.7% |
| 40–50% | 309 | 44.0% | 41.4% |
| 50–100% | 113 | 53.2% | 51.3% |

### 2023

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 443 | 9.0% | 6.3% |
| 10–15% | 1,616 | 12.4% | 10.6% |
| 15–20% | 1,137 | 17.2% | 15.4% |
| 20–25% | 691 | 22.4% | 21.4% |
| 25–30% | 533 | 27.2% | 30.6% |
| 30–40% | 606 | 34.3% | 36.8% |
| 40–50% | 273 | 44.2% | 46.9% |
| 50–100% | 80 | 53.8% | 63.7% |

### 2024

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 1 | 4.6% | 0.0% |
| 5–10% | 662 | 8.6% | 7.3% |
| 10–15% | 1,506 | 12.5% | 12.0% |
| 15–20% | 892 | 17.2% | 18.8% |
| 20–25% | 731 | 22.5% | 24.6% |
| 25–30% | 482 | 27.1% | 28.0% |
| 30–40% | 571 | 34.2% | 36.8% |
| 40–50% | 309 | 45.3% | 46.6% |
| 50–100% | 155 | 54.4% | 57.4% |

### 2025

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 611 | 8.8% | 8.2% |
| 10–15% | 1,519 | 12.4% | 10.9% |
| 15–20% | 920 | 17.4% | 17.6% |
| 20–25% | 704 | 22.4% | 23.0% |
| 25–30% | 537 | 27.3% | 30.4% |
| 30–40% | 628 | 34.3% | 35.5% |
| 40–50% | 313 | 44.2% | 43.8% |
| 50–100% | 145 | 55.7% | 53.1% |

---
*Metrics: Brier = mean squared error of probabilities; Log loss = negative log-likelihood; AUC = ranking (P a scorer outranks a non-scorer); ECE = mean gap between predicted and observed across deciles. Lower is better except AUC.*


## Performance by position (shipped model, pooled 2018–2025)

| Position | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| RB | 11,435 | 0.1742 | 0.5251 | 0.708 | 0.005 | 26.5% | 26.5% |
| WR | 17,240 | 0.1597 | 0.4934 | 0.666 | 0.008 | 21.7% | 21.7% |
| TE | 8,458 | 0.1387 | 0.4456 | 0.642 | 0.013 | 17.1% | 17.5% |
| QB | 4,630 | 0.1305 | 0.4247 | 0.650 | 0.015 | 16.0% | 16.4% |

## Performance by workload tier (trailing carries+targets / game)

| Workload | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| Workhorse (≥15/g) | 2,958 | 0.2450 | 0.6842 | 0.588 | 0.032 | 43.8% | 46.7% |
| Regular (8–15/g) | 6,875 | 0.2110 | 0.6120 | 0.601 | 0.018 | 32.0% | 31.7% |
| Rotational (3–8/g) | 23,194 | 0.1523 | 0.4790 | 0.623 | 0.004 | 19.3% | 19.5% |

## Calibration curve (deciles, shipped model, pooled)

Total predictions: **41,763**. Each row is one decile of predicted probability.

| Decile | n | Mean predicted | Actually scored |
|---|---|---|---|
| 1 | 4,177 | 8.6% | 8.8% |
| 2 | 4,176 | 11.0% | 10.4% |
| 3 | 4,176 | 12.7% | 12.4% |
| 4 | 4,176 | 14.6% | 14.6% |
| 5 | 4,177 | 16.8% | 17.1% |
| 6 | 4,176 | 19.7% | 20.4% |
| 7 | 4,176 | 23.2% | 23.8% |
| 8 | 4,176 | 27.4% | 27.4% |
| 9 | 4,176 | 33.7% | 34.5% |
| 10 | 4,177 | 46.7% | 46.3% |

See `calibration_curve.svg` for the reliability plot (predicted vs actual, with the diagonal = perfect calibration).


## Clean pregame player-eligibility backtest (2022–2025)

The main tables condition on a player getting ≥1 touch — which uses game-day info. This stricter test predicts for **every player projected eligible pregame** (skill position, projected volume ≥3/g, team playing that week, **not** ruled Out/Doubtful/IR), and counts players who were eligible but benched/scratched/held out as the **0s** they were. Model trained on ≤2021 only; features are exact pregame trailing snapshots.

| Universe | n | Base | Brier | Log loss | AUC | ECE |
|---|---|---|---|---|---|---|
| Eligible (pregame) — all | 18,721 | 20.4% | 0.1477 | 0.4634 | 0.711 | 0.028 |
| — of which actually played | 15,657 | 24.3% | 0.1705 | 0.5186 | 0.677 | 0.011 |
| — of which eligible no-shows | 3,064 | 0.0% | 0.0314 | 0.1815 | — | 0.162 |

**What it shows:** even after excluding Out/Doubtful/IR, **16.4% of pregame-eligible players did not play** (surprise inactives, healthy scratches, committee benchings). The model assigns them a normal TD chance but they score 0 — the no-show subgroup's ECE (0.162) is where pregame uncertainty really lives. Ranking across the full eligible slate is strong (AUC 0.711), but calibration is looser than the touch-conditioned view (0.028 vs 0.011) precisely because of those unpredictable no-shows.

*Caveat:* this test uses a fixed ≤2021 training window (a 1–4 season gap to the test years), so its absolute Brier runs higher than the walk-forward tables above; its value is the **relative** comparison and the exposed no-show rate. The live board already drops Out/Doubtful/IR — the residual ~16% is largely irreducible without final inactive reports.
