# EndZone Board — model performance report

Anytime-touchdown model, walk-forward out-of-sample backtest.

- Scoring rates fit only on seasons **before** each test year; the classifier is trained only on prior seasons; every prediction uses pre-kickoff info only.
- Test seasons: **2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025**. Evaluation universe: active, involved skill players (RB/WR/TE/QB with ≥1 touch).
- Features include opportunity (carry/target volume, red-zone & goal-line carry/target share), Vegas implied team total, recent-form (last-3-game) usage & trend, and **Next Gen Stats** trailing form (receiver separation & YAC-over-expected; rusher efficiency, yards-over-expected & light/stacked-box rate).
- This report covers the **conversion** model: P(TD | the player plays). The live board multiplies it by a separate **availability** model — P(the player takes the field) — to show a true pregame probability. See `eligibility_report.md` for that two-stage backtest.
- Pooled held-out sample: **41,800 player-games**, base rate **21.6%**.

## Pooled model comparison (2018–2025)

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | Calib. err (ECE) ↓ |
|---|---|---|---|---|
| Naive: count past TDs | 0.1658 | 0.6131 | 0.650 | 0.060 |
| Opportunity model | 0.1569 | 0.4867 | 0.680 | 0.007 |
| + carry-share + Vegas + NGS (shipped) | 0.1524 | 0.4751 | 0.703 | 0.007 |

## Per-season (shipped model)

| Season | n | Base rate | Brier | Log loss | AUC | ECE |
|---|---|---|---|---|---|---|
| 2018 | 4,955 | 22.4% | 0.1579 | 0.4916 | 0.684 | 0.019 |
| 2019 | 4,934 | 21.2% | 0.1561 | 0.4844 | 0.679 | 0.019 |
| 2020 | 5,122 | 23.8% | 0.1663 | 0.5103 | 0.682 | 0.020 |
| 2021 | 5,372 | 21.6% | 0.1521 | 0.4748 | 0.703 | 0.012 |
| 2022 | 5,333 | 20.5% | 0.1473 | 0.4626 | 0.705 | 0.017 |
| 2023 | 5,393 | 20.2% | 0.1418 | 0.4475 | 0.729 | 0.020 |
| 2024 | 5,314 | 21.8% | 0.1499 | 0.4679 | 0.721 | 0.018 |
| 2025 | 5,377 | 21.2% | 0.1490 | 0.4652 | 0.719 | 0.018 |
| **Pooled** | **41,800** | **21.6%** | **0.1524** | **0.4751** | **0.703** | **0.007** |

## Per-season Brier / log loss, all models

| Season | Naive Brier | Naive LogLoss | Opp Brier | Opp LogLoss | v3 Brier | v3 LogLoss |
|---|---|---|---|---|---|---|
| 2018 | 0.1704 | 0.6208 | 0.1625 | 0.5032 | 0.1579 | 0.4916 |
| 2019 | 0.1652 | 0.6087 | 0.1589 | 0.4914 | 0.1561 | 0.4844 |
| 2020 | 0.1776 | 0.6549 | 0.1706 | 0.5210 | 0.1663 | 0.5103 |
| 2021 | 0.1680 | 0.6243 | 0.1573 | 0.4884 | 0.1521 | 0.4748 |
| 2022 | 0.1629 | 0.6030 | 0.1518 | 0.4751 | 0.1473 | 0.4626 |
| 2023 | 0.1551 | 0.5694 | 0.1463 | 0.4592 | 0.1418 | 0.4475 |
| 2024 | 0.1629 | 0.6138 | 0.1551 | 0.4800 | 0.1499 | 0.4679 |
| 2025 | 0.1651 | 0.6121 | 0.1541 | 0.4784 | 0.1490 | 0.4652 |

## Calibration table — shipped model, pooled (2018–2025)

Predicted-probability band vs. the rate players in that band actually scored. Close = well calibrated.

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 103 | 4.3% | 8.7% |
| 5–10% | 5,437 | 8.4% | 9.4% |
| 10–15% | 12,849 | 12.4% | 12.1% |
| 15–20% | 7,072 | 17.1% | 16.9% |
| 20–25% | 3,724 | 22.4% | 22.5% |
| 25–30% | 3,241 | 27.5% | 28.5% |
| 30–40% | 5,013 | 34.7% | 36.4% |
| 40–50% | 2,394 | 44.5% | 45.0% |
| 50–100% | 1,967 | 57.6% | 54.7% |

## Tier hit-rate — shipped model, pooled

| Tier | Players | Model avg | Actually scored |
|---|---|---|---|
| Elite (45%+) | 2,981 | 54.1% | 52.2% |
| Strong (33-45%) | 4,711 | 38.0% | 40.0% |
| Live (22-33%) | 7,044 | 27.2% | 28.0% |
| Longshot (<22%) | 27,064 | 13.3% | 13.3% |

## Per-season calibration tables (shipped model)


### 2018

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 45 | 4.3% | 11.1% |
| 5–10% | 773 | 8.2% | 11.4% |
| 10–15% | 1,329 | 12.4% | 14.7% |
| 15–20% | 881 | 17.2% | 16.3% |
| 20–25% | 484 | 22.4% | 22.5% |
| 25–30% | 342 | 27.4% | 28.4% |
| 30–40% | 494 | 34.7% | 32.4% |
| 40–50% | 344 | 44.6% | 44.2% |
| 50–100% | 263 | 57.9% | 60.1% |

### 2019

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 38 | 4.3% | 2.6% |
| 5–10% | 512 | 8.5% | 10.2% |
| 10–15% | 1,540 | 12.5% | 12.4% |
| 15–20% | 944 | 17.1% | 18.8% |
| 20–25% | 478 | 22.4% | 23.0% |
| 25–30% | 349 | 27.4% | 29.5% |
| 30–40% | 548 | 34.8% | 33.2% |
| 40–50% | 262 | 44.5% | 42.0% |
| 50–100% | 263 | 58.7% | 45.6% |

### 2020

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 1 | 5.0% | 0.0% |
| 5–10% | 580 | 8.4% | 12.1% |
| 10–15% | 1,384 | 12.5% | 14.7% |
| 15–20% | 933 | 17.0% | 16.8% |
| 20–25% | 518 | 22.4% | 22.2% |
| 25–30% | 464 | 27.6% | 28.2% |
| 30–40% | 769 | 34.5% | 38.0% |
| 40–50% | 247 | 44.4% | 49.8% |
| 50–100% | 226 | 58.9% | 56.2% |

### 2021

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 7 | 4.8% | 42.9% |
| 5–10% | 564 | 8.3% | 8.3% |
| 10–15% | 1,493 | 12.5% | 11.3% |
| 15–20% | 1,035 | 17.2% | 18.0% |
| 20–25% | 524 | 22.2% | 18.1% |
| 25–30% | 445 | 27.5% | 28.1% |
| 30–40% | 705 | 34.6% | 32.5% |
| 40–50% | 330 | 44.8% | 47.6% |
| 50–100% | 269 | 56.8% | 55.0% |

### 2022

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 5 | 4.3% | 0.0% |
| 5–10% | 612 | 8.3% | 10.0% |
| 10–15% | 1,730 | 12.5% | 10.1% |
| 15–20% | 948 | 17.1% | 16.7% |
| 20–25% | 472 | 22.4% | 23.3% |
| 25–30% | 415 | 27.6% | 27.2% |
| 30–40% | 633 | 34.7% | 37.4% |
| 40–50% | 304 | 44.6% | 44.1% |
| 50–100% | 214 | 56.0% | 49.1% |

### 2023

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 6 | 4.5% | 0.0% |
| 5–10% | 769 | 8.6% | 7.4% |
| 10–15% | 1,832 | 12.4% | 11.1% |
| 15–20% | 802 | 17.0% | 14.8% |
| 20–25% | 460 | 22.5% | 23.3% |
| 25–30% | 442 | 27.4% | 26.7% |
| 30–40% | 592 | 34.6% | 39.0% |
| 40–50% | 298 | 44.4% | 47.0% |
| 50–100% | 192 | 55.8% | 59.4% |

### 2024

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 1 | 4.8% | 0.0% |
| 5–10% | 808 | 8.5% | 8.3% |
| 10–15% | 1,810 | 12.3% | 12.7% |
| 15–20% | 746 | 17.0% | 15.0% |
| 20–25% | 366 | 22.5% | 25.7% |
| 25–30% | 398 | 27.5% | 31.7% |
| 30–40% | 661 | 34.4% | 38.3% |
| 40–50% | 256 | 44.4% | 44.9% |
| 50–100% | 268 | 57.0% | 59.7% |

### 2025

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 819 | 8.4% | 8.4% |
| 10–15% | 1,731 | 12.4% | 11.0% |
| 15–20% | 783 | 17.1% | 17.9% |
| 20–25% | 422 | 22.3% | 23.2% |
| 25–30% | 386 | 27.4% | 28.5% |
| 30–40% | 611 | 35.0% | 39.8% |
| 40–50% | 353 | 44.4% | 41.6% |
| 50–100% | 272 | 59.0% | 52.6% |

---
*Metrics: Brier = mean squared error of probabilities; Log loss = negative log-likelihood; AUC = ranking (P a scorer outranks a non-scorer); ECE = mean gap between predicted and observed across deciles. Lower is better except AUC.*


## Performance by position (shipped model, pooled 2018–2025)

| Position | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| RB | 11,470 | 0.1664 | 0.5065 | 0.738 | 0.013 | 26.4% | 26.4% |
| WR | 17,194 | 0.1564 | 0.4842 | 0.690 | 0.007 | 21.6% | 21.7% |
| TE | 8,551 | 0.1372 | 0.4408 | 0.657 | 0.010 | 17.2% | 17.6% |
| QB | 4,585 | 0.1308 | 0.4262 | 0.642 | 0.018 | 15.4% | 16.4% |

## Performance by workload tier (trailing carries+targets / game)

| Workload | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| Workhorse (≥15/g) | 2,958 | 0.2356 | 0.6644 | 0.637 | 0.037 | 44.9% | 46.7% |
| Regular (8–15/g) | 6,875 | 0.2037 | 0.5941 | 0.654 | 0.018 | 32.0% | 31.7% |
| Rotational (3–8/g) | 23,198 | 0.1490 | 0.4697 | 0.654 | 0.007 | 19.1% | 19.5% |

## Calibration curve (deciles, shipped model, pooled)

Total predictions: **41,800**. Each row is one decile of predicted probability.

| Decile | n | Mean predicted | Actually scored |
|---|---|---|---|
| 1 | 4,180 | 7.9% | 8.7% |
| 2 | 4,180 | 10.3% | 10.7% |
| 3 | 4,180 | 11.9% | 11.6% |
| 4 | 4,180 | 13.4% | 12.8% |
| 5 | 4,180 | 15.2% | 15.2% |
| 6 | 4,180 | 17.7% | 17.2% |
| 7 | 4,180 | 22.2% | 22.5% |
| 8 | 4,180 | 28.4% | 28.9% |
| 9 | 4,180 | 35.8% | 38.3% |
| 10 | 4,180 | 50.9% | 49.7% |

See `calibration_curve.svg` for the reliability plot (predicted vs actual, with the diagonal = perfect calibration).
