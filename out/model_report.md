# EndZone Board — model performance report

Anytime-touchdown model, walk-forward out-of-sample backtest.

- Scoring rates fit only on seasons **before** each test year; the classifier is trained only on prior seasons; every prediction uses pre-kickoff info only.
- Test seasons: **2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025**. Evaluation universe: active, involved skill players (RB/WR/TE/QB with ≥1 touch).
- Pooled held-out sample: **41,791 player-games**, base rate **21.6%**.

## Pooled model comparison (2018–2025)

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | Calib. err (ECE) ↓ |
|---|---|---|---|---|
| Naive: count past TDs | 0.1658 | 0.6128 | 0.651 | 0.060 |
| Opportunity model | 0.1565 | 0.4855 | 0.682 | 0.005 |
| + carry-share + Vegas (shipped) | 0.1556 | 0.4833 | 0.687 | 0.006 |

## Per-season (shipped model)

| Season | n | Base rate | Brier | Log loss | AUC | ECE |
|---|---|---|---|---|---|---|
| 2018 | 4,955 | 22.4% | 0.1613 | 0.5001 | 0.668 | 0.025 |
| 2019 | 4,930 | 21.2% | 0.1568 | 0.4868 | 0.669 | 0.009 |
| 2020 | 5,121 | 23.8% | 0.1686 | 0.5152 | 0.671 | 0.019 |
| 2021 | 5,372 | 21.6% | 0.1566 | 0.4860 | 0.680 | 0.012 |
| 2022 | 5,354 | 20.4% | 0.1508 | 0.4724 | 0.681 | 0.015 |
| 2023 | 5,379 | 20.2% | 0.1454 | 0.4570 | 0.714 | 0.022 |
| 2024 | 5,309 | 21.8% | 0.1540 | 0.4777 | 0.707 | 0.016 |
| 2025 | 5,371 | 21.2% | 0.1524 | 0.4741 | 0.701 | 0.013 |
| **Pooled** | **41,791** | **21.6%** | **0.1556** | **0.4833** | **0.687** | **0.006** |

## Per-season Brier / log loss, all models

| Season | Naive Brier | Naive LogLoss | Opp Brier | Opp LogLoss | v3 Brier | v3 LogLoss |
|---|---|---|---|---|---|---|
| 2018 | 0.1704 | 0.6208 | 0.1625 | 0.5029 | 0.1613 | 0.5001 |
| 2019 | 0.1655 | 0.6116 | 0.1574 | 0.4884 | 0.1568 | 0.4868 |
| 2020 | 0.1776 | 0.6549 | 0.1696 | 0.5175 | 0.1686 | 0.5152 |
| 2021 | 0.1680 | 0.6243 | 0.1571 | 0.4874 | 0.1566 | 0.4860 |
| 2022 | 0.1627 | 0.6021 | 0.1514 | 0.4741 | 0.1508 | 0.4724 |
| 2023 | 0.1551 | 0.5676 | 0.1468 | 0.4608 | 0.1454 | 0.4570 |
| 2024 | 0.1628 | 0.6138 | 0.1546 | 0.4791 | 0.1540 | 0.4777 |
| 2025 | 0.1651 | 0.6100 | 0.1536 | 0.4771 | 0.1524 | 0.4741 |

## Calibration table — shipped model, pooled (2018–2025)

Predicted-probability band vs. the rate players in that band actually scored. Close = well calibrated.

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 74 | 4.2% | 8.1% |
| 5–10% | 3,832 | 8.7% | 8.4% |
| 10–15% | 11,638 | 12.5% | 12.1% |
| 15–20% | 7,931 | 17.3% | 17.3% |
| 20–25% | 5,511 | 22.4% | 23.3% |
| 25–30% | 4,357 | 27.3% | 27.0% |
| 30–40% | 5,007 | 34.2% | 35.6% |
| 40–50% | 2,354 | 44.5% | 45.4% |
| 50–100% | 1,087 | 55.0% | 54.1% |

## Tier hit-rate — shipped model, pooled

| Tier | Players | Model avg | Actually scored |
|---|---|---|---|
| Elite (45%+) | 2,094 | 51.3% | 51.9% |
| Strong (33-45%) | 4,354 | 38.1% | 38.6% |
| Live (22-33%) | 9,504 | 26.9% | 27.4% |
| Longshot (<22%) | 25,839 | 14.1% | 14.1% |

## Per-season calibration tables (shipped model)


### 2018

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 73 | 4.2% | 8.2% |
| 5–10% | 742 | 8.1% | 11.2% |
| 10–15% | 1,146 | 12.4% | 15.2% |
| 15–20% | 853 | 17.4% | 19.2% |
| 20–25% | 611 | 22.4% | 22.1% |
| 25–30% | 429 | 27.4% | 25.4% |
| 30–40% | 542 | 34.4% | 29.7% |
| 40–50% | 336 | 44.3% | 46.4% |
| 50–100% | 223 | 57.8% | 54.3% |

### 2019

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 162 | 9.5% | 7.4% |
| 10–15% | 1,558 | 12.6% | 11.6% |
| 15–20% | 1,070 | 17.3% | 17.9% |
| 20–25% | 593 | 22.3% | 24.8% |
| 25–30% | 612 | 27.4% | 25.2% |
| 30–40% | 622 | 34.1% | 34.4% |
| 40–50% | 270 | 44.6% | 46.7% |
| 50–100% | 43 | 52.5% | 48.8% |

### 2020

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 264 | 9.0% | 10.6% |
| 10–15% | 1,298 | 12.6% | 13.2% |
| 15–20% | 1,104 | 17.3% | 18.4% |
| 20–25% | 766 | 22.3% | 25.3% |
| 25–30% | 626 | 27.3% | 26.4% |
| 30–40% | 691 | 34.0% | 38.4% |
| 40–50% | 252 | 44.8% | 52.0% |
| 50–100% | 120 | 53.5% | 51.7% |

### 2021

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 362 | 8.9% | 6.1% |
| 10–15% | 1,433 | 12.5% | 12.4% |
| 15–20% | 1,070 | 17.3% | 17.6% |
| 20–25% | 775 | 22.4% | 22.2% |
| 25–30% | 571 | 27.3% | 23.3% |
| 30–40% | 667 | 34.2% | 35.5% |
| 40–50% | 292 | 44.5% | 42.1% |
| 50–100% | 202 | 54.9% | 53.0% |

### 2022

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 277 | 9.3% | 10.1% |
| 10–15% | 1,728 | 12.5% | 11.3% |
| 15–20% | 996 | 17.3% | 15.8% |
| 20–25% | 719 | 22.6% | 20.2% |
| 25–30% | 562 | 27.3% | 26.7% |
| 30–40% | 661 | 34.3% | 36.6% |
| 40–50% | 339 | 44.7% | 41.0% |
| 50–100% | 72 | 51.7% | 52.8% |

### 2023

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 575 | 8.8% | 7.5% |
| 10–15% | 1,539 | 12.4% | 9.9% |
| 15–20% | 1,049 | 17.3% | 15.2% |
| 20–25% | 695 | 22.3% | 22.0% |
| 25–30% | 528 | 27.3% | 32.2% |
| 30–40% | 619 | 34.3% | 35.7% |
| 40–50% | 273 | 44.4% | 46.5% |
| 50–100% | 101 | 53.9% | 61.4% |

### 2024

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 1 | 4.8% | 0.0% |
| 5–10% | 727 | 8.7% | 6.3% |
| 10–15% | 1,513 | 12.5% | 13.2% |
| 15–20% | 880 | 17.2% | 17.3% |
| 20–25% | 664 | 22.5% | 25.9% |
| 25–30% | 504 | 27.3% | 28.0% |
| 30–40% | 565 | 34.4% | 37.5% |
| 40–50% | 282 | 44.8% | 49.3% |
| 50–100% | 173 | 54.8% | 53.8% |

### 2025

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 723 | 8.6% | 8.3% |
| 10–15% | 1,423 | 12.3% | 11.2% |
| 15–20% | 909 | 17.3% | 17.3% |
| 20–25% | 688 | 22.4% | 24.0% |
| 25–30% | 525 | 27.3% | 29.7% |
| 30–40% | 640 | 34.3% | 35.9% |
| 40–50% | 310 | 44.1% | 41.3% |
| 50–100% | 153 | 55.6% | 54.9% |

---
*Metrics: Brier = mean squared error of probabilities; Log loss = negative log-likelihood; AUC = ranking (P a scorer outranks a non-scorer); ECE = mean gap between predicted and observed across deciles. Lower is better except AUC.*


## Performance by position (shipped model, pooled 2018–2025)

| Position | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| RB | 11,423 | 0.1728 | 0.5216 | 0.715 | 0.008 | 26.4% | 26.5% |
| WR | 17,278 | 0.1594 | 0.4927 | 0.667 | 0.006 | 21.6% | 21.7% |
| TE | 8,462 | 0.1385 | 0.4449 | 0.645 | 0.012 | 17.1% | 17.5% |
| QB | 4,628 | 0.1300 | 0.4237 | 0.652 | 0.016 | 16.3% | 16.4% |

## Performance by workload tier (trailing carries+targets / game)

| Workload | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| Workhorse (≥15/g) | 2,958 | 0.2448 | 0.6833 | 0.589 | 0.040 | 43.4% | 46.7% |
| Regular (8–15/g) | 6,875 | 0.2090 | 0.6071 | 0.615 | 0.018 | 31.9% | 31.7% |
| Rotational (3–8/g) | 23,205 | 0.1520 | 0.4780 | 0.627 | 0.006 | 19.2% | 19.5% |

## Calibration curve (deciles, shipped model, pooled)

Total predictions: **41,791**. Each row is one decile of predicted probability.

| Decile | n | Mean predicted | Actually scored |
|---|---|---|---|
| 1 | 4,180 | 8.7% | 8.5% |
| 2 | 4,179 | 11.1% | 10.9% |
| 3 | 4,179 | 12.8% | 12.0% |
| 4 | 4,179 | 14.6% | 14.2% |
| 5 | 4,179 | 16.8% | 16.4% |
| 6 | 4,179 | 19.6% | 20.9% |
| 7 | 4,179 | 23.2% | 23.7% |
| 8 | 4,179 | 27.5% | 27.3% |
| 9 | 4,179 | 33.5% | 34.8% |
| 10 | 4,179 | 46.3% | 46.9% |

See `calibration_curve.svg` for the reliability plot (predicted vs actual, with the diagonal = perfect calibration).
