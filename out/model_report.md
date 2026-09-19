# EndZone Board — model performance report

Anytime-touchdown model, walk-forward out-of-sample backtest.

- Scoring rates fit only on seasons **before** each test year; the classifier is trained only on prior seasons; every prediction uses pre-kickoff info only.
- Test seasons: **2022, 2023, 2024, 2025**. Evaluation universe: active, involved skill players (RB/WR/TE/QB with ≥1 touch).
- Pooled held-out sample: **21,437 player-games**, base rate **20.9%**.

## Pooled model comparison (2022–2025)

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | Calib. err (ECE) ↓ |
|---|---|---|---|---|
| Naive: count past TDs | 0.1614 | 0.5986 | 0.658 | 0.057 |
| Opportunity model | 0.1532 | 0.4776 | 0.683 | 0.014 |
| + carry-share + Vegas (shipped) | 0.1523 | 0.4752 | 0.688 | 0.015 |

## Per-season (shipped model)

| Season | n | Base rate | Brier | Log loss | AUC | ECE |
|---|---|---|---|---|---|---|
| 2022 | 5,354 | 20.4% | 0.1533 | 0.4800 | 0.661 | 0.024 |
| 2023 | 5,395 | 20.2% | 0.1478 | 0.4643 | 0.696 | 0.027 |
| 2024 | 5,314 | 21.8% | 0.1550 | 0.4806 | 0.701 | 0.018 |
| 2025 | 5,374 | 21.2% | 0.1532 | 0.4761 | 0.695 | 0.014 |
| **Pooled** | **21,437** | **20.9%** | **0.1523** | **0.4752** | **0.688** | **0.015** |

## Per-season Brier / log loss, all models

| Season | Naive Brier | Naive LogLoss | Opp Brier | Opp LogLoss | v3 Brier | v3 LogLoss |
|---|---|---|---|---|---|---|
| 2022 | 0.1627 | 0.6021 | 0.1539 | 0.4820 | 0.1533 | 0.4800 |
| 2023 | 0.1550 | 0.5691 | 0.1493 | 0.4683 | 0.1478 | 0.4643 |
| 2024 | 0.1629 | 0.6138 | 0.1555 | 0.4820 | 0.1550 | 0.4806 |
| 2025 | 0.1650 | 0.6098 | 0.1541 | 0.4784 | 0.1532 | 0.4761 |

## Calibration table — shipped model, pooled (2022–2025)

Predicted-probability band vs. the rate players in that band actually scored. Close = well calibrated.

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 5 | 4.7% | 0.0% |
| 5–10% | 1,793 | 8.5% | 7.9% |
| 10–15% | 5,856 | 12.7% | 11.3% |
| 15–20% | 4,962 | 17.3% | 16.4% |
| 20–25% | 3,110 | 22.3% | 23.7% |
| 25–30% | 1,958 | 27.3% | 30.6% |
| 30–40% | 2,433 | 34.3% | 36.0% |
| 40–50% | 962 | 44.1% | 46.8% |
| 50–100% | 358 | 55.3% | 55.6% |

## Tier hit-rate — shipped model, pooled

| Tier | Players | Model avg | Actually scored |
|---|---|---|---|
| Elite (45%+) | 720 | 51.2% | 52.6% |
| Strong (33-45%) | 2,102 | 37.8% | 40.3% |
| Live (22-33%) | 4,583 | 26.7% | 28.7% |
| Longshot (<22%) | 14,032 | 14.6% | 13.8% |

## Per-season calibration tables (shipped model)


### 2022

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 19 | 9.4% | 15.8% |
| 10–15% | 1,237 | 13.3% | 11.2% |
| 15–20% | 1,854 | 17.2% | 15.0% |
| 20–25% | 919 | 22.3% | 21.7% |
| 25–30% | 585 | 27.2% | 29.1% |
| 30–40% | 623 | 34.0% | 39.0% |
| 40–50% | 113 | 42.8% | 53.1% |
| 50–100% | 4 | 50.5% | 75.0% |

### 2023

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 5 | 4.7% | 0.0% |
| 5–10% | 334 | 8.3% | 6.3% |
| 10–15% | 1,479 | 12.8% | 10.8% |
| 15–20% | 1,409 | 17.4% | 15.1% |
| 20–25% | 847 | 22.3% | 23.7% |
| 25–30% | 464 | 27.3% | 30.2% |
| 30–40% | 602 | 34.3% | 36.9% |
| 40–50% | 229 | 43.7% | 50.2% |
| 50–100% | 26 | 51.8% | 69.2% |

### 2024

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 684 | 8.4% | 8.0% |
| 10–15% | 1,672 | 12.4% | 11.8% |
| 15–20% | 844 | 17.3% | 20.0% |
| 20–25% | 675 | 22.4% | 25.6% |
| 25–30% | 415 | 27.4% | 32.8% |
| 30–40% | 549 | 34.5% | 35.7% |
| 40–50% | 315 | 44.4% | 45.1% |
| 50–100% | 160 | 54.2% | 55.0% |

### 2025

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 756 | 8.5% | 8.3% |
| 10–15% | 1,468 | 12.2% | 11.4% |
| 15–20% | 855 | 17.5% | 17.9% |
| 20–25% | 669 | 22.3% | 24.4% |
| 25–30% | 494 | 27.5% | 31.2% |
| 30–40% | 659 | 34.4% | 32.8% |
| 40–50% | 305 | 44.6% | 43.6% |
| 50–100% | 168 | 56.9% | 53.6% |

---
*Metrics: Brier = mean squared error of probabilities; Log loss = negative log-likelihood; AUC = ranking (P a scorer outranks a non-scorer); ECE = mean gap between predicted and observed across deciles. Lower is better except AUC.*


## Performance by position (shipped model, pooled 2022–2025)

| Position | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| RB | 5,779 | 0.1723 | 0.5201 | 0.715 | 0.022 | 26.2% | 26.3% |
| WR | 8,826 | 0.1558 | 0.4844 | 0.664 | 0.014 | 20.4% | 20.8% |
| TE | 4,439 | 0.1325 | 0.4297 | 0.649 | 0.018 | 16.5% | 16.6% |
| QB | 2,393 | 0.1281 | 0.4177 | 0.671 | 0.022 | 16.4% | 16.3% |

## Performance by workload tier (trailing carries+targets / game)

| Workload | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| Workhorse (≥15/g) | 1,608 | 0.2461 | 0.6861 | 0.579 | 0.049 | 41.7% | 45.9% |
| Regular (8–15/g) | 3,268 | 0.2155 | 0.6220 | 0.590 | 0.025 | 30.4% | 32.6% |
| Rotational (3–8/g) | 11,915 | 0.1482 | 0.4692 | 0.622 | 0.008 | 18.7% | 18.8% |

## Calibration curve (deciles, shipped model, pooled)

Total predictions: **21,437**. Each row is one decile of predicted probability.

| Decile | n | Mean predicted | Actually scored |
|---|---|---|---|
| 1 | 2,144 | 8.7% | 8.3% |
| 2 | 2,144 | 11.4% | 10.3% |
| 3 | 2,143 | 13.2% | 12.0% |
| 4 | 2,144 | 14.9% | 13.6% |
| 5 | 2,144 | 16.7% | 15.2% |
| 6 | 2,143 | 19.0% | 18.3% |
| 7 | 2,144 | 21.9% | 22.9% |
| 8 | 2,143 | 25.8% | 29.2% |
| 9 | 2,144 | 31.7% | 33.6% |
| 10 | 2,144 | 43.5% | 45.4% |

See `calibration_curve.svg` for the reliability plot (predicted vs actual, with the diagonal = perfect calibration).
