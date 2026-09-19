# EndZone Board — model performance report

Anytime-touchdown model, walk-forward out-of-sample backtest.

- Scoring rates fit only on seasons **before** each test year; the classifier is trained only on prior seasons; every prediction uses pre-kickoff info only.
- Test seasons: **2022, 2023, 2024, 2025**. Evaluation universe: active, involved skill players (RB/WR/TE/QB with ≥1 touch).
- Pooled held-out sample: **21,440 player-games**, base rate **20.9%**.

## Pooled model comparison (2022–2025)

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | Calib. err (ECE) ↓ |
|---|---|---|---|---|
| Naive: count past TDs | 0.1614 | 0.5992 | 0.657 | 0.057 |
| Opportunity model | 0.1534 | 0.4787 | 0.681 | 0.012 |
| + carry-share + Vegas (shipped) | 0.1525 | 0.4759 | 0.687 | 0.011 |

## Per-season (shipped model)

| Season | n | Base rate | Brier | Log loss | AUC | ECE |
|---|---|---|---|---|---|---|
| 2022 | 5,354 | 20.4% | 0.1544 | 0.4845 | 0.655 | 0.029 |
| 2023 | 5,395 | 20.2% | 0.1466 | 0.4604 | 0.705 | 0.017 |
| 2024 | 5,314 | 21.8% | 0.1552 | 0.4814 | 0.698 | 0.017 |
| 2025 | 5,377 | 21.2% | 0.1537 | 0.4776 | 0.693 | 0.016 |
| **Pooled** | **21,440** | **20.9%** | **0.1525** | **0.4759** | **0.687** | **0.011** |

## Per-season Brier / log loss, all models

| Season | Naive Brier | Naive LogLoss | Opp Brier | Opp LogLoss | v3 Brier | v3 LogLoss |
|---|---|---|---|---|---|---|
| 2022 | 0.1627 | 0.6021 | 0.1561 | 0.4898 | 0.1544 | 0.4845 |
| 2023 | 0.1550 | 0.5691 | 0.1473 | 0.4624 | 0.1466 | 0.4604 |
| 2024 | 0.1629 | 0.6138 | 0.1559 | 0.4831 | 0.1552 | 0.4814 |
| 2025 | 0.1651 | 0.6121 | 0.1545 | 0.4794 | 0.1537 | 0.4776 |

## Calibration table — shipped model, pooled (2022–2025)

Predicted-probability band vs. the rate players in that band actually scored. Close = well calibrated.

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 48 | 4.4% | 12.5% |
| 5–10% | 3,180 | 8.4% | 9.4% |
| 10–15% | 6,309 | 12.3% | 12.2% |
| 15–20% | 3,623 | 17.4% | 18.5% |
| 20–25% | 2,343 | 22.4% | 23.9% |
| 25–30% | 1,810 | 27.4% | 28.7% |
| 30–40% | 2,245 | 34.5% | 34.2% |
| 40–50% | 1,221 | 44.5% | 44.1% |
| 50–100% | 661 | 56.7% | 52.5% |

## Tier hit-rate — shipped model, pooled

| Tier | Players | Model avg | Actually scored |
|---|---|---|---|
| Elite (45%+) | 1,176 | 52.6% | 49.6% |
| Strong (33-45%) | 2,131 | 38.3% | 37.5% |
| Live (22-33%) | 3,978 | 26.9% | 28.5% |
| Longshot (<22%) | 14,155 | 13.3% | 13.9% |

## Per-season calibration tables (shipped model)


### 2022

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 48 | 4.4% | 12.5% |
| 5–10% | 1,115 | 8.0% | 12.5% |
| 10–15% | 1,529 | 12.4% | 14.1% |
| 15–20% | 795 | 17.3% | 18.2% |
| 20–25% | 483 | 22.3% | 23.0% |
| 25–30% | 369 | 27.4% | 23.6% |
| 30–40% | 480 | 34.8% | 32.9% |
| 40–50% | 259 | 44.6% | 40.5% |
| 50–100% | 276 | 58.9% | 46.4% |

### 2023

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 559 | 8.6% | 6.8% |
| 10–15% | 1,704 | 12.4% | 10.7% |
| 15–20% | 1,093 | 17.3% | 17.6% |
| 20–25% | 628 | 22.4% | 24.8% |
| 25–30% | 474 | 27.3% | 27.4% |
| 30–40% | 537 | 34.5% | 35.4% |
| 40–50% | 331 | 44.4% | 47.4% |
| 50–100% | 69 | 52.8% | 65.2% |

### 2024

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 714 | 8.7% | 7.6% |
| 10–15% | 1,594 | 12.4% | 12.7% |
| 15–20% | 887 | 17.3% | 19.6% |
| 20–25% | 591 | 22.4% | 24.2% |
| 25–30% | 512 | 27.4% | 32.0% |
| 30–40% | 586 | 34.3% | 34.1% |
| 40–50% | 294 | 45.0% | 46.3% |
| 50–100% | 136 | 52.8% | 60.3% |

### 2025

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 792 | 8.4% | 8.7% |
| 10–15% | 1,482 | 12.2% | 11.5% |
| 15–20% | 848 | 17.6% | 18.9% |
| 20–25% | 641 | 22.4% | 23.4% |
| 25–30% | 455 | 27.3% | 30.5% |
| 30–40% | 642 | 34.5% | 34.1% |
| 40–50% | 337 | 44.1% | 41.5% |
| 50–100% | 180 | 57.9% | 51.1% |

---
*Metrics: Brier = mean squared error of probabilities; Log loss = negative log-likelihood; AUC = ranking (P a scorer outranks a non-scorer); ECE = mean gap between predicted and observed across deciles. Lower is better except AUC.*


## Performance by position (shipped model, pooled 2022–2025)

| Position | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| RB | 5,777 | 0.1724 | 0.5198 | 0.717 | 0.012 | 27.4% | 26.3% |
| WR | 8,824 | 0.1557 | 0.4844 | 0.665 | 0.011 | 19.9% | 20.8% |
| TE | 4,434 | 0.1328 | 0.4314 | 0.642 | 0.013 | 15.4% | 16.6% |
| QB | 2,405 | 0.1291 | 0.4216 | 0.648 | 0.028 | 15.1% | 16.3% |

## Performance by workload tier (trailing carries+targets / game)

| Workload | n | Brier | Log loss | AUC | ECE | Model avg | Actual |
|---|---|---|---|---|---|---|---|
| Workhorse (≥15/g) | 1,608 | 0.2464 | 0.6867 | 0.576 | 0.039 | 45.9% | 45.9% |
| Regular (8–15/g) | 3,268 | 0.2159 | 0.6235 | 0.590 | 0.022 | 32.0% | 32.6% |
| Rotational (3–8/g) | 11,915 | 0.1486 | 0.4712 | 0.617 | 0.011 | 17.9% | 18.8% |

## Calibration curve (deciles, shipped model, pooled)

Total predictions: **21,440**. Each row is one decile of predicted probability.

| Decile | n | Mean predicted | Actually scored |
|---|---|---|---|
| 1 | 2,144 | 7.7% | 9.7% |
| 2 | 2,144 | 10.0% | 10.4% |
| 3 | 2,144 | 11.5% | 11.0% |
| 4 | 2,144 | 13.1% | 12.7% |
| 5 | 2,144 | 15.2% | 15.1% |
| 6 | 2,144 | 18.0% | 19.3% |
| 7 | 2,144 | 21.5% | 23.6% |
| 8 | 2,144 | 26.5% | 28.2% |
| 9 | 2,144 | 33.6% | 33.9% |
| 10 | 2,144 | 47.6% | 45.1% |

See `calibration_curve.svg` for the reliability plot (predicted vs actual, with the diagonal = perfect calibration).
