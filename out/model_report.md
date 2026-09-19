# EndZone Board — model performance report

Anytime-touchdown model, walk-forward out-of-sample backtest.

- Scoring rates fit only on seasons **before** each test year; the classifier is trained only on prior seasons; every prediction uses pre-kickoff info only.
- Test seasons: **2022, 2023, 2024, 2025**. Evaluation universe: active, involved skill players (RB/WR/TE/QB with ≥1 touch).
- Pooled held-out sample: **21,437 player-games**, base rate **20.9%**.

## Pooled model comparison (2022–2025)

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | Calib. err (ECE) ↓ |
|---|---|---|---|---|
| Naive: count past TDs | 0.1614 | 0.5986 | 0.658 | 0.057 |
| Opportunity model | 0.1531 | 0.4774 | 0.683 | 0.011 |
| + carry-share + Vegas (shipped) | 0.1523 | 0.4753 | 0.688 | 0.011 |

## Per-season (shipped model)

| Season | n | Base rate | Brier | Log loss | AUC | ECE |
|---|---|---|---|---|---|---|
| 2022 | 5,354 | 20.4% | 0.1529 | 0.4794 | 0.660 | 0.020 |
| 2023 | 5,395 | 20.2% | 0.1475 | 0.4636 | 0.697 | 0.022 |
| 2024 | 5,314 | 21.8% | 0.1553 | 0.4816 | 0.699 | 0.016 |
| 2025 | 5,374 | 21.2% | 0.1534 | 0.4767 | 0.694 | 0.012 |
| **Pooled** | **21,437** | **20.9%** | **0.1523** | **0.4753** | **0.688** | **0.011** |

## Per-season Brier / log loss, all models

| Season | Naive Brier | Naive LogLoss | Opp Brier | Opp LogLoss | v3 Brier | v3 LogLoss |
|---|---|---|---|---|---|---|
| 2022 | 0.1627 | 0.6021 | 0.1533 | 0.4805 | 0.1529 | 0.4794 |
| 2023 | 0.1550 | 0.5691 | 0.1490 | 0.4677 | 0.1475 | 0.4636 |
| 2024 | 0.1629 | 0.6138 | 0.1559 | 0.4830 | 0.1553 | 0.4816 |
| 2025 | 0.1650 | 0.6098 | 0.1541 | 0.4785 | 0.1534 | 0.4767 |

## Calibration table — shipped model, pooled (2022–2025)

Predicted-probability band vs. the rate players in that band actually scored. Close = well calibrated.

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 27 | 4.5% | 7.4% |
| 5–10% | 1,783 | 8.5% | 8.3% |
| 10–15% | 5,760 | 12.6% | 11.4% |
| 15–20% | 4,848 | 17.3% | 16.3% |
| 20–25% | 3,117 | 22.3% | 22.9% |
| 25–30% | 2,068 | 27.3% | 29.6% |
| 30–40% | 2,415 | 34.4% | 35.5% |
| 40–50% | 1,099 | 44.2% | 48.1% |
| 50–100% | 320 | 55.5% | 52.8% |

## Tier hit-rate — shipped model, pooled

| Tier | Players | Model avg | Actually scored |
|---|---|---|---|
| Elite (45%+) | 755 | 50.7% | 52.3% |
| Strong (33-45%) | 2,153 | 38.1% | 40.5% |
| Live (22-33%) | 4,710 | 26.7% | 28.0% |
| Longshot (<22%) | 13,819 | 14.6% | 13.7% |

## Per-season calibration tables (shipped model)


### 2022

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 37 | 9.2% | 16.2% |
| 10–15% | 1,125 | 13.3% | 11.9% |
| 15–20% | 1,804 | 17.2% | 14.5% |
| 20–25% | 923 | 22.2% | 20.3% |
| 25–30% | 584 | 27.4% | 26.2% |
| 30–40% | 641 | 34.4% | 36.8% |
| 40–50% | 224 | 43.7% | 49.6% |
| 50–100% | 16 | 52.5% | 37.5% |

### 2023

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 0–5% | 27 | 4.5% | 7.4% |
| 5–10% | 355 | 8.3% | 7.0% |
| 10–15% | 1,507 | 12.8% | 10.4% |
| 15–20% | 1,302 | 17.3% | 15.3% |
| 20–25% | 863 | 22.4% | 23.8% |
| 25–30% | 474 | 27.2% | 30.2% |
| 30–40% | 577 | 34.5% | 35.4% |
| 40–50% | 272 | 43.9% | 53.3% |
| 50–100% | 18 | 51.6% | 55.6% |

### 2024

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 693 | 8.6% | 8.4% |
| 10–15% | 1,569 | 12.4% | 12.0% |
| 15–20% | 920 | 17.3% | 19.3% |
| 20–25% | 676 | 22.4% | 24.4% |
| 25–30% | 492 | 27.3% | 32.9% |
| 30–40% | 536 | 34.5% | 36.4% |
| 40–50% | 294 | 44.9% | 45.9% |
| 50–100% | 134 | 54.2% | 56.0% |

### 2025

| Predicted band | n | Mean predicted | Actually scored |
|---|---|---|---|
| 5–10% | 698 | 8.5% | 8.5% |
| 10–15% | 1,559 | 12.2% | 11.4% |
| 15–20% | 822 | 17.5% | 18.5% |
| 20–25% | 655 | 22.3% | 23.8% |
| 25–30% | 518 | 27.4% | 29.9% |
| 30–40% | 661 | 34.2% | 33.7% |
| 40–50% | 309 | 44.1% | 44.7% |
| 50–100% | 152 | 57.5% | 51.3% |

---
*Metrics: Brier = mean squared error of probabilities; Log loss = negative log-likelihood; AUC = ranking (P a scorer outranks a non-scorer); ECE = mean gap between predicted and observed across deciles. Lower is better except AUC.*
