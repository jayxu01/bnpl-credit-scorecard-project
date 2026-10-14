# Run Report

> Aggregated results of each pipeline step. Update this file as each module is run.

## 1. Data Preprocessing — `src/preprocess.py`

**Input**: `data/raw/lending_club.csv` 

| Step | Result |
|---|---|
| Rows read (21 columns via `usecols`) | 2,260,701 rows × 21 columns |
| After good/bad definition | 1,348,099 rows, bad rate **19.98%** |
| Train (issue ≤ 2016-12) | 1,122,460 |
| Test (2017) | 169,321 |
| OOT (2018) | 56,318 |
| OOT bad rate | **15.76%** |

- **Split ratio**: train : test : oot ≈ **83.3% : 12.6% : 4.2%** (split by issue month, not random).
- **Good/bad definition**: `bad = 1` for Charged Off / Default, `bad = 0` for Fully Paid; Current / Late dropped (no final outcome).
- OOT bad rate is lower than train partly due to **censoring** — many 2018 loans had not matured by the 2018Q4 data cutoff.

## 2. WoE/IV Feature Selection — `src/woe.py`

Binning: chi-square merge (max 5 bins). Selected **8** of 18 features by IV ≥ 0.02 and VIF < 5.

| Feature | IV |
|---|---|
| int_rate | 0.4371 |
| term | 0.1961 |
| fico | 0.1203 |
| dti | 0.0738 |
| verification_status | 0.0523 |
| loan_amnt | 0.0317 |
| annual_inc | 0.0295 |
| home_ownership | 0.0263 |

Full WoE table: `output/woe/woe_table.csv` · plot: `output/woe/woe_plot.png`.

## 3. Scorecard — `src/scorecard.py`

Logistic regression on WoE features, PDO scaling (base score 600, PDO 20, base odds 50:1).

- Base points: **527.7**
- Score range: [475, 599]

| Dataset | Samples | Mean score |
|---|---|---|
| train | 1,122,460 | 532.9 |
| test | 169,321 | 533.2 |
| oot | 56,318 | 536.3 |

Full scorecard table: `output/scorecard/scorecard.csv`.

## 4. Validation — `src/validate.py`

| Dataset | Model | KS | AUC | Gini |
|---|---|---|---|---|
| test | scorecard | 0.2839 | 0.6950 | 0.3900 |
| test | xgb | 0.2935 | 0.7017 | 0.4034 |
| oot | scorecard | 0.2737 | 0.6864 | 0.3727 |
| oot | xgb | 0.2937 | 0.6982 | 0.3965 |

- PSI (scorecard, train vs oot) = **0.0318** (< 0.1 = stable).
- Plots: `output/validate/{roc_curve,ks_curve,score_distribution,calibration,bad_rate_by_score,psi_plot,feature_contribution}.png`.
