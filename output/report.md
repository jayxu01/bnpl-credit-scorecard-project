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

_Pending — run `src/woe.py` and fill in the IV table + selected features._

## 3. Scorecard — `src/scorecard.py`

_Pending._

## 4. Validation — `src/validate.py`

_Pending — KS / AUC / Gini / PSI / calibration._
