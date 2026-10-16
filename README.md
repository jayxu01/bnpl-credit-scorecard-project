# BNPL Credit Scorecard (A-Card) — Lending Club

> An end-to-end credit scorecard built on real Lending Club loan data: WoE/IV → logistic-regression scorecard → PDO → KS/AUC/PSI.

## Project Overview

Buy-now-pay-later (BNPL) products need to assess credit risk before lending. This project builds an **interpretable, production-ready scorecard (A-card)** from scratch on real Lending Club data:

- **WoE/IV** for binning and feature selection;
- **Logistic regression** (not a black box) as the main model — interpretable and regulator-friendly;
- **PDO** scaling to a standard credit score (base 600, odds double every 20 points, base odds 50:1);
- **XGBoost** as a challenger, and **KS/AUC/Gini/PSI** for full validation.

## Project Structure

```
bnpl-credit-scorecard/
├── data/
│   ├── raw/               # raw Lending Club CSV (not committed)
│   └── processed/         # cleaned train/test/oot (generated, not committed)
├── notebooks/             # EDA
├── output/                # binner, scorecard table, validation plots, report
├── src/                   # core code, split by responsibility
│   ├── config.py          # global config: paths, thresholds, scorecard params
│   ├── preprocess.py      # cleaning + good/bad definition + time split
│   ├── woe.py             # chi-square binning + WoE/IV + IV/VIF selection
│   ├── scorecard.py       # logistic regression + PDO scaling
│   ├── challenger.py      # XGBoost comparison
│   └── validate.py        # KS/AUC/Gini/PSI/calibration + plots
├── tests/                 # unit tests
├── docs/                  # docs (e.g. column selection rationale)
├── .gitignore
├── requirements.txt
├── LICENSE
├── README.md
└── main.py                # one-click pipeline entry
```

> Dependencies: pandas / numpy / scipy / scikit-learn / xgboost / matplotlib / seaborn.

## Data

- Source: Kaggle "Lending Club Loan Data" (`accepted_2007_to_2018Q4.csv`)
- Size: 2.26M rows × 151 columns → 1.35M samples after good/bad definition
- Target: `bad = 1` for Charged Off / Default, `bad = 0` for Fully Paid; Current / Late dropped (no final outcome)
- Time split by issue month `issue_d`: train (≤2016-12), test (2017), oot (2018) ≈ **83% : 13% : 4%**
- Column selection rationale (151 → 21): see `docs/column_selection.md`

## Pipeline

1. Data cleaning & time split — `src/preprocess.py`
2. WoE/IV chi-square binning & feature selection (monotonicity + IV + VIF) — `src/woe.py`
3. Logistic-regression scorecard + PDO scaling — `src/scorecard.py`
4. XGBoost challenger comparison — `src/challenger.py`
5. Validation (KS/AUC/Gini/PSI/calibration) — `src/validate.py`

## Results

### Feature selection

Of 18 candidate features, **8** survive IV (≥0.02) + VIF (<5):

`int_rate`, `term`, `fico`, `dti`, `verification_status`, `loan_amnt`, `annual_inc`, `home_ownership`

### Model performance (test set)

| Model | KS | AUC | Gini |
|---|---|---|---|
| Logistic scorecard (champion) | 0.2839 | 0.6950 | 0.3900 |
| XGBoost (challenger) | 0.2935 | 0.7017 | 0.4034 |

- **PSI** (scorecard, train vs oot) = **0.0318** (< 0.1, stable)
- XGBoost is slightly better, but the gap is small (AUC +0.007) — the scorecard is chosen for interpretability.

> Full results: `output/report.md` · scorecard table: `output/scorecard/scorecard.csv` · plots: `output/validate/`.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download `accepted_2007_to_2018Q4.csv` from Kaggle and place it at `data/raw/lending_club.csv`.
3. Run the full pipeline:
   ```bash
   python main.py
   ```
   or run each step individually: `src/preprocess.py → woe.py → scorecard.py → challenger.py → validate.py`.
4. Run unit tests:
   ```bash
   python -m unittest tests.test_metrics tests.test_woe
   ```

## Skills Demonstrated

| Skill | Module |
|---|---|
| Data cleaning / leakage prevention | `src/preprocess.py` |
| Feature engineering & selection (chi-square binning + IV + VIF) | `src/woe.py` |
| Scorecard modeling (logistic regression + PDO) | `src/scorecard.py` |
| Ensemble comparison (XGBoost) | `src/challenger.py` |
| Model validation (KS/AUC/Gini/PSI/calibration + plots) | `src/validate.py` |
| Unit testing | `tests/` |

---

# 先用后付信用评分卡（A卡）— Lending Club

> 基于真实 Lending Club 贷款数据，自建 WoE/IV → 逻辑回归评分卡 → PDO → KS/AUC/PSI 的全流程信用评分卡项目。

## 项目背景与目标

先用后付（BNPL）产品在放款前需要对用户做信用风险评估。本项目基于真实的 Lending Club 贷款数据，从零构建一张**可解释、可落地**的信用评分卡（A 卡）：

- 用 **WoE/IV** 做特征分箱与变量筛选；
- 用**逻辑回归**（而非黑盒模型）做主模型，保证可解释、可监管；
- 用 **PDO** 把违约概率缩放成标准信用分（600 分、每 20 分 odds 翻倍、基准 odds 50:1）；
- 用 **XGBoost** 作挑战者对比，用 **KS/AUC/Gini/PSI** 做完整验证。

## 项目结构

```
bnpl-credit-scorecard/
├── data/
│   ├── raw/               # 原始 Lending Club CSV（不提交）
│   └── processed/         # 清洗切分后的 train/test/oot（脚本生成，不提交）
├── notebooks/             # EDA 探索性分析
├── output/                # 分箱器、评分卡表、验证图、报告
├── src/                   # 核心代码（按职责拆分）
│   ├── config.py          # 全局配置：路径、阈值、评分卡参数
│   ├── preprocess.py      # 数据清洗 + 好坏定义 + 时间切分
│   ├── woe.py             # 卡方分箱 + WoE/IV + IV/VIF 筛选
│   ├── scorecard.py       # 逻辑回归评分卡 + PDO 缩放
│   ├── challenger.py      # XGBoost 挑战者对比
│   └── validate.py        # KS/AUC/Gini/PSI/校准 + 可视化
├── tests/                 # 单元测试
├── docs/                  # 说明文档（如特征选列说明）
├── .gitignore
├── requirements.txt
├── LICENSE
├── README.md
└── main.py                # 一键跑通全流程
```

> 依赖：pandas / numpy / scipy / scikit-learn / xgboost / matplotlib / seaborn。

## 数据

- 来源：Kaggle「Lending Club Loan Data」（`accepted_2007_to_2018Q4.csv`）
- 规模：226 万行 × 151 列，定义好坏后 135 万条样本
- 目标定义：`loan_status` 为 Charged Off / Default → `bad=1`；Fully Paid → `bad=0`；Current / Late 等未到期样本丢弃
- 时间切分（按放款月份 `issue_d`）：train（≤2016-12）、test（2017）、oot（2018），比例约 **83% : 13% : 4%**
- 选列说明（151 列 → 21 列）：见 `docs/column_selection.md`

## 方法论流程

1. 数据清洗与时间切分 —— `src/preprocess.py`
2. WoE/IV 卡方分箱与变量筛选（单调性 + IV + VIF）—— `src/woe.py`
3. 逻辑回归评分卡 + PDO 缩放 —— `src/scorecard.py`
4. XGBoost 挑战者对比 —— `src/challenger.py`
5. 模型验证（KS/AUC/Gini/PSI/校准）—— `src/validate.py`

## 结果

### 变量筛选

18 个特征经 IV（≥0.02）+ VIF（<5）筛选后，保留 **8 个入模**：

`int_rate`、`term`、`fico`、`dti`、`verification_status`、`loan_amnt`、`annual_inc`、`home_ownership`

### 模型性能（test 集）

| 模型 | KS | AUC | Gini |
|---|---|---|---|
| 逻辑回归评分卡（冠军） | 0.2839 | 0.6950 | 0.3900 |
| XGBoost 挑战者 | 0.2935 | 0.7017 | 0.4034 |

- **PSI**（评分卡，train vs oot）= **0.0318**（< 0.1，稳定）
- XGBoost 略胜，但差距很小（AUC 仅 +0.007）——评分卡因可解释、可落地被选为主模型

> 完整结果见 `output/report.md`；评分卡表见 `output/scorecard/scorecard.csv`；验证图见 `output/validate/`。

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 从 Kaggle 下载 `accepted_2007_to_2018Q4.csv`，放到 `data/raw/lending_club.csv`；
3. 一键跑全流程：
   ```bash
   python main.py
   ```
   或按顺序单独运行：`src/preprocess.py → woe.py → scorecard.py → challenger.py → validate.py`
4. 跑单元测试：
   ```bash
   python -m unittest tests.test_metrics tests.test_woe
   ```

## Skills demonstrated

| 技能 | 对应模块 |
|---|---|
| 数据清洗 / 防泄漏 | `src/preprocess.py` |
| 特征工程 / 变量筛选（卡方分箱 + IV + VIF） | `src/woe.py` |
| 评分卡建模（逻辑回归 + PDO 缩放） | `src/scorecard.py` |
| 集成学习对比（XGBoost） | `src/challenger.py` |
| 模型验证（KS/AUC/Gini/PSI/校准 + 可视化） | `src/validate.py` |
| 单元测试 | `tests/` |
