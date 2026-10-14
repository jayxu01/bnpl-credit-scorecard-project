# 先用后付信用评分卡（A卡）— Lending Club

> 基于真实 Lending Club 贷款数据，自建 WoE/IV → 逻辑回归评分卡 → PDO → KS/AUC/PSI 的全流程信用评分卡项目。

## 项目背景与目标

先用后付（BNPL）产品在放款前需要对用户做信用风险评估。本项目基于真实的 Lending Club 贷款数据，从零构建一张**可解释、可落地**的信用评分卡（A 卡）：

- 用 **WoE/IV** 做特征分箱与变量筛选；
- 用**逻辑回归**（而非黑盒模型）做主模型，保证可解释、可解释性可监管；
- 用 **PDO** 把违约概率缩放成标准信用分（600 分、每 20 分 odds 翻倍、基准 odds 50:1）；
- 用 **XGBoost** 作挑战者对比，用 **KS/AUC/Gini/PSI** 做完整验证。

## 项目结构

```
bnpl-credit-scorecard/
├── data/
│   ├── raw/               # 原始 Lending Club CSV（不提交，太大且可重新下载）
│   └── processed/         # 清洗切分后的 train/test/oot（脚本生成，不提交）
├── notebooks/             # EDA 探索性分析
├── output/                # 分箱器、评分卡表、验证图、报告
├── src/                   # 核心代码（按职责拆分）
│   ├── config.py          # 全局配置：路径、阈值、评分卡参数（改参数只动这里）
│   ├── preprocess.py      # 数据清洗 + 好坏定义 + 时间切分
│   ├── woe.py             # WoE/IV 卡方分箱 + 单调性 + IV/VIF 筛选
│   ├── scorecard.py       # 逻辑回归评分卡 + PDO 缩放
│   ├── challenger.py      # XGBoost 挑战者对比
│   └── validate.py        # KS/AUC/Gini/PSI/校准 + 可视化
├── tests/                 # 单元测试
├── docs/                  # 说明文档（如特征选列说明）
├── .gitignore             # 指定哪些文件不进 git（数据、模型二进制等）
├── requirements.txt       # 依赖清单
├── README.md              # 项目说明
└── main.py                # 一键跑通全流程的入口
```

> 依赖：pandas / numpy / scipy / scikit-learn / xgboost / matplotlib / seaborn。

## 数据

- 来源：Kaggle「Lending Club Loan Data」（`accepted_2007_to_2018Q4.csv`）
- 规模：226 万行 × 151 列，定义好坏后 135 万条样本
- 坏样本定义：`loan_status` 为 Charged Off / Default → `bad=1`；Fully Paid → `bad=0`；Current / Late 等未到期样本丢弃
- 时间切分（按放款月份 `issue_d`）：train（≤2016-12）、test（2017）、oot（2018），比例约 **83% : 13% : 4%**
- 选列说明（151 列 → 21 列，为什么删那 130 列）：见 `docs/column_selection.md`

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
2. 下载数据：从 Kaggle 下载 `accepted_2007_to_2018Q4.csv`，放到 `data/raw/lending_club.csv`；
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
