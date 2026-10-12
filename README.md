# 先用后付信用评分卡（A卡）— Lending Club

> 基于真实 Lending Club 贷款数据，自建 WoE/IV → 逻辑回归评分卡 → PDO → KS/AUC/PSI 的全流程信用评分卡项目。

## 项目背景与目标

（待补充：为什么做、解决什么问题）

## 数据

- 来源：Kaggle「Lending Club Loan Data」
- 样本量：xxx 条，字段 xxx 个（跑完后回填）
- 坏样本定义：`loan_status` 为 `Charged Off` / `Default`；好样本为 `Fully Paid`
- 时间切分：按放款月份 `issue_d` 切 train / test / oot（防泄漏）

## 方法论流程

1. 数据清洗与时间切分 —— `src/preprocess.py`
2. WoE/IV 分箱与变量筛选（单调性 + IV + VIF）—— `src/woe.py`
3. 逻辑回归评分卡 + PDO 缩放 —— `src/scorecard.py`
4. XGBoost 挑战者对比 —— `src/challenger.py`
5. 模型验证（KS/AUC/Gini/PSI/校准）—— `src/validate.py`

## 结果

| 模型 | KS | AUC | Gini | PSI |
|---|---|---|---|---|
| 逻辑回归评分卡 | 待填 | 待填 | 待填 | 待填 |
| XGBoost 挑战者 | 待填 | 待填 | 待填 | 待填 |

（评分卡表、ROC/KS 曲线、单调性验证图见 `output/`）

## 快速开始

（待补充：安装依赖、下载数据、运行 `python main.py`）

## Skills demonstrated

| 技能 | 对应模块 |
|---|---|
| 数据清洗 | `src/preprocess.py` |
| 特征工程 / 变量筛选 | `src/woe.py` |
| 评分卡建模 | `src/scorecard.py` |
| 集成学习对比 | `src/challenger.py` |
| 模型验证 | `src/validate.py` |
