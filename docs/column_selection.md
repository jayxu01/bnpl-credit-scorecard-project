# 特征选列说明（151 列 → 21 列）

> 本文说明 `src/preprocess.py` 为什么从原始 **151 列**中只读取 **21 列**，删掉的 **130 列**分别是什么、为什么删。

## 一、选列的三个总原则

1. **防时间泄漏（最重要）**：只保留"放款时点已经知道"的信息。凡是放款**之后**才产生的信息（还款表现、催收、困难救助、债务和解、放款后的征信更新等），一律删——否则就是"用未来预测过去"，模型会虚高，上线必翻车。
2. **有信息量**：保留对"是否会违约"有预测能力的特征。
3. **去冗余 / 去噪声**：删掉标识符、纯文本、重复派生的列。

## 二、概览

| 项目 | 数量 |
|---|---|
| 原始列数 | 151 |
| 保留 | 21（含目标 `loan_status` 和切分键 `issue_d`） |
| 删除 | 130 |

删除的 130 列按原因分 8 类：

| 类别 | 数量 | 原因 |
|---|---|---|
| A. 放款后信息（泄漏） | 38 | 未来信息，必须删 |
| B. LC 风险评级（循环） | 2 | grade / sub_grade 等于"抄答案" |
| C. 标识符与噪声 | 6 | id、url、纯文本，无预测价值 |
| D. 冗余 / 派生 | 4 | 与保留列高度重复 |
| E. 高基数 / 细粒度 | 3 | 文本职位、邮编、州，为聚焦暂不纳入 |
| F. 联合 / 次级申请人 | 16 | 缺失率极高 |
| G. 渠道 / 常量 | 3 | 低信息 |
| H. 详细征信衍生变量 | 58 | 有预测力，为聚焦暂不纳入（可扩展） |

---

## 三、保留的 21 列（按维度分组）

### 目标与切分（2 列）

| 列名 | 含义 | 保留理由 |
|---|---|---|
| `loan_status` | 贷款状态 | 目标的来源，用来定义好/坏样本 |
| `issue_d` | 放款月份 | 时间切分键（train / test / oot） |

### 贷款基本特征（3 列）

| 列名 | 含义 | 保留理由 |
|---|---|---|
| `loan_amnt` | 申请贷款额 | 贷款规模，核心变量 |
| `term` | 期限（36/60 月） | 期限越长风险越高 |
| `int_rate` | 利率 | 放款时按风险定价，强预测 |

### 收入与负债（4 列）

| 列名 | 含义 | 保留理由 |
|---|---|---|
| `annual_inc` | 年收入 | 还款能力 |
| `dti` | 负债收入比 | 偿债压力，核心变量 |
| `revol_bal` | 循环余额 | 负债规模 |
| `revol_util` | 循环额度使用率 | 信用使用程度 |

### 信用历史（8 列）

| 列名 | 含义 | 保留理由 |
|---|---|---|
| `fico_range_low` / `fico_range_high` | FICO 信用分区间 | 信用质量核心指标（取中点合成 `fico`） |
| `earliest_cr_line` | 最早信用记录 | 推算信用年龄（`credit_age_years`） |
| `delinq_2yrs` | 近 2 年逾期次数 | 历史违约行为 |
| `inq_last_6mths` | 近 6 月征信查询 | 借贷饥渴度 |
| `open_acc` | 开放账户数 | 信用活跃度 |
| `pub_rec` | 公开记录数 | 负面记录 |
| `total_acc` | 总账户数 | 信用历史广度 |

### 身份与用途（4 列）

| 列名 | 含义 | 保留理由 |
|---|---|---|
| `emp_length` | 工作年限 | 收入稳定性 |
| `home_ownership` | 房屋状况 | 资产状况 |
| `verification_status` | 收入核验状态 | 信息可信度 |
| `purpose` | 借款用途 | 资金用途风险差异 |

---

## 四、删除的 130 列（分类说明）

### A. 放款后信息 —— 时间泄漏，38 列（必须删）

模型要回答"放款这一刻会不会违约"，所以放款**之后**才知道的信息一律不能用。分四小类：

**A1. 还款表现类（12 列）**：`out_prncp`、`out_prncp_inv`、`total_pymnt`、`total_pymnt_inv`、`total_rec_prncp`、`total_rec_int`、`total_rec_late_fee`、`recoveries`、`collection_recovery_fee`、`last_pymnt_d`、`last_pymnt_amnt`、`next_pymnt_d`
> 这些都是"这笔贷款后来还了多少、剩多少、催收回来多少"，是结果不是原因。

**A2. 放款后征信更新（3 列）**：`last_credit_pull_d`、`last_fico_range_high`、`last_fico_range_low`
> "last_*"是放款后更新的最新 FICO 和查询日期，泄漏。我们只用放款时的 `fico_range_*`。

**A3. 还款计划（1 列）**：`pymnt_plan`
> 是否已进入还款计划，放款后才知道。

**A4. 困难救助与债务和解（22 列）**：
`hardship_flag`、`hardship_type`、`hardship_reason`、`hardship_status`、`hardship_amount`、`hardship_start_date`、`hardship_end_date`、`hardship_length`、`hardship_dpd`、`hardship_loan_status`、`hardship_payoff_balance_amount`、`hardship_last_payment_amount`、`orig_projected_additional_accrued_interest`、`deferral_term`、`payment_plan_start_date`、`debt_settlement_flag`、`debt_settlement_flag_date`、`settlement_status`、`settlement_date`、`settlement_amount`、`settlement_percentage`、`settlement_term`
> 困难救助（hardship）和债务和解（settlement）都是放款后才可能发生的事件。

### B. LC 风险评级 —— 循环论证，2 列（必须删）

`grade`、`sub_grade`
> 这是 Lending Club 自己的风险评级（A1~G5），本身就是基于借款人的信用状况打的分。拿它当特征等于"用答案预测答案"。真实业务里，你要做的恰恰是**自己算出这个分**。

### C. 标识符与噪声，6 列（必须删）

`id`、`member_id`、`url`、`desc`、`title`、`policy_code`
> 编号/URL/借款描述文本，无预测价值；`policy_code` 几乎全为 1，是常量。

### D. 冗余 / 派生，4 列（必须删）

`funded_amnt`、`funded_amnt_inv`、`installment`、`inq_last_12m`
> `funded_amnt`≈`loan_amnt`（实际放款额与申请额几乎相同）；`installment` 是 `loan_amnt`、`int_rate`、`term` 的派生值；`inq_last_12m` 与保留的 `inq_last_6mths` 重叠。

### E. 高基数 / 细粒度，3 列（为聚焦暂不纳入）

`emp_title`、`zip_code`、`addr_state`
> 这三个**有信息量**（职位、地区确实影响违约），但 `emp_title` 是自由文本、`zip_code`/`addr_state` 基数高，处理成本大。为了聚焦核心信用特征，本项目暂不纳入，可作为后续"地理/职业特征"扩展。

### F. 联合 / 次级申请人，16 列（缺失率极高）

`annual_inc_joint`、`dti_joint`、`verification_status_joint`、`revol_bal_joint`、`sec_app_fico_range_low`、`sec_app_fico_range_high`、`sec_app_earliest_cr_line`、`sec_app_inq_last_6mths`、`sec_app_mort_acc`、`sec_app_open_acc`、`sec_app_revol_util`、`sec_app_open_act_il`、`sec_app_num_rev_accts`、`sec_app_chargeoff_within_12_mths`、`sec_app_collections_12_mths_ex_med`、`sec_app_mths_since_last_major_derog`
> 这些是"联合申请人"和"次级申请人"的信息。绝大多数贷款是单人申请，这些列几乎全是空值，缺失率极高，建模价值低。

### G. 渠道 / 常量，3 列（低信息）

`initial_list_status`、`application_type`、`disbursement_method`
> 放款渠道/申请方式等，属于流程性字段，对违约的区分度很低。

### H. 详细征信衍生变量，58 列（有预测力，为聚焦暂不纳入）

`mths_since_last_delinq`、`mths_since_last_record`、`collections_12_mths_ex_med`、`mths_since_last_major_derog`、`acc_now_delinq`、`tot_coll_amt`、`tot_cur_bal`、`open_acc_6m`、`open_act_il`、`open_il_12m`、`open_il_24m`、`mths_since_rcnt_il`、`total_bal_il`、`il_util`、`open_rv_12m`、`open_rv_24m`、`max_bal_bc`、`all_util`、`total_rev_hi_lim`、`inq_fi`、`total_cu_tl`、`acc_open_past_24mths`、`avg_cur_bal`、`bc_open_to_buy`、`bc_util`、`chargeoff_within_12_mths`、`delinq_amnt`、`mo_sin_old_il_acct`、`mo_sin_old_rev_tl_op`、`mo_sin_rcnt_rev_tl_op`、`mo_sin_rcnt_tl`、`mort_acc`、`mths_since_recent_bc`、`mths_since_recent_bc_dlq`、`mths_since_recent_inq`、`mths_since_recent_revol_delinq`、`num_accts_ever_120_pd`、`num_actv_bc_tl`、`num_actv_rev_tl`、`num_bc_sats`、`num_bc_tl`、`num_il_tl`、`num_op_rev_tl`、`num_rev_accts`、`num_rev_tl_bal_gt_0`、`num_sats`、`num_tl_120dpd_2m`、`num_tl_30dpd`、`num_tl_90g_dpd_24m`、`num_tl_op_past_12m`、`pct_tl_nvr_dlq`、`percent_bc_gt_75`、`pub_rec_bankruptcies`、`tax_liens`、`tot_hi_cred_lim`、`total_bal_ex_mort`、`total_bc_limit`、`total_il_high_credit_limit`

> 这些是征信局提供的**详细衍生变量**（各类账户数量、逾期次数、余额、额度等），本身**有预测力**。我没有删掉它们的原因是"没用"，而是：
> 1. 数量太多（58 个），全塞进去会让模型复杂、难以解释；
> 2. 其中很多和已保留的核心信用变量（`delinq_2yrs`、`open_acc`、`revol_bal` 等）高度相关，容易共线；
> 3. 为了先做出一套**清晰、可解释、能讲清楚**的评分卡，只保留 8 个代表性信用变量。

---

## 五、一个重要说明（面试加分点）

被删的列里，**H 类（征信衍生变量）、E 类（地区/职位）其实有预测力**。我删它们不是"没用"，而是**有意识地聚焦**到 18 个核心特征，先把主流程做清晰。

这正是后续**提升模型的抓手**：可以在"特征扩展"阶段，把这些变量按 IV 逐个加回来做增量验证（看 KS/AUC 是否提升）。面试时能说出"我不是全塞进去，而是先做一套精简可解释的基线，再考虑扩展"，比"我把 151 列全用了"更能体现工程判断。

> 对应到本项目：`woe.py` 会从这 18 个特征里，再按 **IV 阈值 + VIF 共线性**自动筛出真正入模的 ~9 个变量——那是第二层"数据驱动的筛选"，与本文的"领域选列"是两步不同的工作。
