# Column Selection Notes (151 → 21)

> Why `src/preprocess.py` reads only 21 of the 151 raw Lending Club columns, and why the other 130 are dropped.

## 1. Selection Principles

1. **No time leakage**: keep only information known at origination. Anything produced after the loan is issued (payment performance, collections, hardship, settlement, post-origination credit updates) is dropped — otherwise the model is "predicting the past" and will overfit.
2. **Predictive value**: keep features that carry signal about whether a borrower will default.
3. **Non-redundant / non-noise**: drop identifiers, free text, and duplicate or derived columns.

## 2. Overview

| Item | Count |
|---|---|
| Raw columns | 151 |
| Kept | 21 (incl. target `loan_status` and split key `issue_d`) |
| Dropped | 130 |

The 130 dropped columns fall into 8 categories:

| Category | Count | Reason |
|---|---|---|
| A. Post-origination info (leakage) | 38 | Future information — must drop |
| B. LC risk grade (circular) | 2 | `grade` / `sub_grade` = "using the answer" |
| C. Identifiers & noise | 6 | id, url, free text — no predictive value |
| D. Redundant / derived | 4 | Highly duplicated with kept columns |
| E. High-cardinality / granular | 3 | Job title, ZIP, state — excluded to stay focused |
| F. Joint / secondary applicant | 16 | Near-empty (very high missing rate) |
| G. Channel / constant | 3 | Low information |
| H. Detailed credit-bureau variables | 58 | Predictive, but excluded to stay focused (extendable) |

---

## 3. Kept Columns (21) by Dimension

### Target & split key (2)

| Column | Meaning | Why kept |
|---|---|---|
| `loan_status` | Loan status | Source of the target (good/bad) |
| `issue_d` | Issue month | Time split key (train / test / oot) |

### Loan characteristics (3)

| Column | Meaning | Why kept |
|---|---|---|
| `loan_amnt` | Requested loan amount | Loan size, core variable |
| `term` | Term (36/60 months) | Longer term = higher risk |
| `int_rate` | Interest rate | Risk-priced at origination, strong predictor |

### Income & debt (4)

| Column | Meaning | Why kept |
|---|---|---|
| `annual_inc` | Annual income | Repayment ability |
| `dti` | Debt-to-income ratio | Debt burden, core variable |
| `revol_bal` | Revolving balance | Debt level |
| `revol_util` | Revolving utilization | Credit usage |

### Credit history (8)

| Column | Meaning | Why kept |
|---|---|---|
| `fico_range_low` / `fico_range_high` | FICO score range | Core credit-quality metric (midpoint → `fico`) |
| `earliest_cr_line` | Earliest credit line | Derives credit age (`credit_age_years`) |
| `delinq_2yrs` | Delinquencies in past 2 years | Historical delinquent behavior |
| `inq_last_6mths` | Inquiries in past 6 months | Credit-hunger signal |
| `open_acc` | Open accounts | Credit activity |
| `pub_rec` | Public records | Derogatory records |
| `total_acc` | Total accounts | Breadth of credit history |

### Identity & purpose (4)

| Column | Meaning | Why kept |
|---|---|---|
| `emp_length` | Employment length | Income stability |
| `home_ownership` | Home ownership | Asset status |
| `verification_status` | Income verification | Information credibility |
| `purpose` | Loan purpose | Risk varies by use of funds |

---

## 4. Dropped Columns (130) by Reason

### A. Post-origination information — leakage (38)

The model must answer "will this loan default at origination", so any information only known *after* origination is unusable. Four sub-groups:

**A1. Payment performance (12)**: `out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`
> How much was later repaid / outstanding / recovered — outcomes, not causes.

**A2. Post-origination credit updates (3)**: `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low`
> "last_*" are the latest FICO / pull date, updated after origination — leakage. Only the origination-time `fico_range_*` are used.

**A3. Payment plan (1)**: `pymnt_plan`
> Whether the loan entered a payment plan — only known after origination.

**A4. Hardship & settlement (22)**:
`hardship_flag`, `hardship_type`, `hardship_reason`, `hardship_status`, `hardship_amount`, `hardship_start_date`, `hardship_end_date`, `hardship_length`, `hardship_dpd`, `hardship_loan_status`, `hardship_payoff_balance_amount`, `hardship_last_payment_amount`, `orig_projected_additional_accrued_interest`, `deferral_term`, `payment_plan_start_date`, `debt_settlement_flag`, `debt_settlement_flag_date`, `settlement_status`, `settlement_date`, `settlement_amount`, `settlement_percentage`, `settlement_term`
> Hardship and debt settlement are events that can only occur after the loan is issued.

### B. LC risk grade — circular (2, must drop)

`grade`, `sub_grade`
> Lending Club's own risk grades (A1–G5), derived from the borrower's credit profile. Using them as features is "predicting the answer with the answer".

### C. Identifiers & noise (6, must drop)

`id`, `member_id`, `url`, `desc`, `title`, `policy_code`
> IDs / URL / loan description text carry no predictive value; `policy_code` is nearly constant (all 1).

### D. Redundant / derived (4, must drop)

`funded_amnt`, `funded_amnt_inv`, `installment`, `inq_last_12m`
> `funded_amnt` ≈ `loan_amnt`; `installment` is derived from `loan_amnt`, `int_rate`, `term`; `inq_last_12m` overlaps with the kept `inq_last_6mths`.

### E. High-cardinality / granular (3, excluded to stay focused)

`emp_title`, `zip_code`, `addr_state`
> These do carry signal (job and geography affect default), but `emp_title` is free text and `zip_code` / `addr_state` are high-cardinality. Excluded for now to keep the project focused on core credit features.

### F. Joint / secondary applicant (16, very high missing rate)

`annual_inc_joint`, `dti_joint`, `verification_status_joint`, `revol_bal_joint`, `sec_app_fico_range_low`, `sec_app_fico_range_high`, `sec_app_earliest_cr_line`, `sec_app_inq_last_6mths`, `sec_app_mort_acc`, `sec_app_open_acc`, `sec_app_revol_util`, `sec_app_open_act_il`, `sec_app_num_rev_accts`, `sec_app_chargeoff_within_12_mths`, `sec_app_collections_12_mths_ex_med`, `sec_app_mths_since_last_major_derog`
> Joint- and secondary-applicant information. Most loans are single-applicant, so these columns are nearly all missing and add little modeling value.

### G. Channel / constant (3, low information)

`initial_list_status`, `application_type`, `disbursement_method`
> Process / channel fields with very low default discrimination.

### H. Detailed credit-bureau variables (58, predictive but excluded to stay focused)

`mths_since_last_delinq`, `mths_since_last_record`, `collections_12_mths_ex_med`, `mths_since_last_major_derog`, `acc_now_delinq`, `tot_coll_amt`, `tot_cur_bal`, `open_acc_6m`, `open_act_il`, `open_il_12m`, `open_il_24m`, `mths_since_rcnt_il`, `total_bal_il`, `il_util`, `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`, `total_rev_hi_lim`, `inq_fi`, `total_cu_tl`, `acc_open_past_24mths`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `chargeoff_within_12_mths`, `delinq_amnt`, `mo_sin_old_il_acct`, `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl`, `mort_acc`, `mths_since_recent_bc`, `mths_since_recent_bc_dlq`, `mths_since_recent_inq`, `mths_since_recent_revol_delinq`, `num_accts_ever_120_pd`, `num_actv_bc_tl`, `num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`, `num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_120dpd_2m`, `num_tl_30dpd`, `num_tl_90g_dpd_24m`, `num_tl_op_past_12m`, `pct_tl_nvr_dlq`, `percent_bc_gt_75`, `pub_rec_bankruptcies`, `tax_liens`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `total_bc_limit`, `total_il_high_credit_limit`

> These detailed credit-bureau derivatives (account counts, delinquencies, balances, limits) are predictive. They are excluded not because they are useless, but because:
> 1. There are too many (58) — including them all makes the model complex and hard to interpret;
> 2. Many are highly correlated with the kept core variables (`delinq_2yrs`, `open_acc`, `revol_bal`, etc.), risking collinearity;
> 3. They are candidates to add later as feature extensions.

---

> In this project, `woe.py` then narrows these 18 features down to the ~9 that actually enter the model, via **IV threshold + VIF collinearity** — a second, data-driven selection step, distinct from the domain-level column selection described here.
