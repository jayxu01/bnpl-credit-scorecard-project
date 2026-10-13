"""数据加载与预处理。

目的：把 Lending Club 原始 CSV（约 220 万行 × 151 列）

流程：
1. 只读取需要的 ~20 列（usecols），降低内存占用；
2. 定义好坏样本：bad = Charged Off / Default，good = Fully Paid，
   丢弃 Current / Late 等未到期样本；
3. 按放款月份 issue_d 切分 train / test / oot（时间保序，防泄漏）：
   train <= 2016-12，test = 2017 全年，oot = 2018 全年（样本外，用来算 PSI）；
4. 清洗：百分号/字符串字段转数值，emp_length 转年限，构造 credit_age_years；
5. 输出 train/test/oot 三份 CSV 到 data/processed/。

设计说明：
- 刻意剔除 grade / sub_grade：它们是 Lending Club 自己的风险分桶，
会虚高 KS，且真实业务中放款前拿不到；
- 所有特征都是放款时点已知信息，无时间泄漏；
- 缺失值保留为 NaN，交给下一步 WoE 分箱（缺失值单独成一箱，业界标准做法）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 把 src 目录加进 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (RAW_DIR, PROCESSED_DIR, RAW_CSV, BAD_STATUS, GOOD_STATUS,
                    TRAIN_END, TEST_END, SEED)

# 字符串型列
STR_COLS = [
    "loan_status", "issue_d", "term", "emp_length", "home_ownership",
    "purpose", "verification_status", "earliest_cr_line",
    "int_rate", "revol_util",
]

# 数值型列
NUM_COLS = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "fico_range_low", "fico_range_high",
]

COLS = STR_COLS + NUM_COLS


def load_raw() -> pd.DataFrame:
    """读取原始 CSV，只保留需要的列，并显式指定 dtype 节省内存。"""
    dtype = {c: str for c in STR_COLS}
    dtype.update({c: float for c in NUM_COLS})
    return pd.read_csv(RAW_DIR / RAW_CSV, usecols=COLS, dtype=dtype)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """清洗：定义目标、字符串转数值、构造派生特征。"""
    df = df.copy()

    # 目标：坏 = Charged Off/Default，好 = Fully Paid，其余（Current/Late 等未到期）丢弃
    # 用 contains 而不是 ==，是为了覆盖 "Does not meet the credit policy. Status:XXX" 这类状态
    status = df["loan_status"]
    is_bad = status.str.contains("Charged Off|Default", na=False)
    is_good = status.str.contains("Fully Paid", na=False)
    df["bad"] = np.where(is_bad, 1, np.where(is_good, 0, np.nan))
    df = df.dropna(subset=["bad"]).copy()
    df["bad"] = df["bad"].astype(int)

    # 百分号字段："10.65%" -> 10.65
    df["int_rate"] = pd.to_numeric(
        df["int_rate"].str.strip().str.replace("%", "", regex=False), errors="coerce")
    df["revol_util"] = pd.to_numeric(
        df["revol_util"].str.strip().str.replace("%", "", regex=False), errors="coerce")

    # 期限：" 36 months" -> 36
    df["term"] = pd.to_numeric(df["term"].str.extract(r"(\d+)")[0], errors="coerce")

    # 工作年限：映射为数值（"< 1 year" -> 0.5，"10+ years" -> 10，"n/a" -> NaN）
    emp_map = {
        "< 1 year": 0.5, "1 year": 1.0, "2 years": 2.0, "3 years": 3.0,
        "4 years": 4.0, "5 years": 5.0, "6 years": 6.0, "7 years": 7.0,
        "8 years": 8.0, "9 years": 9.0, "10+ years": 10.0, "n/a": np.nan,
    }
    df["emp_length"] = df["emp_length"].map(emp_map)

    # FICO 取区间中点
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # 信用年龄（年）= 放款时间 - 最早信用记录时间
    issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    earliest_dt = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_age_years"] = (issue_dt - earliest_dt).dt.days / 365.25

    # 用于切分的月份（"YYYY-MM" 字符串，方便比较大小）
    df["issue_ym"] = issue_dt.dt.strftime("%Y-%m")

    # 丢掉不再需要的列
    df = df.drop(columns=["loan_status", "issue_d", "earliest_cr_line",
                          "fico_range_low", "fico_range_high"])
    return df


def split(df: pd.DataFrame):
    """按放款月份切分 train / test / oot，切完丢掉 issue_ym。"""
    train = df[df["issue_ym"] <= TRAIN_END].drop(columns=["issue_ym"])
    test = df[(df["issue_ym"] > TRAIN_END) & (df["issue_ym"] <= TEST_END)].drop(columns=["issue_ym"])
    oot = df[df["issue_ym"] > TEST_END].drop(columns=["issue_ym"])
    return train, test, oot


def main() -> None:
    print("1/3 读取原始数据 ...")
    df = load_raw()
    print(f"    原始读取 {len(df):,} 行 × {df.shape[1]} 列")

    print("2/3 清洗 + 定义好坏样本 ...")
    df = clean(df)
    print(f"    清洗后 {len(df):,} 行，坏样本率 {df['bad'].mean():.2%}")

    print("3/3 按放款月份切分 train / test / oot ...")
    train, test, oot = split(df)
    print(f"    train {len(train):,} | test {len(test):,} | oot {len(oot):,}")
    print(f"    oot 坏样本率 {oot['bad'].mean():.2%}（与 train 对比可观察漂移）")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    oot.to_csv(PROCESSED_DIR / "oot.csv", index=False)
    print(f"已保存 train.csv / test.csv / oot.csv 到 {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
