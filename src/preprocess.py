"""Data loading and preprocessing.

Purpose: clean the raw Lending Club CSV (~2.2M rows x 151 columns) into the
files needed for downstream modeling.

Pipeline:
1. Read only the 21 needed columns (usecols) to reduce memory usage;
2. Define good/bad: bad = Charged Off / Default, good = Fully Paid,
   drop Current / Late (loans without a final outcome);
3. Split by issue_d (loan issue month) into train / test / oot (time-ordered):
   train <= 2016-12, test = 2017, oot = 2018 (out-of-time, used for PSI);
4. Clean: percent/string fields to numeric, emp_length to years, derive credit_age_years;
5. Write train/test/oot CSVs to data/processed/.

Notes:
- Why only 21 of 151 columns are kept: see docs/column_selection.md;
- Missing values are kept as NaN and handled later in WoE binning (missing = its own bin).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src/ to sys.path so config can be imported regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (RAW_DIR, PROCESSED_DIR, RAW_CSV, BAD_STATUS, GOOD_STATUS,
                    TRAIN_END, TEST_END, SEED)

# String columns (need cleaning or are categorical)
STR_COLS = [
    "loan_status", "issue_d", "term", "emp_length", "home_ownership",
    "purpose", "verification_status", "earliest_cr_line",
    "int_rate", "revol_util",
]

# Numeric columns
NUM_COLS = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "fico_range_low", "fico_range_high",
]

COLS = STR_COLS + NUM_COLS


def load_raw() -> pd.DataFrame:
    """Read the raw CSV, keeping only needed columns with explicit dtypes to save memory."""
    dtype = {c: str for c in STR_COLS}
    dtype.update({c: float for c in NUM_COLS})
    return pd.read_csv(RAW_DIR / RAW_CSV, usecols=COLS, dtype=dtype)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean: define target, convert strings to numeric, derive features."""
    df = df.copy()

    # Target: bad = Charged Off/Default, good = Fully Paid, drop the rest (Current/Late etc.)
    status = df["loan_status"]
    is_bad = status.str.contains("Charged Off|Default", na=False)
    is_good = status.str.contains("Fully Paid", na=False)
    df["bad"] = np.where(is_bad, 1, np.where(is_good, 0, np.nan))
    df = df.dropna(subset=["bad"]).copy()
    df["bad"] = df["bad"].astype(int)

    # Percent fields: "10.65%" -> 10.65
    df["int_rate"] = pd.to_numeric(
        df["int_rate"].str.strip().str.replace("%", "", regex=False), errors="coerce")
    df["revol_util"] = pd.to_numeric(
        df["revol_util"].str.strip().str.replace("%", "", regex=False), errors="coerce")

    # Term: " 36 months" -> 36
    df["term"] = pd.to_numeric(df["term"].str.extract(r"(\d+)")[0], errors="coerce")

    # Employment length: map to years ("< 1 year" -> 0.5, "10+ years" -> 10, "n/a" -> NaN)
    emp_map = {
        "< 1 year": 0.5, "1 year": 1.0, "2 years": 2.0, "3 years": 3.0,
        "4 years": 4.0, "5 years": 5.0, "6 years": 6.0, "7 years": 7.0,
        "8 years": 8.0, "9 years": 9.0, "10+ years": 10.0, "n/a": np.nan,
    }
    df["emp_length"] = df["emp_length"].map(emp_map)

    # FICO: midpoint of the range
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # Credit age (years) = loan issue date - earliest credit line date
    issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    earliest_dt = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_age_years"] = (issue_dt - earliest_dt).dt.days / 365.25

    # Issue month ("YYYY-MM" string) used for splitting
    df["issue_ym"] = issue_dt.dt.strftime("%Y-%m")

    # Drop columns no longer needed
    df = df.drop(columns=["loan_status", "issue_d", "earliest_cr_line",
                          "fico_range_low", "fico_range_high"])
    return df


def split(df: pd.DataFrame):
    """Split into train / test / oot by issue month, dropping issue_ym afterwards."""
    train = df[df["issue_ym"] <= TRAIN_END].drop(columns=["issue_ym"])
    test = df[(df["issue_ym"] > TRAIN_END) & (df["issue_ym"] <= TEST_END)].drop(columns=["issue_ym"])
    oot = df[df["issue_ym"] > TEST_END].drop(columns=["issue_ym"])
    return train, test, oot


def main() -> None:
    print("1/3 Reading raw data ...")
    df = load_raw()
    print(f"    raw: {len(df):,} rows x {df.shape[1]} columns")

    print("2/3 Cleaning + defining good/bad ...")
    df = clean(df)
    print(f"    cleaned: {len(df):,} rows, bad rate {df['bad'].mean():.2%}")

    print("3/3 Splitting by issue month into train / test / oot ...")
    train, test, oot = split(df)
    print(f"    train {len(train):,} | test {len(test):,} | oot {len(oot):,}")
    print(f"    oot bad rate {oot['bad'].mean():.2%}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    oot.to_csv(PROCESSED_DIR / "oot.csv", index=False)
    print(f"Saved train.csv / test.csv / oot.csv to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
