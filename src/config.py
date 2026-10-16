"""Global configuration: paths, constants, and model parameters in one place.

Centralizing magic numbers into a single file is the first step of good
engineering — change parameters here without touching business logic.
"""
from pathlib import Path

# Project root (one level above config.py)
ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"    # cleaned train/test/oot
OUTPUT_DIR = ROOT / "output"
WOE_DIR = OUTPUT_DIR / "woe"              # WoE/IV binning outputs
SCORECARD_DIR = OUTPUT_DIR / "scorecard"  # scorecard outputs
VALIDATE_DIR = OUTPUT_DIR / "validate"    # validation outputs

# Raw data filename
RAW_CSV = "lending_club.csv"

# Random seed
SEED = 42

# Target definition: which loan_status counts as bad / good.
# Other statuses (Current / Late, i.e. loans without a final outcome) are dropped.
BAD_STATUS = ["Charged Off", "Default"]
GOOD_STATUS = ["Fully Paid"]

# Time split by issue month (format "YYYY-MM"):
#   train: <= TRAIN_END
#   test : TRAIN_END < issue <= TEST_END
#   oot  : > TEST_END (out-of-time, used for PSI)
# Data ends at 2018Q4, so 2018 is used as OOT and train/test are earlier.
TRAIN_END = "2016-12"
TEST_END = "2017-12"

# Scorecard parameters (industry standard: base 600, odds double every 20 points)
BASE_SCORE = 600
PDO = 20
BASE_ODDS = 50  # base odds (good:bad); base score corresponds to 50:1, i.e. ~2% bad rate

# Feature selection thresholds
IV_MIN = 0.02  # drop features with IV below this
VIF_MAX = 5.0  # drop features with VIF above this (collinearity)
