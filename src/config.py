"""项目全局配置：路径、常量、模型参数集中管理。

把散落在各处的"魔法数字"（magic numbers）收敛到一个文件，
是工程化的第一步——改参数时只动这里，不碰业务代码。
"""
from pathlib import Path

# 项目根目录（config.py 的上一级）
ROOT = Path(__file__).resolve().parent.parent

# 目录
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              
PROCESSED_DIR = DATA_DIR / "processed"  # 清洗切分后的 train/test/oot
OUTPUT_DIR = ROOT / "output"            # 生成的图、报告、评分卡表
WOE_DIR = OUTPUT_DIR / "woe"            # WoE/IV 分箱输出
SCORECARD_DIR = OUTPUT_DIR / "scorecard"  # 评分卡输出
VALIDATE_DIR = OUTPUT_DIR / "validate"    # 验证输出

# 原始数据文件名
RAW_CSV = "lending_club.csv"

# 随机种子
SEED = 42

# 目标定义：哪些 loan_status 算"坏"，哪些算"好"
# 其余状态（Current / Late 等未到期样本）在预处理里丢弃
BAD_STATUS = ["Charged Off", "Default"]
GOOD_STATUS = ["Fully Paid"]

# 时间切分：按放款月份 issue_d（格式 "YYYY-MM"）切三份
#   train: <= TRAIN_END
#   test : TRAIN_END < issue <= TEST_END
#   oot  : > TEST_END（样本外，用来算 PSI）
# 注意：数据止于 2018Q4，所以用 2018 全年当 oot，train/test 往前推
TRAIN_END = "2016-12"
TEST_END = "2017-12"

# 评分卡参数（业界标准：base 600 分，每 20 分 odds 翻倍）
BASE_SCORE = 600
PDO = 20

# 变量筛选阈值
IV_MIN = 0.02   # IV 低于该值的变量剔除
VIF_MAX = 5.0   # VIF 高于该值的变量剔除（共线性）
