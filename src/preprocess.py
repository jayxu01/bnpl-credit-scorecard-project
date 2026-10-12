"""数据加载与预处理。

职责：把 Lending Club 原始 CSV 变成可建模的三份干净数据。

流程：
1. 读取 data/raw/ 下的原始数据，只保留需要的列；
2. 定义好坏样本：bad = Charged Off / Default，good = Fully Paid，
   丢弃 Current / Late 等未到期样本；
3. 按放款月份 issue_d 切分 train / test / oot（时间保序，防泄漏）；
4. 缺失值处理、异常值 winsorize、类别变量清洗；
5. 输出 train/test/oot 三份 CSV 到 data/processed/。
"""
