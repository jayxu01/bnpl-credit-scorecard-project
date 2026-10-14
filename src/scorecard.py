"""逻辑回归评分卡与 PDO 缩放。

流程：
1. 读入 train、分箱器、保留变量清单；
2. 把 train/test/oot 的特征变换成 WoE 值；
3. 在 train 的 WoE 特征上拟合逻辑回归（无正则，评分卡惯例）；
4. PDO 缩放：base 600 分，每 20 分 odds 翻倍；
5. 输出评分卡表（每个变量的每个分箱对应多少分）和三份数据的分数。
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (PROCESSED_DIR, WOE_DIR, SCORECARD_DIR, BASE_SCORE, PDO, BASE_ODDS, SEED)
from woe import WoEBinner  # 反序列化分箱器时需要该类可用


def load_artifacts():
    """读入 train、分箱器和保留变量清单。"""
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    with open(WOE_DIR / "woe_binner.pkl", "rb") as f:
        binner = pickle.load(f)
    with open(WOE_DIR / "selected_features.json", "r", encoding="utf-8") as f:
        selected = json.load(f)
    return train, binner, selected


def _ordered_bins(binner, feat) -> list:
    """按有意义的顺序返回一个特征的分箱标签（连续按区间，类别按 WoE 降序）。"""
    kind, _ = binner._rules[feat]
    woe = binner.woe_[feat]
    if kind == "interval":
        order = list(binner._bin_order[feat])
        if "MISSING" in woe:
            order.append("MISSING")
        return order
    keys = sorted([k for k in woe if k not in ("MISSING", "OTHER")], key=lambda k: -woe[k])
    keys += [k for k in ("MISSING", "OTHER") if k in woe]
    return keys


def main() -> None:
    train, binner, selected = load_artifacts()

    X = train[selected]
    y = train["bad"]
    Xw = binner.transform(X)

    # 逻辑回归（无正则）
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, random_state=SEED)
    lr.fit(Xw, y)

    # PDO 缩放：score = base_score + factor * ln(odds_good / BASE_ODDS)
    factor = PDO / np.log(2)
    beta0 = lr.intercept_[0]
    coefs = dict(zip(selected, lr.coef_[0]))

    # 评分卡表：base points + 每个分箱的分数
    base_points = BASE_SCORE - factor * beta0 - factor * np.log(BASE_ODDS)
    rows = []
    for feat in selected:
        beta = coefs[feat]
        for bin_label in _ordered_bins(binner, feat):
            woe = binner.woe_[feat][bin_label]
            rows.append({
                "feature": feat,
                "bin": bin_label,
                "woe": round(woe, 4),
                "points": round(-factor * beta * woe, 2),
            })
    card = pd.DataFrame(rows)

    # 打印评分卡表
    print(f"===== 评分卡表（base points = {base_points:.1f}）=====")
    for feat in selected:
        sub = card[card["feature"] == feat]
        print(f"\n[{feat}]  coef={coefs[feat]:.4f}")
        for _, r in sub.iterrows():
            print(f"    {r['bin']:<26} -> {r['points']:>7.2f} 分")

    # 计算并保存三份数据的分数
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["train", "test", "oot"]:
        df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
        log_odds = lr.decision_function(binner.transform(df[selected]))
        df["score"] = BASE_SCORE - factor * (log_odds + np.log(BASE_ODDS))
        df["prob"] = 1 / (1 + np.exp(-log_odds))
        df[["score", "prob", "bad"]].to_csv(SCORECARD_DIR / f"{name}_scores.csv", index=False)
        print(f"\n{name}: 样本 {len(df):,}，分数均值 {df['score'].mean():.1f}，"
              f"分数范围 [{df['score'].min():.0f}, {df['score'].max():.0f}]")

    card.to_csv(SCORECARD_DIR / "scorecard.csv", index=False)
    with open(SCORECARD_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(SCORECARD_DIR / "scorecard_meta.json", "w", encoding="utf-8") as f:
        json.dump({"base_points": base_points, "factor": factor,
                   "pdo": PDO, "base_score": BASE_SCORE}, f, indent=2)

    print(f"\n评分卡表 / 模型 / 分数已保存到 {SCORECARD_DIR}")


if __name__ == "__main__":
    main()
