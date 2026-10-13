"""WoE/IV 特征分箱与变量筛选（评分卡核心模块）。

职责：
1. 连续变量用决策树自动找最优切分点分箱，类别变量按类别分箱（稀有类别合并）；
2. 计算每箱 WoE 与整体 IV；
3. 检查连续变量 WoE 单调性（业务可解释性）；
4. 按 IV 阈值筛选变量，再按 VIF 剔除共线性；
5. 输出：WoE 分箱表（CSV）、筛选后的变量清单（JSON）、供 scorecard 复用的分箱器（pickle）。

核心概念（面试必考）：
- 分箱：把连续变量切成几段，每段叫一个"箱"；
- WoE = ln(箱内坏占比 / 箱内好占比)，>0 风险偏高、<0 风险偏低；
- IV = 各箱 WoE 的加权和，衡量变量预测力（<0.02 无用，0.1~0.3 中等，>0.5 可疑）；
- 用 WoE 而非原始值：把非线性关系变单调、天然处理缺失/异常、编码后直接进 LR 且可解释。
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_DIR, OUTPUT_DIR, SEED, IV_MIN, VIF_MAX

EPS = 0.5  # Laplace 平滑，避免某箱好/坏样本为 0 时出现 log(0)


class WoEBinner:
    """分箱 + WoE 编码器，支持 fit（在 train 上学习）和 transform（应用到任意数据）。"""

    def __init__(self, max_bins: int = 5, min_leaf_ratio: float = 0.02):
        self.max_bins = max_bins
        self.min_leaf_ratio = min_leaf_ratio
        self._rules = {}      # 特征 -> ("interval", edges) 或 ("categorical", kept_categories)
        self._bin_order = {}  # 特征 -> 连续分箱的区间顺序（用于单调性检查）
        self.woe_ = {}        # 特征 -> {箱 -> WoE}
        self.iv_ = {}         # 特征 -> IV
        self.stats_ = {}      # 特征 -> 分箱明细 DataFrame

    # ---------------- fit ----------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        total_bad = y.sum()
        total_good = len(y) - total_bad
        for col in X.columns:
            x = X[col]
            if self._is_continuous(x):
                edges = self._tree_edges(x, y)
                keys, order = self._interval_keys(x, edges)
                self._rules[col] = ("interval", edges)
                self._bin_order[col] = order
            else:
                kept = self._kept_categories(x)
                keys = self._categorical_keys(x, kept)
                self._rules[col] = ("categorical", kept)
            woe, iv, stats = self._woe_iv(keys, y, total_bad, total_good)
            self.woe_[col] = woe
            self.iv_[col] = iv
            stats["feature"] = col
            self.stats_[col] = stats
        return self

    # ---------------- 分箱辅助 ----------------
    def _is_continuous(self, x: pd.Series) -> bool:
        """数值型且取值较多（> max_bins）才按连续变量分箱，否则按类别处理。"""
        return pd.api.types.is_numeric_dtype(x) and x.nunique() > self.max_bins

    def _tree_edges(self, x: pd.Series, y: pd.Series) -> list:
        """决策树找切分点，返回 [-inf, 阈值..., inf] 的边界列表。"""
        valid = ~x.isna()
        xv, yv = x[valid], y[valid]
        if len(xv) < self.max_bins * 5:
            # 有效样本太少，用等频分箱兜底
            qs = np.unique(xv.quantile(np.linspace(0, 1, self.max_bins + 1)).to_numpy())
            return [-np.inf] + list(qs[1:-1]) + [np.inf]
        clf = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins,
            min_samples_leaf=max(int(len(yv) * self.min_leaf_ratio), 1),
            random_state=SEED,
        )
        clf.fit(xv.to_numpy().reshape(-1, 1), yv.to_numpy())
        thresholds = sorted(set(clf.tree_.threshold[clf.tree_.threshold > 0]))
        return [-np.inf] + thresholds + [np.inf]

    def _interval_keys(self, x: pd.Series, edges: list):
        """把连续变量映射成箱标签，NaN 归入 MISSING 箱。"""
        binned = pd.cut(x, bins=edges, include_lowest=True, duplicates="drop")
        order = [str(c) for c in binned.cat.categories]
        codes = binned.cat.codes
        keys = pd.Series([order[c] if c != -1 else "MISSING" for c in codes], index=x.index)
        return keys, order

    def _kept_categories(self, x: pd.Series) -> list:
        """类别变量：占比 <1% 的稀有类别合并进 OTHER，返回保留类别。"""
        counts = x.astype("string").value_counts(dropna=False)
        total = len(x)
        return [str(c) for c in counts.index if pd.notna(c) and counts[c] >= total * 0.01]

    def _categorical_keys(self, x: pd.Series, kept: list) -> pd.Series:
        def to_key(v):
            if pd.isna(v):
                return "MISSING"
            return v if str(v) in kept else "OTHER"
        return x.map(to_key)

    # ---------------- WoE / IV ----------------
    def _woe_iv(self, keys: pd.Series, y: pd.Series, total_bad, total_good):
        frame = pd.DataFrame({"bin": keys, "y": y})
        woe, rows, iv_total = {}, [], 0.0
        for k, grp in frame.groupby("bin"):
            n = len(grp)
            bad = grp["y"].sum()
            good = n - bad
            # WoE = ln( (bad+平滑)/(总坏+平滑) ) - ln( (good+平滑)/(总好+平滑) )
            w = np.log((bad + EPS) / (total_bad + EPS)) - np.log((good + EPS) / (total_good + EPS))
            iv_bin = (bad / total_bad - good / total_good) * w
            woe[k] = w
            iv_total += iv_bin
            rows.append({"bin": k, "n": n, "bad": int(bad), "bad_rate": bad / n,
                         "woe": w, "iv": iv_bin})
        return woe, iv_total, pd.DataFrame(rows)

    # ---------------- transform ----------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """把原始特征映射成 WoE 值（未见过的箱给 0，即中性）。"""
        Xw = pd.DataFrame(index=X.index)
        for col in X.columns:
            kind, rule = self._rules[col]
            if kind == "interval":
                keys, _ = self._interval_keys(X[col], rule)
            else:
                keys = self._categorical_keys(X[col], rule)
            Xw[col] = keys.map(self.woe_[col]).fillna(0.0)
        return Xw

    def monotonic(self) -> pd.Series:
        """检查连续变量 WoE 是否单调（类别变量返回 False）。"""
        res = {}
        for col, (kind, _) in self._rules.items():
            if kind != "interval":
                res[col] = False
                continue
            w = [self.woe_[col].get(k, 0.0) for k in self._bin_order[col]]
            diffs = np.diff(w)
            res[col] = bool((diffs >= -1e-9).all() or (diffs <= 1e-9).all())
        return pd.Series(res)


def calc_vif(Xw: pd.DataFrame) -> pd.Series:
    """VIF 多重共线性：VIF_j = 1/(1-R_j^2)，>5 说明该特征能被其他特征线性解释。

    用相关系数矩阵求逆的对角线快速计算（先把各列标准化）。
    """
    A = Xw.to_numpy()
    means, stds = A.mean(axis=0), A.std(axis=0)
    stds[stds == 0] = 1.0  # 防常数列
    Z = (A - means) / stds
    R_inv = np.linalg.pinv(np.corrcoef(Z, rowvar=False))
    vif = np.nan_to_num(np.diag(R_inv), nan=999.0, posinf=999.0, neginf=999.0)
    return pd.Series(vif, index=Xw.columns)


def main() -> None:
    print("读取 train 数据 ...")
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    X = train.drop(columns=["bad"])
    y = train["bad"]
    print(f"   {len(X):,} 行 × {X.shape[1]} 个特征")

    print("分箱 + 计算 WoE/IV ...")
    binner = WoEBinner(max_bins=5, min_leaf_ratio=0.02).fit(X, y)
    Xw = binner.transform(X)

    iv = pd.Series(binner.iv_).sort_values(ascending=False)
    vif = calc_vif(Xw)
    mono = binner.monotonic()

    summary = pd.DataFrame({"IV": iv, "VIF": vif.reindex(iv.index), "单调": mono.reindex(iv.index)})
    summary["保留"] = (summary["IV"] >= IV_MIN) & (summary["VIF"] < VIF_MAX)

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n===== 变量筛选汇总（按 IV 降序）=====")
    print(summary)
    print(f"\nIV 阈值 {IV_MIN}，VIF 阈值 {VIF_MAX}")
    selected = summary.index[summary["保留"].to_numpy()].tolist()
    print(f"保留变量（{len(selected)} 个）：{selected}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "woe_binner.pkl", "wb") as f:
        pickle.dump(binner, f)
    with open(OUTPUT_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    woe_table = pd.concat([binner.stats_[c] for c in summary.index])
    woe_table.to_csv(OUTPUT_DIR / "woe_table.csv", index=False)
    print(f"\n已保存 woe_binner.pkl / selected_features.json / woe_table.csv 到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
