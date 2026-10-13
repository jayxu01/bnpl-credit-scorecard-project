"""WoE/IV binning and feature selection.

Steps:
1. Bin continuous variables via chi-square merging, categorical variables by
   category (rare categories merged into OTHER);
2. Compute WoE per bin and IV per variable;
3. Check WoE monotonicity for continuous variables;
4. Select variables by IV threshold, then drop collinear ones by VIF;
5. Output: WoE table (CSV), selected feature list (JSON), and the fitted binner (pickle).
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_DIR, WOE_DIR, IV_MIN, VIF_MAX

EPS = 0.5  # Laplace smoothing to avoid log(0) when a bin has zero good/bad


class WoEBinner:
    """Bin each feature and encode it into WoE; supports fit / transform."""

    def __init__(self, max_bins: int = 5):
        self.max_bins = max_bins
        self._rules = {}      # feature -> ("interval", edges) or ("categorical", kept_categories)
        self._bin_order = {}  # feature -> ordered interval labels (for monotonicity check)
        self.woe_ = {}        # feature -> {bin label -> WoE}
        self.iv_ = {}         # feature -> IV
        self.stats_ = {}      # feature -> per-bin stats DataFrame

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        total_bad = y.sum()
        total_good = len(y) - total_bad
        for col in X.columns:
            x = X[col]
            if self._is_continuous(x):
                edges = self._chi_merge_edges(x, y)
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

    def _is_continuous(self, x: pd.Series) -> bool:
        """Treat as continuous only if numeric with more unique values than max_bins."""
        return pd.api.types.is_numeric_dtype(x) and x.nunique() > self.max_bins

    def _chi_merge_edges(self, x: pd.Series, y: pd.Series) -> list:
        """Chi-square binning: merge adjacent bins with the most similar good/bad
        distribution until the number of bins drops to max_bins.

        A smaller chi-square between two adjacent bins means their distributions
        are more similar, so they should be merged first.
        """
        valid = ~x.isna()
        xv = x[valid].to_numpy()
        yv = y[valid].to_numpy()

        n_unique = len(np.unique(xv))
        if n_unique <= self.max_bins:
            return sorted(np.unique(xv))

        # Initial fine bins: equal-frequency, at most 100, as the merging start point
        n_init = min(100, n_unique)
        q_edges = np.unique(np.quantile(xv, np.linspace(0, 1, n_init + 1)))

        binned = pd.cut(xv, bins=q_edges, include_lowest=True, duplicates="drop")
        frame = pd.DataFrame({"bin": binned, "y": yv})
        agg = frame.groupby("bin", observed=True)["y"].agg(["sum", "count"])
        bads = agg["sum"].to_numpy().astype(float)
        goods = (agg["count"] - agg["sum"]).to_numpy().astype(float)
        rights = np.array([iv.right for iv in agg.index], dtype=float)  # right edge of each bin

        while len(bads) > self.max_bins:
            chi2s = np.empty(len(bads) - 1)
            for i in range(len(bads) - 1):
                a, b = bads[i], goods[i]
                c, d = bads[i + 1], goods[i + 1]
                n = a + b + c + d
                denom = (a + b) * (c + d) * (a + c) * (b + d)
                chi2s[i] = n * (a * d - b * c) ** 2 / denom if denom > 0 else 0.0
            i = int(np.argmin(chi2s))
            bads[i] += bads[i + 1]
            goods[i] += goods[i + 1]
            rights[i] = rights[i + 1]  # merged bin takes the right edge of the right bin
            bads = np.delete(bads, i + 1)
            goods = np.delete(goods, i + 1)
            rights = np.delete(rights, i + 1)

        # Edges: [-inf, r1, ..., r_{k-1}, inf]; the last bin extends to inf to cover future data
        return [-np.inf] + list(rights[:-1]) + [np.inf]

    def _interval_keys(self, x: pd.Series, edges: list):
        """Map a continuous variable to bin labels; NaN goes to the MISSING bin."""
        binned = pd.cut(x, bins=edges, include_lowest=True, duplicates="drop")
        order = [str(c) for c in binned.cat.categories]
        codes = binned.cat.codes
        keys = pd.Series([order[c] if c != -1 else "MISSING" for c in codes], index=x.index)
        return keys, order

    def _kept_categories(self, x: pd.Series) -> list:
        """Keep categories with >= 1% frequency; rarer ones are merged into OTHER."""
        counts = x.astype("string").value_counts(dropna=False)
        total = len(x)
        return [str(c) for c in counts.index if pd.notna(c) and counts[c] >= total * 0.01]

    def _categorical_keys(self, x: pd.Series, kept: list) -> pd.Series:
        def to_key(v):
            if pd.isna(v):
                return "MISSING"
            return v if str(v) in kept else "OTHER"
        return x.map(to_key)

    def _woe_iv(self, keys: pd.Series, y: pd.Series, total_bad, total_good):
        frame = pd.DataFrame({"bin": keys, "y": y})
        woe, rows, iv_total = {}, [], 0.0
        for k, grp in frame.groupby("bin"):
            n = len(grp)
            bad = grp["y"].sum()
            good = n - bad
            w = np.log((bad + EPS) / (total_bad + EPS)) - np.log((good + EPS) / (total_good + EPS))
            iv_bin = (bad / total_bad - good / total_good) * w
            woe[k] = w
            iv_total += iv_bin
            rows.append({"bin": k, "n": n, "bad": int(bad), "bad_rate": bad / n,
                         "woe": w, "iv": iv_bin})
        return woe, iv_total, pd.DataFrame(rows)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode raw features into WoE; unseen bins map to 0 (neutral)."""
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
        """Check whether each continuous feature's WoE is monotonic (False for categorical)."""
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
    """VIF_j = 1 / (1 - R_j^2), computed via the diagonal of the inverse correlation matrix."""
    A = Xw.to_numpy()
    means, stds = A.mean(axis=0), A.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (A - means) / stds
    R_inv = np.linalg.pinv(np.corrcoef(Z, rowvar=False))
    vif = np.nan_to_num(np.diag(R_inv), nan=999.0, posinf=999.0, neginf=999.0)
    return pd.Series(vif, index=Xw.columns)


def _bin_labels(edges: list) -> list:
    """Turn interval edges into short labels, e.g. [-inf, 12, 18, inf] -> ['<= 12', '12~18', '> 18']."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isneginf(lo):
            labels.append(f"<= {hi:g}")
        elif np.isposinf(hi):
            labels.append(f"> {lo:g}")
        else:
            labels.append(f"{lo:g}~{hi:g}")
    return labels


def plot_woe(binner: "WoEBinner", features: list, save_path: Path) -> None:
    """Plot WoE per bin for each feature and save to PNG (red = WoE > 0, blue = WoE < 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = [f for f in features if f in binner.woe_]
    n = len(features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
    axes = np.asarray(axes).ravel()

    for i, feat in enumerate(features):
        ax = axes[i]
        kind, rule = binner._rules[feat]
        woe = binner.woe_[feat]
        if kind == "interval":
            labels = _bin_labels(rule)
            values = [woe.get(k, 0.0) for k in binner._bin_order[feat]]
            if "MISSING" in woe:
                labels.append("Missing")
                values.append(woe["MISSING"])
        else:
            keys = sorted([k for k in woe if k not in ("MISSING", "OTHER")],
                          key=lambda k: -woe[k])
            keys += [k for k in ("MISSING", "OTHER") if k in woe]
            labels, values = keys, [woe[k] for k in keys]

        colors = ["#d64541" if v > 0 else "#4a86c8" for v in values]
        ax.bar(range(len(values)), values, color=colors)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{feat}  (IV={binner.iv_[feat]:.3f})", fontsize=9)
        ax.set_ylabel("WoE", fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved WoE plot -> {save_path}")


def main() -> None:
    print("Reading train data ...")
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    X = train.drop(columns=["bad"])
    y = train["bad"]
    print(f"    {len(X):,} rows x {X.shape[1]} features")

    print("Binning + computing WoE/IV ...")
    binner = WoEBinner(max_bins=5).fit(X, y)
    Xw = binner.transform(X)

    iv = pd.Series(binner.iv_).sort_values(ascending=False)
    vif = calc_vif(Xw)
    mono = binner.monotonic()

    summary = pd.DataFrame({"IV": iv, "VIF": vif.reindex(iv.index),
                            "monotonic": mono.reindex(iv.index)})
    summary["selected"] = (summary["IV"] >= IV_MIN) & (summary["VIF"] < VIF_MAX)

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n===== Feature selection summary (sorted by IV) =====")
    print(summary)
    print(f"\nIV threshold {IV_MIN}, VIF threshold {VIF_MAX}")
    selected = summary.index[summary["selected"].to_numpy()].tolist()
    print(f"Selected features ({len(selected)}): {selected}")

    WOE_DIR.mkdir(parents=True, exist_ok=True)
    with open(WOE_DIR / "woe_binner.pkl", "wb") as f:
        pickle.dump(binner, f)
    with open(WOE_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    woe_table = pd.concat([binner.stats_[c] for c in summary.index])
    woe_table.to_csv(WOE_DIR / "woe_table.csv", index=False)

    plot_woe(binner, selected, WOE_DIR / "woe_plot.png")

    print(f"\nSaved woe_binner.pkl / selected_features.json / woe_table.csv / woe_plot.png to {WOE_DIR}")


if __name__ == "__main__":
    main()
