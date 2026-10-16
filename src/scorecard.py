"""Logistic-regression scorecard with PDO scaling.

Steps:
1. Load train, the fitted binner, and the selected feature list;
2. Transform train/test/oot features into WoE values;
3. Fit logistic regression on train's WoE features (unregularized, scorecard convention);
4. PDO scaling: base 600, odds double every 20 points;
5. Output the scorecard table (points per feature per bin) and scores for all three sets.
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
from woe import WoEBinner  # needed to unpickle the fitted binner


def load_artifacts():
    """Load train, the fitted binner, and the selected feature list."""
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    with open(WOE_DIR / "woe_binner.pkl", "rb") as f:
        binner = pickle.load(f)
    with open(WOE_DIR / "selected_features.json", "r", encoding="utf-8") as f:
        selected = json.load(f)
    return train, binner, selected


def _ordered_bins(binner, feat) -> list:
    """Return a feature's bins in a meaningful order (intervals for continuous, WoE-desc for categorical)."""
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

    # Unregularized logistic regression
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, random_state=SEED)
    lr.fit(Xw, y)

    # PDO scaling: score = base_score + factor * ln(odds_good / BASE_ODDS)
    factor = PDO / np.log(2)
    beta0 = lr.intercept_[0]
    coefs = dict(zip(selected, lr.coef_[0]))

    # Scorecard table: base points + points per bin
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

    # Print scorecard table
    print(f"===== Scorecard (base points = {base_points:.1f}) =====")
    for feat in selected:
        sub = card[card["feature"] == feat]
        print(f"\n[{feat}]  coef={coefs[feat]:.4f}")
        for _, r in sub.iterrows():
            print(f"    {r['bin']:<26} -> {r['points']:>7.2f}")

    # Score the three datasets
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["train", "test", "oot"]:
        df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
        log_odds = lr.decision_function(binner.transform(df[selected]))
        df["score"] = BASE_SCORE - factor * (log_odds + np.log(BASE_ODDS))
        df["prob"] = 1 / (1 + np.exp(-log_odds))
        df[["score", "prob", "bad"]].to_csv(SCORECARD_DIR / f"{name}_scores.csv", index=False)
        print(f"\n{name}: {len(df):,} samples, mean score {df['score'].mean():.1f}, "
              f"range [{df['score'].min():.0f}, {df['score'].max():.0f}]")

    card.to_csv(SCORECARD_DIR / "scorecard.csv", index=False)
    with open(SCORECARD_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(SCORECARD_DIR / "scorecard_meta.json", "w", encoding="utf-8") as f:
        json.dump({"base_points": base_points, "factor": factor,
                   "pdo": PDO, "base_score": BASE_SCORE}, f, indent=2)

    print(f"\nSaved scorecard table / model / scores to {SCORECARD_DIR}")


if __name__ == "__main__":
    main()
