"""XGBoost challenger model.

Trains XGBoost on the raw values of the selected features and compares its
KS / AUC against the logistic-regression scorecard. The scorecard is the
"champion" (interpretable, regulator-friendly); XGBoost is the "challenger"
(non-linear, potentially stronger).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_DIR, WOE_DIR, SCORECARD_DIR, SEED

CAT_COLS = ["home_ownership", "verification_status"]


def ks_stat(y_true, y_score) -> float:
    """KS statistic: max gap between cumulative good/bad distributions when sorted by score."""
    order = np.argsort(-np.asarray(y_score))
    y = np.asarray(y_true)[order]
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - int(y.sum()), 1)
    return float(np.max(np.abs(np.cumsum(y) / n_pos - np.cumsum(1 - y) / n_neg)))


def eval_ks_auc(y_true, y_score):
    return ks_stat(y_true, y_score), roc_auc_score(y_true, y_score)


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    oot = pd.read_csv(PROCESSED_DIR / "oot.csv")
    with open(WOE_DIR / "selected_features.json") as f:
        selected = json.load(f)

    # Label-encode categorical columns (mapping fitted on train, applied to all three sets)
    for c in CAT_COLS:
        if c not in selected:
            continue
        cats = sorted(train[c].dropna().unique())
        mapping = {v: i for i, v in enumerate(cats)}
        train[c] = train[c].map(mapping)
        test[c] = test[c].map(mapping)
        oot[c] = oot[c].map(mapping)

    Xtr, ytr = train[selected], train["bad"]
    Xte, yte = test[selected], test["bad"]
    Xoo, yoo = oot[selected], oot["bad"]

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\n===== XGBoost challenger =====")
    for name, X, y in [("train", Xtr, ytr), ("test", Xte, yte), ("oot", Xoo, yoo)]:
        prob = model.predict_proba(X)[:, 1]
        ks, auc = eval_ks_auc(y, prob)
        rows.append({"dataset": name, "model": "xgb", "ks": round(ks, 4), "auc": round(auc, 4)})
        print(f"    {name}: KS={ks:.4f}, AUC={auc:.4f}")
        pd.DataFrame({"prob": prob, "bad": y}).to_csv(SCORECARD_DIR / f"{name}_xgb.csv", index=False)

    print("\n===== Scorecard (logistic regression) =====")
    for name in ["train", "test", "oot"]:
        s = pd.read_csv(SCORECARD_DIR / f"{name}_scores.csv")
        ks, auc = eval_ks_auc(s["bad"], -s["score"])  # higher score = lower risk; negate for AUC
        rows.append({"dataset": name, "model": "scorecard", "ks": round(ks, 4), "auc": round(auc, 4)})
        print(f"    {name}: KS={ks:.4f}, AUC={auc:.4f}")

    with open(SCORECARD_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    pd.DataFrame(rows).to_csv(SCORECARD_DIR / "model_comparison.csv", index=False)
    print(f"\nComparison saved to {SCORECARD_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
