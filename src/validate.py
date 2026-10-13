"""模型验证与稳定性监控。

计算 KS / AUC / Gini、PSI（train vs oot）、校准曲线，并画出评估图：
ROC、KS 曲线、分数分布、校准图，保存到 output/validate/。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SCORECARD_DIR, VALIDATE_DIR

EPS = 1e-4


def ks_stat(y_true, y_score) -> float:
    """KS 统计量。"""
    order = np.argsort(-np.asarray(y_score))
    y = np.asarray(y_true)[order]
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - int(y.sum()), 1)
    return float(np.max(np.abs(np.cumsum(y) / n_pos - np.cumsum(1 - y) / n_neg)))


def gini(y_true, y_score) -> float:
    """Gini = 2 * AUC - 1。"""
    return 2 * roc_auc_score(y_true, y_score) - 1


def psi(expected, actual, bins=10) -> float:
    """PSI：用 expected 的分位点切箱，比较两分布占比。"""
    breaks = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breaks[0], breaks[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, bins=breaks)[0] / len(expected), EPS, None)
    a = np.clip(np.histogram(actual, bins=breaks)[0] / len(actual), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def load_data():
    sc = {n: pd.read_csv(SCORECARD_DIR / f"{n}_scores.csv") for n in ["train", "test", "oot"]}
    xgb = {n: pd.read_csv(SCORECARD_DIR / f"{n}_xgb.csv") for n in ["train", "test", "oot"]}
    with open(SCORECARD_DIR / "scorecard_meta.json") as f:
        meta = json.load(f)
    return sc, xgb, meta


def score_to_prob(score, meta) -> np.ndarray:
    """把分数反解回违约概率：prob = sigmoid((base_score - score) / factor)。"""
    return 1 / (1 + np.exp(-(meta["base_score"] - score) / meta["factor"]))


def plot_roc(sc, xgb, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(sc["test"]["bad"], -sc["test"]["score"])
    ax.plot(fpr, tpr, label=f"Scorecard (AUC={roc_auc_score(sc['test']['bad'], -sc['test']['score']):.3f})")
    fpr2, tpr2, _ = roc_curve(xgb["test"]["bad"], xgb["test"]["prob"])
    ax.plot(fpr2, tpr2, label=f"XGBoost (AUC={roc_auc_score(xgb['test']['bad'], xgb['test']['prob']):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ks_curve(sc, xgb, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_score, label, c in [
        ("scorecard", -sc["test"]["score"], "Scorecard", "#4a86c8"),
        ("xgb", xgb["test"]["prob"], "XGBoost", "#d64541"),
    ]:
        order = np.argsort(-y_score)
        y = sc["test"]["bad"].to_numpy()[order]
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        x = np.arange(len(y)) / len(y)
        ax.plot(x, np.cumsum(y) / n_pos, color=c, label=f"{label} bad")
        ax.plot(x, np.cumsum(1 - y) / n_neg, color=c, linestyle="--", label=f"{label} good")
    ax.set_xlabel("Score percentile")
    ax.set_ylabel("Cumulative ratio")
    ax.set_title("KS curve (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(sc, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    df = sc["test"]
    ax.hist(df.loc[df["bad"] == 0, "score"], bins=40, alpha=0.6, color="#4a86c8", label="Good")
    ax.hist(df.loc[df["bad"] == 1, "score"], bins=40, alpha=0.6, color="#d64541", label="Bad")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(sc, meta, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    prob = score_to_prob(sc["test"]["score"].to_numpy(), meta)
    prob_true, prob_pred = calibration_curve(sc["test"]["bad"], prob, n_bins=10)
    ax.plot(prob_pred, prob_true, marker="o", label="Scorecard")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual bad rate")
    ax.set_title("Calibration (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sc, xgb, meta = load_data()
    VALIDATE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in ["test", "oot"]:
        for model_name, y_score in [
            ("scorecard", -sc[name]["score"]),
            ("xgb", xgb[name]["prob"]),
        ]:
            y = sc[name]["bad"]
            rows.append({
                "dataset": name, "model": model_name,
                "ks": round(ks_stat(y, y_score), 4),
                "auc": round(roc_auc_score(y, y_score), 4),
                "gini": round(gini(y, y_score), 4),
            })

    # PSI：评分卡分数分布 train vs oot
    psi_value = psi(sc["train"]["score"], sc["oot"]["score"])
    print(f"\nPSI (scorecard, train vs oot) = {psi_value:.4f}")

    metrics = pd.DataFrame(rows)
    print("\n===== 验证指标 =====")
    print(metrics.to_string(index=False))

    plot_roc(sc, xgb, VALIDATE_DIR / "roc_curve.png")
    plot_ks_curve(sc, xgb, VALIDATE_DIR / "ks_curve.png")
    plot_score_distribution(sc, VALIDATE_DIR / "score_distribution.png")
    plot_calibration(sc, meta, VALIDATE_DIR / "calibration.png")

    metrics.to_csv(VALIDATE_DIR / "metrics.csv", index=False)
    with open(VALIDATE_DIR / "psi.json", "w") as f:
        json.dump({"psi": psi_value}, f, indent=2)
    print(f"\n指标和图已保存到 {VALIDATE_DIR}")


if __name__ == "__main__":
    main()
