"""单元测试：ks_stat / gini / psi 三个指标函数。

运行：python -m unittest tests.test_metrics -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from validate import ks_stat, gini, psi


class TestMetrics(unittest.TestCase):
    def test_ks_perfect_separation(self):
        """坏样本分数都高于好样本时，KS 应为 1。"""
        y = np.array([1, 1, 0, 0])
        score = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(ks_stat(y, score), 1.0, places=9)

    def test_ks_random(self):
        """完全随机分数，KS 应接近 0。"""
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, size=20000)
        score = rng.rand(20000)
        self.assertLess(ks_stat(y, score), 0.05)

    def test_psi_identical(self):
        """相同分布 PSI 应为 0。"""
        rng = np.random.RandomState(0)
        a = rng.normal(size=10000)
        self.assertAlmostEqual(psi(a, a), 0.0, places=9)

    def test_gini_matches_auc(self):
        """Gini = 2 * AUC - 1。"""
        from sklearn.metrics import roc_auc_score
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(gini(y, score), 2 * roc_auc_score(y, score) - 1, places=9)


if __name__ == "__main__":
    unittest.main()
