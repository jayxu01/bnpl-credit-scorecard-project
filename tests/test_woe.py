"""单元测试：WoEBinner 分箱。

运行：python -m unittest tests.test_woe -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from woe import WoEBinner


def _make_data(n=1000, seed=0):
    """生成 x 越大违约概率越高的合成数据。"""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 100, n)
    p = 1 / (1 + np.exp(-(-2 + 0.05 * x)))
    y = (rng.rand(n) < p).astype(int)
    return pd.DataFrame({"x": x}), pd.Series(y)


class TestWoE(unittest.TestCase):
    def test_fit_transform_shape(self):
        X, y = _make_data()
        binner = WoEBinner(max_bins=5).fit(X, y)
        Xw = binner.transform(X)
        self.assertEqual(Xw.shape, X.shape)

    def test_iv_non_negative(self):
        X, y = _make_data()
        binner = WoEBinner(max_bins=5).fit(X, y)
        self.assertGreaterEqual(binner.iv_["x"], 0.0)

    def test_monotonic_for_linear_risk(self):
        """风险随 x 单调上升时，WoE 应单调。"""
        X, y = _make_data(n=5000)
        binner = WoEBinner(max_bins=5).fit(X, y)
        self.assertTrue(binner.monotonic()["x"])


if __name__ == "__main__":
    unittest.main()
