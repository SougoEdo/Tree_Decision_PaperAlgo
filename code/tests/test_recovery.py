"""Checks for the rows of the planted-threshold test: alignment and no look-ahead.

Run from the project folder:  python -m unittest discover -s tests
"""
import unittest

import numpy as np

from planted_threshold_test import Case, decision_rows, simulate
from synthetic_market import ou_process

CASE = Case(train_days=300, test_days=100, window=40, step=10)   # 30 + 10 decisions


class DecisionRowsTest(unittest.TestCase):
    def setUp(self):
        self.series = simulate(CASE, seed=1)
        self.rows = decision_rows(*self.series, CASE)

    def test_one_decision_every_step_days_after_the_window(self):
        self.assertEqual((CASE.n_train, CASE.n_test), (30, 10))
        np.testing.assert_array_equal(self.rows.days, 40 + 10 * np.arange(40))
        self.assertEqual(self.rows.X.shape, (40, 1))
        self.assertEqual(self.rows.Sigma.shape, (40, 2, 2))
        self.assertEqual(self.rows.days[-1] + CASE.step, len(self.series[1]) - 1)  # last label ends on the last price

    def test_features_and_returns_by_hand(self):
        x, prices = self.series
        for row in (0, 7, 39):
            t = self.rows.days[row]
            np.testing.assert_allclose(self.rows.X[row], [x[t]], rtol=1e-12)
            np.testing.assert_allclose(self.rows.R[row], [prices[t + 10] / prices[t] - 1, 0.0],
                                       rtol=1e-12)

    def test_covariance_by_hand(self):
        prices = self.series[1]
        t = self.rows.days[3]
        past = prices[t - 40 + 1:t + 1] / prices[t - 40:t] - 1      # the 40 daily returns up to t
        expected = np.array([[10 * past.var(ddof=1) + 1e-6, 0.0], [0.0, 1e-6]])
        np.testing.assert_allclose(self.rows.Sigma[3], expected, rtol=1e-12)

    def test_nothing_after_a_decision_enters_its_features_or_covariance(self):
        x, prices = (series.copy() for series in self.series)
        row = 12
        t = self.rows.days[row]
        x[t + 1:] += 5.0
        prices[t + 1:] *= 1.5
        changed = decision_rows(x, prices, CASE)
        np.testing.assert_array_equal(changed.X[:row + 1], self.rows.X[:row + 1])
        np.testing.assert_array_equal(changed.Sigma[:row + 1], self.rows.Sigma[:row + 1])
        self.assertNotEqual(changed.R[row, 0], self.rows.R[row, 0])     # only the label moved

    def test_the_drift_follows_the_planted_threshold(self):
        case = Case(train_days=200, test_days=0, step=1)
        x, prices = simulate(case, seed=3)
        # Replay the generator: the draws of x come first, then the price noise.
        rng = np.random.default_rng(3)
        ou_process(case.mean, case.feature_variance, case.k, n_steps=len(x) - 1, rng=rng)
        noise = rng.normal(0, case.volatility, size=len(x) - 1)
        drift = np.log(prices[1:] / prices[:-1]) - noise + 0.5 * case.price_variance
        np.testing.assert_allclose(drift, np.where(x[:-1] < case.d, -case.mu, case.mu), atol=1e-12)
        self.assertAlmostEqual(case.mu, case.edge * case.volatility)

    def test_the_threshold_sits_at_the_mean_of_x(self):
        self.assertEqual(CASE.d, CASE.mean)


if __name__ == "__main__":
    unittest.main()
