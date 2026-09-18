"""Checks for the rows of the planted-threshold test: alignment and no look-ahead.

Run from the project folder:  python -m unittest discover -s tests
"""
import unittest

import numpy as np

from recovery_test import Case, decision_rows, simulate

CASE = Case(n_train=30, n_test=10, window=40, ma_window=5, step=10)


class DecisionRowsTest(unittest.TestCase):
    def setUp(self):
        self.series = simulate(CASE, seed=1)
        self.rows = decision_rows(*self.series, CASE)

    def test_one_decision_every_step_steps_after_the_window(self):
        np.testing.assert_array_equal(self.rows.steps, 40 + 10 * np.arange(40))
        self.assertEqual(self.rows.X.shape, (40, 4))
        self.assertEqual(self.rows.Sigma.shape, (40, 2, 2))
        self.assertEqual(self.rows.steps[-1] + CASE.step, len(self.series[3]) - 1)  # last label ends on the last price

    def test_features_and_returns_by_hand(self):
        x, decoy, noise, prices = self.series
        for row in (0, 7, 39):
            t = self.rows.steps[row]
            np.testing.assert_allclose(
                self.rows.X[row], [x[t], x[t - 4:t + 1].mean(), decoy[t], noise[t]], rtol=1e-12)
            np.testing.assert_allclose(self.rows.R[row], [prices[t + 10] / prices[t] - 1, 0.0],
                                       rtol=1e-12)

    def test_covariance_by_hand(self):
        prices = self.series[3]
        t = self.rows.steps[3]
        past = prices[t - 40 + 1:t + 1] / prices[t - 40:t] - 1      # the 40 one-step returns up to t
        expected = np.array([[10 * past.var(ddof=1) + 1e-6, 0.0], [0.0, 1e-6]])
        np.testing.assert_allclose(self.rows.Sigma[3], expected, rtol=1e-12)

    def test_nothing_after_a_decision_enters_its_features_or_covariance(self):
        x, decoy, noise, prices = (series.copy() for series in self.series)
        row = 12
        t = self.rows.steps[row]
        x[t + 1:] += 5.0
        decoy[t + 1:] -= 3.0
        noise[t + 1:] *= 2.0
        prices[t + 1:] *= 1.5
        changed = decision_rows(x, decoy, noise, prices, CASE)
        np.testing.assert_array_equal(changed.X[:row + 1], self.rows.X[:row + 1])
        np.testing.assert_array_equal(changed.Sigma[:row + 1], self.rows.Sigma[:row + 1])
        self.assertNotEqual(changed.R[row, 0], self.rows.R[row, 0])     # only the label moved

    def test_the_drift_follows_the_planted_threshold(self):
        case = Case(mu=0.01, price_variance=0.0, n_train=200, n_test=0, step=1)
        x, _, _, prices = simulate(case, seed=3)
        growth = np.log(prices[1:] / prices[:-1])
        np.testing.assert_allclose(growth, np.where(x[:-1] < case.d, -0.01, 0.01), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
