"""Checks for the rows of the planted-threshold test: alignment and no look-ahead.

Run from the project folder:  python -m unittest discover -s tests
"""
import unittest

import numpy as np

from planted_threshold_test import Case, decision_rows, expected_return_curve, simulate
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
        self.assertEqual(self.rows.days[-1] + CASE.step, len(self.series[2]) - 1)  # last label ends on the last price

    def test_features_and_returns_by_hand(self):
        x, observed, prices = self.series
        np.testing.assert_array_equal(observed, x)          # no observation noise by default
        for row in (0, 7, 39):
            t = self.rows.days[row]
            np.testing.assert_allclose(self.rows.X[row], [x[t]], rtol=1e-12)
            np.testing.assert_allclose(self.rows.R[row], [prices[t + 10] / prices[t] - 1, 0.0],
                                       rtol=1e-12)

    def test_covariance_by_hand(self):
        prices = self.series[2]
        t = self.rows.days[3]
        past = prices[t - 40 + 1:t + 1] / prices[t - 40:t] - 1      # the 40 daily returns up to t
        expected = np.array([[10 * past.var(ddof=1) + 1e-6, 0.0], [0.0, 1e-6]])
        np.testing.assert_allclose(self.rows.Sigma[3], expected, rtol=1e-12)

    def test_nothing_after_a_decision_enters_its_features_or_covariance(self):
        x, observed, prices = (series.copy() for series in self.series)
        row = 12
        t = self.rows.days[row]
        x[t + 1:] += 5.0
        observed[t + 1:] += 5.0
        prices[t + 1:] *= 1.5
        changed = decision_rows(x, observed, prices, CASE)
        np.testing.assert_array_equal(changed.X[:row + 1], self.rows.X[:row + 1])
        np.testing.assert_array_equal(changed.Sigma[:row + 1], self.rows.Sigma[:row + 1])
        self.assertNotEqual(changed.R[row, 0], self.rows.R[row, 0])     # only the label moved

    def test_the_drift_follows_the_planted_threshold(self):
        case = Case(train_days=200, test_days=0, step=1)
        x, _, prices = simulate(case, seed=3)
        # Replay the generator: the draws of x come first, then the price noise.
        rng = np.random.default_rng(3)
        ou_process(case.mean, case.feature_variance, case.k, n_steps=len(x) - 1, rng=rng)
        noise = rng.normal(0, case.volatility, size=len(x) - 1)
        drift = np.log(prices[1:] / prices[:-1]) - noise + 0.5 * case.price_variance
        np.testing.assert_allclose(drift, np.where(x[:-1] < case.d, -case.mu, case.mu), atol=1e-12)
        self.assertAlmostEqual(case.mu, case.edge * case.volatility)

    def test_the_threshold_sits_at_the_mean_of_x(self):
        self.assertEqual(CASE.d, CASE.mean)

    def test_observation_noise_blurs_the_feature_but_not_the_price(self):
        noisy = Case(train_days=300, test_days=100, window=40, step=10, feature_noise=0.5)
        x, observed, prices = simulate(noisy, seed=1)
        clean_x, _, clean_prices = simulate(CASE, seed=1)
        np.testing.assert_array_equal(x, clean_x)
        np.testing.assert_array_equal(prices, clean_prices)          # the noise is drawn last
        self.assertAlmostEqual((observed - x).std(), 0.5, delta=0.08)
        rows = decision_rows(x, observed, prices, noisy)
        np.testing.assert_array_equal(rows.X[:, 0], observed[rows.days])
        np.testing.assert_allclose(rows.Sigma_true[:, 0, 0], 10 * noisy.volatility ** 2 + 1e-6)

    def test_the_expected_return_given_a_noisy_feature_is_flatter(self):
        clean, noisy = Case(step=5), Case(step=5, feature_noise=1.0)
        grid = np.linspace(-4, 4, 81)
        g_clean = expected_return_curve(clean, grid, n_paths=4000)
        g_noisy = expected_return_curve(noisy, grid, n_paths=4000)
        near = np.argmin(np.abs(grid - 0.5))                         # the transition widens, the plateaus stay
        self.assertLess(abs(g_noisy[near]), 0.7 * abs(g_clean[near]))
        self.assertAlmostEqual(np.abs(g_noisy).max(), np.abs(g_clean).max(), delta=0.001)
        self.assertGreater(g_noisy[-1], 0)
        self.assertLess(g_noisy[0], 0)

    def test_a_volatility_threshold_switches_the_price_volatility(self):
        from synthetic_market import signal_process
        rng = np.random.default_rng(0)
        calm = signal_process(np.full(20001, 2.0), 0.0, 0.0, 1e-4, rng=rng,
                              up_volatility_ratio=0.5, volatility_threshold=1.0)
        stress = signal_process(np.full(20001, 0.5), 0.0, 0.0, 1e-4, rng=rng,
                                up_volatility_ratio=0.5, volatility_threshold=1.0)
        self.assertAlmostEqual(np.diff(np.log(calm)).std(), 0.005, delta=0.0003)
        self.assertAlmostEqual(np.diff(np.log(stress)).std(), 0.010, delta=0.0006)
        case = Case(up_volatility_ratio=0.5, volatility_threshold=1.0)
        np.testing.assert_allclose(case.volatility_of(np.array([-1.0, 0.5, 1.0, 2.0])), [0.01, 0.01, 0.005, 0.005])


if __name__ == "__main__":
    unittest.main()
