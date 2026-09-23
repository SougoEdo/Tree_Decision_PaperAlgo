"""Checks for weekly_rows and weekly_covariance: alignment and no look-ahead.

Run from the project folder:  python -m unittest discover -s tests
"""
import datetime
import unittest

import numpy as np

from spo_tree import (PortfolioConfig, PortfolioOptimizer, SPOPortfolioTree, TreeConfig,
                  weekly_covariance, weekly_rows)


def daily_prices(days=500, assets=3, seed=0, start="2024-01-01"):
    """Consecutive daily closes of a random walk (2024-01-01 is a Monday)."""
    rng = np.random.default_rng(seed)
    dates = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(days, "D"))
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.03, (days, assets)), axis=0)
    return dates, closes


def causal_features(closes):
    """20-day and 5-day momentum: row d only uses closes up to day d."""
    features = np.full((len(closes), 2), np.nan)
    features[20:, 0] = closes[20:, 0] / closes[:-20, 0] - 1
    features[5:, 1] = closes[5:, 1] / closes[:-5, 1] - 1
    return features


class WeeklyRowsTest(unittest.TestCase):
    def test_decisions_are_on_the_chosen_weekday_one_week_apart(self):
        dates, closes = daily_prices()
        for weekday in (0, 4, 6):
            rows = weekly_rows(dates, closes, weekday=weekday)
            weekdays = {datetime.date.fromisoformat(str(d)).weekday() for d in rows.dates}
            self.assertEqual(weekdays, {weekday})
            self.assertTrue(np.all(np.diff(rows.day_index) == 7))
            np.testing.assert_array_equal(rows.dates, dates[rows.day_index])

    def test_rows_start_after_a_full_window_and_end_before_the_last_week(self):
        dates, closes = daily_prices()
        rows = weekly_rows(dates, closes, window=180)
        self.assertGreaterEqual(rows.day_index[0], 180)
        self.assertLess(rows.day_index[0] - 7, 180)            # no eligible decision skipped
        self.assertLess(rows.day_index[-1] + 7, len(dates))
        self.assertGreaterEqual(rows.day_index[-1] + 14, len(dates))

    def test_returns_run_from_one_decision_close_to_the_next(self):
        dates, closes = daily_prices()
        rows = weekly_rows(dates, closes)
        expected = closes[rows.day_index + 7] / closes[rows.day_index] - 1
        np.testing.assert_allclose(rows.R, expected, rtol=0, atol=1e-15)

    def test_covariance_is_seven_times_the_daily_sample_covariance_plus_ridge(self):
        dates, closes = daily_prices()
        rows = weekly_rows(dates, closes, window=180, ridge=1e-6)
        t = 10
        d = rows.day_index[t]
        daily = closes[d - 179:d + 1] / closes[d - 180:d] - 1     # returns of days d-179 .. d
        expected = 7 * np.cov(daily.T) + 1e-6 * np.eye(3)
        np.testing.assert_allclose(rows.Sigma[t], expected, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(weekly_covariance(closes, d), rows.Sigma[t])

    def test_nothing_after_a_decision_close_changes_that_row(self):
        dates, closes = daily_prices()
        rows = weekly_rows(dates, closes, causal_features(closes))
        k = len(rows.R) // 2
        cut = rows.day_index[k]
        future = closes.copy()
        future[cut + 1:] *= np.random.default_rng(1).uniform(0.5, 1.5, future[cut + 1:].shape)
        changed = weekly_rows(dates, future, causal_features(future))
        # Rows decided at or before the cut see exactly the same inputs...
        np.testing.assert_array_equal(changed.X[:k + 1], rows.X[:k + 1])
        np.testing.assert_array_equal(changed.Sigma[:k + 1], rows.Sigma[:k + 1])
        np.testing.assert_array_equal(changed.R[:k], rows.R[:k])
        # ...while the outcome of row k and all later rows do change.
        self.assertFalse(np.allclose(changed.R[k], rows.R[k]))
        self.assertFalse(np.allclose(changed.Sigma[k + 1], rows.Sigma[k + 1]))
        self.assertFalse(np.allclose(changed.X[k + 1], rows.X[k + 1]))

    def test_rows_feed_fit_and_replay(self):
        dates, closes = daily_prices(days=700, assets=2)
        rows = weekly_rows(dates, closes, causal_features(closes))
        train, test = slice(0, len(rows.R) - 20), slice(len(rows.R) - 20, None)
        tree = SPOPortfolioTree(
            PortfolioOptimizer(2, PortfolioConfig(fee_rate=0.001, risk_aversion=2.0)),
            TreeConfig(max_depth=1, min_samples_leaf=15, search_passes=1, search_grid_size=3))
        tree.fit(rows.X[train], rows.R[train], rows.Sigma[train])
        first = tree.replay(rows.X[train], rows.R[train], rows.Sigma[train])
        second = tree.replay(rows.X[test], rows.R[test], rows.Sigma[test],
                             initial_weights=first.final_holdings)
        self.assertEqual(len(second.net_returns), 20)

    def test_invalid_inputs_are_rejected(self):
        dates, closes = daily_prices()
        with self.assertRaises(ValueError):                      # a missing day
            weekly_rows(np.delete(dates, 100), np.delete(closes, 100, axis=0))
        bad = closes.copy()
        bad[50, 1] = 0.0
        with self.assertRaises(ValueError):                      # nonpositive price
            weekly_rows(dates, bad)
        features = causal_features(closes)
        features[weekly_rows(dates, closes).day_index[3], 0] = np.nan
        with self.assertRaises(ValueError):                      # NaN on a decision date
            weekly_rows(dates, closes, features)
        with self.assertRaises(ValueError):                      # window longer than the data
            weekly_rows(dates, closes, window=600)
        with self.assertRaises(ValueError):                      # weekday out of range
            weekly_rows(dates, closes, weekday=7)


if __name__ == "__main__":
    unittest.main()
