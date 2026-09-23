"""Checks for evaluation: summaries, max drawdown, equal weight, train_and_test.

Run from the project folder:  python -m unittest discover -s tests
"""
import unittest

import numpy as np

from spo_tree import (PathResult, PortfolioConfig, PortfolioOptimizer, TreeConfig,
                  performance_report, replay_constant_weights, train_and_test, weekly_rows)
from test_data import causal_features, daily_prices


def path_with(net_returns, fees=None, turnover=None):
    """A PathResult whose accounting fields only matter through these arrays."""
    net = np.asarray(net_returns, dtype=float)
    zeros = np.zeros_like(net)
    fees = zeros if fees is None else np.asarray(fees, dtype=float)
    turnover = zeros if turnover is None else np.asarray(turnover, dtype=float)
    weights = np.full((len(net), 2), 0.5)
    return PathResult(holdings=weights, weights=weights, turnover=turnover, fees=fees,
                      gross_returns=net + fees, net_returns=net, utility=net,
                      regret=zeros, final_holdings=np.array([0.5, 0.5]))


class ReportTest(unittest.TestCase):
    def test_max_drawdown_is_measured_from_the_running_peak(self):
        # Wealth 1.10, 0.88, 0.924, 0.8316: the low is 24.4% below the 1.10 peak.
        self.assertAlmostEqual(path_with([0.10, -0.20, 0.05, -0.10]).max_drawdown,
                               0.8316 / 1.10 - 1, places=12)
        # A loss in the first period counts from the starting wealth of 1.
        self.assertAlmostEqual(path_with([-0.10, 0.05]).max_drawdown, -0.10, places=12)
        self.assertEqual(path_with([0.01, 0.02]).max_drawdown, 0.0)

    def test_summary_annualizes_weekly_figures(self):
        rng = np.random.default_rng(0)
        net = rng.normal(0.003, 0.05, 104)                       # two years of weeks
        summary = path_with(net, fees=np.full(104, 0.0002),
                            turnover=np.full(104, 0.2)).summary(periods_per_year=52)
        self.assertAlmostEqual(summary["return"], np.prod(1 + net) ** 0.5 - 1, places=12)
        self.assertAlmostEqual(summary["volatility"], net.std(ddof=1) * np.sqrt(52), places=12)
        self.assertAlmostEqual(summary["sharpe"],
                               net.mean() / net.std(ddof=1) * np.sqrt(52), places=12)
        self.assertAlmostEqual(summary["turnover"], 0.2, places=12)
        self.assertAlmostEqual(summary["fees"], 0.0002 * 52, places=12)

    def test_equal_weight_trades_back_to_the_target_every_period(self):
        # Week 1: asset 1 +20%, asset 2 -10% drifts 50/50 to 0.6/1.05, 0.45/1.05;
        # week 2 trades exactly 1/7 of equity back to 50/50 and pays 0.001/7.
        optimizer = PortfolioOptimizer(2, PortfolioConfig(fee_rate=0.001))
        returns = np.array([[0.20, -0.10], [0.0, 0.0]])
        path = replay_constant_weights(optimizer, [0.5, 0.5], returns, 0.01 * np.eye(2))
        np.testing.assert_array_equal(path.weights, 0.5)
        np.testing.assert_allclose(path.holdings[1], [0.6 / 1.05, 0.45 / 1.05], atol=1e-15)
        np.testing.assert_allclose(path.turnover, [0.0, 1 / 7], atol=1e-15)
        np.testing.assert_allclose(path.net_returns, [0.05, -0.001 / 7], atol=1e-15)
        with self.assertRaises(ValueError):                      # weights sum to 1.4
            replay_constant_weights(optimizer, [0.7, 0.7], returns, 0.01 * np.eye(2))

    def test_train_and_test_compares_three_strategies_on_consecutive_periods(self):
        dates, closes = daily_prices(days=700, assets=2)
        rows = weekly_rows(dates, closes, causal_features(closes))
        portfolio = PortfolioOptimizer(2, PortfolioConfig(fee_rate=0.001, risk_aversion=2.0))
        config = TreeConfig(max_depth=1, min_samples_leaf=15, search_passes=1,
                            search_grid_size=3)
        tree, train_paths, test_paths = train_and_test(portfolio, config, rows, test_periods=20)
        names = ["SPO tree", "tree without splits", "equal weight"]
        self.assertEqual(list(train_paths), names)
        self.assertEqual(list(test_paths), names)
        for name in names:
            self.assertEqual(len(train_paths[name].net_returns), len(rows.R) - 20)
            self.assertEqual(len(test_paths[name].net_returns), 20)
            # Each strategy starts the test period from its own training holdings.
            np.testing.assert_allclose(test_paths[name].holdings[0],
                                       train_paths[name].final_holdings, atol=1e-15)
        np.testing.assert_array_equal(
            train_paths["SPO tree"].weights,
            tree.replay(rows.X[:-20], rows.R[:-20], rows.Sigma[:-20]).weights)
        report = performance_report(test_paths)
        self.assertEqual(len(report.splitlines()), 1 + len(names))
        for name in names:
            self.assertIn(name, report)
        with self.assertRaises(ValueError):
            train_and_test(portfolio, config, rows, test_periods=len(rows.R))


if __name__ == "__main__":
    unittest.main()
