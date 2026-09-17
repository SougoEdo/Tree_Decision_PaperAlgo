"""Checks for replay_path: drifting holdings, fees, returns and Sharpe ratio.

Run from the project folder:  python -m unittest discover -s tests
"""
import unittest

import numpy as np

from main import (PortfolioConfig, PortfolioOptimizer, SPOPortfolioTree, TreeConfig,
                  replay_path)
from test_optimizer import random_covariance


class ReplayTest(unittest.TestCase):
    def test_drift_and_fees_match_the_hand_computed_example(self):
        # Equal scores, equal variances and a huge risk aversion keep the target
        # at 50/50. Week 1: asset 1 +20%, asset 2 -10%, so holdings drift to
        # 0.6/1.05 and 0.45/1.05. Week 2: trading back to 50/50 moves 1/7 of
        # equity and costs 0.001 * 1/7.
        optimizer = PortfolioOptimizer(2, PortfolioConfig(fee_rate=0.001, risk_aversion=1e6))
        returns = np.array([[0.20, -0.10], [0.0, 0.0]])
        path = replay_path(optimizer, np.zeros((2, 2)), returns, 0.01 * np.eye(2))
        np.testing.assert_allclose(path.holdings, [[0.5, 0.5], [0.6 / 1.05, 0.45 / 1.05]],
                                   atol=1e-12)
        np.testing.assert_allclose(path.weights, [[0.5, 0.5], [0.5, 0.5]], atol=1e-6)
        np.testing.assert_allclose(path.turnover, [0.0, 1 / 7], atol=1e-6)
        np.testing.assert_allclose(path.fees, [0.0, 0.001 / 7], atol=1e-9)
        np.testing.assert_allclose(path.gross_returns, [0.05, 0.0], atol=1e-6)
        np.testing.assert_allclose(path.net_returns, [0.05, -0.001 / 7], atol=1e-6)

    def test_accounting_identities_on_a_random_path(self):
        rng = np.random.default_rng(3)
        T, n, fees, lam = 60, 3, np.array([0.001, 0.002, 0.003]), 5.0
        optimizer = PortfolioOptimizer(n, PortfolioConfig(
            max_weight=0.8, fee_rate=tuple(fees), risk_aversion=lam))
        scores = rng.uniform(-0.1, 0.1, (T, n))
        returns = rng.normal(0.002, 0.08, (T, n))
        covariances = np.array([random_covariance(rng, n) for _ in range(T)])
        start = np.array([0.2, 0.3, 0.5])
        path = replay_path(optimizer, scores, returns, covariances, initial_weights=start)

        np.testing.assert_allclose(path.holdings[0], start, atol=1e-12)
        for t in range(T):
            w, r = path.weights[t], returns[t]
            np.testing.assert_allclose(
                w, optimizer.solve(scores[t], path.holdings[t], covariances[t]), atol=1e-12)
            following = path.holdings[t + 1] if t + 1 < T else path.final_holdings
            np.testing.assert_allclose(following, w * (1 + r) / (1 + r @ w), atol=1e-9)
        trades = np.abs(path.weights - path.holdings)
        np.testing.assert_allclose(path.turnover, trades.sum(axis=1), atol=1e-12)
        np.testing.assert_allclose(path.fees, trades @ fees, atol=1e-12)
        np.testing.assert_allclose(path.net_returns,
                                   (returns * path.weights).sum(axis=1) - path.fees, atol=1e-12)
        risk = np.einsum("ti,tij,tj->t", path.weights, covariances, path.weights)
        np.testing.assert_allclose(path.utility, path.net_returns - lam * risk, atol=1e-12)
        self.assertTrue(np.all(path.regret >= -1e-12))
        self.assertAlmostEqual(path.sharpe,
                               path.net_returns.mean() / path.net_returns.std(ddof=1),
                               places=12)

    def test_fees_above_the_score_gap_stop_a_strategy_that_flips_every_week(self):
        # Scores swap sign every week: a gap of 0.10 between the two assets.
        T = 52
        flips = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)[:, None] * np.array([0.05, -0.05])
        returns, covariance = np.zeros((T, 2)), np.array([[0.004, 0.001], [0.001, 0.004]])
        free = PortfolioOptimizer(2, PortfolioConfig(fee_rate=0.0, risk_aversion=5.0))
        costly = PortfolioOptimizer(2, PortfolioConfig(fee_rate=0.06, risk_aversion=5.0))
        # Without fees: 50/50 -> 100/0 (turnover 1), then a full swap (2) every week.
        self.assertAlmostEqual(replay_path(free, flips, returns, covariance).turnover.sum(),
                               1 + 2 * (T - 1), places=9)
        # A round trip costs 2 * 0.06 = 0.12 > 0.10, so it never leaves 50/50.
        self.assertAlmostEqual(replay_path(costly, flips, returns, covariance).turnover.sum(),
                               0.0, places=9)

    def test_a_path_can_be_continued_from_its_final_holdings(self):
        rng = np.random.default_rng(4)
        optimizer = PortfolioOptimizer(3, PortfolioConfig(fee_rate=0.002, risk_aversion=3.0))
        scores = rng.uniform(-0.1, 0.1, (40, 3))
        returns = rng.normal(0.0, 0.05, (40, 3))
        covariance = random_covariance(rng, 3)
        whole = replay_path(optimizer, scores, returns, covariance)
        first = replay_path(optimizer, scores[:25], returns[:25], covariance)
        second = replay_path(optimizer, scores[25:], returns[25:], covariance,
                             initial_weights=first.final_holdings)
        np.testing.assert_allclose(np.r_[first.net_returns, second.net_returns],
                                   whole.net_returns, atol=1e-12)
        np.testing.assert_allclose(second.final_holdings, whole.final_holdings, atol=1e-12)

    def test_tree_replay_trades_the_tree_scores(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40, 2))
        returns = (np.where(X[:, :1] > 0, 0.02, -0.02) * np.array([1.0, -1.0])
                   + rng.normal(0, 0.01, (40, 2)))
        covariance = np.array([[1e-4, 2e-5], [2e-5, 1e-4]])
        optimizer = PortfolioOptimizer(2, PortfolioConfig(fee_rate=0.001, risk_aversion=50.0))
        tree = SPOPortfolioTree(optimizer, TreeConfig(
            max_depth=1, min_samples_leaf=10, search_passes=1, search_grid_size=3))
        tree.fit(X, returns, np.full((40, 2), 0.5), covariance)
        path = tree.replay(X, returns, covariance)
        expected = replay_path(optimizer, tree.predict_returns(X), returns, covariance)
        np.testing.assert_array_equal(path.weights, expected.weights)
        np.testing.assert_array_equal(path.net_returns, expected.net_returns)

    def test_sharpe_is_undefined_for_a_single_period(self):
        optimizer = PortfolioOptimizer(2, PortfolioConfig())
        path = replay_path(optimizer, np.zeros((1, 2)), [[0.01, 0.02]], 0.01 * np.eye(2))
        self.assertTrue(np.isnan(path.sharpe))

    def test_invalid_inputs_are_rejected(self):
        optimizer = PortfolioOptimizer(2, PortfolioConfig())
        scores, covariance = np.zeros((3, 2)), 0.01 * np.eye(2)
        with self.assertRaises(ValueError):   # a simple return of -100% or worse
            replay_path(optimizer, scores, [[0.0, 0.0], [-1.0, 0.1], [0.0, 0.0]], covariance)
        with self.assertRaises(ValueError):   # holdings must sum to 1
            replay_path(optimizer, scores, np.zeros((3, 2)), covariance,
                        initial_weights=[0.6, 0.6])
        with self.assertRaises(ValueError):   # one score row per return row
            replay_path(optimizer, np.zeros((2, 2)), np.zeros((3, 2)), covariance)


if __name__ == "__main__":
    unittest.main()
