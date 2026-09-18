"""Checks for path-aware tree growth: regret screen, replays, Sharpe acceptance.

Run from the project folder:  python -m unittest discover -s tests
"""
import unittest

import numpy as np

from main import PortfolioConfig, PortfolioOptimizer, SPOPortfolioTree, TreeConfig


def regime_data(seed, weeks=120):
    """Two assets whose better one alternates with a persistent 10-week regime.

    x0 observes the regime with a little noise; x1 is pure noise.
    """
    rng = np.random.default_rng(seed)
    regime = np.tile(np.repeat([1.0, -1.0], 10), weeks // 20)
    X = np.column_stack([regime + rng.normal(0, 0.2, weeks), rng.normal(size=weeks)])
    covariance = np.array([[0.0004, 0.0001], [0.0001, 0.0004]])
    means = np.where(regime[:, None] > 0, [0.01, -0.005], [-0.005, 0.01])
    returns = means + rng.multivariate_normal(np.zeros(2), covariance, weeks)
    return X, returns, covariance, regime


def make_tree(fee=0.001, risk_aversion=2.0, **settings):
    optimizer = PortfolioOptimizer(2, PortfolioConfig(fee_rate=fee, risk_aversion=risk_aversion))
    defaults = dict(max_depth=1, min_samples_leaf=20, max_thresholds=None,
                    prediction_bound=0.05, search_passes=2, search_grid_size=5)
    defaults.update(settings)
    return SPOPortfolioTree(optimizer, TreeConfig(**defaults))


def _parts(tree):
    return tree.optimizer, tree.config


class SplitProbe(SPOPortfolioTree):
    """Records what the first round of growth saw: every shortlisted split of the
    root with its regret reduction and replayed Sharpe, and the fallback scores
    handed to each leaf fit."""

    def _best_split(self, leaves, scores, sharpe):
        first_round = not hasattr(self, "candidates")
        if first_round:
            node, rows, _ = leaves[0]
            self.stale_prediction = node.prediction.copy()
            parent = self._fit_leaf(rows, node.prediction)
            self.refitted_prediction = parent.prediction.copy()
            self.sharpe_before, self.candidates, self.fallbacks = sharpe, [], []
            for feature, threshold, left_rows, right_rows in self._shortlist(rows):
                left = self._fit_leaf(left_rows, parent.prediction)
                right = self._fit_leaf(right_rows, parent.prediction)
                replayed = scores.copy()
                replayed[left_rows], replayed[right_rows] = left.prediction, right.prediction
                self.candidates.append({
                    "split": (feature, float(threshold)),
                    "reduction": parent.regret_sum - left.regret_sum - right.regret_sum,
                    "sharpe": self._replay(replayed).sharpe})
        self.recording = first_round
        choice = super()._best_split(leaves, scores, sharpe)
        self.recording = False
        return choice

    def _fit_leaf(self, indices, parent_prediction=None):
        if getattr(self, "recording", False):
            self.fallbacks.append((len(indices), parent_prediction.copy()))
        return super()._fit_leaf(indices, parent_prediction)


def probed_tree():
    """Seed 4: the largest regret reduction and the highest Sharpe are different
    splits, and refitting the root on the replayed holdings changes its scores."""
    X, returns, covariance, _ = regime_data(seed=4)
    tree = SplitProbe(*_parts(make_tree(shortlist_size=8))).fit(X, returns, covariance)
    return tree, len(X)


def leaves(node, depth=0):
    if node.feature is None:
        return [(node, depth)]
    return leaves(node.left, depth + 1) + leaves(node.right, depth + 1)


class PathAwareTreeTest(unittest.TestCase):
    def test_splits_on_the_regime_feature_and_raises_the_sharpe(self):
        X, returns, covariance, regime = regime_data(seed=0)
        tree = make_tree().fit(X, returns, covariance)
        self.assertEqual(len(tree.growth_log), 1)
        split = tree.growth_log[0]
        self.assertEqual(split.feature, 0)
        self.assertGreater(split.sharpe_after, split.sharpe_before + 0.1)
        # The threshold may fit some noise in-sample, but it must separate the
        # two regimes for the vast majority of weeks.
        agreement = np.mean((X[:, 0] <= split.threshold) == (regime < 0))
        self.assertGreaterEqual(agreement, 0.9)

    def test_rejects_weekly_flipping_once_fees_exceed_the_edge(self):
        # The feature flips sign every week and so does the better asset, by
        # 0.8% each way. Following it earns 0.8% a week but a full swap every
        # week costs 2 * fee: worth it without fees, not at a 0.5% fee.
        weeks = 120
        rng = np.random.default_rng(0)
        flip = np.where(np.arange(weeks) % 2 == 0, 1.0, -1.0)
        X = (flip + rng.normal(0, 0.01, weeks))[:, None]
        returns = flip[:, None] * np.array([0.008, -0.008]) + rng.normal(0, 0.002, (weeks, 2))
        covariance = 4e-6 * np.eye(2)
        free = make_tree(fee=0.0, risk_aversion=1.0, max_thresholds=5)
        costly = make_tree(fee=0.005, risk_aversion=1.0, max_thresholds=5)
        self.assertGreater(free.fit(X, returns, covariance)
                           .replay(X, returns, covariance).turnover.sum(), 200)
        self.assertLess(costly.fit(X, returns, covariance)
                        .replay(X, returns, covariance).turnover.sum(), 5)

    def test_the_largest_regret_reduction_wins_among_splits_that_raise_the_sharpe(self):
        tree, n_rows = probed_tree()
        passing = [c for c in tree.candidates if c["sharpe"] > tree.sharpe_before]
        by_regret = max(passing, key=lambda c: c["reduction"])
        by_sharpe = max(passing, key=lambda c: c["sharpe"])
        self.assertNotEqual(by_regret["split"], by_sharpe["split"])   # the two rules differ here
        chosen = tree.growth_log[0]
        self.assertEqual((chosen.feature, chosen.threshold), by_regret["split"])
        # The logged gain is measured against the root refitted on the same holdings.
        self.assertAlmostEqual(chosen.regret_gain, by_regret["reduction"] / n_rows, places=12)

    def test_children_are_compared_with_their_leaf_refitted_on_the_same_holdings(self):
        tree, n_rows = probed_tree()
        self.assertFalse(np.array_equal(tree.stale_prediction, tree.refitted_prediction))
        # First the whole leaf is refitted, starting from its old scores...
        self.assertEqual(tree.fallbacks[0][0], n_rows)
        np.testing.assert_array_equal(tree.fallbacks[0][1], tree.stale_prediction)
        # ...then every child is fitted with the refitted scores as its fallback.
        self.assertGreater(len(tree.fallbacks), 1)
        for size, fallback in tree.fallbacks[1:]:
            self.assertLess(size, n_rows)
            np.testing.assert_array_equal(fallback, tree.refitted_prediction)

    def test_every_accepted_split_raises_the_training_sharpe(self):
        X, returns, covariance, _ = regime_data(seed=1)
        margin = 0.01
        tree = make_tree(max_depth=2, min_samples_leaf=15, min_sharpe_improvement=margin)
        tree.fit(X, returns, covariance)
        log = tree.growth_log
        self.assertGreaterEqual(len(log), 1)
        for record in log:
            self.assertGreater(record.sharpe_after, record.sharpe_before + margin)
        for earlier, later in zip(log, log[1:]):
            self.assertEqual(later.sharpe_before, earlier.sharpe_after)
        # The final refit of leaf scores is kept only if it does not lower the Sharpe.
        self.assertGreaterEqual(tree.replay(X, returns, covariance).sharpe,
                                log[-1].sharpe_after - 1e-12)

    def test_a_large_sharpe_margin_blocks_every_split(self):
        X, returns, covariance, _ = regime_data(seed=0)
        tree = make_tree(max_depth=2, min_sharpe_improvement=10.0).fit(X, returns, covariance)
        self.assertEqual(tree.growth_log, [])
        self.assertIsNone(tree.root.feature)
        scores = tree.predict_returns(X)
        np.testing.assert_array_equal(scores, np.tile(tree.root.prediction, (len(X), 1)))

    def test_a_refit_that_lowers_the_sharpe_is_discarded(self):
        X, returns, covariance, _ = regime_data(seed=0)

        class SabotagedRefit(SPOPortfolioTree):
            """Negates every refit leaf score, which must hurt the Sharpe."""
            def _refit_leaves(self, leaves, scores, path):
                self.scores_before_refit = scores.copy()
                self.refitting = True
                super()._refit_leaves(leaves, scores, path)

            def _fit_leaf(self, indices, parent_prediction=None):
                node = super()._fit_leaf(indices, parent_prediction)
                if getattr(self, "refitting", False):
                    node.prediction = -node.prediction
                return node

        tree = SabotagedRefit(*_parts(make_tree())).fit(X, returns, covariance)
        np.testing.assert_array_equal(tree.predict_returns(X), tree.scores_before_refit)

    def test_a_large_regret_threshold_blocks_every_split(self):
        X, returns, covariance, _ = regime_data(seed=0)
        tree = make_tree(max_depth=2, min_regret_improvement=1.0).fit(X, returns, covariance)
        self.assertEqual(tree.growth_log, [])
        self.assertIsNone(tree.root.feature)

    def test_leaves_respect_depth_and_size_limits(self):
        X, returns, covariance, _ = regime_data(seed=2)
        tree = make_tree(max_depth=2, min_samples_leaf=25).fit(X, returns, covariance)
        found = leaves(tree.root)
        self.assertEqual(sum(node.n_samples for node, _ in found), len(X))
        for node, depth in found:
            self.assertLessEqual(depth, 2)
            self.assertGreaterEqual(node.n_samples, 25)
        for name in ("_X", "_R", "_S", "_U", "_benchmarks"):
            self.assertFalse(hasattr(tree, name))   # no training data kept after fit

    def test_invalid_tree_settings_are_rejected(self):
        optimizer = PortfolioOptimizer(2, PortfolioConfig())
        with self.assertRaises(ValueError):
            SPOPortfolioTree(optimizer, TreeConfig(shortlist_size=0))
        with self.assertRaises(ValueError):
            SPOPortfolioTree(optimizer, TreeConfig(min_sharpe_improvement=-0.1))


if __name__ == "__main__":
    unittest.main()
