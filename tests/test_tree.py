"""Checks for path-aware tree growth: regret screen, replays, Sharpe acceptance,
and pruning on checking rows that took no part in growth.

Run from the project folder:  python -m unittest discover -s tests
"""
import unittest

import numpy as np

from main import (PortfolioConfig, PortfolioOptimizer, SPOPortfolioTree, TreeConfig,
                  replay_path)


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


class ThresholdQuantilesTest(unittest.TestCase):
    def test_only_the_partitions_nearest_to_the_quantiles_are_screened(self):
        optimizer = PortfolioOptimizer(2, PortfolioConfig())
        tree = SPOPortfolioTree(optimizer, TreeConfig(min_samples_leaf=5,
                                                     threshold_quantiles=(0.25, 0.5, 0.75)))
        values = np.arange(100.0)
        np.testing.assert_allclose(tree._thresholds(values), [24.5, 49.5, 74.5])
        every = SPOPortfolioTree(optimizer, TreeConfig(min_samples_leaf=5, max_thresholds=None))
        self.assertEqual(len(every._thresholds(values)), 91)
        with self.assertRaises(ValueError):
            SPOPortfolioTree(optimizer, TreeConfig(threshold_quantiles=(0.0, 0.5)))


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
        for fraction in (-0.1, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                SPOPortfolioTree(optimizer, TreeConfig(validation_fraction=fraction))


# With validation_fraction=0.25 and 160 weeks, the tree grows on the first 120
# (fitting rows) and is pruned on the last 40 (checking rows).
FITTING, WEEKS = 120, 160


def unlinked_regime_data(seed):
    """The returns of regime_data in a random order: no feature says anything."""
    X, returns, covariance, _ = regime_data(seed, WEEKS)
    return X, np.random.default_rng(1000 + seed).permutation(returns), covariance


def interaction_data(seed, weeks=240, fitting=180):
    """Asset 0 is the better one when a and b have the same sign, asset 1 otherwise.

    In the fitting rows a also tilts the returns a little, so growth splits on
    a first. In the checking rows a says nothing without b.
    """
    rng = np.random.default_rng(seed)
    a = np.tile(np.repeat([1.0, 1.0, -1.0, -1.0], 5), weeks // 20)
    b = np.tile(np.repeat([1.0, -1.0, -1.0, 1.0], 5), weeks // 20)
    covariance = np.array([[0.0004, 0.0001], [0.0001, 0.0004]])
    means = np.where((a * b)[:, None] > 0, [0.01, -0.005], [-0.005, 0.01])
    means[:fitting] += a[:fitting, None] * np.array([0.004, -0.004])
    returns = means + rng.multivariate_normal(np.zeros(2), covariance, weeks)
    return np.column_stack([a, b]), returns, covariance


class PruningTest(unittest.TestCase):
    def test_a_split_confirmed_by_the_checking_rows_is_kept(self):
        X, returns, covariance, _ = regime_data(seed=0, weeks=WEEKS)
        tree = make_tree(validation_fraction=0.25).fit(X, returns, covariance)
        self.assertEqual(len(tree.pruning_log), 1)
        check = tree.pruning_log[0]
        self.assertTrue(check.kept)
        self.assertLess(check.regret_with, check.regret_without)
        self.assertGreater(check.sharpe_with, check.sharpe_without)
        self.assertEqual((check.feature, check.threshold),
                         (tree.growth_log[0].feature, tree.growth_log[0].threshold))
        self.assertEqual(tree.root.feature, 0)
        self.assertEqual(tree.growth_log[0].n_samples, FITTING)   # grown on the fitting rows
        for name in ("_X", "_R", "_S", "_U", "_benchmarks"):
            self.assertFalse(hasattr(tree, name))
        # The judged tree is the one the fitting rows alone give. It trades the
        # fitting rows, then the checking rows from the holdings it ends with.
        early = slice(0, FITTING)
        late = slice(FITTING, WEEKS)
        grown = make_tree().fit(X[early], returns[early], covariance)
        holdings = grown.replay(X[early], returns[early], covariance).final_holdings
        checked = grown.replay(X[late], returns[late], covariance, holdings)
        self.assertAlmostEqual(check.sharpe_with, checked.sharpe, places=12)
        self.assertAlmostEqual(check.regret_with, checked.regret.mean(), places=12)

    def test_a_split_contradicted_by_the_checking_rows_is_removed(self):
        X, returns, covariance, _ = regime_data(seed=0, weeks=WEEKS)
        returns[FITTING:] = returns[FITTING:, ::-1]      # the two assets swap roles
        tree = make_tree(validation_fraction=0.25).fit(X, returns, covariance)
        self.assertEqual(len(tree.growth_log), 1)         # the split was grown...
        check = tree.pruning_log[0]
        self.assertFalse(check.kept)                      # ...and then removed
        self.assertGreater(check.regret_with, check.regret_without)
        self.assertLess(check.sharpe_with, check.sharpe_without)
        self.assertIsNone(tree.root.feature)
        self.assertIsNone(tree.root.left)
        scores = tree.predict_returns(X)
        np.testing.assert_array_equal(scores, np.tile(tree.root.prediction, (WEEKS, 1)))

    def test_the_single_leaf_it_is_compared_with_is_refitted_on_the_fitting_rows(self):
        class Recorder(SPOPortfolioTree):
            def _prune(self, *args):
                self.pruning, self.single_leaves = True, []
                super()._prune(*args)
                self.pruning = False

            def _fit_leaf(self, indices, parent_prediction=None):
                node = super()._fit_leaf(indices, parent_prediction)
                if getattr(self, "pruning", False):
                    self.single_leaves.append((len(indices), node.prediction.copy()))
                return node

        X, returns, covariance, _ = regime_data(seed=0, weeks=WEEKS)
        returns[FITTING:] = returns[FITTING:, ::-1]
        tree = Recorder(*_parts(make_tree(validation_fraction=0.25))).fit(X, returns, covariance)
        self.assertEqual([size for size, _ in tree.single_leaves], [FITTING])
        # Its checking-row figures are those of that one leaf, traded from the start.
        prediction = tree.single_leaves[0][1]
        optimizer = tree.optimizer
        early, late = slice(0, FITTING), slice(FITTING, WEEKS)
        holdings = replay_path(optimizer, np.tile(prediction, (FITTING, 1)),
                               returns[early], covariance).final_holdings
        checked = replay_path(optimizer, np.tile(prediction, (WEEKS - FITTING, 1)),
                              returns[late], covariance, holdings)
        self.assertAlmostEqual(tree.pruning_log[0].sharpe_without, checked.sharpe, places=12)
        self.assertAlmostEqual(tree.pruning_log[0].regret_without, checked.regret.mean(), places=12)

    def test_the_checking_rows_take_no_part_in_growth(self):
        X, returns, covariance, _ = regime_data(seed=0, weeks=WEEKS)
        changed_X, changed_returns = X.copy(), returns.copy()
        changed_X[FITTING:] = -changed_X[FITTING:]
        changed_returns[FITTING:] = changed_returns[FITTING:, ::-1] * 2
        first = make_tree(max_depth=2, validation_fraction=0.25).fit(X, returns, covariance)
        second = make_tree(max_depth=2, validation_fraction=0.25).fit(
            changed_X, changed_returns, covariance)
        self.assertGreaterEqual(len(first.growth_log), 1)
        self.assertEqual(first.growth_log, second.growth_log)
        # The same rows, all used for growth, give another tree: the last
        # quarter really was kept aside.
        everything = make_tree(max_depth=2).fit(X, returns, covariance)
        self.assertNotEqual(first.growth_log, everything.growth_log)

    def test_both_judges_must_agree_on_the_checking_rows(self):
        # Seed 16: on the checking rows the split lowers the regret but also the
        # Sharpe. Seed 160: it raises the Sharpe but also the regret.
        for seed, regret_falls, sharpe_rises in ((16, True, False), (160, False, True)):
            X, returns, covariance = unlinked_regime_data(seed)
            tree = make_tree(validation_fraction=0.25).fit(X, returns, covariance)
            check = tree.pruning_log[0]
            self.assertEqual(check.regret_with < check.regret_without, regret_falls)
            self.assertEqual(check.sharpe_with > check.sharpe_without, sharpe_rises)
            self.assertFalse(check.kept)
            self.assertIsNone(tree.root.feature)

    def test_a_pruning_margin_applies_to_the_checking_rows(self):
        X, returns, covariance, _ = regime_data(seed=0, weeks=WEEKS)
        free = make_tree(validation_fraction=0.25).fit(X, returns, covariance).pruning_log[0]
        gain = free.sharpe_with - free.sharpe_without
        # In-sample the split clears this margin easily; on the checking rows it does not.
        demanding = make_tree(validation_fraction=0.25, min_sharpe_improvement=gain + 0.01)
        demanding.fit(X, returns, covariance)
        self.assertEqual(len(demanding.growth_log), 1)
        self.assertFalse(demanding.pruning_log[0].kept)

    def test_a_split_is_judged_with_the_splits_below_it(self):
        X, returns, covariance = interaction_data(seed=1)
        # Alone, the split on a is not confirmed by the checking rows.
        alone = make_tree(max_depth=1, validation_fraction=0.25).fit(X, returns, covariance)
        self.assertEqual([record.feature for record in alone.growth_log], [0])
        self.assertFalse(alone.pruning_log[0].kept)
        # With the two splits on b below it, it is: pruning works from the bottom up.
        tree = make_tree(max_depth=2, validation_fraction=0.25).fit(X, returns, covariance)
        self.assertEqual([(r.depth, r.feature) for r in tree.growth_log], [(0, 0), (1, 1), (1, 1)])
        self.assertEqual([(c.depth, c.feature, c.kept) for c in tree.pruning_log],
                         [(1, 1, True), (1, 1, True), (0, 0, True)])
        self.assertEqual(len(leaves(tree.root)), 4)

    def test_leaf_scores_are_finally_refitted_on_all_rows(self):
        class Recorder(SPOPortfolioTree):
            def _refit_leaves(self, leaves, scores, path, on_all_rows=False):
                self.refitted = getattr(self, "refitted", []) + [
                    sum(len(rows) for _, rows, _ in leaves)]
                super()._refit_leaves(leaves, scores, path, on_all_rows)

        X, returns, covariance, _ = regime_data(seed=0, weeks=WEEKS)
        tree = Recorder(*_parts(make_tree(validation_fraction=0.25))).fit(X, returns, covariance)
        self.assertEqual(tree.refitted, [FITTING, WEEKS])
        self.assertEqual(sum(node.n_samples for node, _ in leaves(tree.root)), WEEKS)

        # One leaf. Asset 0 is the better one in the fitting rows, asset 1 by far
        # in the checking rows: the final scores must have seen the checking rows.
        rng = np.random.default_rng(0)
        means = np.where(np.arange(WEEKS)[:, None] < FITTING, [0.01, 0.0], [-0.02, 0.06])
        returns = means + rng.multivariate_normal(np.zeros(2), covariance, WEEKS)
        fitting_only = make_tree(max_depth=0).fit(X[:FITTING], returns[:FITTING], covariance)
        all_rows = make_tree(max_depth=0, validation_fraction=0.25).fit(X, returns, covariance)
        self.assertGreater(fitting_only.root.prediction[0], fitting_only.root.prediction[1])
        self.assertGreater(all_rows.root.prediction[1], all_rows.root.prediction[0])

    def test_the_refit_on_all_rows_is_kept_even_if_the_training_sharpe_falls(self):
        # The scores it replaces never saw the checking rows, so the Sharpe of
        # the training path is no fair judge between the two (seed 7: it falls).
        class RefitProbe(SPOPortfolioTree):
            def _refit_leaves(self, leaves, scores, path, on_all_rows=False):
                super()._refit_leaves(leaves, scores, path, on_all_rows)
                if on_all_rows:
                    refitted = scores.copy()
                    for leaf, rows, _ in leaves:
                        refitted[rows] = leaf.prediction
                    self.sharpe_before = path.sharpe
                    self.sharpe_after = self._replay(refitted).sharpe

        X, returns, covariance, _ = regime_data(seed=7, weeks=WEEKS)
        tree = RefitProbe(*_parts(make_tree(validation_fraction=0.25))).fit(X, returns, covariance)
        self.assertLess(tree.sharpe_after, tree.sharpe_before - 0.01)
        self.assertAlmostEqual(tree.replay(X, returns, covariance).sharpe, tree.sharpe_after,
                               places=12)

    def test_too_few_rows_for_the_two_parts_are_rejected(self):
        X, returns, covariance, _ = regime_data(seed=0)
        with self.assertRaises(ValueError):
            make_tree(validation_fraction=0.5).fit(X[:3], returns[:3], covariance)


if __name__ == "__main__":
    unittest.main()
