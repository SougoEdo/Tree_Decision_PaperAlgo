#!/usr/bin/env python3
"""Can the tree find a planted threshold? One risky asset and cash, on daily data.

synthetic_data.py makes a mean-reverting feature x and a price whose drift is
-mu while x < d and +mu otherwise. The clock is the trading day: x and the
price are observed every day. A decision is taken every `step` days and held
until the next one, the way main.py takes weekly decisions on daily data. The
tree sees x alone; the second asset is cash (constant price, no fee).

The threshold d sits at the long-run mean of x. The expected return over any
holding period is then antisymmetric around d, so the value of x where a
correct tree changes its decision is d itself, at every trading rate;
expected_return_curve computes that expected return, whose size shows how much
edge is left once the regime has had `step` days to change.

The parameters are sized on daily markets: a volatility of 1% a day (16% a
year), a drift of a few basis points a day (edge = drift / volatility), a
feature that mean-reverts over weeks, 15 years of training data and a fee of
3 basis points per trade.

Run:  python recovery_test.py        one case, printed in detail
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from main import (
    PortfolioConfig,
    PortfolioOptimizer,
    SPOPortfolioTree,
    TreeConfig,
    replay_constant_weights,
    replay_path,
)
from synthetic_data import ou_process, signal_process

FEATURES = ["x"]
DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Case:
    """One synthetic setting. The time unit is one trading day."""

    k: float = 0.02  # reversion speed of x per day: a half-life of 35 days
    feature_variance: float = 1.0  # long-run variance of x (a standardized feature)
    mean: float = 0.0  # long-run mean of x
    d: float = 0.0  # the planted threshold on x, at its long-run mean
    volatility: float = 0.01  # price volatility per day: 16% a year
    edge: float = 0.10  # drift / volatility per day: mu = edge * volatility
    step: int = 5  # days between two decisions: the trading rate
    window: int = 180  # days of past returns behind each covariance
    train_days: int = 15 * DAYS_PER_YEAR  # days of training data (fitting + checking)
    test_days: int = 5 * DAYS_PER_YEAR  # days kept for the test
    ridge: float = 1e-6
    # the decision problem and the tree (defaults: the run of the note)
    risk_aversion: float = 1.0  # lambda of the optimizer
    fee: float = 0.0003  # per traded notional on the asset; cash is free
    max_depth: int = 2
    min_samples_leaf: int = 20
    max_thresholds: int | None = None  # None: every partition is screened
    threshold_quantiles: tuple[float, ...] | None = None  # e.g. (0.25, 0.5, 0.75)

    @property
    def mu(self):
        """Drift per day: -mu while x < d, +mu otherwise."""
        return self.edge * self.volatility

    @property
    def price_variance(self):
        return self.volatility**2

    @property
    def n_train(self):
        """Decisions used for training (fitting + checking)."""
        return self.train_days // self.step

    @property
    def n_test(self):
        """Decisions kept for the test."""
        return self.test_days // self.step

    @property
    def decisions_per_year(self):
        return DAYS_PER_YEAR / self.step


@dataclass(frozen=True)
class Rows:
    """One row per decision: asset 0 is the risky asset, asset 1 is cash."""

    days: np.ndarray  # (T,) the day of each decision
    X: np.ndarray  # (T, 1) x read on the decision day
    R: np.ndarray  # (T, 2) simple return from this decision to the next
    Sigma: np.ndarray  # (T, 2, 2) covariance of that return, from past days only


def simulate(case, seed):
    """x and the price, one value per day."""
    rng = np.random.default_rng(seed)
    n_days = case.window + case.step * (case.n_train + case.n_test)
    x = ou_process(case.mean, case.feature_variance, case.k, n_steps=n_days, rng=rng)
    prices = signal_process(x, case.mu, case.d, case.price_variance, rng=rng)
    return x, prices


def decision_rows(x, prices, case):
    """Rows for a decision every case.step days, after case.window days of history.

    On a decision day t:
      X[t]     = x[t]
      R[t]     = prices[t + step] / prices[t] - 1 for the asset, 0 for cash
      Sigma[t] = step * variance of the last `window` daily returns up to t
                 (days treated as independent), plus the ridge on both assets
    """
    t = np.arange(case.window, len(prices) - case.step, case.step)
    X = x[t, None]
    R = np.column_stack([prices[t + case.step] / prices[t] - 1, np.zeros(len(t))])
    daily = prices[1:] / prices[:-1] - 1  # daily[s - 1] ends on day s
    Sigma = np.zeros((len(t), 2, 2))
    for row, now in enumerate(t):
        Sigma[row, 0, 0] = case.step * daily[now - case.window : now].var(ddof=1)
    Sigma += case.ridge * np.eye(2)
    return Rows(t, X, R, Sigma)


def expected_return_curve(case, grid, n_paths=20000, seed=0):
    """E[return of the asset over the next case.step days | x = grid value now].

    Monte Carlo on the exact transition of x. With S the number of days spent at
    or above d minus the number spent below, the expected simple return is
    E[exp(mu * S)] - 1: the price noise has mean zero and drops out.
    """
    rng = np.random.default_rng(seed)
    decay = np.exp(-case.k)
    noise_std = np.sqrt(case.feature_variance * -np.expm1(-2 * case.k))
    x = np.repeat(np.asarray(grid, dtype=float)[:, None], n_paths, axis=1)
    balance = np.where(x >= case.d, 1.0, -1.0)
    for _ in range(case.step - 1):
        x = case.mean + decay * (x - case.mean) + rng.normal(0, noise_std, x.shape)
        balance += np.where(x >= case.d, 1.0, -1.0)
    return np.exp(case.mu * balance).mean(axis=1) - 1


def sign_change(grid, curve):
    """The grid value where the curve goes from negative to positive (None if it never does)."""
    crossing = np.flatnonzero((curve[:-1] < 0) & (curve[1:] >= 0))
    if len(crossing) == 0:
        return None
    i = crossing[0]
    return float(
        grid[i] - curve[i] * (grid[i + 1] - grid[i]) / (curve[i + 1] - curve[i])
    )


def splits_of(node, depth=0):
    """(depth, feature name, threshold) of every split left in the tree."""
    if node.feature is None:
        return []
    return (
        [(depth, FEATURES[node.feature], node.threshold)]
        + splits_of(node.left, depth + 1)
        + splits_of(node.right, depth + 1)
    )


def annual(path, case):
    """Annualized figures of a path of decisions held case.step days each."""
    return path.summary(periods_per_year=case.decisions_per_year)


def run_case(case, seed, verbose=False, curve=None):
    """Fit the tree on the training decisions; compare on the test decisions.

    curve = (grid, expected returns) from expected_return_curve adds the ideal
    rule: the optimizer fed with the true expected return given x.
    """
    series = simulate(case, seed)
    rows = decision_rows(*series, case)
    X, R, S = rows.X, rows.R, rows.Sigma
    train, test = slice(0, case.n_train), slice(case.n_train, None)
    optimizer = PortfolioOptimizer(
        2, PortfolioConfig(max_weight=1.0, fee_rate=(case.fee, 0.0),
                           risk_aversion=case.risk_aversion)
    )
    config = TreeConfig(
        max_depth=case.max_depth,
        min_samples_leaf=case.min_samples_leaf,
        max_thresholds=case.max_thresholds,
        threshold_quantiles=case.threshold_quantiles,
        validation_fraction=0.25,
        verbose=verbose,
    )
    tree = SPOPortfolioTree(optimizer, config).fit(X[train], R[train], S[train])
    flat = SPOPortfolioTree(optimizer, replace(config, max_depth=0, verbose=False))
    flat.fit(X[train], R[train], S[train])

    # The rule at d: the asset while x >= d, cash otherwise.
    bound = config.prediction_bound
    at_d = np.column_stack(
        [np.where(X[:, 0] >= case.d, bound, -bound), np.zeros(len(X))]
    )
    strategies = {
        "tree": lambda part, start: tree.replay(X[part], R[part], S[part], start),
        "tree without splits": lambda part, start: flat.replay(
            X[part], R[part], S[part], start
        ),
        "rule at d": lambda part, start: replay_path(
            optimizer, at_d[part], R[part], S[part], start
        ),
        "always the asset": lambda part, start: replay_constant_weights(
            optimizer, np.array([1.0, 0.0]), R[part], S[part], start
        ),
    }
    if curve is not None:
        ideal = np.column_stack([np.interp(X[:, 0], *curve), np.zeros(len(X))])
        strategies["ideal rule"] = lambda part, start: replay_path(
            optimizer, ideal[part], R[part], S[part], start
        )
    paths = {}
    for name, run in strategies.items():
        paths[name] = run(test, run(train, None).final_holdings)
    return {
        "case": case,
        "seed": seed,
        "tree": tree,
        "flat": flat,
        "paths": paths,
        "series": series,
        "rows": rows,
        "optimizer": optimizer,
        "grown": [(g.depth, FEATURES[g.feature], g.threshold) for g in tree.growth_log],
        "kept": splits_of(tree.root),
        "feature_std": float(np.sqrt(case.feature_variance)),
    }


def describe(result):
    case, tree = result["case"], result["tree"]
    print(
        f"\nplanted: drift {case.mu:.2%} a day ({case.mu * DAYS_PER_YEAR:+.0%} a year) while "
        f"x >= {case.d:g}, the opposite below; volatility {case.volatility:.1%} a day "
        f"({case.volatility * np.sqrt(DAYS_PER_YEAR):.0%} a year); edge {case.edge:g}; "
        f"x has mean {case.mean:g}, variance {case.feature_variance:g}, k = {case.k:g} a day "
        f"(half-life {np.log(2) / case.k:.0f} days); one decision every {case.step} days"
    )
    print(
        f"training decisions: {case.n_train} ({case.train_days / DAYS_PER_YEAR:g} years); "
        f"test decisions: {case.n_test} ({case.test_days / DAYS_PER_YEAR:g} years)\n"
    )
    tree.describe(FEATURES)
    print("\nsplits grown :", [(d, f, round(th, 4)) for d, f, th in result["grown"]])
    print("splits kept  :", [(d, f, round(th, 4)) for d, f, th in result["kept"]])
    print(
        f"thresholds kept on x: {[round(th, 4) for _, f, th in result['kept'] if f == 'x']}; "
        f"planted d = {case.d:g}; the expected {case.step}-day return changes sign at "
        f"x = {result['sign_change']:+.3f}"
    )
    print(
        f"\ntest, annualized{'':16}return   volatility   Sharpe   turnover per decision"
    )
    for name, path in result["paths"].items():
        figures = annual(path, case)
        print(
            f"  {name:24s}{figures['return']:>8.1%}{figures['volatility']:>13.1%}"
            f"{figures['sharpe']:>9.2f}{figures['turnover']:>11.2f}"
        )


if __name__ == "__main__":
    case = Case()
    grid = np.linspace(
        case.mean - 4 * np.sqrt(case.feature_variance),
        case.mean + 4 * np.sqrt(case.feature_variance),
        321,
    )
    curve = expected_return_curve(case, grid)
    result = run_case(case, seed=42, verbose=True, curve=(grid, curve))
    result["sign_change"] = sign_change(grid, curve)
    describe(result)
