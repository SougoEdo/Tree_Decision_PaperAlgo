#!/usr/bin/env python3
"""Can the tree find a planted threshold? One risky asset and cash.

synthetic_data.py makes a mean-reverting feature x and a price whose drift is
-mu while x < d and +mu otherwise. Data come at every step; a decision is taken
every `step` steps and held until the next one, the way main.py takes weekly
decisions on daily data. The tree sees x together with decoys: a moving average
of x, an unrelated mean-reverting feature and white noise. The second asset is
cash: constant price, no fee.

A decision is held for `step` steps while x keeps moving, so the tree should
not find d itself but the value of x where the expected return over the holding
period changes sign; expected_return_curve computes that benchmark.

Run:  python recovery_test.py        one case, printed in detail (a few seconds)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from main import (PortfolioConfig, PortfolioOptimizer, SPOPortfolioTree, TreeConfig,
                  replay_constant_weights, replay_path)
from synthetic_data import moving_average, ou_process, signal_process

FEATURES = ["x", "x_ma", "decoy_ou", "decoy_noise"]


@dataclass(frozen=True)
class Case:
    """One synthetic setting. A step is the time unit of the data."""
    k: float = 0.2                  # reversion speed of x, per step
    feature_variance: float = 4.0   # long-run variance of x
    mu: float = 0.03                # drift per step: -mu while x < d, +mu otherwise
    price_variance: float = 0.001   # return variance per step
    d: float = 0.0                  # the planted threshold
    mean: float = 1.0               # long-run mean of x
    step: int = 10                  # steps between two decisions
    window: int = 180               # steps of past returns behind each covariance
    ma_window: int = 20             # steps in the moving average of x
    n_train: int = 443              # decisions used for training (fitting + checking)
    n_test: int = 1000              # decisions kept for the test
    ridge: float = 1e-6


@dataclass(frozen=True)
class Rows:
    """One row per decision: asset 0 is the risky asset, asset 1 is cash."""
    steps: np.ndarray    # (T,) the step of each decision
    X: np.ndarray        # (T, 4) FEATURES read at the decision step
    R: np.ndarray        # (T, 2) simple return from this decision to the next
    Sigma: np.ndarray    # (T, 2, 2) covariance of that return, from past steps only


def simulate(case, seed):
    """x, the unrelated decoy, white noise and the price, one value per step."""
    rng = np.random.default_rng(seed)
    n_steps = case.window + case.step * (case.n_train + case.n_test)
    x = ou_process(case.mean, case.feature_variance, case.k, n_steps=n_steps, rng=rng)
    decoy = ou_process(case.mean, case.feature_variance, case.k, n_steps=n_steps, rng=rng)
    noise = rng.normal(size=n_steps + 1)
    prices = signal_process(x, case.mu, case.d, case.price_variance, rng=rng)
    return x, decoy, noise, prices


def decision_rows(x, decoy, noise, prices, case):
    """Rows for a decision every case.step steps, after case.window steps of history.

    At a decision step t:
      X[t]     = x[t], the mean of the last ma_window values of x, decoy[t], noise[t]
      R[t]     = prices[t + step] / prices[t] - 1 for the asset, 0 for cash
      Sigma[t] = step * variance of the last `window` one-step returns up to t
                 (steps treated as independent), plus the ridge on both assets
    """
    t = np.arange(case.window, len(prices) - case.step, case.step)
    smooth = moving_average(x, case.ma_window)            # aligned with x[ma_window - 1:]
    X = np.column_stack([x[t], smooth[t - case.ma_window + 1], decoy[t], noise[t]])
    R = np.column_stack([prices[t + case.step] / prices[t] - 1, np.zeros(len(t))])
    one_step = prices[1:] / prices[:-1] - 1               # one_step[s - 1] ends at step s
    Sigma = np.zeros((len(t), 2, 2))
    for row, now in enumerate(t):
        Sigma[row, 0, 0] = case.step * one_step[now - case.window:now].var(ddof=1)
    Sigma += case.ridge * np.eye(2)
    return Rows(t, X, R, Sigma)


def expected_return_curve(case, grid, n_paths=20000, seed=0):
    """E[return of the asset over the next case.step steps | x = grid value now].

    Monte Carlo on the exact transition of x. With S the number of steps spent at
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
    return float(grid[i] - curve[i] * (grid[i + 1] - grid[i]) / (curve[i + 1] - curve[i]))


def splits_of(node, depth=0):
    """(depth, feature name, threshold) of every split left in the tree."""
    if node.feature is None:
        return []
    return ([(depth, FEATURES[node.feature], node.threshold)]
            + splits_of(node.left, depth + 1) + splits_of(node.right, depth + 1))


def run_case(case, seed, verbose=False, curve=None):
    """Fit the tree on the training decisions; compare on the test decisions.

    curve = (grid, expected returns) from expected_return_curve adds the ideal
    rule: the optimizer fed with the true expected return given x.
    """
    series = simulate(case, seed)
    rows = decision_rows(*series, case)
    X, R, S = rows.X, rows.R, rows.Sigma
    train, test = slice(0, case.n_train), slice(case.n_train, None)
    optimizer = PortfolioOptimizer(2, PortfolioConfig(
        max_weight=1.0, fee_rate=(0.0003, 0.0), risk_aversion=1.0))
    config = TreeConfig(max_depth=2, min_samples_leaf=20, max_thresholds=None,
                        validation_fraction=0.25, verbose=verbose)
    tree = SPOPortfolioTree(optimizer, config).fit(X[train], R[train], S[train])
    flat = SPOPortfolioTree(optimizer, replace(config, max_depth=0, verbose=False))
    flat.fit(X[train], R[train], S[train])

    # The rule at d: the asset while x >= d, cash otherwise (right for one-step decisions).
    bound = config.prediction_bound
    at_d = np.column_stack([np.where(X[:, 0] >= case.d, bound, -bound), np.zeros(len(X))])
    strategies = {
        "tree": lambda part, start: tree.replay(X[part], R[part], S[part], start),
        "tree without splits": lambda part, start: flat.replay(X[part], R[part], S[part], start),
        "rule at d": lambda part, start: replay_path(optimizer, at_d[part], R[part],
                                                     S[part], start),
        "always the asset": lambda part, start: replay_constant_weights(
            optimizer, np.array([1.0, 0.0]), R[part], S[part], start),
    }
    if curve is not None:
        ideal = np.column_stack([np.interp(X[:, 0], *curve), np.zeros(len(X))])
        strategies["ideal rule"] = lambda part, start: replay_path(
            optimizer, ideal[part], R[part], S[part], start)
    paths = {}
    for name, run in strategies.items():
        paths[name] = run(test, run(train, None).final_holdings)
    return {
        "case": case, "seed": seed, "tree": tree, "flat": flat, "paths": paths,
        "series": series, "rows": rows, "optimizer": optimizer,
        "grown": [(g.depth, FEATURES[g.feature], g.threshold) for g in tree.growth_log],
        "kept": splits_of(tree.root),
        "feature_std": float(np.sqrt(case.feature_variance)),
    }


def describe(result):
    case, tree = result["case"], result["tree"]
    print(f"\nplanted: drift -{case.mu:g} per step while x < {case.d:g}, +{case.mu:g} otherwise; "
          f"price volatility {np.sqrt(case.price_variance):.2%} per step; k = {case.k:g}; "
          f"one decision every {case.step} steps")
    print(f"training decisions: {case.n_train}; test decisions: {case.n_test}\n")
    tree.describe(FEATURES)
    print("\nsplits grown :", [(d, f, round(th, 4)) for d, f, th in result["grown"]])
    print("splits kept  :", [(d, f, round(th, 4)) for d, f, th in result["kept"]])
    print(f"thresholds kept on x: {[round(th, 4) for _, f, th in result['kept'] if f == 'x']}; "
          f"planted d = {case.d:g}; the expected {case.step}-step return changes sign at "
          f"x = {result['sign_change']:+.3f}")
    print(f"\ntest, per decision ({case.step} steps){'':8}mean return   volatility   Sharpe   turnover")
    for name, path in result["paths"].items():
        print(f"  {name:24s}{path.net_returns.mean():>12.3%}{path.net_returns.std(ddof=1):>13.3%}"
              f"{path.sharpe:>9.3f}{path.turnover.mean():>11.2f}")


if __name__ == "__main__":
    case = Case()
    grid = np.linspace(case.mean - 4 * np.sqrt(case.feature_variance),
                       case.mean + 4 * np.sqrt(case.feature_variance), 321)
    curve = expected_return_curve(case, grid)
    result = run_case(case, seed=42, verbose=True, curve=(grid, curve))
    result["sign_change"] = sign_change(grid, curve)
    describe(result)
