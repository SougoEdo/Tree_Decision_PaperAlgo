"""Readable SPO portfolio tree using only NumPy.

Install:  python -m pip install numpy        (the tests also use scipy)
Demo:     python main.py
Tests:    python -m unittest discover -s tests

Arrays supplied by the caller (all rows must be aligned):
    X:     (observations, features), available BEFORE the decision.
    R:     (observations, assets), subsequent SIMPLE holding-period returns.
    U:     (observations, assets), actual or scenario PRE-TRADE weights (fees).
    Sigma: (observations, assets, assets), covariance of the holding-period
           returns, estimated only from data available BEFORE the decision.

Use decimal returns: 0.01 means 1%. Use the same asset order everywhere.

Portfolio problem (long-only, fully invested, 2 to 4 assets):
    maximize    scores @ weights
                - sum(fee_rate * abs(weights - starting_weights))
                - risk_aversion * weights @ Sigma @ weights
    subject to  0 <= weights <= max_weight,  sum(weights) == 1

PortfolioOptimizer solves it exactly. Borrow costs, funding, market impact,
and execution are NOT modeled.

replay_path (or SPOPortfolioTree.replay) trades a strategy period after period
in row order: holdings drift with returns between rebalances, fees are paid on
each trade, and it reports net returns, turnover, regret and the Sharpe ratio.

Training is a fixed-state, one-period approximation: U stays fixed when
comparing candidate trees. This is NOT a sequential trading backtest and does
not generate the strategy's own historical holdings. Validate chronologically
with replay_path, which does both.

Leaf scores come from a bounded coordinate search on the empirical decision
regret, started from the leaf's mean return; it is approximate, not a global
solver. The fitted scores need not be calibrated expected-return forecasts.

Read in order: configurations -> optimizer -> replay -> leaf fitting -> splits
-> demo.
Based on the SPO-tree idea, not a reproduction of the papers' full experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import numpy as np


# 1. All investment and tree choices live in these configurations.
@dataclass(frozen=True)
class PortfolioConfig:
    max_weight: float = 1.0        # long-only: 0 <= weight <= max_weight, sum == 1
    fee_rate: float | tuple[float, ...] = 0.001  # per traded notional, per asset
    risk_aversion: float = 1.0     # > 0; multiplies weights @ Sigma @ weights


@dataclass(frozen=True)
class TreeConfig:
    max_depth: int = 4            # root depth is 0; zero means one leaf
    min_samples_leaf: int = 10
    max_thresholds: int | None = 10  # None tests every distinct partition
    min_regret_improvement: float = 1e-6  # average gain per observation at node
    prediction_bound: float = 0.10  # leaf scores restricted to +/- this
    search_passes: int = 3
    search_grid_size: int = 7
    regret_tolerance: float = 1e-6  # allowable numerical error per observation
    verbose: bool = False


MAX_ASSETS = 4             # the exact solver enumerates 5**n_assets cases
WEIGHT_TOLERANCE = 1e-9


def finite_array(value, name, ndim):
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty, finite {ndim}-D array.")
    return result


def covariance_rows(Sigma, rows, n_assets):
    """Accept one (n, n) covariance for every row, or one per row."""
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.shape == (n_assets, n_assets):
        Sigma = np.broadcast_to(Sigma, (rows, n_assets, n_assets))
    if Sigma.shape != (rows, n_assets, n_assets):
        raise ValueError("Sigma must have shape (n_assets, n_assets) "
                         "or (N, n_assets, n_assets).")
    return Sigma


# 2. A deterministic portfolio optimizer: it is solved, never trained.
class PortfolioOptimizer:
    """Exact solver for the long-only, fully-invested problem described above.

    At the optimum every weight is at 0, at max_weight, at its starting weight
    (fees put a kink there), or strictly above or below it. Each of these
    5**n_assets cases fixes some weights; the free ones solve a small linear
    system (a stationary point under the budget constraint). With
    risk_aversion > 0 and a positive-definite Sigma the objective is strictly
    concave, so the best case that is consistent with its own assumptions is
    the unique optimum. All rows are solved at once.
    """

    def __init__(self, n_assets: int, config: PortfolioConfig):
        self.n_assets = n_assets
        self.config = config
        if not isinstance(n_assets, int) or not 2 <= n_assets <= MAX_ASSETS:
            raise ValueError(f"n_assets must be an integer from 2 to {MAX_ASSETS}.")
        if not np.isfinite([config.max_weight, config.risk_aversion]).all():
            raise ValueError("Portfolio parameters must be finite.")
        if not 0 < config.max_weight <= 1 or n_assets * config.max_weight < 1:
            raise ValueError("max_weight must be in (0, 1] and n_assets * max_weight >= 1.")
        if config.risk_aversion <= 0:
            # A positive penalty makes the optimal portfolio unique, so the
            # regret of a prediction does not depend on how it is computed.
            raise ValueError("risk_aversion must be > 0.")
        self.fees = np.broadcast_to(np.asarray(config.fee_rate, dtype=float),
                                    (n_assets,)).copy()
        if not np.isfinite(self.fees).all() or np.any(self.fees < 0):
            raise ValueError("fee_rate must be nonnegative, scalar or per asset.")

        # Group the cases by which weights are free. A fixed weight is coded
        # 0 (at zero), 1 (at max_weight) or 2 (at its starting weight); a free
        # weight carries the side of its starting weight it lies on (+1 or -1).
        self._cases = []
        for n_free in range(n_assets + 1):
            for free in combinations(range(n_assets), n_free):
                fixed = [i for i in range(n_assets) if i not in free]
                combos = list(product(product((0, 1, 2), repeat=len(fixed)),
                                      product((1.0, -1.0), repeat=n_free)))
                states = np.array([s for s, _ in combos], dtype=int)
                sides = np.array([d for _, d in combos], dtype=float)
                self._cases.append((list(free), fixed,
                                    states.reshape(len(combos), len(fixed)),
                                    sides.reshape(len(combos), n_free)))

    def is_feasible(self, weights):
        """True when every row is long-only, capped, and fully invested."""
        w, tol = np.asarray(weights, dtype=float), WEIGHT_TOLERANCE
        return bool(np.isfinite(w).all() and np.all(w >= -tol)
                    and np.all(w <= self.config.max_weight + tol)
                    and np.all(np.abs(w.sum(axis=-1) - 1) <= tol))

    def utility(self, returns, weights, starting_weights, covariance):
        """returns @ w - fees @ |w - u| - risk_aversion * w @ Sigma @ w, per row."""
        r, w, u, S = (np.asarray(a, dtype=float)
                      for a in (returns, weights, starting_weights, covariance))
        return (np.einsum("...i,...i->...", r, w)
                - np.abs(w - u) @ self.fees
                - self.config.risk_aversion * np.einsum("...i,...ij,...j->...", w, S, w))

    def solve(self, scores, starting_weights, covariance):
        """Optimal weights for each row; all-single inputs give one portfolio.

        scores (n,) or (rows, n); starting_weights (n,) or (rows, n);
        covariance (n, n) or (rows, n, n).
        """
        c, u, S, single = self._rows(scores, starting_weights, covariance)
        rows, cap = len(c), self.config.max_weight
        lam, tol = self.config.risk_aversion, WEIGHT_TOLERANCE
        every_row = np.arange(rows)
        best_value = np.full(rows, -np.inf)
        best = np.zeros((rows, self.n_assets))
        for free, fixed, states, sides in self._cases:
            W = np.empty((len(states), rows, self.n_assets))
            for j, asset in enumerate(fixed):
                state = states[:, j, None]
                W[:, :, asset] = np.where(state == 0, 0.0,
                                          np.where(state == 1, cap, u[:, asset]))
            if free:
                # Free weights solve  A w + nu = g  and  sum(w) = budget,
                # with A = 2 * lam * Sigma[free, free] and nu the budget multiplier.
                A_inv = np.linalg.inv(2 * lam * S[:, free][:, :, free])
                g = (c[:, free] - sides[:, None, :] * self.fees[free]
                     - 2 * lam * np.einsum("rij,krj->kri",
                                           S[:, free][:, :, fixed], W[:, :, fixed]))
                budget = 1.0 - W[:, :, fixed].sum(axis=-1)
                A_inv_g = np.einsum("rij,krj->kri", A_inv, g)
                A_inv_1 = A_inv.sum(axis=-1)
                nu = (A_inv_g.sum(axis=-1) - budget) / A_inv_1.sum(axis=-1)
                w_free = A_inv_g - nu[..., None] * A_inv_1
                W[:, :, free] = w_free
                valid = np.all(sides[:, None, :] * (w_free - u[:, free]) >= -tol, axis=-1)
            else:
                valid = np.abs(W.sum(axis=-1) - 1.0) <= tol
            # Bounds apply to every weight, including one held at a starting
            # weight that drifted outside [0, max_weight].
            valid &= np.all((W >= -tol) & (W <= cap + tol), axis=-1)
            value = np.where(valid, self.utility(c, W, u, S), -np.inf)
            k = value.argmax(axis=0)
            improved = value[k, every_row] > best_value
            best_value[improved] = value[k, every_row][improved]
            best[improved] = W[k, every_row][improved]
        if not np.isfinite(best_value).all():
            raise RuntimeError("No feasible portfolio found; check the inputs.")
        weights = np.clip(best, 0.0, cap)
        return weights[0] if single else weights

    def _rows(self, scores, starting_weights, covariance):
        """Validate inputs and broadcast them to (rows, n), (rows, n), (rows, n, n)."""
        n = self.n_assets
        c, u, S = (np.asarray(a, dtype=float)
                   for a in (scores, starting_weights, covariance))
        single = c.ndim == 1 and u.ndim == 1 and S.ndim == 2
        c, u = np.atleast_2d(c), np.atleast_2d(u)
        S = S[None] if S.ndim == 2 else S
        if (c.ndim != 2 or u.ndim != 2 or S.ndim != 3 or c.shape[1] != n
                or u.shape[1] != n or S.shape[1:] != (n, n)):
            raise ValueError("Expect scores and starting weights with n_assets "
                             "columns and covariance matrices of n_assets x n_assets.")
        rows = max(len(c), len(u), len(S))
        if any(len(a) not in (1, rows) for a in (c, u, S)):
            raise ValueError("Inputs must have one row or the same number of rows.")
        c = np.broadcast_to(c, (rows, n))
        u = np.broadcast_to(u, (rows, n))
        S = np.broadcast_to(S, (rows, n, n))
        if not (np.isfinite(c).all() and np.isfinite(u).all() and np.isfinite(S).all()):
            raise ValueError("Scores, starting weights and covariance must be finite.")
        if not np.allclose(S, np.swapaxes(S, 1, 2)):
            raise ValueError("Covariance matrices must be symmetric.")
        try:
            np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrices must be positive definite.") from None
        return c, u, S, single


# 3. Replay a strategy through time: holdings drift, every trade pays fees.
@dataclass(frozen=True)
class PathResult:
    """What a strategy did in each period, in row order."""
    holdings: np.ndarray        # (T, n) pre-trade weights, drifted from the last period
    weights: np.ndarray         # (T, n) post-trade weights held over the period
    turnover: np.ndarray        # (T,) sum(|weights - holdings|)
    fees: np.ndarray            # (T,) fee_rate @ |weights - holdings|, fraction of equity
    gross_returns: np.ndarray   # (T,) returns @ weights
    net_returns: np.ndarray     # (T,) gross_returns - fees
    utility: np.ndarray         # (T,) net_returns - risk_aversion * weights @ Sigma @ weights
    regret: np.ndarray          # (T,) best utility in hindsight from the same holdings - utility
    final_holdings: np.ndarray  # (n,) drifted weights after the last period

    @property
    def sharpe(self) -> float:
        """Per-period Sharpe ratio of net returns; NaN with fewer than 2 periods."""
        if len(self.net_returns) < 2:
            return float("nan")
        std = self.net_returns.std(ddof=1)
        return float(self.net_returns.mean() / std) if std > 0 else float("nan")


def replay_path(optimizer, scores, returns, covariance, initial_weights=None):
    """Trade period by period, in row order (rows must be chronological).

    In period t the optimizer turns scores[t] into weights w_t, starting from
    the holdings u_t. Fees are charged on |w_t - u_t| and subtracted from the
    period return (the tiny fees-times-return cross term is ignored). Holdings
    then drift with the returns: u_{t+1} = w_t * (1 + r_t) / (1 + r_t @ w_t).
    The first holdings are equal weights unless initial_weights is given, e.g.
    the final_holdings of a previous path.
    """
    n = optimizer.n_assets
    C = finite_array(scores, "scores", 2)
    R = finite_array(returns, "returns", 2)
    if C.shape != R.shape or C.shape[1] != n:
        raise ValueError("scores and returns must both have shape (T, n_assets).")
    if np.any(R <= -1):
        raise ValueError("Simple returns must be greater than -1.")
    S = covariance_rows(covariance, len(R), n)
    if initial_weights is None:
        u = np.full(n, 1.0 / n)
    else:
        u = finite_array(initial_weights, "initial_weights", 1)
        if u.shape != (n,) or np.any(u < 0) or abs(u.sum() - 1) > 1e-6:
            raise ValueError("initial_weights must be n_assets nonnegative "
                             "weights summing to 1.")
        u = u / u.sum()

    holdings, weights = np.empty_like(R), np.empty_like(R)
    for t in range(len(R)):
        holdings[t] = u
        weights[t] = optimizer.solve(C[t], u, S[t])
        grown = weights[t] * (1 + R[t])
        u = grown / grown.sum()          # grown.sum() == 1 + r_t @ w_t

    trades = np.abs(weights - holdings)
    fees = trades @ optimizer.fees
    gross = np.einsum("ti,ti->t", R, weights)
    utility = optimizer.utility(R, weights, holdings, S)
    best = optimizer.utility(R, optimizer.solve(R, holdings, S), holdings, S)
    return PathResult(holdings=holdings, weights=weights, turnover=trades.sum(axis=1),
                      fees=fees, gross_returns=gross, net_returns=gross - fees,
                      utility=utility, regret=best - utility, final_holdings=u)


@dataclass
class Node:
    prediction: np.ndarray
    n_samples: int
    regret_sum: float
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None


# 4. Train one tree by evaluating portfolio decisions at candidate splits.
class SPOPortfolioTree:
    def __init__(self, optimizer: PortfolioOptimizer, config: TreeConfig):
        self.optimizer = optimizer
        self.config = config
        self.root = None
        if (config.max_depth < 0 or config.min_samples_leaf < 1
                or (config.max_thresholds is not None and config.max_thresholds < 1)
                or config.search_passes < 1 or config.search_grid_size < 3):
            raise ValueError("Invalid tree depth, leaf size, or search settings.")
        if (not np.isfinite([config.min_regret_improvement,
                             config.prediction_bound, config.regret_tolerance]).all()
                or config.min_regret_improvement < 0 or config.prediction_bound <= 0
                or config.regret_tolerance <= 0):
            raise ValueError("Invalid prediction bound or regret tolerance.")

    def fit(self, X, R, U, Sigma):
        """U and Sigma are explicit and stay fixed. No automatic dates or label shifting."""
        self.root = None
        X = finite_array(X, "X", 2)
        R = finite_array(R, "R", 2)
        U = finite_array(U, "U", 2)
        if (R.shape != U.shape or len(X) != len(R)
                or R.shape[1] != self.optimizer.n_assets):
            raise ValueError("Require X=(N,P) and R=U=(N,n_assets).")
        Sigma = covariance_rows(Sigma, len(R), self.optimizer.n_assets)
        self.n_features = X.shape[1]
        self._X, self._R, self._U, self._S = X.copy(), R.copy(), U.copy(), Sigma.copy()
        try:
            self._benchmarks = self.optimizer.utility(
                R, self.optimizer.solve(R, U, Sigma), U, Sigma)
            indices = np.arange(len(X))
            root = self._fit_leaf(indices)
            self.root = self._grow(root, indices, depth=0)
        finally:
            # Deployment uses the frozen tree, not stored training outcomes.
            for name in ("_X", "_R", "_U", "_S", "_benchmarks"):
                if hasattr(self, name):
                    delattr(self, name)
        return self

    def _score_prediction(self, prediction, indices):
        R, U, S = self._R[indices], self._U[indices], self._S[indices]
        weights = self.optimizer.solve(prediction, U, S)
        regret = self._benchmarks[indices] - self.optimizer.utility(R, weights, U, S)
        if np.min(regret) < -self.config.regret_tolerance:
            raise RuntimeError("Negative regret exceeds tolerance; check solver accuracy.")
        # Clip only negligible numerical negatives, not meaningful discrepancies.
        return float(np.maximum(regret, 0).sum())

    def _fit_leaf(self, indices, parent_prediction=None):
        # Deterministic bounded coordinate search, started from the leaf mean.
        # With fees or row-specific covariances the mean is only a starting
        # point, not the regret minimizer.
        c = self.config
        bound = c.prediction_bound
        best = np.clip(self._R[indices].mean(axis=0), -bound, bound)
        best_loss = self._score_prediction(best, indices)
        if parent_prediction is not None:
            loss = self._score_prediction(parent_prediction, indices)
            if loss < best_loss:
                best, best_loss = parent_prediction.copy(), loss
        # Including the parent lets a child retain its parent's prediction.
        # The first pass spans all allowed values; later passes refine locally.
        radius = 2 * bound
        for pass_number in range(c.search_passes):
            for asset in range(self.optimizer.n_assets):
                lower = -bound if pass_number == 0 else max(-bound, best[asset] - radius)
                upper = bound if pass_number == 0 else min(bound, best[asset] + radius)
                for value in np.linspace(lower, upper, c.search_grid_size):
                    candidate = best.copy()
                    candidate[asset] = value
                    loss = self._score_prediction(candidate, indices)
                    if loss < best_loss:
                        best, best_loss = candidate, loss
            radius /= (c.search_grid_size - 1)
        return Node(best, len(indices), best_loss)

    def _thresholds(self, values):
        unique, counts = np.unique(values, return_counts=True)
        thresholds = unique[:-1] / 2 + unique[1:] / 2
        # Rounding can put a midpoint on the upper endpoint: retain the intended
        # partition in that case by using the lower endpoint with the <= rule.
        thresholds = np.where(thresholds >= unique[1:], unique[:-1], thresholds)
        left_sizes = np.cumsum(counts)[:-1]
        valid = ((left_sizes >= self.config.min_samples_leaf)
                 & (len(values) - left_sizes >= self.config.min_samples_leaf))
        thresholds = thresholds[valid]
        limit = self.config.max_thresholds
        if limit is not None and len(thresholds) > limit:
            selected = np.linspace(0, len(thresholds) - 1, limit).astype(int)
            thresholds = thresholds[selected]
        return thresholds

    def _grow(self, node, indices, depth):
        c = self.config
        if depth >= c.max_depth or len(indices) < 2 * c.min_samples_leaf:
            return node
        best_split, best_loss = None, node.regret_sum
        for feature in range(self.n_features):
            values = self._X[indices, feature]
            for threshold in self._thresholds(values):
                left_indices = indices[values <= threshold]
                right_indices = indices[values > threshold]
                if min(len(left_indices), len(right_indices)) < c.min_samples_leaf:
                    continue
                left = self._fit_leaf(left_indices, node.prediction)
                right = self._fit_leaf(right_indices, node.prediction)
                split_loss = left.regret_sum + right.regret_sum
                if split_loss < best_loss:
                    best_loss = split_loss
                    best_split = (feature, threshold, left, right,
                                  left_indices, right_indices)
        gain = (node.regret_sum - best_loss) / len(indices)
        if best_split is None or gain <= c.min_regret_improvement:
            return node
        feature, threshold, left, right, left_indices, right_indices = best_split
        node.feature, node.threshold = feature, float(threshold)
        if c.verbose:
            print(f"depth={depth}, N={len(indices)}, feature={feature}, "
                  f"threshold={threshold:.5g}, mean regret improvement={gain:.6g}")
        node.left = self._grow(left, left_indices, depth + 1)
        node.right = self._grow(right, right_indices, depth + 1)
        return node

    def predict_returns(self, X):
        """Frozen leaf scores, with shape (observations, assets)."""
        if self.root is None:
            raise RuntimeError("Call fit before prediction.")
        X = finite_array(X, "X", 2)
        if X.shape[1] != self.n_features:
            raise ValueError("Feature count differs from training.")
        predictions = []
        for row in X:
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            predictions.append(node.prediction.copy())
        return np.vstack(predictions)

    def predict_weights(self, X, U, Sigma):
        """Current holdings and covariance are supplied separately from features."""
        predictions = self.predict_returns(X)
        U = finite_array(U, "U", 2)
        if U.shape != predictions.shape:
            raise ValueError("U must have one holdings vector per prediction.")
        Sigma = covariance_rows(Sigma, len(U), self.optimizer.n_assets)
        return self.optimizer.solve(predictions, U, Sigma)

    def replay(self, X, R, Sigma, initial_weights=None):
        """Trade the frozen tree through the rows of X in order; see replay_path."""
        return replay_path(self.optimizer, self.predict_returns(X), R, Sigma,
                           initial_weights)

    def mean_regret(self, X, R, U, Sigma):
        """Evaluate unseen fixed-state observations; does not refit the tree."""
        R = finite_array(R, "R", 2)
        U = finite_array(U, "U", 2)
        weights = self.predict_weights(X, U, Sigma)
        if R.shape != weights.shape:
            raise ValueError("R shape must match predicted weights.")
        Sigma = covariance_rows(Sigma, len(R), self.optimizer.n_assets)
        benchmarks = self.optimizer.utility(
            R, self.optimizer.solve(R, U, Sigma), U, Sigma)
        regret = benchmarks - self.optimizer.utility(R, weights, U, Sigma)
        if regret.min() < -self.config.regret_tolerance:
            raise RuntimeError("Negative regret exceeds numerical tolerance.")
        return float(np.maximum(regret, 0).mean())

    def describe(self, feature_names=None):
        """Print the frozen rules and leaf scores for inspection."""
        if self.root is None:
            raise RuntimeError("Call fit first.")
        names = (list(feature_names) if feature_names is not None
                 else [f"x[{j}]" for j in range(self.n_features)])
        if len(names) != self.n_features:
            raise ValueError("feature_names has the wrong length.")

        def visit(node, indent=""):
            if node.feature is None:
                print(f"{indent}leaf: N={node.n_samples}, "
                      f"scores={np.round(node.prediction, 6)}")
            else:
                print(f"{indent}if {names[node.feature]} <= {node.threshold:.6g}:")
                visit(node.left, indent + "  ")
                print(f"{indent}else:")
                visit(node.right, indent + "  ")
        visit(self.root)


# 5. Runnable synthetic example. Replace X, R, U, Sigma with your aligned dataset.
def demo():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(36, 2))
    # Expected returns flip with the sign of feature 0. The noise has a known
    # covariance, used directly as Sigma (no estimation in this demo).
    covariance = np.array([[1.6e-5, 0.8e-5],
                           [0.8e-5, 1.6e-5]])
    R = np.column_stack([
        np.where(X[:, 0] > 0, 0.025, -0.015),
        np.where(X[:, 0] > 0, -0.010, 0.020),
    ]) + rng.multivariate_normal(np.zeros(2), covariance, size=36)
    Sigma = np.broadcast_to(covariance, (36, 2, 2))
    # Synthetic fully-invested states, NOT holdings generated by this strategy.
    U = rng.dirichlet(np.ones(R.shape[1]), size=len(R))

    portfolio = PortfolioOptimizer(n_assets=R.shape[1], config=PortfolioConfig(
        max_weight=1.0, fee_rate=0.001, risk_aversion=500.0,
    ))
    tree = SPOPortfolioTree(portfolio, TreeConfig(
        max_depth=1, min_samples_leaf=6, max_thresholds=4,
        prediction_bound=0.06, search_passes=2, search_grid_size=5,
        verbose=True,
    ))
    # Illustrates an ordered holdout, not realistic crypto time-series evidence.
    cut = 24
    tree.fit(X[:cut], R[:cut], U[:cut], Sigma[:cut])
    tree.describe(["feature_0", "feature_1"])
    print("Holdout mean regret (fixed states):",
          round(tree.mean_regret(X[cut:], R[cut:], U[cut:], Sigma[cut:]), 6))
    path = tree.replay(X[cut:], R[cut:], Sigma[cut:])
    print(f"Holdout replay from equal weights: mean net return "
          f"{path.net_returns.mean():.4%}, Sharpe {path.sharpe:.3f} per period, "
          f"turnover {path.turnover.sum():.2f}, fees {path.fees.sum():.4%}")

    current_features = np.array([[0.8, -0.2]])
    current_weights = np.array([[0.60, 0.40]])
    target = tree.predict_weights(current_features, current_weights, covariance)
    print("Leaf scores:", tree.predict_returns(current_features)[0])
    print("Target weights:", target[0])
    print("Weight changes:", (target - current_weights)[0])
    print("Synthetic demonstration only; no market data or orders are sent.")


if __name__ == "__main__":
    demo()
