"""Readable SPO portfolio tree using only NumPy and SciPy.

Install:  python -m pip install numpy scipy
Demo:     python main.py

Arrays supplied by the caller (all rows must be aligned):
    X: (observations, features), available BEFORE the decision.
    R: (observations, assets), subsequent SIMPLE holding-period returns.
    U: (observations, assets), actual or scenario PRE-TRADE weights (fees).

Use decimal returns: 0.01 means 1%. Use the same asset order everywhere.

Portfolio objective:
    predicted_return @ weights
    - sum(fee_rate * abs(weights - starting_weights))
    - l2_penalty * sum(weights**2)

Weights are fractions of portfolio equity. The default portfolio is fully
invested and long-only: weights >= 0 and sum(weights) == 1. For long/short
exposures, set min_weight < 0 and net_exposure=None (sum(abs(weights)) <=
gross_limit); residual cash/collateral and financing then earn zero.
Borrow costs, funding, market impact, margin rules, and execution are NOT modeled.

Training is a fixed-state, one-period approximation: U stays fixed when
comparing candidate trees. This is NOT a sequential trading backtest and does
not generate the strategy's own historical holdings. Validate chronologically
and then run a separate sequential simulation with drifted holdings and costs.

With zero fees, average-return leaves minimize the stated empirical decision
regret (fixed constraints and penalty). With fees, bounded coordinate search
fits decision-inducing return scores; it is approximate, not a global solver.
The fitted scores need not be calibrated expected-return forecasts.

Read in order: configurations -> optimizer -> leaf fitting -> splits -> demo.
Based on the SPO-tree idea, not a reproduction of the papers' full experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


# 1. All investment and tree choices live in these configurations.
@dataclass(frozen=True)
class PortfolioConfig:
    min_weight: float = 0.0
    max_weight: float = 1.0
    gross_limit: float = 1.0       # sum(abs(weights)) <= gross_limit
    net_exposure: float | None = 1.0  # require sum(weights) == value; None disables
    fee_rate: float | tuple[float, ...] = 0.001  # per traded notional, per asset
    l2_penalty: float = 0.01       # > 0; concentration/exposure penalty, not a fee
    solver_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-7
    solver_max_iterations: int = 500


@dataclass(frozen=True)
class TreeConfig:
    max_depth: int = 4            # root depth is 0; zero means one leaf
    min_samples_leaf: int = 10
    max_thresholds: int | None = 10  # None tests every distinct partition
    min_regret_improvement: float = 1e-6  # average gain per observation at node
    prediction_bound: float = 0.10  # fee-aware leaf scores restricted to +/- this
    search_passes: int = 3
    search_grid_size: int = 7
    regret_tolerance: float = 1e-6  # allowable numerical error per observation
    verbose: bool = False


def finite_array(value, name, ndim):
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty, finite {ndim}-D array.")
    return result


# 2. A deterministic portfolio optimizer: it is solved, never trained.
class PortfolioOptimizer:
    def __init__(self, n_assets: int, config: PortfolioConfig):
        self.n_assets = n_assets
        self.config = config
        if not isinstance(n_assets, int) or n_assets < 1:
            raise ValueError("n_assets must be a positive integer.")
        if not np.isfinite([config.min_weight, config.max_weight,
                            config.gross_limit, config.l2_penalty,
                            config.solver_tolerance,
                            config.feasibility_tolerance]).all():
            raise ValueError("Portfolio parameters must be finite.")
        if (config.min_weight >= config.max_weight or config.gross_limit < 0
                or config.solver_tolerance <= 0
                or config.feasibility_tolerance <= 0
                or config.solver_max_iterations < 1):
            raise ValueError("Invalid bounds or solver settings.")
        if config.l2_penalty <= 0:
            # A positive penalty makes the optimal portfolio unique, so the
            # regret of a prediction does not depend on the solver's start.
            raise ValueError("l2_penalty must be > 0.")
        if config.net_exposure is not None and not np.isfinite(config.net_exposure):
            raise ValueError("net_exposure must be finite or None.")
        self.fees = np.broadcast_to(np.asarray(config.fee_rate, dtype=float),
                                    (n_assets,)).copy()
        if not np.isfinite(self.fees).all() or np.any(self.fees < 0):
            raise ValueError("fee_rate must be nonnegative, scalar or per asset.")

        n = n_assets
        eye, zero = np.eye(n), np.zeros((n, n))
        # Variables z = [w, trade, gross]. Auxiliary variables represent:
        # trade >= abs(w - u), gross >= abs(w), sum(gross) <= gross_limit.
        self.A = np.vstack([
            np.hstack([eye, -eye, zero]),
            np.hstack([-eye, -eye, zero]),
            np.hstack([eye, zero, -eye]),
            np.hstack([-eye, zero, -eye]),
            np.concatenate([np.zeros(2 * n), np.ones(n)])[None, :],
        ])
        self.bounds = Bounds(
            np.r_[np.full(n, config.min_weight), np.zeros(2 * n)],
            np.r_[np.full(n, config.max_weight), np.full(2 * n, np.inf)],
        )
        self.Aeq = np.r_[np.ones(n), np.zeros(2 * n)][None, :]
        # Check feasibility once and retain a valid starting portfolio.
        feasible = linprog(
            np.zeros(3 * n), A_ub=self.A, b_ub=self._rhs(np.zeros(n)),
            A_eq=None if config.net_exposure is None else self.Aeq,
            b_eq=None if config.net_exposure is None else [config.net_exposure],
            bounds=list(zip(self.bounds.lb, self.bounds.ub)), method="highs",
        )
        if not feasible.success:
            raise ValueError(f"Infeasible portfolio constraints: {feasible.message}")
        self.initial_weights = feasible.x[:n]

    def _rhs(self, starting_weights):
        return np.r_[starting_weights, -starting_weights,
                     np.zeros(2 * self.n_assets), self.config.gross_limit]

    def is_feasible(self, weights):
        w, c = np.asarray(weights), self.config
        tol = c.feasibility_tolerance
        return bool(
            np.isfinite(w).all()
            and np.all(w >= c.min_weight - tol)
            and np.all(w <= c.max_weight + tol)
            and np.abs(w).sum() <= c.gross_limit + tol
            and (c.net_exposure is None or abs(w.sum() - c.net_exposure) <= tol)
        )

    def utility(self, returns, weights, starting_weights):
        """Realized/predicted utility. Also supports batches via broadcasting."""
        r, w, u = map(np.asarray, (returns, weights, starting_weights))
        return (np.sum(r * w, axis=-1)
                - np.sum(self.fees * np.abs(w - u), axis=-1)
                - self.config.l2_penalty * np.sum(w * w, axis=-1))

    def solve(self, predicted_returns, starting_weights):
        r = finite_array(predicted_returns, "predicted_returns", 1)
        u = finite_array(starting_weights, "starting_weights", 1)
        n, c = self.n_assets, self.config
        if r.shape != (n,) or u.shape != (n,):
            raise ValueError("Return and holdings vectors must match n_assets.")
        # Incoming holdings may violate target limits after price drift.
        w0 = u if self.is_feasible(u) else self.initial_weights
        # Give the auxiliary inequalities slack where possible. Starting every
        # absolute-value constraint at equality can make SLSQP's subproblem
        # degenerate even though this convex portfolio problem is feasible.
        slack = max(0.0, c.gross_limit - np.abs(w0).sum()) / (2 * n)
        z0 = np.r_[w0, np.abs(w0 - u) + 0.1, np.abs(w0) + slack]
        rhs = self._rhs(u)
        constraints = [LinearConstraint(self.A, -np.inf, rhs)]
        if c.net_exposure is not None:
            constraints.append(LinearConstraint(self.Aeq, c.net_exposure,
                                                c.net_exposure))
        # Scaling improves numerical accuracy for small decimal returns.
        scale = max(float(np.max(np.abs(r))), float(self.fees.max()),
                    c.l2_penalty, 1e-6)

        def objective(z):
            w, trade = z[:n], z[n:2 * n]
            return (-r @ w + self.fees @ trade + c.l2_penalty * (w @ w)) / scale

        def gradient(z):
            return np.r_[-r + 2 * c.l2_penalty * z[:n],
                         self.fees, np.zeros(n)] / scale

        result = minimize(objective, z0, jac=gradient, method="SLSQP",
                          bounds=self.bounds, constraints=constraints,
                          options={"ftol": c.solver_tolerance,
                                   "maxiter": c.solver_max_iterations})
        w = result.x[:n]
        violation = np.max(self.A @ result.x - rhs)
        primal_ok = (self.is_feasible(w) and np.isfinite(result.x).all()
                     and violation <= c.feasibility_tolerance
                     and np.max(self.bounds.lb - result.x) <= c.feasibility_tolerance)
        optimal = result.success
        if not optimal and primal_ok:
            # SLSQP can warn at a valid optimum when constraints are redundant.
            # For convex f, min_z grad(f)(x) @ z provides a global optimality
            # certificate: f(x)-f(optimum) <= grad(f)(x) @ x - min_z grad(f)(x) @ z.
            # Accept a warning ONLY when this bound is within tolerance.
            grad = gradient(result.x)
            certificate = linprog(
                grad, A_ub=self.A, b_ub=rhs,
                A_eq=None if c.net_exposure is None else self.Aeq,
                b_eq=None if c.net_exposure is None else [c.net_exposure],
                bounds=list(zip(self.bounds.lb, self.bounds.ub)), method="highs",
            )
            optimal = (certificate.success and
                       grad @ result.x - certificate.fun <= 10 * c.solver_tolerance)
        if not optimal or not primal_ok:
            raise RuntimeError(f"Portfolio solve failed: {result.message}; "
                               f"constraint violation={violation:.3g}")
        return w.copy()  # Never silently substitute a portfolio on failure.


@dataclass
class Node:
    prediction: np.ndarray
    n_samples: int
    regret_sum: float
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None


# 3. Train one tree by evaluating portfolio decisions at candidate splits.
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

    def fit(self, X, R, U):
        """U is explicit and stays fixed. No automatic dates or label shifting."""
        self.root = None
        X = finite_array(X, "X", 2)
        R = finite_array(R, "R", 2)
        U = finite_array(U, "U", 2)
        if (R.shape != U.shape or len(X) != len(R)
                or R.shape[1] != self.optimizer.n_assets):
            raise ValueError("Require X=(N,P) and R=U=(N,n_assets).")
        self.n_features = X.shape[1]
        self._X, self._R, self._U = X.copy(), R.copy(), U.copy()
        try:
            self._benchmarks = np.array([
                self.optimizer.utility(r, self.optimizer.solve(r, u), u)
                for r, u in zip(R, U)
            ])
            indices = np.arange(len(X))
            root = self._fit_leaf(indices)
            self.root = self._grow(root, indices, depth=0)
        finally:
            # Deployment uses the frozen tree, not stored training outcomes.
            for name in ("_X", "_R", "_U", "_benchmarks"):
                if hasattr(self, name):
                    delattr(self, name)
        return self

    def _score_prediction(self, prediction, indices):
        R, U = self._R[indices], self._U[indices]
        if np.all(self.optimizer.fees == 0):
            weights = self.optimizer.solve(prediction, U[0])
        else:
            weights = np.array([self.optimizer.solve(prediction, u) for u in U])
        regret = self._benchmarks[indices] - self.optimizer.utility(R, weights, U)
        if np.min(regret) < -self.config.regret_tolerance:
            raise RuntimeError("Negative regret exceeds tolerance; check solver accuracy.")
        # Clip only negligible numerical negatives, not meaningful discrepancies.
        return float(np.maximum(regret, 0).sum())

    def _fit_leaf(self, indices, parent_prediction=None):
        mean = self._R[indices].mean(axis=0)
        if np.all(self.optimizer.fees == 0):
            # Exact empirical leaf fit for the common, fixed regularized objective.
            return Node(mean, len(indices), self._score_prediction(mean, indices))

        # Fee-aware approximation: deterministic bounded coordinate search.
        c = self.config
        bound = c.prediction_bound
        best = np.clip(mean, -bound, bound)
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

    def predict_weights(self, X, U):
        """Current holdings are supplied separately from market features."""
        predictions = self.predict_returns(X)
        U = finite_array(U, "U", 2)
        if U.shape != predictions.shape:
            raise ValueError("U must have one holdings vector per prediction.")
        return np.array([self.optimizer.solve(r, u) for r, u in zip(predictions, U)])

    def mean_regret(self, X, R, U):
        """Evaluate unseen fixed-state observations; does not refit the tree."""
        R = finite_array(R, "R", 2)
        U = finite_array(U, "U", 2)
        weights = self.predict_weights(X, U)
        if R.shape != weights.shape:
            raise ValueError("R shape must match predicted weights.")
        benchmarks = np.array([
            self.optimizer.utility(r, self.optimizer.solve(r, u), u)
            for r, u in zip(R, U)
        ])
        regret = benchmarks - self.optimizer.utility(R, weights, U)
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


# 4. Runnable synthetic example. Replace X, R, U with your aligned dataset.
def demo():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(36, 2))
    R = np.column_stack([
        np.where(X[:, 0] > 0, 0.025, -0.015),
        np.where(X[:, 0] > 0, -0.010, 0.020),
    ]) + rng.normal(0, 0.004, size=(36, 2))
    # Synthetic fully-invested states, NOT holdings generated by this strategy.
    U = rng.dirichlet(np.ones(R.shape[1]), size=len(R))

    portfolio = PortfolioOptimizer(n_assets=R.shape[1], config=PortfolioConfig(
        min_weight=0.0, max_weight=1.0, gross_limit=1.0,
        net_exposure=1.0, fee_rate=0.001, l2_penalty=0.01,
    ))
    tree = SPOPortfolioTree(portfolio, TreeConfig(
        max_depth=1, min_samples_leaf=6, max_thresholds=4,
        prediction_bound=0.06, search_passes=2, search_grid_size=5,
        verbose=True,
    ))
    # Illustrates an ordered holdout, not realistic crypto time-series evidence.
    cut = 24
    tree.fit(X[:cut], R[:cut], U[:cut])
    tree.describe(["feature_0", "feature_1"])
    print("Holdout mean regret (fixed states):",
          round(tree.mean_regret(X[cut:], R[cut:], U[cut:]), 6))

    current_features = np.array([[0.8, -0.2]])
    current_weights = np.array([[0.60, 0.40]])
    target = tree.predict_weights(current_features, current_weights)
    print("Leaf scores:", tree.predict_returns(current_features)[0])
    print("Target signed weights:", target[0])
    print("Weight changes:", (target - current_weights)[0])
    print("Synthetic demonstration only; no market data or orders are sent.")


if __name__ == "__main__":
    demo()
