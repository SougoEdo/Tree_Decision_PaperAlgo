#!/usr/bin/env python3
"""Readable SPO portfolio tree using only NumPy.

Install:  python -m pip install numpy, scipy
Demo:     python main.py
Tests:    python -m unittest discover -s tests

Arrays supplied by the caller (rows aligned and in time order):
    X:     (observations, features), available BEFORE the decision.
    R:     (observations, assets), subsequent SIMPLE holding-period returns.
    Sigma: (observations, assets, assets), covariance of the holding-period
           returns, estimated only from data available BEFORE the decision.
    U:     (observations, assets), current PRE-TRADE weights, only for a live
           decision (predict_weights); training replays its own holdings.

Use decimal returns: 0.01 means 1%. Use the same asset order everywhere.
weekly_rows builds X, R and Sigma for weekly decisions from consecutive daily
closes and your own daily features.

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
replay_constant_weights does the same for a fixed target such as equal weights.
train_and_test and performance_report compare the tree with  the same tree
without splits and with an equal-weight portfolio, on training and test rows
(annualized return, volatility, Sharpe, max drawdown, turnover and fees).

Training (fit) is path-aware and grows the tree best-first. Each round replays
the current tree to get the holdings it would have had, screens every split of
every leaf with mean leaf scores on those holdings, and fits the scores of the
best few. The decision regret chooses the split: the one with the largest regret
reduction, measured against its leaf refitted on the same holdings. It is
accepted only if it also raises the Sharpe ratio of the replayed path, so
splits that make the strategy flip between leaves and pay fees are rejected.

Both checks above look at the rows the leaf scores were fitted on, so among
hundreds of candidate splits they also accept lucky ones. With
validation_fraction > 0 the last share of the rows (the checking rows) takes no
part in growth. Afterwards every split, judged with all the splits below it, must
lower the mean regret and raise the Sharpe ratio on the checking rows, or it is
removed (bottom-up pruning on validation data, as in the SPOT paper). The leaf
scores are then refit on all rows, always. Test on still later rows with replay.

TODO (known limitation, to address later): noise switches inside a split.
The Sharpe check accepts or rejects a split as a whole. A split that is right
on average but whose feature is noisy around its threshold is still accepted,
and the strategy then switches leaves back and forth within a regime, paying
fees each time (in a synthetic regime example, about three quarters of the
fees came from leaf switches without a regime change). Options to evaluate:
smoother features, a buffer zone around thresholds, partial adjustment towards
the target (Decision-Induced Ranking paper), a turnover penalty in the path score.

Leaf scores come from a bounded coordinate search on the empirical decision
regret, started from the leaf's mean return; it is approximate, not a global
solver. The fitted scores need not be calibrated expected-return forecasts.

Read in order: configurations -> optimizer -> replay -> leaf fitting -> splits
-> weekly rows -> evaluation -> demo.
Based on the SPO-tree idea, not a reproduction of the papers' full experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product
import numpy as np


# 1. All investment and tree choices live in these configurations.
@dataclass(frozen=True)
class PortfolioConfig:
    max_weight: float = 1.0  # long-only: 0 <= weight <= max_weight, sum == 1
    fee_rate: float | tuple[float, ...] = 0.001  # per traded notional, per asset
    risk_aversion: float = 1.0  # > 0; multiplies weights @ Sigma @ weights


@dataclass(frozen=True)
class TreeConfig:
    max_depth: int = 4  # root depth is 0; zero means one leaf
    min_samples_leaf: int = 10
    max_thresholds: int | None = 10  # None tests every distinct partition
    threshold_quantiles: tuple[float, ...] | None = None  # e.g. (0.25, 0.5, 0.75): only the quantiles of the node's values
    min_leaf_fraction: float = 0.0  # each child keeps at least this share of its node's rows (on top of min_samples_leaf)
    refine_thresholds: bool = False  # with threshold_quantiles: also screen every partition around the best quantile
    refine_standard_errors: float = 0.0  # hard splits: a refined threshold must beat the quantile one by this many s.e.
    smoothing: float = 0.0  # > 0: soft splits with Boltzmann weights exp(-z / smoothing) over the candidate thresholds
    min_regret_improvement: float = 1e-6  # average gain per observation at node
    shortlist_size: int = 3  # screened splits per leaf that get a full fit and replay
    min_sharpe_improvement: float = 0.0  # a split must raise the path Sharpe by more
    validation_fraction: float = (
        0.0  # last share of the rows, kept aside to prune splits; 0 = no pruning
    )
    prediction_bound: float = 0.10  # leaf scores restricted to +/- this
    search_passes: int = 3
    search_grid_size: int = 7
    regret_tolerance: float = 1e-6  # allowable numerical error per observation
    verbose: bool = False


MAX_ASSETS = 4  # the exact solver enumerates 5**n_assets cases
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
        raise ValueError(
            "Sigma must have shape (n_assets, n_assets) or (N, n_assets, n_assets)."
        )
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
            raise ValueError(
                "max_weight must be in (0, 1] and n_assets * max_weight >= 1."
            )
        if config.risk_aversion <= 0:
            # A positive penalty makes the optimal portfolio unique, so the
            # regret of a prediction does not depend on how it is computed.
            raise ValueError("risk_aversion must be > 0.")
        self.fees = np.broadcast_to(
            np.asarray(config.fee_rate, dtype=float), (n_assets,)
        ).copy()
        if not np.isfinite(self.fees).all() or np.any(self.fees < 0):
            raise ValueError("fee_rate must be nonnegative, scalar or per asset.")

        # Group the cases by which weights are free. A fixed weight is coded
        # 0 (at zero), 1 (at max_weight) or 2 (at its starting weight); a free
        # weight carries the side of its starting weight it lies on (+1 or -1).
        self._cases = []
        for n_free in range(n_assets + 1):
            for free in combinations(range(n_assets), n_free):
                fixed = [i for i in range(n_assets) if i not in free]
                combos = list(
                    product(
                        product((0, 1, 2), repeat=len(fixed)),
                        product((1.0, -1.0), repeat=n_free),
                    )
                )
                states = np.array([s for s, _ in combos], dtype=int)
                sides = np.array([d for _, d in combos], dtype=float)
                self._cases.append(
                    (
                        list(free),
                        fixed,
                        states.reshape(len(combos), len(fixed)),
                        sides.reshape(len(combos), n_free),
                    )
                )

    def is_feasible(self, weights):
        """True when every row is long-only, capped, and fully invested."""
        w, tol = np.asarray(weights, dtype=float), WEIGHT_TOLERANCE
        return bool(
            np.isfinite(w).all()
            and np.all(w >= -tol)
            and np.all(w <= self.config.max_weight + tol)
            and np.all(np.abs(w.sum(axis=-1) - 1) <= tol)
        )

    def utility(self, returns, weights, starting_weights, covariance):
        """returns @ w - fees @ |w - u| - risk_aversion * w @ Sigma @ w, per row."""
        r, w, u, S = (
            np.asarray(a, dtype=float)
            for a in (returns, weights, starting_weights, covariance)
        )
        return (
            np.einsum("...i,...i->...", r, w)
            - np.abs(w - u) @ self.fees
            - self.config.risk_aversion * np.einsum("...i,...ij,...j->...", w, S, w)
        )

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
                W[:, :, asset] = np.where(
                    state == 0, 0.0, np.where(state == 1, cap, u[:, asset])
                )
            if free:
                # Free weights solve  A w + nu = g  and  sum(w) = budget,
                # with A = 2 * lam * Sigma[free, free] and nu the budget multiplier.
                A_inv = np.linalg.inv(2 * lam * S[:, free][:, :, free])
                g = (
                    c[:, free]
                    - sides[:, None, :] * self.fees[free]
                    - 2
                    * lam
                    * np.einsum("rij,krj->kri", S[:, free][:, :, fixed], W[:, :, fixed])
                )
                budget = 1.0 - W[:, :, fixed].sum(axis=-1)
                A_inv_g = np.einsum("rij,krj->kri", A_inv, g)
                A_inv_1 = A_inv.sum(axis=-1)
                nu = (A_inv_g.sum(axis=-1) - budget) / A_inv_1.sum(axis=-1)
                w_free = A_inv_g - nu[..., None] * A_inv_1
                W[:, :, free] = w_free
                valid = np.all(
                    sides[:, None, :] * (w_free - u[:, free]) >= -tol, axis=-1
                )
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
        c, u, S = (
            np.asarray(a, dtype=float) for a in (scores, starting_weights, covariance)
        )
        single = c.ndim == 1 and u.ndim == 1 and S.ndim == 2
        c, u = np.atleast_2d(c), np.atleast_2d(u)
        S = S[None] if S.ndim == 2 else S
        if (
            c.ndim != 2
            or u.ndim != 2
            or S.ndim != 3
            or c.shape[1] != n
            or u.shape[1] != n
            or S.shape[1:] != (n, n)
        ):
            raise ValueError(
                "Expect scores and starting weights with n_assets "
                "columns and covariance matrices of n_assets x n_assets."
            )
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

    holdings: np.ndarray  # (T, n) pre-trade weights, drifted from the last period
    weights: np.ndarray  # (T, n) post-trade weights held over the period
    turnover: np.ndarray  # (T,) sum(|weights - holdings|)
    fees: np.ndarray  # (T,) fee_rate @ |weights - holdings|, fraction of equity
    gross_returns: np.ndarray  # (T,) returns @ weights
    net_returns: np.ndarray  # (T,) gross_returns - fees
    utility: np.ndarray  # (T,) net_returns - risk_aversion * weights @ Sigma @ weights
    regret: (
        np.ndarray
    )  # (T,) best utility in hindsight from the same holdings - utility
    final_holdings: np.ndarray  # (n,) drifted weights after the last period

    @property
    def sharpe(self) -> float:
        """Per-period Sharpe ratio of net returns; NaN with fewer than 2 periods.

        A path that never takes risk and never earns (all cash, no fees) has a
        Sharpe ratio of 0, so a split can be judged against an all-cash tree.
        """
        if len(self.net_returns) < 2:
            return float("nan")
        std = self.net_returns.std(ddof=1)
        if std > 0:
            return float(self.net_returns.mean() / std)
        return 0.0 if self.net_returns.mean() == 0 else float("nan")

    @property
    def max_drawdown(self) -> float:
        """Largest fall of net wealth below its running peak (-0.25 means -25%)."""
        wealth = np.cumprod(1 + self.net_returns)
        peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]  # wealth starts at 1
        return float(np.min(wealth / peak - 1))

    def summary(self, periods_per_year=52):
        """Annualized figures of the net returns (52 periods a year for weekly rows).

        return: compounded growth per year; volatility and sharpe: from the
        per-period mean and standard deviation; max_drawdown: over the path;
        turnover: average per period; fees: per year, as a fraction of equity.
        """
        periods = len(self.net_returns)
        scale = float(np.sqrt(periods_per_year))
        volatility = self.net_returns.std(ddof=1) if periods > 1 else float("nan")
        return {
            "return": float(
                np.prod(1 + self.net_returns) ** (periods_per_year / periods) - 1
            ),
            "volatility": float(volatility) * scale,
            "sharpe": self.sharpe * scale,
            "max_drawdown": self.max_drawdown,
            "turnover": float(self.turnover.mean()),
            "fees": float(self.fees.mean()) * periods_per_year,
        }


def replay_path(optimizer, scores, returns, covariance, initial_weights=None):
    """Trade period by period, in row order (rows must be chronological).

    In period t the optimizer turns scores[t] into weights w_t, starting from
    the holdings u_t. Fees are charged on |w_t - u_t| and subtracted from the
    period return (the tiny fees-times-return cross term is ignored). Holdings
    then drift with the returns: u_{t+1} = w_t * (1 + r_t) / (1 + r_t @ w_t).
    The first holdings are equal weights unless initial_weights is given, e.g.
    the final_holdings of a previous path.
    """
    R, S, u = _path_inputs(optimizer, returns, covariance, initial_weights)
    C = finite_array(scores, "scores", 2)
    if C.shape != R.shape:
        raise ValueError("scores and returns must both have shape (T, n_assets).")
    return _trade(
        optimizer, lambda t, holdings: optimizer.solve(C[t], holdings, S[t]), R, S, u
    )


def replay_constant_weights(
    optimizer, weights, returns, covariance, initial_weights=None
):
    """Rebalance to the same target weights every period, e.g. equal weights.

    Same accounting as replay_path, but the target ignores costs: every period
    it trades all the way back from the drifted holdings.
    """
    R, S, u = _path_inputs(optimizer, returns, covariance, initial_weights)
    target = finite_array(weights, "weights", 1)
    if target.shape != (optimizer.n_assets,) or not optimizer.is_feasible(target):
        raise ValueError(
            "weights must be a long-only, fully-invested portfolio within max_weight."
        )
    return _trade(optimizer, lambda t, holdings: target, R, S, u)


def _path_inputs(optimizer, returns, covariance, initial_weights):
    """Validated returns, per-period covariances and first holdings."""
    n = optimizer.n_assets
    R = finite_array(returns, "returns", 2)
    if R.shape[1] != n:
        raise ValueError("returns must have shape (T, n_assets).")
    if np.any(R <= -1):
        raise ValueError("Simple returns must be greater than -1.")
    S = covariance_rows(covariance, len(R), n)
    if initial_weights is None:
        return R, S, np.full(n, 1.0 / n)
    u = finite_array(initial_weights, "initial_weights", 1)
    if u.shape != (n,) or np.any(u < 0) or abs(u.sum() - 1) > 1e-6:
        raise ValueError(
            "initial_weights must be n_assets nonnegative weights summing to 1."
        )
    return R, S, u / u.sum()


def _trade(optimizer, decide, R, S, u):
    """Apply decide(t, holdings) period by period and do the accounting."""
    holdings, weights = np.empty_like(R), np.empty_like(R)
    for t in range(len(R)):
        holdings[t] = u
        weights[t] = decide(t, u)
        grown = weights[t] * (1 + R[t])
        u = grown / grown.sum()  # grown.sum() == 1 + r_t @ w_t

    trades = np.abs(weights - holdings)
    fees = trades @ optimizer.fees
    gross = np.einsum("ti,ti->t", R, weights)
    utility = optimizer.utility(R, weights, holdings, S)
    best = optimizer.utility(R, optimizer.solve(R, holdings, S), holdings, S)
    return PathResult(
        holdings=holdings,
        weights=weights,
        turnover=trades.sum(axis=1),
        fees=fees,
        gross_returns=gross,
        net_returns=gross - fees,
        utility=utility,
        regret=best - utility,
        final_holdings=u,
    )


@dataclass
class Node:
    prediction: np.ndarray
    n_samples: int
    regret_sum: float
    feature: int | None = None
    threshold: float | None = None  # a soft split reports the weighted median of `thresholds`
    left: Node | None = None
    right: Node | None = None
    thresholds: np.ndarray | None = None  # soft split: candidate thresholds, ascending ...
    weights: np.ndarray | None = None  # ... and their Boltzmann weights, summing to 1
    spread: float = 0.0  # soft split: thresholds between the 25% and 75% weight quantiles


@dataclass(frozen=True)
class SplitRecord:
    """One accepted split, in the order training accepted them."""

    depth: int
    n_samples: int
    feature: int
    threshold: float
    regret_gain: (
        float  # regret reduction per row of the split leaf, vs the refitted leaf
    )
    sharpe_before: float  # Sharpe ratio of the replayed training path
    sharpe_after: float


@dataclass(frozen=True)
class PruneRecord:
    """One split judged on the checking rows, in the bottom-up order of pruning."""

    depth: int
    feature: int
    threshold: float
    regret_with: (
        float  # mean regret on the checking rows, with the split and all below it
    )
    regret_without: float  # the same with the node as a single leaf
    sharpe_with: float  # Sharpe ratio on the checking rows
    sharpe_without: float
    kept: bool


# 4. Grow one tree on its own replayed training path.
class SPOPortfolioTree:
    """See the module docstring. Three options extend the plain tree; all are
    off by default, and then the tree is exactly the one described there.

    threshold_quantiles (+ min_leaf_fraction): only the partitions nearest to
    these quantiles of a node's values are screened, and every child keeps at
    least that share of its node. With refine_thresholds, every partition
    between the neighbouring quantile candidates of the best one is screened
    as well; with hard splits the refined threshold replaces the quantile one
    only if it beats it by refine_standard_errors standard errors of the
    regret difference.

    smoothing > 0: soft splits. Every screened threshold t of the chosen
    feature keeps a Boltzmann weight exp(-z_t / smoothing), with z_t the
    regret excess of t over the best candidate in standard errors. A row then
    belongs to the left child with the total weight of the thresholds at or
    above its value, so the tree's score is a continuous function of the
    feature, and thresholds that the data cannot tell apart share the weight.
    Leaf scores are fitted on their rows weighted by membership; the regret,
    the replay, the Sharpe gate and the pruning see per-row scores only and
    are unchanged.
    """

    def __init__(self, optimizer: PortfolioOptimizer, config: TreeConfig):
        self.optimizer = optimizer
        self.config = config
        self.root = None
        self.growth_log: list[SplitRecord] = []
        self.pruning_log: list[PruneRecord] = []
        if (
            config.max_depth < 0
            or config.min_samples_leaf < 1
            or (config.max_thresholds is not None and config.max_thresholds < 1)
            or (
                config.threshold_quantiles is not None
                and not all(0 < q < 1 for q in config.threshold_quantiles)
            )
            or config.shortlist_size < 1
            or config.search_passes < 1
            or config.search_grid_size < 3
        ):
            raise ValueError("Invalid tree depth, leaf size, or search settings.")
        if (
            not np.isfinite(
                [
                    config.min_regret_improvement,
                    config.min_sharpe_improvement,
                    config.prediction_bound,
                    config.regret_tolerance,
                    config.validation_fraction,
                    config.min_leaf_fraction,
                    config.refine_standard_errors,
                    config.smoothing,
                ]
            ).all()
            or config.min_regret_improvement < 0
            or config.min_sharpe_improvement < 0
            or config.prediction_bound <= 0
            or config.regret_tolerance <= 0
            or not 0 <= config.validation_fraction < 1
            or not 0 <= config.min_leaf_fraction < 0.5
            or config.refine_standard_errors < 0
            or config.smoothing < 0
        ):
            raise ValueError(
                "Invalid improvement thresholds, prediction bound, regret tolerance, "
                "validation fraction, leaf fraction, refinement or smoothing setting."
            )
        if config.refine_thresholds and config.threshold_quantiles is None:
            raise ValueError("refine_thresholds needs threshold_quantiles.")

    def fit(self, X, R, Sigma, initial_weights=None):
        """Grow the tree on chronological rows while it trades its own holdings.

        initial_weights are the holdings before the first row (equal weights by
        default). Each round replays the current tree, shortlists the
        shortlist_size best splits of every leaf (mean child scores, regret from
        the replayed holdings) and fits their child scores. A split's regret
        reduction is measured against its leaf refitted on the same holdings,
        and must exceed min_regret_improvement per row. The split with the
        largest reduction is replayed and accepted if the Sharpe ratio beats the
        current one by more than min_sharpe_improvement; otherwise the next
        largest is tried. Finally every leaf score is refit on the final
        holdings and kept only if the Sharpe does not drop.

        With validation_fraction > 0 all of the above sees only the earlier rows
        (the fitting rows). The last rows (the checking rows) then prune the
        tree, see _prune, and every leaf score is refit once more, on all rows.
        That last refit is always kept.
        """
        self.root, self.growth_log, self.pruning_log = None, [], []
        X = finite_array(X, "X", 2)
        R = finite_array(R, "R", 2)
        n = self.optimizer.n_assets
        if len(X) != len(R) or R.shape[1] != n:
            raise ValueError("Require X=(N,P) and R=(N,n_assets).")
        Sigma = covariance_rows(Sigma, len(R), n)
        n_check = int(round(self.config.validation_fraction * len(X)))
        n_fit = len(X) - n_check
        if self.config.validation_fraction > 0 and min(n_fit, n_check) < 2:
            raise ValueError(
                "validation_fraction must leave at least 2 fitting rows "
                "and 2 checking rows."
            )
        self.n_features = X.shape[1]
        self._initial_weights = initial_weights
        try:
            root, leaves, scores, path = self._grow(X[:n_fit], R[:n_fit], Sigma[:n_fit])
            self._refit_leaves(leaves, scores, path)
            if n_check:
                self._prune(root, X, R, Sigma, n_fit)
                # The rules are settled: the leaf scores now learn from every row.
                self._X, self._R, self._S = X.copy(), R.copy(), Sigma.copy()
                leaves = self._leaves(root, self._X)
                scores = self._scores(leaves, len(X))
                path = self._replay(scores)
                self._measure_from(path)
                for leaf, rows, weights, _ in leaves:
                    leaf.n_samples = int(round(weights.sum()))
                    leaf.regret_sum = self._score_prediction(leaf.prediction, rows, weights)
                # Always kept: the scores it replaces never saw the checking rows.
                self._refit_leaves(leaves, scores, path, on_all_rows=True)
            self.root = root
        finally:
            # Deployment uses the frozen tree, not stored training outcomes.
            for name in ("_X", "_R", "_S", "_U", "_benchmarks", "_initial_weights"):
                if hasattr(self, name):
                    delattr(self, name)
        return self

    def _grow(self, X, R, Sigma):
        """Best-first growth on these rows: (root, leaves, per-row scores, replayed path).

        A leaf is (node, rows, weights, depth): the rows that reach it and, with
        soft splits, the weight with which they do (1 everywhere otherwise).
        """
        n = self.optimizer.n_assets
        self._X, self._R, self._S = X.copy(), R.copy(), Sigma.copy()
        # The root is scored from equal-weight holdings; from then on,
        # regrets are measured from the holdings of the replayed tree.
        rows, ones = np.arange(len(X)), np.ones(len(X))
        equal = np.full((len(X), n), 1.0 / n)
        self._U = equal
        self._benchmarks = self.optimizer.utility(
            self._R, self.optimizer.solve(self._R, equal, self._S), equal, self._S
        )
        root = self._fit_leaf(rows, weights=ones)
        leaves = [(root, rows, ones, 0)]
        scores = np.tile(root.prediction, (len(X), 1))
        path = self._replay(scores)
        self._measure_from(path)
        if self.config.verbose:
            print(f"root: training Sharpe {path.sharpe:.4f}")
        while True:
            split = self._best_split(leaves, scores, path.sharpe)
            if split is None:
                break
            (position, feature, thresholds, pis, left, right,
             left_rows, left_weights, right_rows, right_weights,
             gain, new_scores, new_path) = split
            node, node_rows, node_weights, depth = leaves.pop(position)
            self._set_split(node, feature, thresholds, pis)
            node.left, node.right = left, right
            leaves += [
                (left, left_rows, left_weights, depth + 1),
                (right, right_rows, right_weights, depth + 1),
            ]
            scores = new_scores
            self.growth_log.append(
                SplitRecord(
                    depth,
                    int(round(node_weights.sum())),
                    feature,
                    float(node.threshold),
                    gain,
                    path.sharpe,
                    new_path.sharpe,
                )
            )
            if self.config.verbose:
                soft = f" (soft, spread {node.spread:.3g})" if node.thresholds is not None else ""
                print(
                    f"split {len(self.growth_log)}: depth={depth}, "
                    f"N={int(round(node_weights.sum()))}, x[{feature}] <= {node.threshold:.5g}{soft}, "
                    f"regret -{gain:.3g} per row, "
                    f"Sharpe {path.sharpe:.4f} -> {new_path.sharpe:.4f}"
                )
            path = new_path
            self._measure_from(path)
        return root, leaves, scores, path

    def _prune(self, root, X, R, Sigma, n_fit):
        """Remove, from the bottom up, every split the checking rows do not confirm.

        The rows from n_fit on took no part in growing the tree, so a split that
        only fitted noise has no reason to help there. Each split is judged with
        everything below it, against its node as a single leaf (scores fitted on
        the fitting rows, from the current holdings). Both trees trade the
        fitting rows and then the checking rows; the split stays only if, on the
        checking rows, the mean regret falls by more than min_regret_improvement
        and the Sharpe ratio rises by more than min_sharpe_improvement. This is
        SPOT's pruning on validation data, with the Sharpe as a second judge.
        """
        c = self.config

        def checking_path():
            scores = self._scores(self._leaves(root, X), len(X))
            holdings = self._replay(scores[:n_fit]).final_holdings
            return replay_path(
                self.optimizer, scores[n_fit:], R[n_fit:], Sigma[n_fit:], holdings
            )

        def visit(node, rows, weights, depth):
            if node.feature is None:
                return
            m = self._membership(node, self._X[rows, node.feature])
            visit(node.left, rows[m > 0], (weights * m)[m > 0], depth + 1)
            visit(node.right, rows[m < 1], (weights * (1 - m))[m < 1], depth + 1)
            with_split = checking_path()
            self._measure_from(
                self._replay(self._scores(self._leaves(root, self._X), n_fit))
            )
            leaf = self._fit_leaf(rows, node.prediction, weights)
            split = (
                node.feature,
                node.threshold,
                node.thresholds,
                node.weights,
                node.spread,
                node.left,
                node.right,
                node.prediction,
                node.regret_sum,
            )
            node.feature = node.threshold = node.thresholds = node.weights = None
            node.left = node.right = None
            node.spread = 0.0
            node.prediction, node.regret_sum = leaf.prediction, leaf.regret_sum
            without = checking_path()
            kept = bool(
                without.regret.mean() - with_split.regret.mean()
                > c.min_regret_improvement
                and with_split.sharpe > without.sharpe + c.min_sharpe_improvement
            )
            if kept:
                (
                    node.feature,
                    node.threshold,
                    node.thresholds,
                    node.weights,
                    node.spread,
                    node.left,
                    node.right,
                    node.prediction,
                    node.regret_sum,
                ) = split
            self.pruning_log.append(
                PruneRecord(
                    depth,
                    split[0],
                    split[1],
                    float(with_split.regret.mean()),
                    float(without.regret.mean()),
                    with_split.sharpe,
                    without.sharpe,
                    kept,
                )
            )
            if c.verbose:
                print(
                    f"prune check: depth={depth}, x[{split[0]}] <= {split[1]:.5g}, "
                    f"checking regret {without.regret.mean():.5f} -> "
                    f"{with_split.regret.mean():.5f} with the split, "
                    f"Sharpe {without.sharpe:.4f} -> {with_split.sharpe:.4f} "
                    f"({'kept' if kept else 'removed'})"
                )

        visit(root, np.arange(n_fit), np.ones(n_fit), 0)

    # ----------------------------------------------------------------- memberships
    @staticmethod
    def _membership_of(thresholds, pis, x):
        """Weight with which values x fall on the left of a split: the total
        weight of the candidate thresholds at or above x (1 or 0 for a hard split)."""
        cumulative = np.concatenate([[0.0], np.cumsum(pis)])
        below = np.searchsorted(thresholds, x, side="left")  # candidates strictly below x
        return np.clip(1.0 - cumulative[below], 0.0, 1.0)

    def _membership(self, node, x):
        if node.thresholds is None:
            return (np.asarray(x) <= node.threshold).astype(float)
        return self._membership_of(node.thresholds, node.weights, x)

    @staticmethod
    def _set_split(node, feature, thresholds, pis):
        """Record a split on the node: hard (one threshold) or soft (several, weighted)."""
        node.feature = feature
        if len(thresholds) == 1:
            node.threshold, node.thresholds, node.weights, node.spread = (
                float(thresholds[0]), None, None, 0.0,
            )
            return
        cumulative = np.cumsum(pis)
        quantile = lambda q: float(thresholds[min(np.searchsorted(cumulative, q), len(thresholds) - 1)])
        node.threshold = quantile(0.5)
        node.thresholds, node.weights = np.asarray(thresholds, dtype=float), np.asarray(pis, dtype=float)
        node.spread = quantile(0.75) - quantile(0.25)

    def _leaves(self, node, X, rows=None, weights=None, depth=0):
        """(leaf, the rows of X that reach it, their weights, depth) for every leaf below node."""
        rows = np.arange(len(X)) if rows is None else rows
        weights = np.ones(len(rows)) if weights is None else weights
        if node.feature is None:
            return [(node, rows, weights, depth)]
        m = self._membership(node, X[rows, node.feature])
        left, right = m > 0, m < 1
        return self._leaves(
            node.left, X, rows[left], (weights * m)[left], depth + 1
        ) + self._leaves(node.right, X, rows[right], (weights * (1 - m))[right], depth + 1)

    def _scores(self, leaves, n_rows):
        """Per-row scores from a list of (leaf, rows, weights, depth): the
        membership-weighted mixture of the leaf scores (one leaf per row when hard)."""
        scores = np.zeros((n_rows, self.optimizer.n_assets))
        for leaf, rows, weights, _ in leaves:
            scores[rows] += weights[:, None] * leaf.prediction
        return scores

    def _replay(self, scores):
        """Replay per-row scores on the training rows (holdings used for regrets unchanged)."""
        return replay_path(
            self.optimizer, scores, self._R, self._S, self._initial_weights
        )

    def _measure_from(self, path):
        """From now on, measure regrets from the holdings of this replayed path."""
        self._U, self._benchmarks = path.holdings, path.utility + path.regret

    def _min_leaf(self, mass):
        c = self.config
        return max(c.min_samples_leaf, int(np.ceil(c.min_leaf_fraction * mass)))

    # ----------------------------------------------------------------- split search
    def _best_split(self, leaves, scores, sharpe):
        """The split with the largest regret reduction that also raises the
        replayed Sharpe, or None."""
        c = self.config
        candidates = []
        for position, (node, rows, weights, depth) in enumerate(leaves):
            mass = weights.sum()
            if depth >= c.max_depth or mass < 2 * self._min_leaf(mass):
                continue
            # Compare like with like. The leaf's scores were fitted on the
            # holdings of an earlier tree, so refit them on the current holdings
            # before measuring what two leaves add over one.
            parent = self._fit_leaf(rows, node.prediction, weights)
            for feature, thresholds, pis in self._shortlist(rows, weights):
                m = self._membership_of(thresholds, pis, self._X[rows, feature])
                on_left, on_right = m > 0, m < 1
                left_rows, left_weights = rows[on_left], (weights * m)[on_left]
                right_rows, right_weights = rows[on_right], (weights * (1 - m))[on_right]
                left = self._fit_leaf(left_rows, parent.prediction, left_weights)
                right = self._fit_leaf(right_rows, parent.prediction, right_weights)
                reduction = parent.regret_sum - left.regret_sum - right.regret_sum
                if reduction / mass > c.min_regret_improvement:
                    candidates.append(
                        (reduction, position, feature, thresholds, pis, m,
                         left, right, left_rows, left_weights, right_rows, right_weights)
                    )
        # The SPO loss chooses (largest total regret reduction over all leaves,
        # as in SPOT); the replayed Sharpe must confirm, else the next one is tried.
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        for (reduction, position, feature, thresholds, pis, m,
             left, right, left_rows, left_weights, right_rows, right_weights) in candidates:
            # Scores of the tree with this split, rebuilt from its leaves (exact).
            trial = (leaves[:position]
                     + [(left, left_rows, left_weights, 0), (right, right_rows, right_weights, 0)]
                     + leaves[position + 1:])
            candidate = self._scores(trial, len(scores))
            path = self._replay(candidate)
            # TODO: a split is judged as a whole, so a profitable split whose
            # feature is noisy around the threshold still gets in and pays
            # fees on back-and-forth leaf switches ("noise switches" in the
            # module docstring).
            if path.sharpe > sharpe + c.min_sharpe_improvement:  # NaN never qualifies
                gain = reduction / weights.sum()
                return (position, feature, thresholds, pis, left, right,
                        left_rows, left_weights, right_rows, right_weights,
                        gain, candidate, path)
        return None

    def _shortlist(self, rows, weights):
        """Cheap screen of a leaf's splits: each child simply uses its clipped
        (weighted) mean return as scores. Returns (feature, thresholds, weights)
        candidates: one threshold with weight 1 for a hard split, the whole set
        of the feature's candidates with their Boltzmann weights for a soft one."""
        c = self.config
        min_leaf = self._min_leaf(weights.sum())
        screened = []
        for feature in range(self.n_features):
            values = self._X[rows, feature]
            candidates = self._thresholds(values, weights, min_leaf)
            if len(candidates) == 0:
                continue
            keep_rows = c.smoothing > 0 or c.refine_thresholds
            sums, per_row = self._screen(rows, weights, values, candidates, keep_rows)
            if c.refine_thresholds:
                every = self._thresholds(values, weights, min_leaf, use_quantiles=False, limit=False)
                best = int(np.argmin(sums))
                lower = candidates[best - 1] if best > 0 else -np.inf
                upper = candidates[best + 1] if best + 1 < len(candidates) else np.inf
                window = self._thin(every[(every > lower) & (every < upper) & ~np.isin(every, candidates)])
                if len(window):
                    window_sums, window_rows = self._screen(rows, weights, values, window, True)
                    if c.smoothing > 0:
                        candidates = np.concatenate([candidates, window])
                        sums = np.concatenate([sums, window_sums])
                        per_row = np.vstack([per_row, window_rows])
                    else:
                        refined = int(np.argmin(window_sums))
                        hurdle = c.refine_standard_errors * self._standard_error(
                            window_rows[refined] - per_row[best]
                        )
                        if sums[best] - window_sums[refined] > hurdle:
                            candidates[best], sums[best] = window[refined], window_sums[refined]
            if c.smoothing > 0:
                best = int(np.argmin(sums))
                z = np.empty(len(candidates))
                for k in range(len(candidates)):
                    excess = sums[k] - sums[best]
                    error = self._standard_error(per_row[k] - per_row[best])
                    z[k] = excess / error if error > 0 else (0.0 if excess <= 0 else np.inf)
                pis = np.exp(-z / c.smoothing)
                pis /= pis.sum()
                order = np.argsort(candidates)
                screened.append((float(sums[best]), feature, candidates[order], pis[order]))
            else:
                for threshold, total in zip(candidates, sums):
                    screened.append((float(total), feature, np.array([threshold]), np.array([1.0])))
        screened.sort(key=lambda split: split[0])
        return [split[1:] for split in screened[: c.shortlist_size]]

    def _screen(self, rows, weights, values, thresholds, keep_rows):
        """Weighted regret of each threshold when each child uses its clipped
        mean return as scores: the sums and, if asked, the per-row regrets."""
        bound = self.config.prediction_bound
        sums = np.empty(len(thresholds))
        per_row = np.empty((len(thresholds), len(rows))) if keep_rows else None
        for k, threshold in enumerate(thresholds):
            left = values <= threshold
            regret = np.empty(len(rows))
            for side in (left, ~left):
                part = rows[side]
                score = np.clip(
                    np.average(self._R[part], axis=0, weights=weights[side]), -bound, bound
                )
                regret[side] = self._regret_rows(score, part)
            regret *= weights
            sums[k] = regret.sum()
            if keep_rows:
                per_row[k] = regret
        return sums, per_row

    @staticmethod
    def _standard_error(differences):
        """Standard error of a sum of per-row differences (rows treated as independent)."""
        if len(differences) < 2:
            return 0.0
        return float(np.std(differences, ddof=1) * np.sqrt(len(differences)))

    def _refit_leaves(self, leaves, scores, path, on_all_rows=False):
        """Refit every leaf score on the final holdings; keep it if Sharpe holds.

        The Sharpe check compares two sets of scores fitted on the same rows. The
        refit on all rows adds the checking rows, which the scores it replaces
        never saw, so it is kept whatever the Sharpe of the training path does.
        """
        refits = [
            self._fit_leaf(rows, node.prediction, weights) for node, rows, weights, _ in leaves
        ]
        trial = [(refit, rows, weights, 0) for refit, (_, rows, weights, _) in zip(refits, leaves)]
        candidate = self._scores(trial, len(scores))
        refit_path = self._replay(candidate)
        kept = on_all_rows or refit_path.sharpe >= path.sharpe
        if kept:
            for refit, (node, _, _, _) in zip(refits, leaves):
                node.prediction, node.regret_sum = refit.prediction, refit.regret_sum
        if self.config.verbose:
            print(
                f"refit leaf scores{' on all rows' if on_all_rows else ''}: "
                f"Sharpe {path.sharpe:.4f} -> {refit_path.sharpe:.4f} "
                f"({'always kept' if on_all_rows else 'kept' if kept else 'discarded'})"
            )

    def _regret_rows(self, prediction, indices):
        """Regret of one score vector on each of these rows, from the current holdings."""
        R, U, S = self._R[indices], self._U[indices], self._S[indices]
        weights = self.optimizer.solve(prediction, U, S)
        regret = self._benchmarks[indices] - self.optimizer.utility(R, weights, U, S)
        if len(regret) and np.min(regret) < -self.config.regret_tolerance:
            raise RuntimeError(
                "Negative regret exceeds tolerance; check solver accuracy."
            )
        # Clip only negligible numerical negatives, not meaningful discrepancies.
        return np.maximum(regret, 0)

    def _score_prediction(self, prediction, indices, weights=None):
        regret = self._regret_rows(prediction, indices)
        if weights is not None:
            regret = regret * weights
        return float(regret.sum())

    def _fit_leaf(self, indices, parent_prediction=None, weights=None):
        # Deterministic bounded coordinate search, started from the leaf mean.
        # With fees or row-specific covariances the mean is only a starting
        # point, not the regret minimizer.
        c = self.config
        bound = c.prediction_bound
        best = np.clip(np.average(self._R[indices], axis=0, weights=weights), -bound, bound)
        best_loss = self._score_prediction(best, indices, weights)
        if parent_prediction is not None:
            loss = self._score_prediction(parent_prediction, indices, weights)
            if loss < best_loss:
                best, best_loss = parent_prediction.copy(), loss
        # Including the parent lets a child retain its parent's prediction.
        # The first pass spans all allowed values; later passes refine locally.
        radius = 2 * bound
        for pass_number in range(c.search_passes):
            for asset in range(self.optimizer.n_assets):
                lower = (
                    -bound if pass_number == 0 else max(-bound, best[asset] - radius)
                )
                upper = bound if pass_number == 0 else min(bound, best[asset] + radius)
                for value in np.linspace(lower, upper, c.search_grid_size):
                    candidate = best.copy()
                    candidate[asset] = value
                    loss = self._score_prediction(candidate, indices, weights)
                    if loss < best_loss:
                        best, best_loss = candidate, loss
            radius /= c.search_grid_size - 1
        n_samples = len(indices) if weights is None else int(round(weights.sum()))
        return Node(best, n_samples, best_loss)

    def _thresholds(self, values, weights=None, min_leaf=None, use_quantiles=True, limit=True):
        """Candidate thresholds: midpoints of consecutive distinct values whose
        two sides each carry at least min_leaf rows (weighted with soft splits)."""
        min_leaf = self.config.min_samples_leaf if min_leaf is None else min_leaf
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        mass = np.ones(len(values)) if weights is None else weights[order]
        unique, first = np.unique(sorted_values, return_index=True)
        per_value = np.add.reduceat(mass, first) if len(first) else np.array([])
        thresholds = unique[:-1] / 2 + unique[1:] / 2
        # Rounding can put a midpoint on the upper endpoint: retain the intended
        # partition in that case by using the lower endpoint with the <= rule.
        thresholds = np.where(thresholds >= unique[1:], unique[:-1], thresholds)
        left_mass = np.cumsum(per_value)[:-1]
        valid = (left_mass >= min_leaf) & (per_value.sum() - left_mass >= min_leaf)
        thresholds = thresholds[valid]
        quantiles = self.config.threshold_quantiles
        if use_quantiles and quantiles is not None and len(thresholds):
            # Only the partitions closest to the requested quantiles of the values.
            nearest = [int(np.argmin(np.abs(thresholds - q))) for q in np.quantile(values, quantiles)]
            thresholds = thresholds[np.unique(nearest)]
        return self._thin(thresholds) if limit else thresholds

    def _thin(self, thresholds):
        """At most max_thresholds candidates, evenly spaced by rank."""
        limit = self.config.max_thresholds
        if limit is not None and len(thresholds) > limit:
            selected = np.linspace(0, len(thresholds) - 1, limit).astype(int)
            thresholds = thresholds[selected]
        return thresholds

    def predict_returns(self, X):
        """Frozen leaf scores, with shape (observations, assets)."""
        if self.root is None:
            raise RuntimeError("Call fit before prediction.")
        X = finite_array(X, "X", 2)
        if X.shape[1] != self.n_features:
            raise ValueError("Feature count differs from training.")
        return self._scores(self._leaves(self.root, X), len(X))

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
        return replay_path(
            self.optimizer, self.predict_returns(X), R, Sigma, initial_weights
        )

    def describe(self, feature_names=None):
        """Print the frozen rules and leaf scores for inspection."""
        if self.root is None:
            raise RuntimeError("Call fit first.")
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x[{j}]" for j in range(self.n_features)]
        )
        if len(names) != self.n_features:
            raise ValueError("feature_names has the wrong length.")

        def visit(node, indent=""):
            if node.feature is None:
                print(
                    f"{indent}leaf: N={node.n_samples}, "
                    f"scores={np.round(node.prediction, 6)}"
                )
            else:
                soft = (
                    f" (soft: half of the weight within a range of {node.spread:.3g})"
                    if node.thresholds is not None
                    else ""
                )
                print(f"{indent}if {names[node.feature]} <= {node.threshold:.6g}{soft}:")
                visit(node.left, indent + "  ")
                print(f"{indent}else:")
                visit(node.right, indent + "  ")

        visit(self.root)


# 5. Weekly decision rows from consecutive daily closes.
@dataclass(frozen=True)
class WeeklyRows:
    """Aligned inputs for fit and replay, one row per weekly decision."""

    dates: np.ndarray  # (T,) decision dates: the daily close of the chosen weekday
    day_index: np.ndarray  # (T,) position of each decision date in the daily arrays
    X: np.ndarray | None  # (T, features) your daily features on the decision dates
    R: np.ndarray  # (T, n) simple return from this decision close to the next
    Sigma: np.ndarray  # (T, n, n) weekly covariance from past daily returns


def weekly_covariance(closes, day, window=180, ridge=1e-6):
    """Covariance of the next week's returns, estimated at the close of `day`.

    Sample covariance of the `window` daily simple returns up to and including
    that close, times 7 (daily returns treated as independent; crypto trades
    every day), plus `ridge` on the diagonal.
    """
    P = finite_array(closes, "closes", 2)
    if not isinstance(window, int) or window < 2 or not window <= day < len(P):
        raise ValueError("Need window >= 2 daily returns before `day`, inside closes.")
    if np.any(P[day - window : day + 1] <= 0) or ridge < 0:
        raise ValueError("Closes must be positive and ridge nonnegative.")
    daily = P[day - window + 1 : day + 1] / P[day - window : day] - 1
    covariance = np.cov(daily, rowvar=False).reshape(P.shape[1], P.shape[1])
    return 7 * covariance + ridge * np.eye(P.shape[1])


def weekly_rows(dates, closes, daily_features=None, weekday=0, window=180, ridge=1e-6):
    """Rows for weekly decisions taken at the close of `weekday` (0 = Monday).

    A weekday close becomes a decision row when `window` daily returns lie
    behind it and a close one week later exists:
      R[t]     = close one week later / close at the decision - 1
      Sigma[t] = weekly_covariance at the decision close
      X[t]     = daily_features at the decision date; row d of daily_features
                 must use data up to day d only (NaN is fine before the first
                 decision). X is None when no features are given.
    Dates must be consecutive calendar days: fill or drop missing days first.
    """
    days = np.asarray(dates, dtype="datetime64[D]")
    P = finite_array(closes, "closes", 2)
    if days.shape != (len(P),):
        raise ValueError("dates must be 1-D with one date per row of closes.")
    if np.any(np.diff(days) != np.timedelta64(1, "D")):
        raise ValueError(
            "dates must be consecutive calendar days; fill or drop gaps first."
        )
    if np.any(P <= 0):
        raise ValueError("closes must be positive prices.")
    if not isinstance(weekday, int) or not 0 <= weekday <= 6:
        raise ValueError("weekday must be an integer from 0 (Monday) to 6 (Sunday).")
    positions = np.arange(len(days))
    day_of_week = (days.astype(np.int64) + 3) % 7  # 1970-01-01 was a Thursday
    decisions = positions[
        (day_of_week == weekday) & (positions >= window) & (positions + 7 < len(days))
    ]
    if len(decisions) == 0:
        raise ValueError(
            "No weekly decision has a full covariance window and a next week."
        )
    X = None
    if daily_features is not None:
        F = np.asarray(daily_features, dtype=float)
        if F.ndim != 2 or len(F) != len(days):
            raise ValueError(
                "daily_features must be (days, features), aligned with dates."
            )
        X = F[decisions]
        if not np.isfinite(X).all():
            raise ValueError("daily_features must be finite on every decision date.")
    return WeeklyRows(
        dates=days[decisions],
        day_index=decisions,
        X=X,
        R=P[decisions + 7] / P[decisions] - 1,
        Sigma=np.array([weekly_covariance(P, d, window, ridge) for d in decisions]),
    )


# 6. Compare the tree with simple baselines on a training and a test period.
def train_and_test(portfolio, config, rows, test_periods=52):
    """Fit on all rows but the last test_periods, then replay three strategies.

    Returns (tree, train_paths, test_paths). Each dict holds the replayed paths
    of the SPO tree, the same tree without splits (max_depth=0) and an
    equal-weight portfolio rebalanced every period. In the test period each
    strategy continues from its own final training holdings.
    """
    if not 0 < test_periods < len(rows.R):
        raise ValueError("test_periods must leave at least one training row.")
    train = slice(0, len(rows.R) - test_periods)
    test = slice(len(rows.R) - test_periods, None)
    X, R, S = rows.X, rows.R, rows.Sigma
    tree = SPOPortfolioTree(portfolio, config).fit(X[train], R[train], S[train])
    no_split = SPOPortfolioTree(portfolio, replace(config, max_depth=0, verbose=False))
    no_split.fit(X[train], R[train], S[train])
    equal = np.full(portfolio.n_assets, 1.0 / portfolio.n_assets)
    strategies = {
        "SPO tree": lambda part, start: tree.replay(X[part], R[part], S[part], start),
        "tree without splits": lambda part, start: no_split.replay(
            X[part], R[part], S[part], start
        ),
        "equal weight": lambda part, start: replay_constant_weights(
            portfolio, equal, R[part], S[part], start
        ),
    }
    train_paths = {name: run(train, None) for name, run in strategies.items()}
    test_paths = {
        name: run(test, train_paths[name].final_holdings)
        for name, run in strategies.items()
    }
    return tree, train_paths, test_paths


def performance_report(paths, periods_per_year=52):
    """One line per named path: annualized return, volatility and Sharpe, max
    drawdown, average turnover per period and fees per year."""
    width = max(len(name) for name in paths) + 2
    lines = [
        f"{'':{width}}{'return':>9}{'volatility':>12}{'Sharpe':>8}"
        f"{'max DD':>9}{'turnover':>10}{'fees/yr':>9}"
    ]
    for name, path in paths.items():
        s = path.summary(periods_per_year)
        lines.append(
            f"{name:{width}}{s['return']:>9.1%}{s['volatility']:>12.1%}"
            f"{s['sharpe']:>8.2f}{s['max_drawdown']:>9.1%}"
            f"{s['turnover']:>10.2f}{s['fees']:>9.2%}"
        )
    return "\n".join(lines)


# 7. Runnable synthetic example of the whole weekly pipeline.
def demo():
    rng = np.random.default_rng(3)
    # Ten years of daily closes for three crypto-like assets (3% daily
    # volatility). A regime of about two months tilts the drifts of the first
    # two assets in opposite directions.
    days = 10 * 365 + 3
    start = np.datetime64("2016-01-01")
    dates = np.arange(start, start + np.timedelta64(days, "D"))
    regime = np.repeat(rng.choice([-1.0, 1.0], size=days // 60 + 1), 60)[:days]
    drift = np.where(regime[:, None] > 0, [0.002, -0.001, 0.0], [-0.001, 0.002, 0.0])
    daily_cov = 0.03**2 * np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.5], [0.5, 0.5, 1.0]])
    closes = 100 * np.cumprod(
        1 + drift + rng.multivariate_normal(np.zeros(3), daily_cov, days), axis=0
    )

    # Daily features: row d only uses closes up to day d.
    names = [
        "momentum_28d_a0",
        "momentum_28d_a1",
        "momentum_7d_a0",
        "volatility_28d_a0",
        "noise",
    ]
    features = np.full((days, len(names)), np.nan)
    features[28:, 0] = closes[28:, 0] / closes[:-28, 0] - 1
    features[28:, 1] = closes[28:, 1] / closes[:-28, 1] - 1
    features[7:, 2] = closes[7:, 0] / closes[:-7, 0] - 1
    daily_a0 = closes[1:, 0] / closes[:-1, 0] - 1
    features[28:, 3] = np.lib.stride_tricks.sliding_window_view(daily_a0, 28).std(
        axis=1
    )
    features[:, 4] = rng.normal(size=days)

    # One row per Monday close; the last 52 weeks are the test year. Of the
    # training weeks, the last quarter only prunes the tree (checking rows).
    rows = weekly_rows(dates, closes, features)
    portfolio = PortfolioOptimizer(
        n_assets=3,
        config=PortfolioConfig(
            max_weight=1.0,
            fee_rate=0.0003,
            risk_aversion=1.0,
        ),
    )
    config = TreeConfig(
        max_depth=2,
        min_samples_leaf=20,
        max_thresholds=None,
        validation_fraction=0.25,
        verbose=True,
    )
    tree, train_paths, test_paths = train_and_test(
        portfolio, config, rows, test_periods=52
    )
    tree.describe(names)
    print(f"\nTrain: {len(train_paths['SPO tree'].net_returns)} weeks, annualized")
    print(performance_report(train_paths))
    print(
        f"\nTest: {len(test_paths['SPO tree'].net_returns)} weeks, annualized, "
        f"holdings carried over from training"
    )
    print(performance_report(test_paths))

    holdings = test_paths["SPO tree"].final_holdings
    target = tree.predict_weights(
        features[-1:], holdings[None, :], weekly_covariance(closes, days - 1)
    )
    print(
        "\nLive decision at the last close: holdings",
        np.round(holdings, 3),
        "-> target",
        np.round(target[0], 3),
    )
    print("Synthetic demonstration only; no market data or orders are sent.")


if __name__ == "__main__":
    demo()
