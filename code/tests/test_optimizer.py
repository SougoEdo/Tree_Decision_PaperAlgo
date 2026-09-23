"""Checks for the exact portfolio solver.

Run from the project folder:  python -m unittest discover -s tests
"""
import unittest

import numpy as np
from scipy.optimize import minimize

from spo_tree import PortfolioConfig, PortfolioOptimizer


def random_covariance(rng, n):
    """Positive-definite covariance with random correlations and volatilities."""
    A = rng.normal(size=(n, n))
    C = A @ A.T + 0.1 * np.eye(n)
    scale = np.sqrt(np.diag(C))
    vols = rng.uniform(0.03, 0.15, n)      # weekly crypto-like volatilities
    return C / np.outer(scale, scale) * np.outer(vols, vols)


def random_problem(rng, n):
    optimizer = PortfolioOptimizer(n, PortfolioConfig(
        max_weight=float(rng.choice([1.0, 0.7])),
        fee_rate=tuple(float(f) for f in rng.choice([0.0, 0.001, 0.005], size=n)),
        risk_aversion=float(rng.choice([0.5, 5.0, 50.0])),
    ))
    scores = rng.uniform(-0.1, 0.1, n) * rng.choice([1.0, 10.0])
    holdings = rng.dirichlet(np.ones(n))    # may exceed max_weight, like drifted holdings
    return optimizer, scores, holdings, random_covariance(rng, n)


def grid_best_utility(optimizer, scores, holdings, covariance, step):
    """Best utility over a dense grid of long-only, fully-invested portfolios."""
    cap = optimizer.config.max_weight
    axis = np.arange(0.0, 1.0 + step / 2, step)
    if optimizer.n_assets == 2:
        W = np.column_stack([axis, 1.0 - axis])
    else:
        a, b = np.meshgrid(axis, axis, indexing="ij")
        keep = a + b <= 1.0 + 1e-12
        W = np.column_stack([a[keep], b[keep], np.maximum(1.0 - a[keep] - b[keep], 0.0)])
    W = W[np.all(W <= cap + 1e-12, axis=1)]
    return optimizer.utility(scores, W, holdings, covariance).max()


def slsqp_best_utility(optimizer, scores, holdings, covariance):
    """Independent reference: SLSQP with trade variables t >= |w - u|."""
    n, fees = optimizer.n_assets, optimizer.fees
    lam, cap = optimizer.config.risk_aversion, optimizer.config.max_weight

    def objective(z):
        w, t = z[:n], z[n:]
        return -(scores @ w) + fees @ t + lam * (w @ covariance @ w)

    constraints = [{"type": "eq", "fun": lambda z: z[:n].sum() - 1.0},
                   {"type": "ineq", "fun": lambda z: z[n:] - (z[:n] - holdings)},
                   {"type": "ineq", "fun": lambda z: z[n:] + (z[:n] - holdings)}]
    start = np.r_[np.full(n, 1.0 / n), np.abs(1.0 / n - holdings) + 0.1]
    result = minimize(objective, start, method="SLSQP",
                      bounds=[(0.0, cap)] * n + [(0.0, None)] * n,
                      constraints=constraints, options={"ftol": 1e-14, "maxiter": 1000})
    return optimizer.utility(scores, result.x[:n], holdings, covariance)


class ExactSolverTest(unittest.TestCase):
    def test_never_beaten_by_a_dense_grid(self):
        rng = np.random.default_rng(0)
        for n, step in ((2, 1e-5), (3, 2e-3)):
            for _ in range(100):
                optimizer, scores, holdings, covariance = random_problem(rng, n)
                weights = optimizer.solve(scores, holdings, covariance)
                self.assertTrue(optimizer.is_feasible(weights))
                exact = optimizer.utility(scores, weights, holdings, covariance)
                grid = grid_best_utility(optimizer, scores, holdings, covariance, step)
                self.assertGreaterEqual(exact, grid - 1e-12)

    def test_agrees_with_slsqp(self):
        rng = np.random.default_rng(1)
        for n in (2, 3, 4):
            for _ in range(40):
                optimizer, scores, holdings, covariance = random_problem(rng, n)
                weights = optimizer.solve(scores, holdings, covariance)
                exact = optimizer.utility(scores, weights, holdings, covariance)
                reference = slsqp_best_utility(optimizer, scores, holdings, covariance)
                self.assertGreaterEqual(exact, reference - 1e-9)
                self.assertLess(exact - reference, 1e-6)

    def test_rows_are_solved_independently(self):
        rng = np.random.default_rng(2)
        optimizer = PortfolioOptimizer(3, PortfolioConfig(fee_rate=0.002, risk_aversion=5.0))
        scores = rng.uniform(-0.1, 0.1, (50, 3))
        holdings = rng.dirichlet(np.ones(3), size=50)
        covariances = np.array([random_covariance(rng, 3) for _ in range(50)])
        together = optimizer.solve(scores, holdings, covariances)
        one_by_one = np.array([optimizer.solve(s, u, S)
                               for s, u, S in zip(scores, holdings, covariances)])
        np.testing.assert_allclose(together, one_by_one, atol=1e-12)

    def test_two_assets_without_fees_match_the_closed_form(self):
        # Interior optimum of c1*w + c2*(1-w) - lam * [w, 1-w] Sigma [w, 1-w]:
        # w = ((c1 - c2) / (2 lam) - s12 + s22) / (s11 - 2 s12 + s22)
        s11, s12, s22, lam = 0.04, 0.012, 0.09, 1.0
        optimizer = PortfolioOptimizer(2, PortfolioConfig(fee_rate=0.0, risk_aversion=lam))
        scores = np.array([0.03, 0.02])
        weights = optimizer.solve(scores, [0.5, 0.5], [[s11, s12], [s12, s22]])
        expected = ((scores[0] - scores[1]) / (2 * lam) - s12 + s22) / (s11 - 2 * s12 + s22)
        np.testing.assert_allclose(weights, [expected, 1 - expected], atol=1e-12)

    def test_keeps_holdings_when_fees_outweigh_the_score_gap(self):
        optimizer = PortfolioOptimizer(2, PortfolioConfig(fee_rate=0.05, risk_aversion=0.1))
        holdings = np.array([0.3, 0.7])
        weights = optimizer.solve([0.02, 0.01], holdings, [[0.0064, 0.002], [0.002, 0.0081]])
        np.testing.assert_allclose(weights, holdings, atol=1e-12)

    def test_utility_formula(self):
        optimizer = PortfolioOptimizer(2, PortfolioConfig(fee_rate=(0.001, 0.002),
                                                          risk_aversion=2.0))
        r, w, u = np.array([0.05, -0.02]), np.array([0.6, 0.4]), np.array([0.5, 0.5])
        S = np.array([[0.01, 0.002], [0.002, 0.02]])
        expected = (0.05 * 0.6 - 0.02 * 0.4) - (0.001 * 0.1 + 0.002 * 0.1) - 2.0 * (w @ S @ w)
        self.assertAlmostEqual(optimizer.utility(r, w, u, S), expected, places=15)

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            PortfolioOptimizer(2, PortfolioConfig(risk_aversion=0.0))
        with self.assertRaises(ValueError):
            PortfolioOptimizer(2, PortfolioConfig(max_weight=0.4))    # 2 * 0.4 < 1
        with self.assertRaises(ValueError):
            PortfolioOptimizer(5, PortfolioConfig())                  # beyond MAX_ASSETS
        with self.assertRaises(ValueError):
            PortfolioOptimizer(2, PortfolioConfig(fee_rate=-0.001))
        optimizer = PortfolioOptimizer(2, PortfolioConfig())
        with self.assertRaises(ValueError):                           # not positive definite
            optimizer.solve([0.01, 0.02], [0.5, 0.5], [[0.01, 0.02], [0.02, 0.01]])
        with self.assertRaises(ValueError):                           # wrong shape
            optimizer.solve([0.01, 0.02, 0.0], [0.5, 0.5], np.eye(2))


if __name__ == "__main__":
    unittest.main()
