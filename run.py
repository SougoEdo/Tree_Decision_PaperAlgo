import numpy as np

from main import (
    PortfolioConfig,
    PortfolioOptimizer,
    TreeConfig,
    SPOPortfolioTree,
    weekly_covariance,
    weekly_rows,
)


def load_data():
    """Return daily data in time order, one row per calendar day (no gaps).

    dates:           (days,) numpy datetime64 or 'YYYY-MM-DD' strings
    closes:          (days, assets) daily closing prices, same asset order everywhere
    daily_features:  (days, features), row d computed from data up to day d only
                     (NaN is fine during warm-up, before the first decision)
    current_weights: (assets,) holdings now, for the live decision, summing to 1
    """
    raise NotImplementedError("Fill in load_data() in run.py with your data.")


dates, closes, daily_features, current_weights = load_data()

# One row per Monday close, with 180 days of returns behind it; the last
# 52 weeks are held out to test the tree after training.
rows = weekly_rows(dates, closes, daily_features, weekday=0, window=180)
test_weeks = 52
train = slice(0, len(rows.R) - test_weeks)
test = slice(len(rows.R) - test_weeks, None)

portfolio = PortfolioOptimizer(
    n_assets=rows.R.shape[1],
    config=PortfolioConfig(
        max_weight=1.0,       # Long-only, fully invested; cap per asset
        fee_rate=0.001,       # Illustrative; replace with your fees
        risk_aversion=1.0,    # Must be > 0; select using historical validation
    ),
)

tree = SPOPortfolioTree(
    optimizer=portfolio,
    config=TreeConfig(
        max_depth=2,
        min_samples_leaf=20,
        max_thresholds=None,         # Every threshold; affordable with screening
        min_sharpe_improvement=0.0,  # Raise (e.g. 0.05 per week) to ignore small in-sample gains
        prediction_bound=0.10,       # Leaf scores bounded to ±10%
        search_passes=3,
        search_grid_size=7,
        verbose=True,
    ),
)

tree.fit(rows.X[train], rows.R[train], rows.Sigma[train])   # holdings start from equal weights
tree.describe()
train_path = tree.replay(rows.X[train], rows.R[train], rows.Sigma[train])
test_path = tree.replay(rows.X[test], rows.R[test], rows.Sigma[test],
                        initial_weights=train_path.final_holdings)
for name, path in (("Train", train_path), ("Test", test_path)):
    print(f"{name}: {len(path.net_returns)} weeks, mean net return "
          f"{path.net_returns.mean():.3%} per week, Sharpe {path.sharpe:.3f} per week, "
          f"turnover {path.turnover.sum():.2f}, fees {path.fees.sum():.3%}")

# Live decision at the latest close (run on the decision weekday).
X_live = np.asarray(daily_features, dtype=float)[-1:]
Sigma_live = weekly_covariance(closes, len(closes) - 1, window=180)
target_weights = tree.predict_weights(X_live, np.asarray(current_weights)[None, :], Sigma_live)
print("Target weights:", target_weights[0])
