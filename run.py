from main import (
    PortfolioConfig,
    PortfolioOptimizer,
    TreeConfig,
    SPOPortfolioTree,
)


def load_data():
    """Return aligned arrays, one row per decision date, in time order.

    X_train:     (observations, features), known before each decision
    R_train:     (observations, assets), simple returns over the next holding period
    Sigma_train: (observations, assets, assets), covariance of the next
                 holding-period returns, estimated from past data only
    X_live:      (1, features), latest features
    U_live:      (1, assets), current weights
    Sigma_live:  (1, assets, assets), latest covariance estimate
    """
    raise NotImplementedError("Fill in load_data() in run.py with your data.")


X_train, R_train, Sigma_train, X_live, U_live, Sigma_live = load_data()

portfolio = PortfolioOptimizer(
    n_assets=R_train.shape[1],
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

tree.fit(X_train, R_train, Sigma_train)   # holdings start from equal weights
tree.describe()
train_path = tree.replay(X_train, R_train, Sigma_train)
print(f"Training Sharpe per period: {train_path.sharpe:.3f}")

# Both inputs are 2-D, even for one live decision.
predicted_returns = tree.predict_returns(X_live)
target_weights = tree.predict_weights(X_live, U_live, Sigma_live)
