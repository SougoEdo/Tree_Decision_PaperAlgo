from spo_portfolio_tree import (
    PortfolioConfig,
    PortfolioOptimizer,
    TreeConfig,
    SPOPortfolioTree,
)

portfolio = PortfolioOptimizer(
    n_assets=R_train.shape[1],
    config=PortfolioConfig(
        min_weight=-1.0,
        max_weight=1.0,
        gross_limit=1.0,       # Sum of absolute weights <= 1
        net_exposure=None,    # No fixed sum of signed weights
        fee_rate=0.001,       # Illustrative; replace with your fees
        l2_penalty=0.01,      # Select using historical validation
    ),
)

tree = SPOPortfolioTree(
    optimizer=portfolio,
    config=TreeConfig(
        max_depth=2,
        min_samples_leaf=20,
        max_thresholds=None,     # Test every valid threshold
        prediction_bound=0.10,   # Fee-aware scores bounded to ±10%
        search_passes=3,
        search_grid_size=7,
    ),
)

tree.fit(X_train, R_train, U_train)
tree.describe()

# Both inputs are 2-D, even for one live decision.
predicted_returns = tree.predict_returns(X_live)
target_weights = tree.predict_weights(X_live, U_live)