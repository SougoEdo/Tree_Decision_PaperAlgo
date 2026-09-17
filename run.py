from main import (
    PortfolioConfig,
    PortfolioOptimizer,
    TreeConfig,
    SPOPortfolioTree,
)


def load_data():
    """Return aligned arrays, one row per decision date.

    X_train: (observations, features), known before each decision
    R_train: (observations, assets), simple returns over the next holding period
    U_train: (observations, assets), pre-trade weights, each row summing to 1
    X_live:  (1, features), latest features
    U_live:  (1, assets), current weights, summing to 1
    """
    raise NotImplementedError("Fill in load_data() in run.py with your data.")


X_train, R_train, U_train, X_live, U_live = load_data()

portfolio = PortfolioOptimizer(
    n_assets=R_train.shape[1],
    config=PortfolioConfig(
        min_weight=0.0,       # Long-only
        max_weight=1.0,
        gross_limit=1.0,      # Sum of absolute weights <= 1
        net_exposure=1.0,     # Fully invested: sum of weights == 1
        fee_rate=0.001,       # Illustrative; replace with your fees
        l2_penalty=0.01,      # Must be > 0; select using historical validation
    ),
)

tree = SPOPortfolioTree(
    optimizer=portfolio,
    config=TreeConfig(
        max_depth=2,
        min_samples_leaf=20,
        max_thresholds=10,       # None tests every threshold: too slow with fees
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
