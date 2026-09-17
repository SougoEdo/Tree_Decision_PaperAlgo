# SPO portfolio tree: report

**Scope.** `main.py` (814 lines), `run.py` and `tests/` (33 tests) at commit `e0c02fe`, checked against the four
articles in `../Biblio`. Date: 2026-09-17. **No code was changed.** Extra checks ran through subclasses in a
scratch folder. Every number below comes from **synthetic data** (5 years of daily prices, 3 crypto-like
assets with 3% daily volatility, settings of `run.py`) unless stated otherwise.

**Short names used for the articles.**
- **SPOT**: Elmachtoub, Liang & McNellis (2020), *Decision Trees for Decision-Making under the Predict-then-Optimize Framework*.
- **W&H-SPO**: Wang & Hasuike (2026), *Smart Predict-then-Optimize Paradigm for Portfolio Optimization in Real Markets*.
- **DIR**: Wang & Hasuike (2026), *Decision-Induced Ranking Explains Prediction Inflation and Excessive Turnover…*
- **DSPO**: Zhong et al. (2024), *DSPO: An End-to-End Framework for Direct Sorted Portfolio Construction*.

---

## 1. Summary

1. **Correctness.** I found no bug in what the code claims to do.
   - The 33 unit tests pass.
   - 17 bugs planted on purpose during development were each caught by a test.
   - The extra audit checks pass (section 5). Among them, the regret minimised in training equals the regret of
     the replayed path to 9 decimals.
   - The optimizer is exact: it agrees with a dense grid and with SLSQP to 5e-14.
2. **It implements your specification.**
   - A decision tree outputs a score vector per leaf.
   - Leaf scores minimise an SPO loss (decision regret) that contains the fee and the covariance penalty.
   - A split is kept only if the Sharpe ratio of the tree's own replayed trading path improves.
3. **Consistency with the articles.**
   - The decision problem and the loss are exactly DIR eq. 8 and eq. 10.
   - The tree side follows SPOT's greedy partitioning, with deviations that were deliberate choices (section 6).
   - Four elements of the articles are absent: pruning or validation, rolling retraining, partial adjustment
     and forests.
4. **Three behaviours that are not bugs but differ from what one would assume.**
   - **The Sharpe test is measured on the training weeks and almost never says no.** At the root, 32% to 99.7%
     (median 81%) of the ~715 candidate splits pass both gates.
   - **The regret gate is non-binding.** 715 of 715 candidates pass it in 9 of 10 datasets.
   - **With `risk_aversion = 1` the covariance penalty is about 10% of the size of weekly returns.** 83% of the
     tree's portfolios are all-in on one asset, and turnover is 0.6 to 1.0 per week.
5. **The mechanics work when the signal is identifiable.**
   - I added a feature that reveals the true regime and halved the noise. The tree split on that feature first
     in 20 of 20 datasets.
   - At depth 1 its test Sharpe was within 0.05 of the ceiling (standard error 0.05).
   - At crypto-level noise the tree found the feature in 14 of 20 datasets. So the earlier disappointing
     experiments are explained by statistics, not by a coding error: the tree picks the best of ~700 candidate
     splits on ~180 weeks.

---

## 2. Big picture: the pipeline

```
daily closes + your daily features                                   (load_data, run.py)
      │  weekly_rows                                                 (main.py:678)
      ▼
one row per Monday close:  X_t features · R_t next-week returns · Σ_t weekly covariance
      │  train_and_test: every week except the last 52               (main.py:721)
      ▼
SPOPortfolioTree.fit ──uses──► PortfolioOptimizer.solve (exact) + replay_path (drift, fees)
      ▼
frozen tree:  x → leaf → score vector c
      │ replay: train, then test continuing the holdings             │ live: predict_weights(x, holdings, Σ)
      ▼                                                              ▼
performance_report: return, volatility, Sharpe, max drawdown,        target weights
turnover and fees vs "tree without splits" and "equal weight"
```

| Stage | What it does |
|---|---|
| **Weekly rows** (`weekly_rows`, `main.py:678`) | A weekday close becomes a row once 180 daily returns lie behind it and a close 7 days later exists. `R_t = close[d+7]/close[d] − 1`. `Σ_t` is 7 × the sample covariance of the last 180 daily returns (up to and including the decision close), plus a 1e-6 ridge. `X_t` is your feature row on the decision date. Tests prove that no later price changes a row. The code cannot check your features, so you must make sure they only use past data. |
| **Optimizer** (`PortfolioOptimizer`, `main.py:115`) | Turns a score vector and the current holdings into weights. It is deterministic and never trained. |
| **Replay** (`replay_path`, `main.py:300`) | Trades week by week: solve from the current holdings, pay fees, earn the net return, then let holdings drift with returns. |
| **Training** (`fit`, `main.py:410`) | Grows the tree on the training weeks while it trades its own holdings (section 3). |
| **Evaluation** (`train_and_test`, `performance_report`) | Fits the tree and a no-split tree, then replays both and an equal-weight portfolio on the training and test weeks. |
| **Live decision** (`predict_weights`, `weekly_covariance`) | Uses the same optimizer and the same Σ estimate as training. |

---

## 3. What is optimised, precisely

**3.1 Decision problem** (`solve`, `main.py:175`). In week *t*:
- `c` is the score vector from the tree leaf.
- `u_t` is the holdings before trading.
- `f` is `fee_rate` (one value per asset).
- `λ` is `risk_aversion`.

```
w_t(c) = argmax_w   c·w − f·|w − u_t| − λ·wᵀ Σ_t w      subject to   0 ≤ w_i ≤ max_weight,   Σ_i w_i = 1
```
The solver is exact. It enumerates the KKT cases: each weight sits at 0, at the cap, at `u_i` (the fee kink), or
is free above or below `u_i`. That gives 5ⁿ cases, which is why the code allows at most 4 assets. Each case is a
small linear system. Since λ > 0 and Σ is positive definite, the optimum is unique, so there are no ties.

**3.2 Utility and SPO loss** (`utility`, `main.py:167`). The utility of a decision `w` under the realised returns `r_t` is:
```
U_t(w; r_t) = r_t·w − f·|w − u_t| − λ·wᵀ Σ_t w
```
The SPO loss (regret) of scores `c` in week *t* is:
```
ℓ_t(c) = U_t(w_t(r_t); r_t) − U_t(w_t(c); r_t)   ≥ 0
```
Two points about this regret:
- The hindsight decision `w_t(r_t)` starts from the same holdings `u_t`. It is a one-week oracle, as in the articles.
- Fees and the covariance penalty are inside the regret, as we decided on 2026-09-17 and as in DIR eq. 10.

**3.3 Path and Sharpe ratio** (`_trade`, `main.py:349`).
```
u_1 = equal weights
w_t = w_t(c_t)
net_t = r_t·w_t − f·|w_t − u_t|
u_{t+1} = w_t∘(1 + r_t) / (1 + r_t·w_t)
Sharpe = mean(net_t) / std(net_t)  per week     (×√52 in reports; no risk-free rate)
```

**3.4 Leaf scores** (`_fit_leaf`, `main.py:556`). Take a leaf with rows `I`. The holdings `u_t` are frozen at the last
replay of the current tree. The leaf scores are:
```
ĉ ≈ argmin over c in [−b, b]ⁿ of   Σ_{t∈I} ℓ_t(c)          with b = prediction_bound
```
- **Search:** it starts from the clipped mean return of the leaf and also tries the parent's scores. It then runs
  `search_passes` rounds of coordinate-wise grid search. Each round tries `search_grid_size` values per asset,
  and the window shrinks each round. The search is deterministic and approximate.
- **Why a search is needed:** SPOT's Theorem 1 says the leaf mean is optimal, but only when all rows of a leaf
  share the same problem. I verified this on the current code (900 random candidates per case):

| Rows of the leaf share… | Candidates that beat the mean |
|---|---|
| the same `u` and the same Σ | 0 of 900 |
| the same Σ, but each row has its own holdings | 73 of 900 |
| the same holdings, but each row has its own Σ | 181 of 900 |

**3.5 Growing the tree** (`fit`, `_best_split`, `_shortlist`; `main.py:410`, `485`, `512`).
```
0. Root: fit scores on all rows with every row's holdings set to equal weights.
   Replay to get the holdings u_t, the hindsight utilities and the Sharpe S.
1. For every leaf that may still split (depth < max_depth and rows ≥ 2·min_samples_leaf):
   a. SCREEN every (feature, threshold). Give each child its clipped mean return as scores.
      Rank the splits by the summed regret of the two children. Keep the shortlist_size (3) lowest.
   b. For each shortlisted split, fit both children as in 3.4.
      gain = (leaf regret − children's regret) / rows.  Skip the split if gain ≤ min_regret_improvement.
   c. Replay the whole tree with that split to get its Sharpe S′.
2. Take the candidate with the highest S′ over all leaves.
   If S′ > S + min_sharpe_improvement: accept it, replay, refresh u_t and S, and go back to step 1.
   Otherwise stop.
3. Refit every leaf on the final holdings. Keep the refit only if the replayed Sharpe does not fall.
```

**3.6 There is no single global objective.**
- Leaf values minimise regret, which is the SPO loss.
- The tree structure maximises the training-path Sharpe among regret-shortlisted candidates.
- The search is greedy, and holdings are frozen within a round.

---

## 4. Parameters

| Parameter | Default | `run.py` | Role |
|---|---|---|---|
| `max_weight` | 1.0 | 1.0 | Cap per asset. Portfolios are long-only and fully invested (weights sum to 1). |
| `fee_rate` | 0.001 | 0.001 | Proportional fee per traded notional. The same value is used in the optimizer, the regret and the replay. |
| `risk_aversion` λ | 1.0 | 1.0 | Multiplies `wᵀΣw`. Must be > 0. **Its scale decides whether the covariance matters** (finding F4). |
| `max_depth` | **4** | 2 | Root depth is 0. |
| `min_samples_leaf` | **10** | 20 | Minimum weeks per leaf. |
| `max_thresholds` | **10** | None | Candidate thresholds per feature, evenly spaced by rank. `None` tests every midpoint. |
| `shortlist_size` | 3 | 3 | Screened splits per leaf that get a full fit and a replay. |
| `min_regret_improvement` | 1e-6 | 1e-6 | Regret gate, per row of the split leaf (finding F2). |
| `min_sharpe_improvement` | 0.0 | 0.0 | Margin on the weekly training Sharpe needed to accept a split. |
| `prediction_bound` b | 0.10 | 0.10 | Leaf scores are limited to ±b, the counterpart of DIR's clipping. It was not binding in the audit runs (largest score 0.06). |
| `search_passes`, `search_grid_size` | 3, 7 | 3, 7 | Leaf search; the final resolution is about 0.002. |
| `regret_tolerance` | 1e-6 | 1e-6 | Numerical guard only. |
| `weekday`, `window`, `ridge` | 0, 180, 1e-6 | same | Decision day (Monday), days of data in the covariance, and the ridge added to it. |
| `test_periods` | 52 | 52 | Weeks held out at the end. |

`MAX_ASSETS = 4` and `WEIGHT_TOLERANCE = 1e-9` are fixed constants.

The defaults in bold are much looser than `run.py` (finding F7).

---

## 5. How the code was verified

**5.1 Unit tests.** 33 tests run in 4 s:
- **Optimizer (7):** exactness against a dense grid and against SLSQP for 2 to 4 assets, including holdings that
  drifted above the cap; the two-asset closed form; the no-trade zone; row independence; input validation.
- **Replay (7):** a hand-computed drift and fee example; accounting identities on a random path; continuing a path.
- **Tree (8):** splits on a clean regime feature; rejects a split that flips weekly once 2 × fee exceeds the edge;
  the Sharpe rises with every accepted split; both gates; a worse refit is discarded; depth and leaf-size limits.
- **Data (7):** weekday and spacing; returns and Σ against hand computations; **no look-ahead** when later prices change.
- **Report (4):** drawdown, annualisation, the equal-weight baseline, holdings carried from train to test.

**5.2 Planted bugs.** 17 mutations, for example no drift, fees ignored, Σ window including the next day, features
read one day late, the Sharpe check removed, the depth limit ignored. Each one makes at least one test fail.

**5.3 Audit checks that no unit test covers.**

| Check | Result |
|---|---|
| Final tree routes every training row to the leaf it had during growth | OK in 6 of 6 fits |
| Sharpe logged at the last split equals the replayed Sharpe of that tree | OK in 6 of 6 |
| Regret minimised in training equals the regret reported by the replay | Equal to 9 decimals in 6 of 6 |
| Fitting twice gives the same tree | OK in 6 of 6 |
| Adding a constant to all scores leaves the portfolio unchanged | Largest weight change 2e-15 |
| Leaf search against a dense search over ~7,900 score pairs plus local refinement (20 leaves) | Coordinate search is within **0.03%** (median) of the best regret found, worst **2.1%**. The plain leaf mean is 0.8% above (median), worst 3.9%. |
| Shortlist of 3 against a full evaluation of all 715 root splits (10 datasets) | The chosen split is the best of all in **8 of 10**. The 2 misses cost 0.004 and 0.006 of weekly Sharpe. |
| Positive control with a perfect regime feature | See summary point 5 and finding F3. |
| Final refit of leaf scores (40 fits) | It proposes new scores in 36 fits; 16 are kept and 20 discarded. No systematic effect on the replayed regret (lower in 18 of 36) or on the Sharpe (higher in 16 of 36). |
| A constant-price "cash" column | Runs unchanged; the ridge keeps Σ positive definite. |

---

## 6. Consistency with the articles

Status legend: **=** identical · **≈** same idea, different detail · **Δ** deliberate deviation that we agreed · **✗** not implemented.

**6.1 SPOT (the tree).**

| Element | Article | Code | Status |
|---|---|---|---|
| Decision problem | `min cᵀw` over one fixed set S (§2) | `max c·w − g_t(w)`, where `g_t` is fees plus risk and differs by row | ≈ generalised as in W&H |
| SPO loss | eq. 2, worst case over ties | Regret on the full objective; the optimum is unique, so no tie rule is needed | = |
| Leaf value | Mean of the leaf (Theorem 1) | Mean as the starting point, then a bounded search (needed, see 3.4) | ≈ necessary extension |
| Split criterion | Lowest summed SPO loss of the children (eq. 5) | That criterion is used only to shortlist (with mean scores). The final choice is the highest replayed Sharpe. | Δ your idea |
| Growth | Recursive, each leaf on its own | Best-first, one split per round, because the path couples the leaves | Δ implied by path-awareness |
| Thresholds | Quantiles of the sorted unique values | Midpoints of unique values, optional evenly spaced subset | = |
| Stopping | Maximum depth, minimum leaf size | The same, plus the two gates | = |
| **Pruning** | CART pruning on a 20% validation set, with SPO loss as the metric (§4.1, §5.1) | None | ✗ |
| Forests, MILP | §4.3, §4.2 | None | ✗ optional |
| Independent rows | Assumed (§2) | Violated on purpose, because holdings follow the path | Δ |

**6.2 W&H-SPO (the optimizer and the protocol).**

| Element | Article | Code | Status |
|---|---|---|---|
| Objective | `r̂ᵀw − γ‖w − w_{t−1}‖₁ − λ‖w‖²` (eq. 6) | Same fee term, with per-asset rates allowed. The risk term is `λwᵀΣw`. | Δ covariance instead of L2 |
| `w_{t−1}` | "Portfolio held in the previous period" | Holdings drifted by returns, which is what the account really holds | Δ refinement |
| Training | SPO+ surrogate with gradients, linear model (eq. 4) | True regret minimised by search, tree model | ≈ follows SPOT instead |
| Backtest fee | 0.005 × turnover, deducted from returns (§4.2) | `fee_rate` × the traded amount `|w − u|`, deducted from returns | = in form |
| Protocol | Monthly rebalancing; retrained every month on 9 months of training plus 3 of validation, tuned with Optuna | Weekly rows, one train/test split, no tuning | ✗ |
| Metrics | Return, volatility, Sharpe, **Sortino**, max drawdown | All except Sortino | ≈ |
| RobustSPO, SoftmaxDFL | §3.1.4, §3.2 | None | ✗ out of scope |

The article's own evidence on penalties is mixed: adding the L2 penalty lowered its Sharpe from 0.715 to 0.550 (Table 4).

**6.3 DIR (the loss, and the behaviour it predicts).**

| Element | Article | Code | Status |
|---|---|---|---|
| Decision problem | eq. 8, on the simplex (eq. 6) | Identical when `max_weight = 1` | **=** |
| Regret | eq. 10, on the full objective `U_t` | Identical | **=** |
| Previous portfolio | "Induced by past predictions" | The tree's own replayed holdings (drifted) | = |
| KKT structure | eq. 24–27 | The exact solver enumerates precisely these cases | = |
| Σ | 220 trading days, regularisation 1e-6 | 180 days × 7, ridge 1e-6 | ≈ |
| Initial portfolio | Equal weights (eq. 34) | Same | = |
| Clipping | `clip(r̂, ±0.1)` applied after prediction | `prediction_bound = 0.10`, imposed during leaf fitting | ≈ |
| **Partial adjustment** | `w_t = w_{t−1} + δ(w* − w_{t−1})` with δ = 0.1 (eq. 35). This was **their most effective turnover control**. | None (it is the TODO in the docstring) | ✗ |
| Min–max rescaling | eq. 33. The article found it ineffective when used alone. | None | ✗ |
| Predicted pathology | Inflated scores, concentrated portfolios, 84–95% monthly turnover (Table 1) | Reproduced: 83% all-in portfolios, turnover 0.6–1.0 per week | = same behaviour |

**6.4 DSPO.** Nothing from it is used. It ranks 4,000+ stocks every day with a neural network and has no optimizer
and no regret. It is not relevant to 1 or 2 coins.

**6.5 Against your specification.**
- **Met:** the SPO loss with the fee and covariance penalty; the path-Sharpe decision on splits; the tree's own
  drifting holdings; long-only, fully invested portfolios.
- **One nuance to confirm:** the Sharpe does not only veto a split. It also chooses which of the shortlisted
  splits wins (question Q1).

---

## 7. Findings (nothing was changed; suggested fixes are in section 9)

**F1. High, design: the Sharpe acceptance is not a filter.**
- The Sharpe is measured on the same weeks the scores were fitted on, so almost any partition raises it. At the
  root, 231 to 713 of 715 candidates (median 81%) pass both gates.
- On data with **no signal at all**, 20 of 20 trees still split. Training turnover went from 0.9 to 142 (summed over ~182 weeks).
- Training Sharpe was 1.1–1.4 against 0.3–0.5 in the test year.
- A fixed margin does not solve it: lucky gains on noise have a median of 0.067 per week.
- SPOT avoids this with validation pruning, and W&H with time-ordered validation. The code has neither.

**F2. Medium, design: the regret gate is non-binding.**
- `min_regret_improvement = 1e-6` compares the children, which are fitted on the current holdings, with a parent
  score fitted on older holdings.
- Simply refitting the root gives a "free" gain of up to 5.7e-4 per row, and above 1e-6 in 8 of 10 datasets.
- All 715 candidates pass the gate in 9 of 10 datasets.
- So the SPO loss acts through leaf values and the shortlist, not through this gate. A split whose two children
  are almost identical can still be accepted.

**F3. Medium, design: the shortlist of 3 can exclude the right split.**
- At crypto-level noise the perfect regime split ranked #4, #5, #15, #19, #111 and #324 in the six misses.
- In two of those six it would have had a higher Sharpe than the split that was chosen.
- Normally the shortlist is a good proxy (8 of 10 in section 5.3), and it limits the number of Sharpe comparisons,
  which helps against F1. I would only enlarge it together with out-of-sample acceptance.

**F4. Medium, calibration: at λ = 1 the covariance penalty is almost inactive.** Weekly averages on the training path:

| λ | Absolute gross return | Fees | Risk penalty | Regret | All-in: hindsight / tree | Turnover | Train Sharpe |
|---|---|---|---|---|---|---|---|
| 1 | 0.066 | 0.0006 | **0.006** | 0.034 | 92% / 83% | 0.62 | 1.42 |
| 10 | 0.056 | 0.0003 | 0.045 | 0.027 | 31% / 0% | 0.31 | 1.05 |
| 50 | 0.054 | 0.0001 | 0.212 | 0.009 | 0% / 0% | 0.10 | 0.63 |

- At λ = 1 the algorithm is in practice "pick one asset per leaf". Fees are 3–5% a year.
- λ must be chosen on validation data. DIR scanned values from 0.1 to 50.

**F5. Low, approximation.**
- Leaf scores are fitted with holdings frozen at the previous replay.
- The root is first fitted as if every week started from equal weights.
- The only re-fit happens at the end, and it is neutral on average (section 5.3).
- This is acceptable as designed, but the final scores are not regret-optimal on the final path.

**F6. Low.** The coordinate search is approximate. It is at most 2.1% above the best regret found, with a median of 0.03%.

**F7. Low.**
- The `TreeConfig` defaults (depth 4, leaf 10, 10 thresholds) are far looser than `run.py`.
- They are unsafe for 200–500 weekly rows. A caller that relies on the defaults would overfit badly.

**F8. Info, backtest conventions.**
- The strategy trades at the same daily close it decides on.
- The fee × return cross term is ignored (about 5e-5 per week).
- No spread or slippage is modelled.
- The Sharpe uses no risk-free rate and is annualised with √52.
- Holdings drift. This is deliberate; the articles use the previous target.

**F9. Structural for your goal.**
- Portfolios are always fully invested.
- With **one coin** the code cannot run, because it needs at least 2 assets and the single weight would always be 1.
- With **BTC and ETH** only, the strategy can rotate but never reduce market exposure. Every strategy in the
  experiments had a median max drawdown of 36–48%.
- A constant-price **cash column works today without any code change**. λ then becomes the exposure dial: at
  λ = 10 the test portfolio held 87% cash.

**F10. Cosmetic.**
- `Node.regret_sum` of internal nodes is stale.
- Sortino is missing compared with W&H.
- There are no sample weights.

---

## 8. What the experiments say so far (synthetic, 20 datasets per line, 52-week test)

Differences are in annualised test Sharpe; the standard error is in brackets.

| Setting | Tree − equal weight | Tree − simple momentum rule | Tree finds the informative feature(s) |
|---|---|---|---|
| 5 years, no signal | +0.06 (0.13) | +0.20 (0.17) | n/a |
| 5 years, strong simple signal | −0.10 (0.10) | **−0.35 (0.16)** | 8/20 |
| 10 years, strong simple signal | **+0.28 (0.14)** | −0.20 (0.14) | 13/20 |
| 10 years, interaction signal (volatility regime × trend) | **+0.67 (0.16)** | **+0.70 (0.21)** | 16/20 |
| 5 years, perfect regime feature, depth 1, half noise | +1.61 (0.17) | −0.05 (0.05) vs the ceiling rule | 20/20 |

- The tree earns its place when the signal is an **interaction** that a one-feature rule cannot see, and when
  there are about **10 years** of data.
- With short histories or simple signals it does not beat simple rules.
- One test year gives a Sharpe uncertain by about ±1.0, so real-data conclusions need several out-of-sample years.

---

## 9. Improvement ideas, in priority order for BTC/ETH

1. **Accept splits on weeks the tree did not learn from (fixes F1).**
   - Grow the tree on the first ~70% of the training window.
   - Keep a split only if the replayed Sharpe also improves on the remaining ~30%, with holdings carried over.
   - This is SPOT's validation pruning in path-aware form.
2. **Walk-forward evaluation.**
   - Retrain monthly or quarterly, as both W&H articles do, on 10 years of BTC and ETH.
   - Stitch the out-of-sample weeks together.
   - Report against buy-and-hold BTC, buy-and-hold ETH, 50/50, a momentum rule, and a **placebo with shuffled features**.
3. **Add a cash or stablecoin column (F9).**
   - It is mandatory for one coin, and it lets the strategy cut its exposure.
   - No code change is needed.
4. **Tune λ, `prediction_bound`, depth and leaf size on validation data (F4).** The articles do this with Optuna.
5. **Control turnover.**
   - Add partial adjustment (DIR eq. 35) to both the replay and live trading.
   - Optionally add a buffer zone around thresholds.
   - This is the existing TODO. With fees of 0.1% and turnover near 1 per week, costs are 3–5% a year.
6. **Make the regret gate real (F2).**
   - Refit the parent on the current holdings before measuring the gain.
   - Express the gate relative to the leaf's regret, for example at least 1–2%.
7. **Shrink the search.**
   - Use about 10 quantile thresholds per feature, depth ≤ 2, leaves ≥ 40–50 weeks once there are 10 years of
     data, and at most 5 well-motivated features.
   - Align the `TreeConfig` defaults with `run.py` (F7).
8. **Report uncertainty.**
   - Add bootstrap intervals for the Sharpe and for the difference against each baseline in `performance_report`.
   - Add Sortino.
9. **Optional.**
   - SPO forest with a block bootstrap (SPOT §4.3).
   - Trade at the next open and include spread and slippage.
   - A helper that tests your feature function for look-ahead by truncating the data, as the tests do for Σ.

---

## 10. Questions for you

- **Q1. Split choice.** Today, among the 3 lowest-regret candidates of each leaf, the one with the **highest
  training Sharpe** wins. Is that what you want? The alternative is to take the **lowest-regret** split, which is
  SPOT's rule, and use the Sharpe only as a veto.
- **Q2. Assets.** Do you want BTC and ETH fully invested (pure rotation), or with a cash or stablecoin column so
  that exposure can change? For a single coin, cash is required.
- **Q3. Role of the covariance penalty.** Should it really shape the weights (then λ ≈ 5–20 on a weekly Σ), or is
  near all-in rotation acceptable (λ ≈ 1)?
- **Q4. Walk-forward cadence.** How often should it retrain (monthly or quarterly), and with a training window of how many years?
- **Q5. Priority.** Still open from before: a tradable strategy, or a paper on path-aware SPO trees? It changes
  which of the items in section 9 come first.
