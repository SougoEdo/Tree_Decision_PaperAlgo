# SPO portfolio tree: report

**Scope.** `main.py` (814 lines), `run.py` and `tests/` (33 tests) at commit `e0c02fe`, checked against the four
articles in `../Biblio`. Date: 2026-09-17, revised the same day after your questions (sections 3.5, 7, 8, 9 and 11
changed most). **The audit itself changed no code.** Afterwards, at your request, two changes were made in
`_best_split`: the parent refit, and the largest regret reduction chooses the split (section 3.5; 35 tests now).
Extra checks ran through subclasses in a scratch folder. Every number
comes from **synthetic data** (5 years of daily prices, 3 crypto-like assets with 3% daily volatility, settings of
`run.py`) unless stated otherwise.

**Short names used for the articles.**
- **SPOT**: Elmachtoub, Liang & McNellis (2020), *Decision Trees for Decision-Making under the Predict-then-Optimize Framework*.
- **W&H-SPO**: Wang & Hasuike (2026), *Smart Predict-then-Optimize Paradigm for Portfolio Optimization in Real Markets*.
- **DIR**: Wang & Hasuike (2026), *Decision-Induced Ranking Explains Prediction Inflation and Excessive Turnover…*
- **DSPO**: Zhong et al. (2024), *DSPO: An End-to-End Framework for Direct Sorted Portfolio Construction*.

**Vocabulary.** Each leaf holds one **predicted-return vector** ĉ, with one number per asset (`prediction` in the
code). The first version of this report called it a "score vector", which is DIR's word for predictions trained
on regret rather than on forecast error. It is the same object.

**Runs added after your second round of questions:** the comparison at λ = 1, 10 and 50 with a "tree without
splits" reference (section 8), and the calibration of the regret threshold (section 9.3).

---

## 1. Summary

1. **Correctness.** I found no bug in what the code claims to do.
   - The 33 unit tests pass.
   - 17 bugs planted on purpose during development were each caught by a test.
   - The extra audit checks pass (section 5). Among them, the regret minimised in training equals the regret of
     the replayed path to 9 decimals.
   - The optimizer is exact: it agrees with a dense grid and with SLSQP to 5e-14.
2. **It implements your specification.**
   - A decision tree whose leaves hold predicted returns.
   - Leaf values minimise an SPO loss (decision regret) that contains the fee and the covariance penalty.
   - A split is kept only if it **lowers the regret and raises the Sharpe ratio** of the tree's own replayed
     trading path.
   - The regret condition exists in the code, but today it has no effect (finding F2). That is why the first
     version of this report described the decision as "Sharpe only".
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
   - At crypto-level noise the tree found the feature in 14 of 20 datasets. So the disappointing experiments are
     explained by statistics, not by a coding error: the tree picks the best of ~700 candidate splits on ~180 weeks.
6. **On realistic synthetic signals, the splits add no measurable value out of sample (section 8).**
   - Against the same tree without splits, the difference in test Sharpe is between −0.05 and +0.12, always
     within about one standard error. This holds at λ = 1, 10 and 50.
   - Where the tree beats equal weight, the tree without splits beats it just as much.
   - Lucky splits on noise remove as much regret (10.2% at λ = 1) as real splits on realistic signals (8.5–9.0%).

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
frozen tree:  x → leaf → predicted returns ĉ
      │ replay: train, then test continuing the holdings             │ live: predict_weights(x, holdings, Σ)
      ▼                                                              ▼
performance_report: return, volatility, Sharpe, max drawdown,        target weights
turnover and fees vs "tree without splits" and "equal weight"
```

| Stage | What it does |
|---|---|
| **Weekly rows** (`weekly_rows`, `main.py:678`) | One row per Monday close once 180 daily returns lie behind it. `R_t = close[d+7]/close[d] − 1`. `Σ_t` is 7 × the covariance of the last 180 daily returns, plus a 1e-6 ridge. `X_t` is your feature row on that date. Tests prove that no later price changes a row. Your features are not checked: they must only use past data. |
| **Optimizer** (`PortfolioOptimizer`, `main.py:115`) | Turns predicted returns and the current holdings into weights. It is deterministic and never trained. |
| **Replay** (`replay_path`, `main.py:300`) | Trades week by week: solve from the current holdings, pay fees, earn the net return, then let holdings drift with returns. |
| **Training** (`fit`, `main.py:410`) | Grows the tree on the training weeks while it trades its own holdings (section 3). |
| **Evaluation** (`train_and_test`, `performance_report`) | Fits the tree and a no-split tree, then replays both and an equal-weight portfolio on the training and test weeks. |
| **Live decision** (`predict_weights`, `weekly_covariance`) | Uses the same optimizer and the same Σ estimate as training. |

---

## 3. What is optimised, precisely

**3.1 Decision problem** (`solve`, `main.py:175`). In week *t*:
- `c` is the predicted-return vector from the tree leaf.
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
The SPO loss (regret) of predicted returns `c` in week *t* is:
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

**3.4 Leaf predictions** (`_fit_leaf`, `main.py:556`). Take a leaf with rows `I`. The holdings `u_t` are frozen at
the last replay of the current tree. The leaf's predicted returns are:
```
ĉ ≈ argmin over c in [−b, b]ⁿ of   Σ_{t∈I} ℓ_t(c)          with b = prediction_bound
```
- **Search:** it starts from the clipped mean return of the leaf and also tries the parent's predicted returns.
  It then runs `search_passes` rounds of coordinate-wise grid search. Each round tries `search_grid_size` values
  per asset, and the window shrinks each round. The search is deterministic and approximate.
- **Why a search is needed:** SPOT's Theorem 1 says the leaf mean is optimal, but only when all rows of a leaf
  share the same problem. I checked this on the current code with 900 random candidates per case:
  - when the rows share the same holdings and the same Σ, 0 of 900 candidates beat the mean;
  - when each row has its own holdings, 73 of 900 do;
  - when each row has its own Σ, 181 of 900 do.

**3.5 Growing the tree** (`fit`, `_best_split`, `_shortlist`; `main.py:410`, `485`, `512`).

0. **Root.**
   - Fit one predicted-return vector on all weeks, as if every week started from equal weights.
   - Replay the tree. This gives the holdings `u_t` the tree would really have had, and its Sharpe `S`.
1. **For every leaf that may still split** (depth < `max_depth` and at least 2 × `min_samples_leaf` weeks):
   - **a. Screen.** For every feature and every threshold, give each child its mean return as predicted returns
     and add up the regret of the two children. Keep the 3 splits with the lowest regret (`shortlist_size`).
   - **b. Regret test.**
     - First refit the leaf itself on the current holdings. This is the **parent refit**, added after the audit.
     - For each of the 3 splits, find the best predicted returns of the left child and of the right child with
       the search of section 3.4. The search starts from the refitted parent.
     - Compare the refitted leaf's regret with the summed regret of the two children. The difference is the
       split's **regret reduction**.
     - If the reduction per week is not above `min_regret_improvement`, the split is dropped.
2. **Decision**, changed after the audit.
   - Rank the remaining candidates of all leaves by total regret reduction, largest first. This is SPOT's criterion.
   - Replay the tree with the first candidate to get its Sharpe `S′`.
   - If `S′ > S + min_sharpe_improvement`, accept it, replay, refresh `u_t` and `S`, and go back to step 1.
   - Otherwise try the next candidate. If none passes, stop.
3. **Final refit.** Refit every leaf on the final holdings. Keep the refit only if the replayed Sharpe does not fall.

Before this change, the gain was measured against the leaf's old predicted returns, and the winner was the
shortlisted split with the highest Sharpe. Sections 5 to 8 were measured with that earlier rule.

Example of step 1.b on the demo data (`python main.py`), with the earlier rule; the new rule accepts the same three
splits. Feature `x[4]` is pure noise, `x[0]` is 28-day momentum (informative), and `x[2]` is 7-day momentum:

| Accepted split | Feature | Leaf size (weeks) | Regret per week, unsplit → split | Gain | Training Sharpe per week |
|---|---|---|---|---|---|
| 1 | `x[4]`, **pure noise** | 182 | 0.0385 → 0.0332 | 0.0053 = **13.7%** | 0.168 → 0.215 |
| 2 | `x[0]`, informative | 133 | 0.0327 → 0.0306 | 0.0021 = 6.5% | 0.215 → 0.237 |
| 3 | `x[2]` | 49 | 0.0350 → 0.0334 | 0.0016 = 4.6% | 0.237 → 0.249 |

- All three gains are far above the gate of 0.000001, so step b rejects nothing (finding F2).
- The split on noise has the **largest** regret gain and the largest Sharpe gain. This matters for the size of the
  threshold (section 9.3).

**3.6 There is no single global objective.**
- Leaf values minimise regret, which is the SPO loss.
- The tree structure follows the largest regret reduction, and the training-path Sharpe must confirm it.
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
| `min_regret_improvement` | 1e-6 | 1e-6 | Regret gate, per week of the split leaf (finding F2). |
| `min_sharpe_improvement` | 0.0 | 0.0 | Margin on the weekly training Sharpe needed to accept a split. |
| `prediction_bound` b | 0.10 | 0.10 | Predicted returns are limited to ±b, the counterpart of DIR's clipping. It was not binding in the audit runs (largest value 0.06). |
| `search_passes`, `search_grid_size` | 3, 7 | 3, 7 | Leaf search; the final resolution is about 0.002. |
| `regret_tolerance` | 1e-6 | 1e-6 | Numerical guard only. |
| `weekday`, `window`, `ridge` | 0, 180, 1e-6 | same | Decision day (Monday), days of data in the covariance, and the ridge added to it. |
| `test_periods` | 52 | 52 | Weeks held out at the end. |

`MAX_ASSETS = 4` and `WEIGHT_TOLERANCE = 1e-9` are fixed constants.

The defaults in bold are much looser than `run.py` (finding F7).

---

## 5. How the code was verified

**5.1 Unit tests and planted bugs.** 33 tests run in 4 s: optimizer exactness (7), replay accounting (7), tree
growth and both gates (8), weekly rows with a **no-look-ahead** test (7), and the report figures (4). 17 planted
bugs (no drift, fees ignored, Σ window including the next day, features read one day late, the Sharpe check
removed, and others) each make at least one test fail.

**5.2 Audit checks that no unit test covers.**

| Check | Result |
|---|---|
| Final tree routes every training row to the leaf it had during growth | OK in 6 of 6 fits |
| Sharpe logged at the last split equals the replayed Sharpe of that tree | OK in 6 of 6 |
| Regret minimised in training equals the regret reported by the replay | Equal to 9 decimals in 6 of 6 |
| Fitting twice gives the same tree | OK in 6 of 6 |
| Adding a constant to all predicted returns leaves the portfolio unchanged | Largest weight change 2e-15 |
| Leaf search against a dense search over ~7,900 pairs of values plus local refinement (20 leaves) | Coordinate search is within **0.03%** (median) of the best regret found, worst **2.1%**. The plain leaf mean is 0.8% above (median), worst 3.9%. |
| Shortlist of 3 against a full evaluation of all 715 root splits (10 datasets) | The chosen split is the best of all in **8 of 10**. The 2 misses cost 0.004 and 0.006 of weekly Sharpe. |
| Final refit of leaf predictions (40 fits) | It proposes new values in 36 fits; 16 are kept and 20 discarded. No systematic effect on the replayed regret or on the Sharpe. |

---

## 6. Consistency with the articles

Status legend: **=** identical · **≈** same idea, different detail · **Δ** deliberate deviation that we agreed · **✗** not implemented.

**6.1 SPOT (the tree).**

| Element | Article | Code | Status |
|---|---|---|---|
| Decision problem | `min cᵀw` over one fixed set S (§2) | `max c·w − g_t(w)`, where `g_t` is fees plus risk and differs by row | ≈ generalised as in W&H |
| SPO loss | eq. 2, worst case over ties | Regret on the full objective; the optimum is unique, so no tie rule is needed | = |
| Leaf value | Mean of the leaf (Theorem 1) | Mean as the starting point, then a bounded search (needed, see 3.4) | ≈ necessary extension |
| Split criterion | Lowest summed SPO loss of the children (eq. 5) | Now the same: the largest regret reduction wins. It must also raise the replayed Sharpe (your addition). | ≈ |
| Growth | Recursive, each leaf on its own | Best-first, one split per round, because the path couples the leaves | Δ implied by path-awareness |
| Thresholds, stopping | Quantiles of unique values; maximum depth, minimum leaf size | Midpoints of unique values, optional subset; the same stops plus the two gates | = |
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

RobustSPO and SoftmaxDFL (§3.1.4, §3.2) are out of scope. The article's own evidence on penalties is mixed: adding
the L2 penalty lowered its Sharpe from 0.715 to 0.550 (Table 4).

**6.3 DIR (the loss, and the behaviour it predicts).**

| Element | Article | Code | Status |
|---|---|---|---|
| Decision problem | eq. 8, on the simplex (eq. 6) | Identical when `max_weight = 1` | **=** |
| Regret | eq. 10, on the full objective `U_t` | Identical | **=** |
| Previous portfolio | "Induced by past predictions" | The tree's own replayed holdings (drifted) | = |
| KKT structure | eq. 24–27 | The exact solver enumerates precisely these cases | = |
| Σ and initial portfolio | 220 trading days, regularisation 1e-6; equal weights (eq. 34) | 180 days × 7, ridge 1e-6; equal weights | ≈ |
| Clipping | `clip(r̂, ±0.1)` applied after prediction | `prediction_bound = 0.10`, imposed during leaf fitting | ≈ |
| **Partial adjustment** | `w_t = w_{t−1} + δ(w* − w_{t−1})` with δ = 0.1 (eq. 35). This was **their most effective turnover control**. | None (it is the TODO in the docstring) | ✗ |
| Min–max rescaling | eq. 33. The article found it ineffective when used alone. | None | ✗ |
| Predicted pathology | Inflated predictions, concentrated portfolios, 84–95% monthly turnover (Table 1) | Reproduced: 83% all-in portfolios, turnover 0.6–1.0 per week | = same behaviour |

**6.4 DSPO.** Nothing from it is used. It ranks 4,000+ stocks every day with a neural network and has no optimizer
and no regret. It is not relevant to 1 or 2 coins.

**6.5 Against your specification.**
- **Met:** leaves hold predicted returns; the SPO loss with the fee and covariance penalty; a split needs both a
  regret improvement and a path-Sharpe improvement; the tree's own drifting holdings; long-only, fully invested
  portfolios.
- **One gap left:** the regret threshold is still 1e-6, so the regret condition rejects almost nothing (F2).
  The regret now chooses the split, which closes the other gap.

---

## 7. Findings (as measured before the two changes; your proposed corrections are in section 9)

**F1. High, design: the Sharpe acceptance is not a filter.**
- The Sharpe is measured on the same weeks the leaf predictions were fitted on, so almost any partition raises
  it. At the root, 231 to 713 of 715 candidates (median 81%) pass both gates.
- On data with **no signal at all**, 20 of 20 trees still split. Training turnover went from 0.9 to 142 (summed over ~182 weeks).
- Training Sharpe was 1.1–1.4 against 0.3–0.5 in the test year.
- A fixed margin does not solve it: lucky gains on noise have a median of 0.067 per week.
- SPOT avoids this with validation pruning, and W&H with time-ordered validation. The code has neither.

**F2. Medium, design: the regret gate is non-binding.**
- `min_regret_improvement = 1e-6`, while real gains are around 0.002 to 0.005 per week (table in 3.5).
- The children are fitted on the current holdings but compared with a parent value fitted on older holdings.
  Simply refitting the root gives a "free" gain of up to 5.7e-4 per week, and above 1e-6 in 8 of 10 datasets.
- All 715 candidates pass the gate in 9 of 10 datasets.
- So the SPO loss acted through leaf values and the shortlist, not through this gate.
- **Status:** the parent refit is now in the code, so the "free" gain is gone. The threshold is still 1e-6
  (section 9.3).

**F3. Medium, design: the shortlist of 3 can exclude the right split.**
- At crypto-level noise the perfect regime split ranked #4, #5, #15, #19, #111 and #324 in the six misses. In two
  of them it would have had a higher Sharpe than the split that was chosen.
- Normally the shortlist is a good proxy (8 of 10 in section 5.2), and it limits the number of Sharpe
  comparisons. I would only enlarge it together with out-of-sample acceptance.

**F4. Medium, calibration: at λ = 1 the covariance penalty is almost inactive.** Weekly averages on the training path:

| λ | Absolute gross return | Fees | Risk penalty | Regret | All-in: hindsight / tree | Turnover | Train Sharpe |
|---|---|---|---|---|---|---|---|
| 1 | 0.066 | 0.0006 | **0.006** | 0.034 | 92% / 83% | 0.62 | 1.42 |
| 10 | 0.056 | 0.0003 | 0.045 | 0.027 | 31% / 0% | 0.31 | 1.05 |
| 50 | 0.054 | 0.0001 | 0.212 | 0.009 | 0% / 0% | 0.10 | 0.63 |

- **Why the experiments used λ = 1.** It was a placeholder I set when the covariance term replaced the L2 penalty.
  I never calibrated it, and every experiment inherited it. I only measured the size of each term during this
  audit. This is my oversight.
- **Intuition.** Take two assets with 8% weekly volatility and correlation 0.6. The gap in predicted returns that
  moves the portfolio from 50/50 to all-in is `2λσ²(1−ρ)`, about 0.5% × λ.
  - λ = 1: a gap of 0.5% is enough, so at λ = 1 the algorithm is in practice "pick one asset per leaf", with fees of 3–5% a year.
  - λ = 10: it takes a gap of 5%.
  - λ = 50: it would take 25%, more than the ±10% bound allows, so the portfolio can never be all-in.
- **BTC and ETH.** Their correlation is closer to 0.8, which makes the needed gap even smaller (about 0.3% × λ).
  For these two coins λ has to be roughly 10 to 30 before the covariance term matters.
- **λ = 50 is not automatically right.**
  - There the risk penalty (0.21 per week) is 4 times the size of the returns.
  - The optimizer then sits near the minimum-variance portfolio and the predictions barely move it; the training Sharpe fell to 0.63.
  - Return and risk have comparable size around λ = 10.
- **λ must be chosen on validation data.** DIR scanned values from 0.1 to 50. Section 8 gives the out-of-sample
  comparison at λ = 1, 10 and 50.

**F5. Low, approximation.** Leaf predictions are fitted with holdings frozen at the previous replay, and the root
as if every week started from equal weights. The only re-fit happens at the end and is neutral on average
(section 5.2). The coordinate search is at most 2.1% above the best regret found (median 0.03%).

**F6. Low.** The `TreeConfig` defaults (depth 4, leaf 10, 10 thresholds) are far looser than `run.py` and unsafe
for 200–500 weekly rows.

**F7. Info, backtest conventions.** The strategy trades at the same daily close it decides on. The fee × return
cross term is ignored (about 5e-5 per week). No spread or slippage is modelled. The Sharpe uses no risk-free rate
and is annualised with √52. Holdings drift, which is deliberate; the articles use the previous target.

**F8. Structural for your goal.**
- Portfolios are always fully invested.
- With **one coin** the code cannot run, because it needs at least 2 assets and the single weight would always be 1.
- With **BTC and ETH** only, the strategy can rotate but never reduce market exposure. Every strategy in the
  experiments had a median max drawdown of 36–48%.
- A constant-price **cash column works today without any code change**. λ then becomes the exposure dial: at
  λ = 10 the test portfolio held 87% cash.

---

## 8. What the experiments say so far (synthetic, 20 datasets per line, 52-week test)

**What "single-feature" and "interaction" mean.** These words describe **my synthetic data generators, not BTC or ETH**.
- In the *single-feature* data, expected returns depend on one hidden regime that one feature (28-day momentum)
  partly reveals. A one-threshold rule can exploit it. I started with this easiest case as a sanity check of
  statistical power.
- You then pointed out that real signals are interactions, so I built the *interaction* data: the best asset
  depends on the volatility regime **and** the trend, which no one-feature rule can see.
- I agree the interaction case is the more realistic one. It is also the favourable case for a tree.
- None of these results is evidence about BTC or ETH. Only real data can give that.

Paired differences in annualised test Sharpe; the standard error is in brackets.

| Synthetic setting | λ | Tree − equal weight | Tree − **same tree without splits** | Tree − one-feature momentum rule |
|---|---|---|---|---|
| 5 years, single-feature signal | 1 | −0.10 (0.10) | +0.12 (0.16) | −0.35 (0.16) |
| | 10 | +0.04 (0.04) | +0.04 (0.05) | −0.21 (0.13) |
| | 50 | +0.03 (0.03) | +0.01 (0.01) | −0.20 (0.08) |
| 5 years, interaction signal | 1 | +0.24 (0.23) | +0.01 (0.20) | +0.43 (0.28) |
| | 10 | +0.43 (0.18) | +0.05 (0.09) | +0.49 (0.21) |
| | 50 | +0.42 (0.20) | +0.01 (0.01) | 0.00 (0.10) |
| 10 years, interaction signal | 1 | **+0.67 (0.16)** | **−0.02 (0.16)** | +0.70 (0.21) |
| | 10 | +0.64 (0.22) | −0.05 (0.05) | +0.49 (0.23) |
| | 50 | +0.69 (0.21) | −0.01 (0.01) | +0.08 (0.06) |

Other lines, all at λ = 1:
- 5 years, no signal: tree − equal weight +0.06 (0.13).
- 10 years, single-feature signal: tree − equal weight +0.28 (0.14).
- Perfect regime feature at half the noise, depth 1: +1.61 (0.17) against equal weight, and −0.05 (0.05) from the ceiling.

What the tree costs in the test year, on the single-feature data:

| λ | Sharpe, train / test | Turnover per week | Fees per year | All-in weeks | Median max drawdown, tree / equal weight |
|---|---|---|---|---|---|
| 1 | 1.36 / 0.52 | 0.70 | 3.7% | 80% | −43% / −36% |
| 10 | 0.90 / 0.76 | 0.37 | 1.9% | 0% | −39% / −36% |
| 50 | 0.36 / 0.63 | 0.11 | 0.6% | 0% | −34% / −36% |

- **Correction to the first version of this report.**
  - I wrote that the tree "earns its place" on interaction signals with 10 years of data. The middle column
    shows that the same tree **without splits** does just as well (−0.02 ± 0.16).
  - In that synthetic data one asset is defensive and the two others lose money in stressed periods. A single
    leaf already learns to hold the defensive asset. That is what beats equal weight and the momentum rule.
  - The splits add nothing measurable out of sample, in any setting and at any λ. This is so even though 16 of
    20 trees split on the right features.
- **Raising λ removes the damage but does not make the splits useful.**
  - At λ = 10 the tree no longer trails equal weight, the train/test gap shrinks from 0.84 to 0.14, and fees halve.
  - At λ = 50 the tree and the tree without splits are the same portfolio to within ±0.01: the predictions no longer matter.
  - So λ ≈ 10 is a better default than 1 for these synthetic assets.
- **The only setting where a split demonstrably pays is the positive control** (perfect feature, half the noise).
- One test year gives a Sharpe uncertain by about ±1.0, so real-data conclusions need several out-of-sample years.

---

## 9. Your three proposed corrections

**9.1 Train on daily data while trading weekly.**
- **How I understand it.** There is one training row per **day** `d`: the features at `d`, the return from `d` to
  `d+7`, and `Σ_d`. Trading stays weekly. Because holdings follow a path, training would replay 7 interleaved
  weekly paths (a Monday path, a Tuesday path, and so on) and pool their returns for the Sharpe.
- **What it brings.** It does not bring 7 times the information, because consecutive rows share 6 of their 7 days of return.
  - For a persistent feature, such as a regime that lasts weeks, the gain is small: mainly a better alignment at regime changes.
  - For fast-moving features it can approach the equivalent of about 1.5 times more data.
  - It also removes the dependence on an arbitrary weekday and gives 7 test paths instead of one.
- **What it needs.** `min_samples_leaf` multiplied by 7, a 7-day gap between training and test, sub-sampled
  thresholds, fits about 7 times slower, and in-sample tests that do not treat overlapping rows as independent.
- **My view.** It is worth doing, but expect a modest gain. It does not replace out-of-sample acceptance.

**9.2 Test the Sharpe first, then the regret.**
- If a split must pass **both** tests, the order cannot change which splits pass. It only changes the cost.
- The regret test is about 100 times cheaper: it only needs the leaf's own rows, while the Sharpe test needs
  fitted children and a replay of the whole path. Testing the Sharpe first on all ~715 candidates took about
  3 minutes per leaf in the audit, against 6 seconds for the whole current fit.
- What does change the result is **which candidate wins when several pass both tests**: the highest Sharpe
  (today) or the largest regret reduction (SPOT's rule). This is question Q1.
- I would choose by regret and use the Sharpe as a confirmation. Picking the maximum Sharpe among candidates
  is exactly what inflates the training Sharpe (F1).

**9.3 Increase the regret threshold "largely".**
- **Agreed that 1e-6 is meaningless.** Two changes would make the gate real:
  - Express it **relative to the leaf's regret**. The regret per week changes with λ and with volatility: it was
    0.034 at λ = 1 and 0.009 at λ = 50.
  - Compare the children with a parent **refitted on the same holdings**.
- **Calibration run.** 16 datasets per case, 5 years each. The table gives the gain of the best root split, as a
  share of the regret of a parent refitted on the same holdings:

| Data | λ = 1 | λ = 10 |
|---|---|---|
| **No signal** (the best of ~715 splits wins by luck) | median **10.2%**, 90th pct 14.0%, max 17.2% | median **3.4%**, 90th pct 5.3%, max 7.1% |
| Single-feature signal, best informative split | 8.5% | 3.0% |
| Interaction signal, best informative split | 9.0% | 3.4% |
| Perfect regime feature | 14.3% | 5.7% |

| Threshold at λ = 1 (the Sharpe must also rise) | No-signal datasets that still split | Informative split accepted: single-feature / interaction / perfect feature |
|---|---|---|
| 10.2% (median of luck) | 8/16 | 5/16 · 7/16 · 10/16 |
| 14.0% (90th pct of luck) | 2/16 | 2/16 · 2/16 · 8/16 |
| 17.2% (max of luck) | 0/16 | 0/16 · 0/16 · 5/16 |

- **Reading.**
  - The holdings mismatch inflates gains by only 0.3 points (10.5% measured as today, against 10.2% with the
    refitted parent). Fixing it is right in principle but small in effect. The lucky gain comes from picking
    the best of ~715 splits.
  - Realistic signals produce gains no larger than luck. This is the same limit as the Sharpe margin (F1).
  - A high threshold gives precision without recall. At 14% the tree rarely splits on noise, and 12 of the 13
    splits it accepts on data with a signal are informative, but about 85% of realistic signals are missed.
    For trading, that is the safer side.
  - The threshold depends on λ, on the number of weeks and on the number of candidate splits. A fixed value
    does not carry over.
- **Recommended approach.** Calibrate the threshold on the data at hand. Shuffle the feature rows several times,
  record the best lucky gain each time, and use the 90th percentile as the threshold.

---

## 10. Other improvement ideas, in priority order for BTC/ETH

1. **Accept splits on weeks the tree did not learn from (fixes F1; the example in 3.5 shows why).** Grow the tree
   on the first ~70% of the training window. Keep a split only if the regret and the replayed Sharpe also improve
   on the remaining ~30%, with holdings carried over. This is SPOT's validation pruning in path-aware form.
2. **Walk-forward evaluation.** Retrain monthly or quarterly, as both W&H articles do, on 10 years of BTC and ETH,
   and stitch the out-of-sample weeks together. Report against buy-and-hold BTC, buy-and-hold ETH, 50/50, a
   momentum rule, and a **placebo with shuffled features**.
3. **Add a cash or stablecoin column (F8).** It is mandatory for one coin, it lets the strategy cut its exposure,
   and it needs no code change.
4. **Tune λ, `prediction_bound`, depth and leaf size on validation data (F4).** The articles do this with Optuna.
5. **Control turnover.** Add partial adjustment (DIR eq. 35) to the replay and to live trading, and optionally a
   buffer zone around thresholds. This is the existing TODO.
6. **Shrink the search.** Use about 10 quantile thresholds per feature, depth ≤ 2, larger leaves, and at most 5
   well-motivated features. Every candidate removed lowers the lucky gain (section 9.3). Align the `TreeConfig`
   defaults with `run.py` (F6).
7. **Smaller items.** Bootstrap intervals for the Sharpe and Sortino in the report; an SPO forest with a block
   bootstrap (SPOT §4.3); trading at the next open with spread and slippage; a look-ahead test for your features.

---

## 11. Questions for you

- **Answered and applied:** the parent refit, and the largest regret reduction chooses the split.
- **Q1. Regret threshold.** Should it become a share of the parent's regret, calibrated by shuffling the features
  (section 9.3)? It is still 1e-6.
- **Q2. Daily rows.** Is 9.1 what you meant: one row per day with the next-7-day return, 7 interleaved weekly paths in training?
- **Q3. Assets.** Do you want BTC and ETH fully invested (pure rotation), or with a cash or stablecoin column so
  that exposure can change? For a single coin, cash is required.
- **Q4. Role of the covariance penalty.** Should it really shape the weights (then λ ≈ 10–30 for BTC and ETH), or
  is near all-in rotation acceptable (λ ≈ 1)?
- **Q5. Walk-forward cadence.** How often should it retrain (monthly or quarterly), and with a training window of how many years?
- **Q6. Priority.** Still open from before: a tradable strategy, or a paper on path-aware SPO trees?
