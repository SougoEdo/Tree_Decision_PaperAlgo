# History of the model and of the experiments

Dates are commit dates (`git log`) or report dates. Pipeline versions are labelled P0–P3b and synthetic
markets M1–M2', and the experiment sections of the two reports state which ones they used.

## Timeline

| date | commit | change |
|---|---|---|
| 2026-09-16 | eed348b, 760d093, 0d550f6 | project created; first tree with an SLSQP portfolio optimizer ("a first try") |
| 2026-09-17 12:12 | 54abe2b | defaults aligned with the fully-invested, long-only design |
| 2026-09-17 12:25 | 2135373 | exact portfolio solver (enumeration of the 5^n KKT patterns) replaces SLSQP |
| 2026-09-17 12:36 | 580cab1 | `replay_path`: sequential backtest with drifting holdings and fees on every trade |
| 2026-09-17 13:05 | 8408c07 | the tree grows on its own replayed path; splits ranked by regret reduction and confirmed by the replayed Sharpe ratio |
| 2026-09-17 13:25 | 9bc4394 | weekly decision rows from daily closes (`weekly_rows`); the "noise switch" TODO |
| 2026-09-17 14:42 | 4fd0a75 | strategies reported against baselines on training and test periods |
| 2026-09-17 15:21 | e0c02fe | code audit (`AUDIT_REPORT.md`, findings F1–F8; deleted on 2026-09-23, in git history) |
| 2026-09-17 / 18 | 36b2ffc, aabae8d | README; `pipeline_report.tex` dated 18 September (deleted on 2026-09-23, in git history) |
| 2026-09-18 16:25 | c20af0e | pruning on the checking rows (last quarter of the training rows); leaf scores refitted on all rows |
| 2026-09-18 17:58 | 218c473 | first planted-threshold experiment and its note (`recovery_note.tex` of 18 September) |
| 2026-09-22 | eae145c, 9e321db | daily-data experiment: one feature, threshold at the mean of x, market-sized parameters; Sharpe ratio of an all-cash path set to 0 (was NaN); regime-dependent volatility made an option of the synthetic market |
| 2026-09-23 09:50 | de8d3f3 | note rewritten for the daily experiment; risk aversion / fee / depth / quartiles experiment |
| 2026-09-23 11:06 | 5c2425b | tree options: quantile candidates (of the node's values), minimum leaf fraction, local refinement, soft splits with Boltzmann weights |
| 2026-09-23 12:58 | 69de742 | quantile candidates by rank among the admissible thresholds; pruning hurdle in standard errors; `training_pipeline.tex` |
| 2026-09-23 14:10 | 9d5bfdc | reorganisation: `code/` and `report/`, explicit file names, one model document and one experiments report |
| 2026-09-23 (uncommitted) | | observation noise on the feature, volatility threshold, benchmark with the true regime variance; resolution and regime-volatility experiments (E6, E7); this history; P4 options and their experiment (E8) |
| 2026-09-24 (uncommitted) | | model v1 frozen (a configuration of P4, see below); its validation on 100 datasets (E9); `run-v1` and the paired summary in `planted_threshold_experiments.py`; the beamer deck `presentation_model_v1.tex`; Configuration v1 section in the model document |

## Pipeline versions (the tree and its training)

The current description of the training is `model_and_training_pipeline.tex`; each version below says what differed.

| version | date | what it is |
|---|---|---|
| P0 | 2026-09-16 | a decision tree with an SLSQP mean-variance optimizer; not documented, replaced the next day |
| P1 | 2026-09-17 to 18 | exact solver; regret with fees and risk from the replayed holdings; best-first growth (root fitted from equal weights, parent refitted on the current holdings, every threshold screened, shortlist of 3, largest regret reduction accepted if the replayed Sharpe ratio rises, ε_R = 1e-6, ε_S = 0); final refit kept if the Sharpe does not fall; leaf scores in ±10% by a 3×7 coordinate search; from 18 September 16:25, pruning on the last quarter of the training rows and refit on all rows. Known defect: the Sharpe ratio of an all-cash path was NaN, so pruning rejected every split whose alternative sat in cash |
| P2 | 2026-09-22 | P1 with the Sharpe ratio of an all-cash, zero-return path defined as 0 |
| P3 | 2026-09-23 11:06 | P2 plus four options, off by default: `threshold_quantiles` (quantiles of the node's values), `min_leaf_fraction`, `refine_thresholds` / `refine_standard_errors`, `smoothing` (Boltzmann weights over candidate thresholds, membership-weighted leaves) |
| P3b | 2026-09-23 12:58 | `threshold_quantiles` redefined as quantiles by rank among the admissible thresholds; `prune_standard_errors` (pruning hurdle in standard errors of the checking-row gain) added in the afternoon. With every option off, P3 and P3b are identical to P2 |
| P4 | 2026-09-23 (evening) | P3b plus `smoothing_kernel` ("laplace" e^{-u} or "gaussian" e^{-u²/2} weights over the candidate thresholds) and `prune_hurdle_from_depth` (the pruning hurdle applies only to splits at or below that depth; 1 spares the root). Off by default: identical to P3b |
| **v1** | 2026-09-24 | not a new pipeline but the first frozen *configuration* of P4: λ = 1, fee 3 bp on the asset, weekly decisions (every 2 days as the alternative), depth at most 3, 20 decisions and 10% of the node per leaf, deciles by rank + local refinement (1 s.e.), soft splits off, growth gates unchanged (regret reduction, then the replayed Sharpe must rise), pruning on the last 25% with a hurdle of 2 s.e. below the root, 180-day covariance, cash column, all-cash Sharpe = 0. Setting `V1` of `planted_threshold_experiments.py`; Section "Configuration v1" of `model_and_training_pipeline.tex`; `presentation_model_v1.tex` |

## Synthetic markets

| market | date | what it is |
|---|---|---|
| M1 | 2026-09-17 | `synthetic_data.py`: Ornstein–Uhlenbeck feature and a price whose log return is ±μ − σ²/2 + σε by the side of a threshold d; time unit "one step" |
| M2 | 2026-09-22 | the same equations on a daily clock: x standardized (mean 0, variance 1), threshold at the mean, σ = 1% a day, drift as an edge e = μ/σ, 15 years of training and 5 of test, covariance from the last 180 days |
| M2' | 2026-09-22 / 23 | M2 with options: `up_volatility_ratio` (volatility above a threshold relative to below; the user's idea of 22 September), `volatility_threshold` (23 September), `feature_noise` (the tree observes x plus white noise; 23 September). Off by default, M2' is M2 |

## Experiments

| id | date | market | pipeline | parameters | size | documented in | headline |
|---|---|---|---|---|---|---|---|
| E0 | 2026-09-17 | demo world of `main.py` (3 assets, 5 features, weekly, 60-day regimes) | P1 before pruning | λ = 1, fee 0.1%, depth 2, 20 rows per leaf | 40 fits | `AUDIT_REPORT.md`, `pipeline_report.tex` (git history) | in-sample gates say yes to almost any split (F1); λ = 1 switches the covariance off; pruning on checking rows recommended and then added |
| E1 | 2026-09-18 | M1: μ = 3%/step, σ = 3.2%/step, d = 0, m = 1, s² = 4, k = 0.2; features x, 20-step average of x, an unrelated OU, white noise | P1 with pruning | h = 10 steps (and 1); 443 training decisions; depth 2, 20 per leaf, all thresholds, pruning 25% | 800 fits (k × volatility map, two histograms) | `experiments_planted_threshold.tex`, Section 5 | x found first in 100/100 (h = 1) and 86/100 (h = 10), never a decoy first; the tree finds where the decision changes, below d when d < m; drift far too large to be realistic |
| E2 | 2026-09-22 | M2, edge 0.02 / 0.05 / 0.10, k = 0.01–0.10, rates 1/2/5/10/21 days | P2 | depth 2, 20 per leaf, all thresholds, pruning 25%, λ = 1, fee 3 bp | 2000 fits | Sections 1–4 and 6–7 | threshold recovered at every rate at ±25%/yr, three times in four at ±13%, not at ±5%; monthly loses precision, more so with a fast feature |
| E3 | 2026-09-22 | M2, edges 0.05 / 0.10, rates 2/5/10/21 | P2 (+ the values' quartiles as candidates) | λ ∈ {1, 5, 10}, fee ∈ {3, 10} bp, depth ∈ {2, 3}, quartile candidates | 1920 fits | Section 8 | depth alone changes nothing; λ = 5–10 makes positions gradual and the tree learns them at a cost of ~0.1 Sharpe; fees keep positions at 0/1 |
| E4 | 2026-09-23 (run twice: values' deciles, then rank deciles) | M2, edges 0.05 / 0.10, rates 2/5/10/21 | P3, then P3b (reported) | deciles + 10% leaf fraction, refinement (1 s.e.), smoothing κ = 1, all three | 1600 fits | Section 9 | deciles + refinement ≈ base with steadier thresholds at slow rates; smoothing κ = 1 too wide and inside the no-trade band at λ = 1 |
| E5 | 2026-09-23 | M2, edges 0.05 / 0.10, rates 2/5/10/21 | P3b | pruning hurdle 1 or 2 s.e., alone and with deciles + refinement | 1600 fits | Section 10 | removes the spurious second-level leaves but loses true root splits faster; hurdle kept off |
| E6 | 2026-09-23 | M2', drift ±25%/yr | P3b, all options off (= P2) | volatility 0.4–4%/day; observation noise 0–2 std of x; rates 5 and 21 | 1920 fits | Section 11 | threshold error ≈ 0.012/edge until the edge falls below ≈ 0.05, where the split is guessed; observation noise of τ std acts like dividing the edge by 1 + τ²; noise up to 0.25 std costs nothing, 0.5 doubles the error |
| E7 | 2026-09-23 | M2', volatility 1% below d_σ and 0.5% above, d_σ ∈ {d, d + 1} | P3b, all options off (= P2) | λ ∈ {1, 10}; covariance window 180 or 20 days; edges 0.05 / 0.10; rates 2/5/10/21 | 2240 fits | `experiments_regime_volatility.tex` | with d_σ = d the drift threshold is found better (calm good regime) and every Sharpe nearly doubles; the volatility boundary at d + 1 is not learnt because the regret uses the same lagged covariance as the optimizer; a 20-day window tracks the regime but its noise costs more; next: a regime-aware variance estimate |
| E8 | 2026-09-23 | M2, edges 0.05 / 0.10, rates 2/5/10/21 | P4 | smoothing κ = 0.5 and 0.25 (Laplace), κ = 1 (Gaussian); hurdle 2 s.e. below the root; each alone and on top of deciles + refinement | 2240 fits | `experiments_planted_threshold.tex`, Section 12 | κ = 0.25 cures the no-trade problem (spread 0.2, base turnover) and matches the base from weekly on, still −0.26 Sharpe at 2 days; the hurdle below the root removes the second-level leaves (25/10/5/5% → 5/0/0/0%) at no Sharpe cost from weekly on, −0.13 at 2 days; no option beats the base beyond noise |
| E9 | 2026-09-24 | M2, edges 0.05 / 0.10, rates 2/5/10/21 | P4, configuration v1 | model v1 as a whole (depth ≤ 3, deciles by rank + refinement 1 s.e., 10% leaf fraction, hurdle 2 s.e. below the root) on 100 datasets, seeds 0–99 = the base of E2, so the comparison is paired | 800 fits (plus the 800 base fits of E2) | `experiments_planted_threshold.tex`, Section 13; `presentation_model_v1.tex` | v1 equals the base within noise from weekly on (paired Sharpe differences 0.00 ± 0.04, +0.01 ± 0.05, −0.01 ± 0.03 at edge 0.10; −0.03 ± 0.05, +0.03 ± 0.04, −0.03 ± 0.03 at edge 0.05) with the second-level leaves gone (18/14/9% → 0/2/0% at edge 0.10) and 1.9 leaves instead of 2.1; at every 2 days it costs 0.08 ± 0.07 at edge 0.10; the depth budget of 3 is never used in this one-threshold market |

## Conventions kept throughout

- Two clocks: data daily, decisions every h days; the label is the h-day return; Σ is h times the daily variance of the past window.
- Test decisions never enter training; every strategy starts the test from its own final training holdings.
- Comparability: each experiment reports the annualized test Sharpe ratio (per-decision Sharpe × √(252/h)) and the share of the ideal rule's gain captured.
