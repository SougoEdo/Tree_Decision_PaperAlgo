# Why an Improved Decision Tree Can Be a Serious Portfolio Model
## A Fair Comparison with an Over-Parameterized Deep Neural Network

## Purpose

This document motivates the use of a decision-tree-based architecture for portfolio management while being explicit about what a deep neural network can do better.

The argument is **not** that trees are universally superior to neural networks. They are not. A sufficiently expressive neural network can approximate far richer nonlinear mappings, learn complex feature interactions, exploit temporal structure, and potentially discover representations that a shallow tree cannot express.

The argument is instead that portfolio management is not purely a function-approximation problem. It is a **decision problem under noise, non-stationarity, transaction costs, risk constraints, and limited effective sample size**. In that environment, a deliberately constrained, decision-aware, and path-aware tree can be powerful because its inductive bias may be closer to the structure of the economic problem.

The practical research question is therefore not:

> Can a deep neural network fit more complicated relationships than a tree?

It clearly can.

The relevant question is:

> Does the additional representational capacity of a deep neural network translate into more stable, implementable, and out-of-sample portfolio decisions than a carefully designed decision-aware tree?

That is an empirical question, and the proposed tree architecture is interesting precisely because it offers a strong and testable alternative.

---

# 1. The Portfolio Problem Is Not the Same as a Return-Prediction Problem

A conventional machine-learning approach would learn a mapping

$$
x_t \longrightarrow \hat r_t,
$$

where $x_t$ contains the information available at decision time and $\hat r_t$ is a vector of predicted future returns.

The predictions are then passed to a portfolio optimizer. In the current tree implementation, the downstream decision problem is approximately

$$
w_t^*
=
\arg\max_w
\left[
\hat r_t^\top w
-
f^\top |w-u_t|
-
\lambda w^\top \Sigma_t w
\right]
$$

subject to long-only, fully-invested, and maximum-weight constraints.

Here:

- $u_t$ is the **actual pre-trade portfolio** carried into time $t$,
- $f$ represents proportional transaction costs,
- $\Sigma_t$ is the covariance estimate available before the decision,
- $\lambda$ controls risk aversion.

This distinction matters. The end goal is not necessarily to minimize

$$
\|\hat r_t-r_t\|^2.
$$

The real goal is to make good decisions after the prediction interacts with fees, risk, constraints, and previous holdings.

A model can therefore be numerically inaccurate as a return forecaster while still generating good portfolio decisions. Conversely, a model with lower forecasting error can generate inferior allocations.

This is the central motivation of Smart Predict-then-Optimize and decision-focused learning:

> **Prediction quality and decision quality are different objectives.**

The portfolio literature motivating Smart Predict-then-Optimize explicitly emphasizes that improvements in forecasting accuracy need not translate into improvements in the quality of the resulting portfolio.

---

# 2. Why an Extremely Powerful DNN Is an Obvious Competitor

A deep neural network has several genuine advantages.

With enough layers, nonlinearities, and parameters, a DNN can represent complicated mappings such as

$$
x_t
\mapsto
\hat r_t
\mapsto
w_t
$$

with interactions that are difficult or impossible for a shallow axis-aligned tree to reproduce compactly.

A sufficiently rich architecture can potentially learn:

- nonlinear interactions between momentum, volatility, and liquidity;
- cross-asset relationships;
- conditional effects that vary continuously rather than by hard regime;
- temporal dependencies;
- lead-lag structures;
- latent representations from raw market data;
- high-order interactions among many features.

This becomes particularly attractive when the input itself is high dimensional. For example,

$$
X_t
\in
\mathbb R^{
N_{\mathrm{assets}}
\times
T_{\mathrm{history}}
\times
P_{\mathrm{features}}
}.
$$

Architectures using convolution, recurrence, attention, or Transformers can extract structure directly from such tensors.

The DSPO framework is a good example of this approach. It combines temporal processing, multi-frequency fusion, and an inter-stock Transformer in order to learn cross-sectional ranking signals directly from high-dimensional market data.

Therefore, if the research problem were simply

> Which function class can represent the richest relationship between market observations and future outcomes?

the DNN would have the stronger claim.

That is not, however, the whole portfolio-management problem.

---

# 3. Capacity Is Useful Only When the Data Can Identify the Additional Structure

A model with more parameters has a larger hypothesis space.

That gives it lower approximation bias, but it also gives it many more ways to fit accidental relationships.

This matters particularly in finance because the raw sample count can be misleading.

Suppose a strategy contains 100,000 sequential observations. These observations are rarely equivalent to 100,000 independent samples.

Adjacent observations may share:

- overlapping return windows;
- moving-average windows;
- volatility estimates;
- market regimes;
- liquidity states;
- macroeconomic conditions;
- the same underlying price path.

The effective statistical sample size can therefore be much smaller than the row count suggests.

At the same time, the conditional relationship itself may move:

$$
P(r_{t+1}\mid x_t)
$$

is not guaranteed to remain stable across years or market regimes.

Consequently, the ability of a large neural network to fit extremely detailed structure is simultaneously its strength and one of its main risks.

A relationship can be:

1. statistically real in the training sample;
2. captured accurately by a DNN;
3. economically irrelevant or temporary;
4. absent in the future.

A restricted model can sometimes generalize better precisely because it is **incapable of learning many weak, unstable patterns**.

This is not an argument that simple models always win. It is an argument that in low-signal, non-stationary environments, additional capacity must justify itself out of sample.

The financial decision-focused literature itself highlights weak periodicity, low signal-to-noise ratios, and frequent regime shifts as reasons why highly complex architectures do not automatically translate their representational capacity into better portfolio decisions.

---

# 4. What the Decision Tree Assumes

A shallow decision tree imposes a strong structural hypothesis:

> The feature space can be partitioned into a relatively small number of regions within which approximately the same decision-relevant signal is appropriate.

For example,

$$
x_1 < s_1
$$

might define a low-volatility regime, followed by

$$
x_2 > s_2
$$

to separate positive and negative momentum conditions.

A leaf may then contain a score vector such as

$$
\hat r_L
=
(0.018,-0.006,0.002).
$$

That vector is not necessarily best interpreted as a calibrated forecast. Its role is to induce a suitable portfolio through the optimizer.

The resulting model therefore resembles

$$
\text{market state}
\rightarrow
\text{decision regime}
\rightarrow
\text{portfolio}.
$$

This is a strong inductive bias.

It will be wrong if the true decision surface is highly irregular and continuously changing.

But it can be useful if the economically relevant structure is closer to a small number of persistent regimes.

---

# 5. Why Regime Learning Is a Natural Interpretation of a Portfolio Tree

Many trading hypotheses are naturally conditional.

Examples include statements such as:

- momentum works differently in high and low volatility;
- a directional signal is useful only when liquidity is sufficient;
- one asset becomes preferable to another after a relative-strength threshold;
- a risk signal matters only after a volatility threshold has been crossed.

A tree represents this logic directly.

A path such as

$$
\mathrm{volatility} < 0.035,
$$

followed by

$$
\mathrm{momentum}_{28d} > 0.08,
$$

and perhaps

$$
\mathrm{relative\ strength}_{BTC/ETH} > s
$$

defines an explicit market region.

The associated leaf then answers:

> Given that the market is in this region, what decision-inducing score vector leads to a good portfolio?

A neural network can absolutely learn an equivalent relationship.

The distinction is that the tree is **forced to expose the relationship as a small set of regimes** rather than distributing it over thousands or millions of parameters.

That constraint may be useful scientifically and operationally even when it costs some predictive flexibility.

This interpretation is closely aligned with the original SPO Tree motivation, in which trees provide an interpretable segmentation of contextual features into groups associated with different decisions.

---

# 6. The Strongest Motivation: Learn Decision Boundaries, Not the Entire Return Surface

Consider two assets with conditional future returns

$$
r_1(x),
\qquad
r_2(x).
$$

A complex model may attempt to estimate both functions accurately everywhere.

But suppose the optimizer's relevant decision is primarily determined by whether

$$
r_1(x) > r_2(x).
$$

Then the economically important object may be the boundary

$$
x^*:
\qquad
r_1(x^*) = r_2(x^*),
$$

rather than the exact shape of both functions.

This is one of the important insights behind SPO Trees.

A decision-aware tree can place a split close to the point at which the optimal action changes even when its piecewise-constant predictions are crude approximations of the true underlying response functions.

A prediction-focused tree may instead spend splits reducing numerical forecast error in regions where the decision does not change.

This gives a different notion of model efficiency:

$$
\text{useful complexity}
=
\text{complexity devoted to changing the decision}.
$$

In a portfolio context, that can be more relevant than fitting the full conditional expectation surface.

The illustrative example in the original SPO Tree paper shows precisely this phenomenon: an SPO Tree can recover the relevant decision boundary with a very small tree even when a prediction-focused CART requires substantially more depth to obtain comparable decision quality.

---

# 7. The Current Implementation Goes Beyond a Standard SPO Tree

The tree in the `path-aware-spo-tree` branch of the repository is not a conventional CART model and is not a literal reproduction of the original SPOT algorithm.

It combines several ideas.

## 7.1 The Leaf Prediction Is Evaluated Through a Portfolio Optimizer

Each leaf stores a bounded **score vector**.

These scores are passed to a deterministic optimizer that solves

$$
\max_w
\left[
c^\top w
-
f^\top |w-u|
-
\lambda w^\top \Sigma w
\right].
$$

The tree therefore does not directly predict portfolio weights.

It predicts decision-inducing scores and lets an explicit optimizer enforce the economic structure.

This separation has practical advantages.

Risk aversion, fee assumptions, weight constraints, and covariance information remain transparent and can be changed without asking the learning model to rediscover their meaning implicitly.

---

## 7.2 Leaf Scores Are Optimized for Decision Regret

For each observation, the implementation computes the best hindsight utility obtainable from the same starting holdings.

The regret of the model's score vector is then the difference between that oracle utility and the utility induced by the predicted scores.

Schematically,

$$
\mathcal R_t(\hat r)
=
U_t(w_t^*;r_t)
-
U_t(\hat w_t;r_t),
$$

where

$$
w_t^*
=
\arg\max_w U_t(w;r_t)
$$

is the hindsight oracle decision and

$$
\hat w_t
=
\arg\max_w U_t(w;\hat r)
$$

is the decision induced by the leaf score.

Consequently, leaf fitting is not ordinary least squares.

The leaf score is selected to reduce downstream decision regret.

The implementation currently uses a bounded coordinate search initialized from the leaf's mean return. It is deterministic and practical for the small asset universe considered, but it is approximate and should not be described as a global solver.

That limitation matters.

---

## 7.3 The Tree Is Path-Aware

This is one of the most important departures from an ordinary supervised tree.

At time $t$, transaction costs depend on

$$
|w_t-u_t|,
$$

where $u_t$ is the portfolio already held before rebalancing.

But $u_t$ itself depends on every previous decision and on the subsequent price evolution.

Therefore, the quality of a split cannot be evaluated completely independently row by row.

The repository addresses this by repeatedly **replaying the current tree through time**.

The replay:

1. begins from initial holdings;
2. applies the tree's scores;
3. solves the portfolio problem;
4. charges transaction costs;
5. realizes returns;
6. lets the portfolio drift;
7. carries the resulting holdings into the next period.

Thus,

$$
u_{t+1}
=
\frac{
w_t \odot (1+r_t)
}{
1+w_t^\top r_t
}.
$$

The next portfolio decision is consequently evaluated from the holdings that the strategy itself would actually have produced.

This makes the training criterion substantially closer to a trading process than a static row-wise prediction problem.

---

# 8. Split Selection Uses Two Different Criteria

The current algorithm does not accept a split simply because it improves an impurity measure.

A candidate must satisfy two conceptually different tests.

## 8.1 Local Decision Criterion: Regret Reduction

For a candidate split,

$$
R
\rightarrow
(R_L,R_R),
$$

the parent leaf is first refitted using the current replayed holdings.

The two children are then fitted.

The split gain is approximately

$$
G_{\mathrm{regret}}
=
L_{\mathrm{parent}}
-
L_{\mathrm{left}}
-
L_{\mathrm{right}}.
$$

The split must reduce regret by more than the configured minimum improvement.

This asks:

> Does dividing this state space actually lead to better downstream portfolio decisions?

---

## 8.2 Global Sequential Criterion: Path Sharpe

A locally attractive split can still be harmful to the strategy.

For example, suppose it creates two leaves whose optimal portfolios are very different.

If the feature oscillates around the threshold, the strategy can repeatedly move

$$
w_A
\rightarrow
w_B
\rightarrow
w_A
\rightarrow
w_B,
$$

paying fees each time.

A row-wise regret calculation can underestimate this effect because the trading path is sequential.

The implementation therefore replays the entire candidate tree and requires

$$
\operatorname{Sharpe}_{\mathrm{new}}
>
\operatorname{Sharpe}_{\mathrm{old}}
+
\Delta_{\min}.
$$

This is unusual and important.

The local SPO-style criterion asks whether the split improves decisions.

The path-level Sharpe gate asks whether those improvements survive after the induced sequence of trades is reconstructed.

The architecture therefore tries to combine

$$
\boxed{\text{local decision quality}}
$$

with

$$
\boxed{\text{global trading-path quality}}.
$$

---

# 9. Why This Can Be More Appropriate Than Simply Making the Predictor Bigger

A large DNN can learn a much finer mapping than this tree.

But a finer mapping is not automatically a better trading strategy.

Suppose the neural network produces small but frequent changes:

$$
\hat r_t
\rightarrow
\hat r_{t+1}
\rightarrow
\hat r_{t+2}.
$$

Even if those changes improve predictive loss, they can repeatedly move the optimizer across decision boundaries.

The resulting weights may exhibit high turnover.

This is particularly important because the optimizer can amplify seemingly small changes in scores.

The actual mapping is

$$
x
\rightarrow
\hat r
\rightarrow
\arg\max_w U(w;\hat r),
$$

and the $\arg\max$ can be considerably more sensitive than the predictor itself.

Thus,

$$
\|\hat r_t-\hat r_{t-1}\|
$$

being small does **not** imply that

$$
\|w_t-w_{t-1}\|
$$

is small.

A larger neural predictor does not remove this problem. It can in some circumstances make the input to the optimizer even more variable.

The recent work on decision-induced ranking and prediction inflation provides an important warning: SPO-based predictors can learn values that behave more like **decision-inducing scores** than calibrated expected returns.

The optimizer may exploit relative orderings and score gaps rather than realistic forecast magnitudes.

This can produce:

- inflated scores;
- concentrated allocations;
- unstable reallocations;
- excessive turnover.

The implication is important:

> The critical stability problem can originate from the interaction between the predictor and optimizer, not simply from the choice of neural network versus tree.

---

# 10. Interpretability Is Not Merely Cosmetic in This Problem

Interpretability is sometimes presented as a secondary advantage: a transparent model is pleasant to look at but a black box is acceptable if it performs better.

For this application, interpretability can have a more substantive role.

Suppose a model begins trading aggressively because

$$
\mathrm{volatility}_{28d}<2.4\%
$$

and

$$
\mathrm{momentum}_{7d}>5.1\%.
$$

With a tree, this behavior is visible.

One can inspect:

- the feature used;
- the threshold;
- the observations in each regime;
- the score vector in each leaf;
- the portfolio induced by that score;
- the regret improvement attributed to the split;
- the change in path Sharpe;
- whether the split survives validation pruning.

This allows economic criticism.

A researcher can ask:

- Does the threshold correspond to a plausible regime?
- Is the split based on enough observations?
- Is the result driven by one market episode?
- Does it survive a nearby threshold?
- Does it generate excessive turnover?
- Does it remain useful under higher fees?
- Does the split recur across windows?

A DNN can be analyzed with attribution techniques, but these explanations are generally posterior analyses of a distributed function.

A tree's logic is the model itself.

That distinction is valuable when the goal includes understanding **why** the strategy works rather than merely producing a backtest.

---

# 11. Simplicity Creates Useful Failure Modes

A tree can fail conspicuously.

For example,

$$
x < 0.5
\rightarrow
w_A
$$

while

$$
x \ge 0.5
\rightarrow
w_B.
$$

If $x$ oscillates near $0.5$, the resulting turnover is immediately understandable.

This is a weakness, but it is also diagnostically useful.

By contrast, a high-capacity model can generate unstable decisions through complex combinations of features that are much harder to diagnose.

In systematic finance, transparent failure modes can be valuable because they can be specifically targeted.

The current project already illustrates this process.

The hard-threshold tree exposed a concrete weakness:

> Observations close to a split can receive very different portfolio signals even when their feature values are almost identical.

That observation motivates the next extension:

> **split smoothing.**

---

# 12. Hard Splits Are Also One of the Tree's Largest Weaknesses

A standard tree defines

$$
m(x;s)
=
\begin{cases}
1, & x\le s,\\
0, & x>s.
\end{cases}
$$

Therefore,

$$
s-\epsilon
$$

and

$$
s+\epsilon
$$

can receive completely different leaf scores even for arbitrarily small $\epsilon$.

For portfolio management this can be economically undesirable because the discontinuity can become a trade.

If two neighboring leaves imply substantially different allocations, noise around a threshold can create

$$
\text{leaf A}
\rightarrow
\text{leaf B}
\rightarrow
\text{leaf A}.
$$

Transaction costs convert statistical instability directly into economic loss.

This is a genuine advantage of continuous models such as neural networks.

A DNN generally produces a smoother predictor unless its learned mapping or the downstream optimizer itself creates sharp decision boundaries.

An improved tree therefore needs to address this weakness rather than pretend it does not exist.

---

# 13. Soft Split Smoothing: an Intermediate Model Between a Tree and a Continuous Learner

The current branch already contains an experimental soft-split mechanism.

Instead of assuming that one estimated threshold $s^*$ is certainly correct, it can retain several plausible thresholds

$$
s_1,\ldots,s_K
$$

and associate probabilities

$$
\pi_1,\ldots,\pi_K,
\qquad
\sum_k\pi_k=1.
$$

Candidate thresholds are weighted according to how much worse their regret is than the best candidate, measured relative to an estimated standard error.

Schematically,

$$
\pi_k
\propto
\exp\left(
-\frac{z_k}{\tau}
\right),
$$

where

$$
z_k
=
\frac{
L(s_k)-L(s^*)
}{
\operatorname{SE}[L(s_k)-L(s^*)]
}
$$

and $\tau$ is the smoothing parameter.

For an observation with feature value $x$, its left-child membership becomes

$$
m_L(x)
=
\sum_{k:s_k\ge x}\pi_k.
$$

The right membership is

$$
m_R(x)=1-m_L(x).
$$

The final score is therefore a membership-weighted combination of leaf scores.

This creates an important interpretation:

> The model does not claim that the estimated split threshold is known exactly. It averages decisions over several statistically plausible nearby thresholds.

That is not simply cosmetic smoothing.

It is a form of **decision-boundary uncertainty modeling**.

---

# 14. Why Smoothing Is Especially Interesting for This Project

The motivation differs from ordinary random-forest averaging.

A random forest typically reduces variance by changing the training sample and/or feature subset:

$$
T_1(x),T_2(x),\ldots,T_B(x)
$$

and averaging the resulting predictions.

That reduces dependence on one particular tree.

The smoothing idea attacks a more local source of variance:

> uncertainty about the location of an individual split.

If several thresholds produce almost indistinguishable decision loss, choosing one threshold with certainty introduces unnecessary estimator variance.

Instead of

$$
s=s^*
$$

the model effectively works with

$$
S\sim\pi(s)
$$

and averages over the induced routing.

This can make the transition between regimes gradual while retaining the underlying tree structure.

Conceptually, the model moves from

$$
\text{hard regime classifier}
$$

toward

$$
\text{probabilistic regime membership}.
$$

That is a compelling middle ground between a conventional decision tree and a fully distributed neural representation.

---

# 15. Why Not Simply Use a Random Forest?

A forest is a natural extension and should be an important benchmark.

The original SPO Tree work itself proposes SPO Forests by bootstrapping observations and performing feature bagging.

A forest can reduce the variance of individual trees:

$$
\operatorname{Var}
\left(
\frac{1}{B}
\sum_{b=1}^{B}T_b(x)
\right)
<
\operatorname{Var}(T(x))
$$

when the individual trees are not perfectly correlated.

There are nevertheless reasons to study the smoothed single-tree formulation separately.

First, it isolates **threshold uncertainty** rather than mixing together many sources of randomness.

Second, it preserves much more of the interpretability of one tree.

Third, smoothing may reduce pointless switching close to an otherwise economically meaningful boundary.

Fourth, understanding the single-tree behavior is valuable before introducing ensemble complexity.

The two ideas are complementary rather than mutually exclusive.

A natural eventual architecture is therefore

$$
\boxed{\text{Smoothed SPO Forest}}
$$

where each tree handles local split uncertainty and the ensemble handles sample-level model variance.

---

# 16. Why the Explicit Optimizer Remains Useful Even If a DNN Is Introduced

There are two broad neural alternatives.

## 16.1 DNN Predicts Returns or Scores

The first possibility is

$$
x_t
\xrightarrow{\mathrm{DNN}}
\hat r_t
\xrightarrow{\mathrm{optimizer}}
w_t.
$$

This is the closest neural competitor to the current tree.

The optimizer remains explicit, preserving fees, covariance penalties, and constraints.

This is a particularly important benchmark because it tests whether the tree's inductive bias is useful while holding the downstream decision problem fixed.

---

## 16.2 DNN Predicts Portfolio Weights Directly

The second possibility is

$$
x_t
\xrightarrow{\mathrm{DNN}}
w_t.
$$

This is more flexible but changes the problem more substantially.

The network must implicitly learn, or be explicitly trained to account for:

- risk;
- transaction costs;
- concentration constraints;
- dependence on current holdings;
- portfolio feasibility.

These elements can be encoded into the architecture or loss, but doing so increases modeling complexity.

An explicit optimization layer has an important conceptual benefit:

$$
\text{learn what is uncertain}
+
\text{solve what is known}.
$$

The expected-return signal is uncertain and learned.

The portfolio budget constraint is known.

The fee schedule is known or assumed.

The covariance penalty is a modeling choice.

Maximum-weight constraints are known.

There is no intrinsic reason to ask a black-box network to rediscover these deterministic structures.

---

# 17. The Tree Is Particularly Compelling for a Small Asset Universe

The current implementation is intentionally designed for only a few assets.

The exact portfolio optimizer supports approximately two to four assets and enumerates the possible KKT regimes created by:

- zero weights;
- maximum weights;
- current holdings;
- weights above current holdings;
- weights below current holdings.

For a BTC/ETH or BTC/ETH/cash-like research setting, this is reasonable.

The downstream optimization remains transparent and computationally manageable.

In this regime, the main learning problem is not a 4,000-stock cross-sectional representation problem.

It is closer to:

> Which observable market conditions justify changing the relative exposure among a handful of assets?

That is precisely where a regime-based model becomes interesting.

By contrast, if the objective later becomes to process thousands of securities, hundreds of raw signals, and long intraday sequences, the balance may shift strongly toward neural representation learning.

---

# 18. The Path-Aware Component Introduces a Form of Economic Regularization

A normal tree asks whether a partition explains the target better.

The current tree additionally asks whether the partition produces a better trading path after fees.

This effectively penalizes complexity that expresses itself as unnecessary trading.

Suppose a split marginally improves decision regret but causes frequent switching.

The replayed path may have

$$
\operatorname{Sharpe}_{\mathrm{split}}
<
\operatorname{Sharpe}_{\mathrm{parent}}.
$$

The split is rejected.

Thus the model contains a form of **economically induced regularization**.

Instead of penalizing depth purely because depth is statistically dangerous, the algorithm can reject complexity because that complexity creates economically poor behavior.

This suggests a useful principle:

$$
\text{complexity is useful only when it survives implementation}.
$$

It does not eliminate overfitting. The Sharpe gate itself can overfit the training path.

But it aligns the notion of complexity control more closely with the intended application.

---

# 19. Validation Pruning Is Therefore Essential

A danger in the architecture is multiple testing.

During tree construction, many:

- features;
- thresholds;
- leaves;
- score vectors

are examined.

Even if every candidate were pure noise, some would appear to improve both regret and Sharpe by chance.

The current implementation acknowledges this.

When `validation_fraction > 0`, the latest section of the training data is kept out of tree growth.

The grown tree is then evaluated bottom-up on these checking observations.

A split is retained only when it improves both

$$
\text{validation regret}
$$

and

$$
\text{validation Sharpe}.
$$

This is particularly important because the criterion is already adaptive.

A high training Sharpe from a tree whose structure was itself selected partly according to training Sharpe is not strong evidence.

Chronological validation makes the claim more credible, although a final untouched test period is still required.

The original SPO Tree methodology also uses validation-based pruning to control overfitting. The present implementation adds a path-level Sharpe condition on top of the decision-quality criterion.

---

# 20. Why Chronological Evaluation Matters More Than Random Train/Test Splitting

Financial models are meant to operate forward through time.

A random split can allow observations from the same regime, or even overlapping feature windows, to appear on both sides of the train/test boundary.

The current architecture is designed around ordered rows and sequential replay.

A credible evaluation should therefore preserve chronology:

$$
\text{train}
\rightarrow
\text{validation}
\rightarrow
\text{test}.
$$

Better still, use rolling or expanding windows:

$$
\text{train}_1
\rightarrow
\text{test}_1,
$$

$$
\text{train}_2
\rightarrow
\text{test}_2,
$$

$$
\ldots
$$

The question is not whether the tree can reconstruct historical regimes after seeing them.

The question is whether a regime partition estimated from the past remains useful later.

---

# 21. Where the DNN Should Genuinely Be Expected to Dominate

A fair motivation document must identify situations in which the tree is likely the wrong tool.

A neural architecture becomes increasingly compelling when the following conditions hold.

## 21.1 The Input Is Raw and High Dimensional

Examples include:

- limit-order-book tensors;
- tick sequences;
- text;
- hundreds or thousands of correlated assets;
- long multivariate histories.

A tree on manually engineered summary features throws away substantial structure before learning begins.

## 21.2 Representation Learning Is Central

If useful information exists in complicated latent patterns rather than interpretable indicators, the DNN's ability to construct features can be decisive.

## 21.3 There Is Genuinely Abundant Effective Data

If millions of sufficiently diverse and relevant observations are available, strong regularization plus large-scale training can justify high-capacity models.

The important word is **effective**.

Millions of highly overlapping observations from one market regime do not provide the same statistical information as millions of independent observations from diverse conditions.

## 21.4 The True Relationship Is Smooth and Highly Multivariate

Axis-aligned trees can approximate smooth functions only through many partitions.

A neural network may represent the same mapping much more efficiently.

## 21.5 Cross-Asset Interactions Are the Main Source of Alpha

An attention model can explicitly learn relationships among many assets.

A small tree using asset-level summary features may be too restrictive.

The DSPO architecture is a strong example of a setting where neural structure has a clear motivation because it was specifically designed to process multi-frequency information and model interdependencies across a large stock universe.

---

# 22. Where the Improved Tree Has a Particularly Credible Case

The tree becomes especially interesting when:

- the number of assets is small;
- features are already economically meaningful;
- effective data are limited;
- signal-to-noise is low;
- regime changes matter;
- transaction costs are important;
- interpretability matters;
- portfolio constraints are naturally represented by an optimizer;
- the research question concerns **when the optimal decision changes**, rather than exact return forecasting.

This is close to the current BTC/ETH-oriented setting.

That does not imply the tree will win.

It means the architecture has a defensible inductive bias and therefore deserves to be tested rather than dismissed simply because it is less expressive.

---

# 23. A Useful Way to Think About the Model-Capacity Spectrum

The candidate models can be organized along a continuum.

$$
\boxed{\text{Linear model}}
$$

very strong bias, low variance

$$
\downarrow
$$

$$
\boxed{\text{Hard SPO Tree}}
$$

nonlinear regimes, sharp boundaries

$$
\downarrow
$$

$$
\boxed{\text{Smoothed SPO Tree}}
$$

nonlinear regimes with uncertain boundaries

$$
\downarrow
$$

$$
\boxed{\text{SPO Forest}}
$$

many nonlinear regime partitions, lower ensemble variance

$$
\downarrow
$$

$$
\boxed{\text{Smoothed SPO Forest}}
$$

ensemble variance reduction plus local boundary smoothing

$$
\downarrow
$$

$$
\boxed{\text{Small MLP + explicit optimizer}}
$$

continuous flexible mapping

$$
\downarrow
$$

$$
\boxed{\text{Large DNN / Transformer}}
$$

high-capacity representation learning.

This suggests that the project should not attempt to prove abstractly that trees are superior.

Instead, it should measure:

> **How much model capacity does the portfolio problem actually reward out of sample?**

---

# 24. The Correct DNN Benchmark Is Not an Intentionally Weak Neural Network

To make a credible empirical argument, the neural competitor should be well designed.

A straw-man DNN would undermine the entire comparison.

At minimum, the DNN benchmark should use:

- the same information set;
- the same training periods;
- the same validation periods;
- the same downstream optimizer where possible;
- equivalent transaction-cost assumptions;
- equivalent covariance estimates;
- equivalent portfolio constraints;
- appropriate regularization;
- early stopping;
- multiple random seeds;
- hyperparameter tuning restricted to validation data.

One useful progression would be:

1. linear predictor + optimizer;
2. hard SPO tree;
3. smoothed SPO tree;
4. SPO forest;
5. small MLP + optimizer;
6. regularized deeper MLP + optimizer;
7. sequence model only if temporal raw data justify it.

The most informative comparison is not merely performance.

It is performance **relative to complexity and instability**.

---

# 25. Metrics Should Go Well Beyond Sharpe

The final evaluation should include at least

$$
\text{annualized return},
$$

$$
\text{annualized volatility},
$$

$$
\text{Sharpe ratio},
$$

$$
\text{maximum drawdown},
$$

$$
\text{turnover},
$$

and

$$
\text{fee drag}.
$$

For this project, additional stability metrics are particularly important.

## 25.1 Seed Instability

For stochastic models, evaluate

$$
\operatorname{Var}_{\mathrm{seed}}
\left[
\operatorname{Sharpe}_{\mathrm{OOS}}
\right].
$$

A DNN that obtains a very high Sharpe in one run but very different portfolios across seeds is less persuasive than its best run suggests.

## 25.2 Retraining Instability

Measure quantities such as

$$
\|w_t^{(k)}-w_t^{(k+1)}\|
$$

for models trained on neighboring rolling windows.

## 25.3 Decision-Boundary Stability

For the tree, record whether the same features and nearby thresholds recur across training windows.

## 25.4 Regime Persistence

Measure how long observations remain in the same leaf or dominant soft regime.

## 25.5 Turnover Attributable to Regime Switching

Separate total turnover into:

- drift/rebalancing turnover;
- economically meaningful regime transitions;
- repeated switching around effectively the same boundary.

This last diagnostic directly tests the motivation for smoothing.

---

# 26. An Important Limitation: Sharpe-Aware Split Selection Can Itself Overfit

The path-level Sharpe gate is economically intuitive, but it is not statistically free.

If hundreds of splits are evaluated and only those increasing in-sample Sharpe are accepted, the algorithm is implicitly optimizing Sharpe over many candidates.

A lucky split can therefore pass the gate.

This means the path-aware criterion should not be interpreted as proof that a split has discovered a genuine economic regime.

It is a model-selection mechanism.

The chronological pruning set and untouched out-of-sample tests are essential.

Further robustness checks could include:

- block bootstrap;
- walk-forward evaluation;
- sensitivity to fee assumptions;
- sensitivity to covariance estimation;
- threshold perturbations;
- feature perturbations;
- multiple training-window lengths.

---

# 27. Another Limitation: the Current Leaf Optimization Is Approximate

The current implementation fits leaf scores using bounded coordinate search.

This is attractive because it is simple and deterministic.

But it also means that differences between candidate trees can partly reflect the quality of the score search rather than only the underlying partition.

A deeper or more flexible model should not be compared unfairly against a tree whose leaf optimization is accidentally weak or overly coarse.

This suggests testing convergence with respect to:

- prediction bounds;
- coordinate-search passes;
- grid resolution.

If performance changes materially with these numerical settings, that instability must be reported.

---

# 28. Another Limitation: Soft Splitting Does Not Solve Every Source of Instability

Smoothing the threshold can reduce discontinuity around the split.

It does not automatically solve:

- unstable feature selection;
- unstable tree depth;
- regime drift;
- incorrect covariance estimates;
- mis-specified transaction costs;
- score inflation;
- optimizer sensitivity;
- overfitting to historical Sharpe.

Nor does it guarantee lower turnover.

If two leaves imply radically different decisions and the membership changes materially through time, turnover can remain high.

Therefore smoothing should be viewed as one targeted intervention:

$$
\boxed{
\text{reduce uncertainty generated by arbitrary threshold selection}
}
$$

rather than a universal stability solution.

---

# 29. The Central Scientific Hypothesis

A concise version of the research hypothesis is:

> In a low-dimensional, low-signal, and non-stationary portfolio problem, a model that learns a small number of decision-relevant regimes and explicitly accounts for transaction costs, risk, and path dependence may generalize more reliably than an over-parameterized model that attempts to learn the richest possible feature-to-return mapping.

There are several distinct sub-hypotheses inside this statement.

## H1 — Decision-Focused Splits

Splits selected by downstream portfolio regret are more useful than splits selected by return-prediction error.

## H2 — Path Awareness

Evaluating candidate structures through sequential replay removes splits whose row-wise improvements are erased by turnover and fees.

## H3 — Soft Boundaries

Averaging over statistically similar thresholds reduces unnecessary decision discontinuity.

## H4 — Controlled Complexity

A shallow, interpretable regime model exhibits lower out-of-sample variance than a high-capacity DNN when the effective dataset is limited.

## H5 — Capacity Eventually Wins When Information Justifies It

As the information set becomes richer and the effective sample becomes larger, neural models should become increasingly competitive and may dominate.

These hypotheses are falsifiable.

That is important.

The project is stronger if it is designed so that a neural network is allowed to win.

---

# 30. What Would Constitute Evidence in Favor of the Tree?

The strongest case would not be:

> The tree had the highest backtest Sharpe once.

A much stronger result would look like:

- similar or better median out-of-sample Sharpe;
- materially lower variance across training windows and random perturbations;
- lower turnover;
- lower fee drag;
- shallower drawdowns;
- stable feature selection;
- stable split locations;
- interpretable regime transitions;
- graceful degradation when fees or noise are increased;
- much lower model complexity.

That would support the claim that the tree's restricted structure is useful.

Even if a DNN produces slightly higher raw return, the tree could remain interesting if it achieves a better stability/complexity/interpretability trade-off.

Conversely, if the DNN consistently dominates on untouched data without excessive instability or turnover, that would be evidence that the tree's inductive bias is too restrictive.

That result should be accepted rather than explained away.

---

# 31. What Would Falsify the Motivation?

The tree-based research direction would be weakened if:

1. split features and thresholds vary almost arbitrarily across windows;
2. smoothing does not reduce turnover or decision variance;
3. the path-Sharpe gate mainly improves training results but not validation/test results;
4. tree performance collapses under modest changes to fees;
5. a regularized small MLP consistently dominates across rolling tests;
6. the tree requires increasing depth until it effectively becomes a complicated partition approximator;
7. regime interpretations are added only after seeing the results and are not stable out of sample.

These are not reasons to avoid the experiment.

They are precisely the tests that make the research credible.

---

# 32. Why an "Overkill" DNN Can Actually Be a Useful Experimental Adversary

A deliberately high-capacity DNN is useful because it tests the core hypothesis directly.

If the DNN has much more capacity but does not improve out-of-sample decisions, that suggests the limiting factor is not function approximation.

Possible limiting factors would instead include:

- insufficient signal;
- non-stationarity;
- sample dependence;
- optimizer sensitivity;
- transaction costs;
- estimation noise.

If the DNN wins decisively, the conclusion is equally informative:

> The portfolio problem contains exploitable structure that the regime-tree hypothesis is failing to capture.

Thus the DNN should not be treated as an enemy of the tree project.

It is one of the most important controls.

---

# 33. The Broader Modeling Philosophy

The tree follows a useful principle:

$$
\boxed{
\text{Do not learn with ML what can be represented explicitly and reliably.}
}
$$

The model learns the uncertain relationship

$$
x_t \rightarrow c_t.
$$

The optimizer handles the known decision structure:

$$
c_t,u_t,\Sigma_t
\rightarrow
w_t.
$$

The replay handles the sequential accounting:

$$
w_t,r_t
\rightarrow
u_{t+1}.
$$

And the tree architecture constrains the learned mapping to a small collection of regimes.

Each component has a specific role.

This modularity is scientifically attractive because errors can be diagnosed separately.

Was the problem:

- the feature representation?
- the split?
- the leaf score?
- the covariance matrix?
- the risk-aversion parameter?
- the fee assumption?
- the holdings transition?
- the sequential instability?

A monolithic neural allocator can absorb many of these effects into its parameters, which may increase performance but makes diagnosis harder.

---

# 34. A Fair Summary of the Competition

| Dimension | Improved decision tree | Deep neural network |
|---|---|---|
| Representational capacity | Limited | Very high |
| Nonlinearity | Yes | Yes, much richer |
| High-order interactions | Limited by depth | Strong |
| Representation learning | Weak | Strong |
| Raw sequence modeling | Weak | Strong |
| Small effective sample | Attractive inductive bias | Greater overfitting risk |
| Regime interpretation | Native | Indirect |
| Decision-boundary targeting | Natural | Possible |
| Continuous mapping | Requires smoothing | Natural at predictor level |
| Explicit optimizer integration | Natural | Also possible |
| Transaction-cost awareness | Explicit in current model | Can be explicit |
| Path-aware training | Explicit in current design | Possible but more complex |
| Transparency | High | Low to moderate |
| Training stochasticity | Low in current implementation | Often significant |
| Scalability to thousands of assets | Poor in current implementation | Potentially strong |
| Ability to exploit very rich datasets | Limited | Strong |
| Scientific diagnosis | Strong | More difficult |

No row in this table establishes a universal winner.

The appropriate model depends on the structure and scale of the problem.

---

# 35. Conclusion

The motivation for the improved decision tree is **not** that trees are more expressive than deep neural networks.

They are not.

Its motivation is that portfolio management rewards more than expressive prediction.

The current architecture imposes a particular structure on the problem:

$$
\text{features}
\rightarrow
\text{decision-relevant regimes}
\rightarrow
\text{bounded scores}
\rightarrow
\text{explicit portfolio optimizer}
\rightarrow
\text{sequential trading path}.
$$

The tree uses decision regret to search for regime boundaries, sequential replay to expose transaction-cost consequences, a path-level Sharpe criterion to reject locally attractive but globally harmful splits, chronological validation to prune unsupported structure, and an experimental soft-split mechanism to represent uncertainty in the precise location of a decision boundary.

Those properties give the model a coherent economic and statistical motivation for a small-asset portfolio problem.

At the same time, the architecture has real limitations:

- axis-aligned partitioning is restrictive;
- hard trees are unstable around thresholds;
- soft splitting only addresses part of that instability;
- split search and Sharpe selection can overfit;
- leaf fitting is approximate;
- the current exact optimizer does not scale naturally to large asset universes;
- trees cannot compete naturally with deep representation learning on rich raw data.

For that reason, the most defensible research position is not

> Trees are better than neural networks.

It is:

> **A decision-aware, path-aware, and uncertainty-smoothed tree is a deliberately constrained alternative whose inductive bias may be especially appropriate for low-dimensional, noisy, and non-stationary portfolio problems. Its value should be established by direct, fair out-of-sample comparison against well-regularized neural models.**

That is both a motivation for the model and a standard by which it can legitimately fail.

---

# References and Project Basis

1. Adam N. Elmachtoub, Jason Cheuk Nam Liang, and Ryan McNellis, **"Decision Trees for Decision-Making under the Predict-then-Optimize Framework."**

   Key ideas used here: SPO loss, decision-aware tree partitions, decision boundaries, validation pruning, and SPO Forests.

2. Yi Wang and Takashi Hasuike, **"Smart Predict-then-Optimize Paradigm for Portfolio Optimization in Real Markets."**

   Key ideas used here: decision-focused portfolio learning, explicit transaction costs and risk/regularization, comparison with direct differentiable allocation, and the motivation for simple robust predictors in noisy financial environments.

3. Yi Wang and Takashi Hasuike, **"Decision-Induced Ranking Explains Prediction Inflation and Excessive Turnover in SPO-Based Portfolio Optimization."**

   Key ideas used here: decision-inducing rather than necessarily calibrated predictions, optimizer-induced ranking, excessive turnover, and the importance of portfolio-level stabilization.

4. Jianyuan Zhong et al., **"DSPO: An End-to-End Framework for Direct Sorted Portfolio Construction."**

   Used as an example of the genuine strengths of deep architectures for high-dimensional temporal and cross-sectional representation learning.

5. GitHub repository: `SougoEdo/Tree_Decision_PaperAlgo`, branch `path-aware-spo-tree`, inspected on 23 September 2026.

   Implementation-specific discussion refers to the current `main.py` architecture: exact small-universe portfolio optimization, chronological replay, regret-fitted leaf scores, best-first split growth, path-Sharpe acceptance, validation pruning, threshold refinement, and Boltzmann-weighted soft splitting.