#!/usr/bin/env python3
"""Experiments and figures for the planted-threshold test on daily data (see recovery_test.py).

  python recovery_figures.py time    one job per trading rate, to estimate the run time
  python recovery_figures.py run     every experiment, in parallel; saves report/figures/recovery_results.json
  python recovery_figures.py run-settings   the settings experiment (risk aversion, fee, depth, quartile
                                     thresholds); saves report/figures/recovery_settings.json
  python recovery_figures.py run-enhancements   the split-search options (quantile candidates, refinement,
                                     soft splits); saves report/figures/recovery_enhancements.json
  python recovery_figures.py run-pruning   the pruning hurdle in standard errors; saves report/figures/recovery_pruning.json
  python recovery_figures.py plot    the figures, from the saved results, into report/figures/

Every cell of the experiment is one trading rate (days between decisions) with
one strength of the planted drift (edge = drift / volatility per day) or one
reversion speed k of the feature; a cell holds independent datasets that
differ only by their seed.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import json
import os
import sys
import time

import numpy as np

from recovery_test import (DAYS_PER_YEAR, FEATURES, Case, annual, expected_return_curve,
                           run_case, sign_change)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report", "figures")
RESULTS = os.path.join(OUT, "recovery_results.json")

BASE = Case()                                   # the parameters set in recovery_test.py
STEPS = [1, 2, 5, 10, 21]                       # days between two decisions: daily to monthly
EDGES = [0.02, 0.05, 0.10]                      # drift / volatility per day
K_VALUES = [0.01, 0.02, 0.05, 0.10]             # reversion speed per day: half-lives 69, 35, 14, 7 days
DATASETS_PER_CELL = 50
HISTOGRAM_EDGES = [0.05, 0.10]                  # these cells get more datasets, for the histograms
DATASETS_PER_HISTOGRAM = 100
RATE_NAMES = {1: "daily", 2: "twice a week", 5: "weekly", 10: "every 2 weeks", 21: "monthly"}

# A second experiment: the decision problem and the tree are changed, one or two things at a time.
SETTINGS = {
    "base": {},                                                  # the run of the note
    "depth 3": {"max_depth": 3},
    "λ = 5, depth 3": {"risk_aversion": 5.0, "max_depth": 3},
    "λ = 10, depth 3": {"risk_aversion": 10.0, "max_depth": 3},
    "fee 10 bp": {"fee": 0.001},
    "quartiles, depth 3": {"threshold_quantiles": (0.25, 0.5, 0.75), "max_depth": 3},
}
SETTING_STEPS = [2, 5, 10, 21]
SETTING_EDGES = [0.05, 0.10]
DATASETS_PER_SETTING = 40
SETTINGS_RESULTS = os.path.join(OUT, "recovery_settings.json")
POSITION_BINS = np.linspace(-2.5, 2.5, 26)      # the mean position by bin of x, on the test rows
SCORE_GRID = np.linspace(-2.5, 2.5, 51)         # the tree's score as a function of x

# A third experiment: the options that change how a split is chosen (see SPOPortfolioTree).
DECILES = tuple(float(q) for q in np.round(np.arange(0.1, 1.0, 0.1), 2))
ENHANCEMENTS = {
    "base": {},
    "deciles": {"threshold_quantiles": DECILES, "min_leaf_fraction": 0.10},
    "deciles + refinement": {"threshold_quantiles": DECILES, "min_leaf_fraction": 0.10,
                             "refine_thresholds": True, "refine_standard_errors": 1.0},
    "smoothing": {"smoothing": 1.0},
    "deciles + refinement + smoothing": {"threshold_quantiles": DECILES, "min_leaf_fraction": 0.10,
                                         "refine_thresholds": True, "smoothing": 1.0},
}
ENHANCEMENTS_RESULTS = os.path.join(OUT, "recovery_enhancements.json")

# A fourth experiment: the pruning hurdle, in standard errors of the checking-row regret gain.
PRUNING = {
    "base": {},
    "hurdle 1 s.e.": {"prune_standard_errors": 1.0},
    "hurdle 2 s.e.": {"prune_standard_errors": 2.0},
    "deciles + refinement": {"threshold_quantiles": DECILES, "min_leaf_fraction": 0.10,
                             "refine_thresholds": True, "refine_standard_errors": 1.0},
    "deciles + refinement + hurdle 2 s.e.": {"threshold_quantiles": DECILES, "min_leaf_fraction": 0.10,
                                             "refine_thresholds": True, "refine_standard_errors": 1.0,
                                             "prune_standard_errors": 2.0},
}
PRUNING_RESULTS = os.path.join(OUT, "recovery_pruning.json")


def x_grid(case):
    std = np.sqrt(case.feature_variance)
    return np.linspace(case.mean - 4 * std, case.mean + 4 * std, 321)


@lru_cache(maxsize=None)
def curve_of(k, feature_variance, edge, volatility, d, mean, step):
    case = replace(BASE, k=k, feature_variance=feature_variance, edge=edge,
                   volatility=volatility, d=d, mean=mean, step=step)
    grid = x_grid(case)
    return grid, expected_return_curve(case, grid, n_paths=10000)


def curve(case):
    return curve_of(case.k, case.feature_variance, case.edge, case.volatility,
                    case.d, case.mean, case.step)


def position_profile(x, weights):
    """Mean weight by bin of x (None where fewer than 3 decisions fall)."""
    which = np.digitize(x, POSITION_BINS) - 1
    return [float(weights[which == b].mean()) if np.sum(which == b) >= 3 else None
            for b in range(len(POSITION_BINS) - 1)]


def job(task):
    tag, case, seed, extra = task
    grid, expected = curve(case)
    result = run_case(case, seed, curve=(grid, expected))
    paths = result["paths"]
    x_test = result["rows"].X[case.n_train:, 0]
    return {
        "tag": tag, "step": case.step, "edge": case.edge, "k": case.k,
        "volatility": case.volatility, "seed": seed, "sign_change": sign_change(grid, expected),
        "grown": result["grown"], "kept": result["kept"], "n_leaves": len(result["kept"]) + 1,
        "spreads": result["spreads"],
        "score_profile": [float(v) for v in result["tree"].predict_returns(SCORE_GRID[:, None])[:, 0]],
        "test_mean": {name: float(path.net_returns.mean()) for name, path in paths.items()},
        "test_sharpe": {name: float(path.sharpe) for name, path in paths.items()},
        "test_turnover": {name: float(path.turnover.mean()) for name, path in paths.items()},
        "profile": {name: position_profile(x_test, paths[name].weights[:, 0])
                    for name in ("tree", "ideal rule")},
        **extra,
    }


def all_tasks():
    tasks = []
    for step in STEPS:                          # the daily jobs are the slowest: submit them first
        for edge in EDGES:
            n = DATASETS_PER_HISTOGRAM if edge in HISTOGRAM_EDGES else DATASETS_PER_CELL
            tasks += [("edge", replace(BASE, step=step, edge=edge), seed, {}) for seed in range(n)]
        for k in K_VALUES:
            if k != BASE.k:
                tasks += [("speed", replace(BASE, step=step, k=k), seed, {})
                          for seed in range(DATASETS_PER_CELL)]
    return tasks


def settings_tasks():
    return [("settings", replace(BASE, step=step, edge=edge, **changes), seed, {"setting": name})
            for step in SETTING_STEPS for name, changes in SETTINGS.items()
            for edge in SETTING_EDGES for seed in range(DATASETS_PER_SETTING)]


def setting_tasks(tag, settings):
    return [(tag, replace(BASE, step=step, edge=edge, **changes), seed, {"setting": name})
            for step in SETTING_STEPS for name, changes in settings.items()
            for edge in SETTING_EDGES for seed in range(DATASETS_PER_SETTING)]


def enhancement_tasks():
    return setting_tasks("enhancements", ENHANCEMENTS)


def run(tasks, path):
    results, start = [], time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 50 == 0 or done == len(tasks):
                print(f"progress: {done}/{len(tasks)} fits after {time.perf_counter() - start:.0f}s",
                      flush=True)
    with open(path, "w") as handle:
        json.dump(results, handle)
    print(f"saved {path}")


# ----------------------------------------------------------------------------- summaries
def records(results, step, edge=BASE.edge, k=BASE.k):
    """The datasets of one cell."""
    return [r for r in results if r["step"] == step and r["edge"] == edge and r["k"] == k]


def root_of(record):
    roots = [(f, th) for depth, f, th in record["kept"] if depth == 0]
    return roots[0] if roots else (None, None)


def annual_sharpe(cell, name, step):
    """Mean over the datasets of the annualized test Sharpe ratio of one strategy."""
    return float(np.nanmean([r["test_sharpe"][name] for r in cell])) * np.sqrt(DAYS_PER_YEAR / step)


def annual_gain(cell, name, step):
    """Mean test return of a strategy over the tree without splits, per year."""
    return float(np.mean([r["test_mean"][name] - r["test_mean"]["tree without splits"]
                          for r in cell])) * DAYS_PER_YEAR / step


def cell_summary(cell, step):
    roots = [root_of(r) for r in cell]
    first = [th for f, th in roots if f == "x"]
    return {
        "n": len(cell),
        "on_x": 100 * sum(f == "x" for f, _ in roots) / len(cell),
        "none": 100 * sum(f is None for f, _ in roots) / len(cell),
        "median": float(np.median(first)) if first else np.nan,
        "quartiles": tuple(np.percentile(first, [25, 75])) if first else (np.nan, np.nan),
        "sharpe": {name: annual_sharpe(cell, name, step)
                   for name in ("tree", "ideal rule", "rule at d", "always the asset")},
        "captured": (100 * annual_gain(cell, "tree", step) / annual_gain(cell, "ideal rule", step)
                     if annual_gain(cell, "ideal rule", step) > 0.005 else np.nan),
    }


def print_summary(results):
    print(f"\n{'rate':>14} {'edge':>5} {'k':>5} {'n':>4} {'on x':>5} {'none':>5} {'median':>7} "
          f"{'quartiles':>15} {'tree':>6} {'ideal':>6} {'at d':>6} {'asset':>6} {'captured':>9}")
    print(f"{'':>14} {'':>5} {'':>5} {'':>4} {'%':>5} {'%':>5} {'':>7} {'':>15} "
          f"{'annualized test Sharpe ratio':^27} {'%':>9}")
    cells = [(step, edge, BASE.k) for edge in EDGES for step in STEPS]
    cells += [(step, BASE.edge, k) for k in K_VALUES if k != BASE.k for step in STEPS]
    for step, edge, k in cells:
        cell = records(results, step, edge, k)
        if not cell:
            continue
        s = cell_summary(cell, step)
        print(f"{RATE_NAMES[step]:>14} {edge:>5g} {k:>5g} {s['n']:>4d} {s['on_x']:>5.0f} {s['none']:>5.0f} "
              f"{s['median']:>+7.2f} {s['quartiles'][0]:>+7.2f}{s['quartiles'][1]:>+8.2f} "
              f"{s['sharpe']['tree']:>6.2f} {s['sharpe']['ideal rule']:>6.2f} "
              f"{s['sharpe']['rule at d']:>6.2f} {s['sharpe']['always the asset']:>6.2f} "
              f"{s['captured']:>9.0f}")
    print("\nrate: days between decisions; on x / none: share of datasets whose first split is on x / "
          "that keep no split;\nmedian and quartiles: first threshold on x over the datasets that "
          "split (x has standard deviation 1, the target is d = 0);\ncaptured: test gain of the tree "
          "over the tree without splits, as a share of the ideal rule's gain.")


def setting_records(results, name, step, edge):
    return [r for r in results if r.get("setting") == name and r["step"] == step and r["edge"] == edge]


def setting_summary(cell, step):
    s = cell_summary(cell, step)
    s["leaves"] = float(np.mean([r["n_leaves"] for r in cell]))
    s["turnover"] = {name: float(np.mean([r["test_turnover"][name] for r in cell]))
                     for name in ("tree", "ideal rule")}
    first = [th for r in cell for f, th in [root_of(r)] if f == "x"]
    s["within"] = 100 * float(np.mean(np.abs(first) <= 0.25)) if first else np.nan
    root_spreads = [r["spreads"][0] for r in cell if r.get("spreads") and root_of(r)[0] == "x"]
    s["spread"] = float(np.mean(root_spreads)) if root_spreads else 0.0
    s["second"] = 100 * float(np.mean([any(d >= 1 for d, _, _ in r["kept"]) for r in cell]))
    return s


def print_settings_summary(results, names, title):
    for edge in SETTING_EDGES:
        print(f"\n{title}, edge {edge:g}, {DATASETS_PER_SETTING} datasets per cell")
        print(f"{'setting':>32} {'rate':>14} {'leaves':>6} {'on x':>5} {'none':>5} {'median':>7} "
              f"{'quartiles':>15} {'<=.25':>5} {'spread':>6} {'2nd':>4} {'tree':>6} {'ideal':>6} {'captured':>9} {'turnover':>14}")
        for name in names:
            for step in SETTING_STEPS:
                cell = setting_records(results, name, step, edge)
                if not cell:
                    continue
                s = setting_summary(cell, step)
                print(f"{name:>32} {RATE_NAMES[step]:>14} {s['leaves']:>6.2f} {s['on_x']:>5.0f} "
                      f"{s['none']:>5.0f} {s['median']:>+7.2f} {s['quartiles'][0]:>+7.2f}{s['quartiles'][1]:>+8.2f} "
                      f"{s['within']:>5.0f} {s['spread']:>6.2f} {s['second']:>4.0f} {s['sharpe']['tree']:>6.2f} "
                      f"{s['sharpe']['ideal rule']:>6.2f} {s['captured']:>9.0f} "
                      f"{s['turnover']['tree']:>6.2f} ({s['turnover']['ideal rule']:.2f})")
    print("\nleaves: mean number of leaves kept; <=.25: share of first thresholds within 0.25 of d; "
          "spread: mean spread of the root split (soft splits); 2nd: share of datasets with a second-level split; turnover: per decision, tree (ideal rule in brackets).")


# ----------------------------------------------------------------------------- figures
GREEN, RED, BLUE, ORANGE, GREY = "#2e8b57", "#c0392b", "#1f4e9c", "#e67e22", "#555555"
EDGE_COLORS = {0.02: "#f5c99b", 0.05: ORANGE, 0.10: "#a04000"}


def shade_regimes(axis, days, x, d):
    """Light green while x >= d (drift +mu), light red while x < d (drift -mu)."""
    above = x >= d
    edges = np.flatnonzero(np.diff(above.astype(int))) + 1
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(x)]):
        axis.axvspan(days[start], days[min(stop, len(x) - 1)],
                     color=GREEN if above[start] else RED, alpha=0.13, lw=0)


def tree_scores(result, x_values):
    """The tree's predicted return of the asset when x takes these values."""
    X = np.tile(result["rows"].X[:result["case"].n_train].mean(axis=0), (len(x_values), 1))
    X[:, 0] = x_values
    return result["tree"].predict_returns(X)


def position(result, scores):
    """Weight of the asset chosen by the optimizer, from half-and-half holdings."""
    n = len(scores)
    sigma = np.tile(result["rows"].Sigma[:result["case"].n_train].mean(axis=0), (n, 1, 1))
    return result["optimizer"].solve(scores, np.full((n, 2), 0.5), sigma)[:, 0]


def binned_means(x, r, n_bins=16):
    """Mean realised return by quantile bin of x, with its standard error."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    centers, means, errors = [], [], []
    for b in range(n_bins):
        inside = which == b
        if inside.sum() >= 2:
            centers.append(x[inside].mean())
            means.append(r[inside].mean())
            errors.append(r[inside].std(ddof=1) / np.sqrt(inside.sum()))
    return np.array(centers), np.array(means), np.array(errors)


def figure_world(result, plt):
    case = result["case"]
    x, prices = result["series"]
    days = np.arange(case.window, case.window + 2 * DAYS_PER_YEAR)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for axis in (top, bottom):
        shade_regimes(axis, days, x[days], case.d)
    top.plot(days, x[days], color=BLUE, lw=1.0)
    top.axhline(case.d, color="black", ls="--", lw=1,
                label=f"planted threshold d = {case.d:g}, the long-run mean of x")
    top.set_ylabel("feature x")
    top.legend(loc="upper right", fontsize=8)
    top.set_title(f"The planted world (two years): drift {case.mu:+.2%} a day while x is above d "
                  f"(green), {-case.mu:+.2%} a day below (red); volatility {case.volatility:.0%} a day",
                  fontsize=9)
    bottom.semilogy(days, prices[days], color="black", lw=1.0)
    decisions = days[(days - case.window) % case.step == 0]
    bottom.plot(decisions, np.full(len(decisions), prices[days].min() * 0.98), "|", color=GREY, ms=6)
    bottom.set_ylabel("price of the asset (log scale)")
    bottom.set_xlabel(f"day (grey ticks: one decision every {case.step} days)")
    fig.tight_layout()
    return fig


def figure_learned(result, grid, expected, plt):
    case = result["case"]
    rows = result["rows"]
    train = slice(0, case.n_train)
    crossing = sign_change(grid, expected)
    thresholds = [th for _, f, th in result["kept"] if f == "x"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    centers, means, errors = binned_means(rows.X[train, 0], rows.R[train, 0])
    top.errorbar(centers, means, yerr=errors, fmt="o", color=GREY, ms=4, lw=1, capsize=2,
                 label=f"realised {case.step}-day returns, mean by bin of x (training decisions, "
                       f"± 1 standard error)")
    top.plot(grid, expected, color=BLUE, lw=2, label=f"true expected {case.step}-day return given x")
    limit = 1.25 * max(np.abs(expected).max(), (np.abs(means) + errors).max())
    scores = tree_scores(result, grid)[:, 0]
    top.plot(grid, np.clip(scores, -limit, limit), color=ORANGE, lw=2.2,
             label="the tree: score of each leaf (clipped to the axis: scores are inputs "
                   "to the optimizer, not forecasts)")
    top.axhline(0, color="black", lw=0.6)
    top.set_ylabel(f"return of the asset over {case.step} days")
    top.set_ylim(-limit, limit)
    top.set_title("What the tree should learn, and what it learned (one dataset)", fontsize=10)
    ideal = position(result, np.column_stack([expected, np.zeros(len(grid))]))
    bottom.plot(grid, ideal, color=BLUE, lw=2, label="ideal position (optimizer fed with the true curve)")
    bottom.plot(grid, position(result, tree_scores(result, grid)), color=ORANGE, lw=2.2,
                label="position chosen with the tree's leaves")
    bottom.set_ylabel("weight of the asset")
    bottom.set_xlabel("feature x on the decision day")
    bottom.set_ylim(-0.05, 1.08)
    for axis in (top, bottom):
        axis.axvline(case.d, color="black", ls="--", lw=1)
        if crossing is not None:
            axis.axvline(crossing, color=BLUE, ls=":", lw=1.4)
        for th in thresholds:
            axis.axvline(th, color=ORANGE, lw=1)
    if crossing is not None and abs(crossing - case.d) < 0.05:
        top.annotate(f"d = {case.d:g}: the sign change\n(x = {crossing:.2f}) is at d",
                     (case.d, 0.9 * limit), fontsize=8, ha="right", xytext=(-4, 0),
                     textcoords="offset points")
    else:
        top.annotate(f"d = {case.d:g}", (case.d, 0.9 * limit), fontsize=8, ha="left", xytext=(3, 0),
                     textcoords="offset points")
        if crossing is not None:
            top.annotate(f"sign change\nx = {crossing:.2f}", (crossing, 0.75 * limit), fontsize=8,
                         ha="right", color=BLUE, xytext=(-3, 0), textcoords="offset points")
    for th in thresholds:
        top.annotate(f"tree\n{th:+.2f}", (th, -0.9 * limit), fontsize=8, ha="left", color=ORANGE,
                     xytext=(3, 0), textcoords="offset points")
    top.legend(loc="lower right", fontsize=7.5)
    bottom.legend(loc="center right", fontsize=8)
    top.set_xlim(case.mean - 3.2 * result["feature_std"], case.mean + 3.2 * result["feature_std"])
    fig.tight_layout()
    return fig


def figure_decisions(result, plt):
    case = result["case"]
    rows = result["rows"]
    x = result["series"][0]
    shown = 60
    first = case.n_train
    decisions = np.arange(first, first + shown)
    days = np.arange(rows.days[first], rows.days[first + shown])
    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(9, 8.5),
                                              gridspec_kw={"height_ratios": [2, 1.3, 2.2]})
    middle.sharex(top)
    for axis in (top, middle):
        shade_regimes(axis, days, x[days], case.d)
    top.plot(days, x[days], color=BLUE, lw=1)
    top.axhline(case.d, color="black", ls="--", lw=1)
    for th in [th for _, f, th in result["kept"] if f == "x"]:
        top.axhline(th, color=ORANGE, lw=1.2)
    top.set_ylabel("feature x")
    top.set_title(f"Test period, never seen in training: the first {shown} decisions "
                  f"(regimes in green/red, the tree's threshold in orange)", fontsize=10)
    middle.step(rows.days[decisions], result["paths"]["tree"].weights[:shown, 0], where="post",
                color=ORANGE, lw=1.6, label="tree")
    middle.step(rows.days[decisions], result["paths"]["ideal rule"].weights[:shown, 0],
                where="post", color=BLUE, lw=1, ls="--", label="ideal rule")
    middle.set_ylabel("weight of the asset")
    middle.set_ylim(-0.08, 1.5)
    middle.set_xlabel("day")
    middle.legend(loc="upper right", fontsize=8, ncol=2)
    styles = {"tree": (ORANGE, "-", 2), "ideal rule": (BLUE, "--", 1.4), "rule at d": (GREEN, ":", 1.6),
              "always the asset": ("black", "-", 1)}
    years = (rows.days[first:] + case.step - rows.days[first]) / DAYS_PER_YEAR
    for name, (color, ls, lw) in styles.items():
        path = result["paths"][name]
        wealth = np.cumprod(1 + path.net_returns)
        bottom.plot(years, wealth, color=color, ls=ls, lw=lw,
                    label=f"{name} (Sharpe {annual(path, case)['sharpe']:.2f})")
    bottom.axhline(1, color=GREY, lw=0.8)
    bottom.set_ylabel("wealth (start = 1)")
    bottom.set_xlabel("years into the test period")
    bottom.set_title(f"The whole test period, {case.test_days / DAYS_PER_YEAR:g} years "
                     f"(annualized Sharpe ratios in brackets)", fontsize=10)
    bottom.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8, ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def figure_histograms(results, plt):
    fig, axes = plt.subplots(len(STEPS), len(HISTOGRAM_EDGES), figsize=(8.5, 11.5), sharex=True)
    bins = np.linspace(-1.5, 1.5, 31)
    for i, step in enumerate(STEPS):
        for j, edge in enumerate(HISTOGRAM_EDGES):
            axis = axes[i, j]
            cell = records(results, step, edge=edge)
            if not cell:
                continue
            found = {depth: [th for r in cell for dp, f, th in r["kept"] if f == "x" and dp == depth]
                     for depth in (0, 1)}
            axis.hist([np.clip(found[0], bins[0], bins[-1]), np.clip(found[1], bins[0], bins[-1])],
                      bins=bins, stacked=True, color=[ORANGE, "#f5c99b"],
                      label=["first split", "second-level split"])
            axis.axvline(BASE.d, color="black", ls="--", lw=1.2,
                         label=f"planted d = {BASE.d:g}, the mean of x: the target at every rate")
            s_ = cell_summary(cell, step)
            axis.set_title(f"{RATE_NAMES[step]} (every {step} d), edge {edge:g} "
                           f"({edge * BASE.volatility * DAYS_PER_YEAR:+.0%} a year)\n"
                           f"first split on x {s_['on_x']:.0f}%, no split {s_['none']:.0f}%, "
                           f"median {s_['median']:+.2f}", fontsize=9)
            if j == 0:
                axis.set_ylabel("number of splits")
            if i == len(STEPS) - 1:
                axis.set_xlabel("threshold found on x")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    n = len(records(results, STEPS[0], edge=HISTOGRAM_EDGES[0]))
    fig.suptitle(f"Where the tree puts its thresholds on x (k = {BASE.k:g}, {n} datasets per panel; "
                 f"x has standard deviation 1)", fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    return fig


def heatmap(axis, values, title, text, cmap, vmin, vmax, xlabels, xlabel, steps=STEPS):
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axis.set_xticks(range(len(xlabels)), xlabels, fontsize=9)
    axis.set_yticks(range(len(steps)), [f"{RATE_NAMES[s]}\n(every {s} d)" for s in steps], fontsize=9)
    axis.set_xlabel(xlabel, fontsize=9.5)
    axis.set_title(title, fontsize=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                dark = cmap != "RdYlGn" and (values[i, j] - vmin) / (vmax - vmin) > 0.6
                axis.text(j, i, text(i, j), ha="center", va="center", fontsize=10,
                          color="white" if dark else "black")
    return image


def figure_maps(results, plt, key, columns, xlabels, xlabel):
    """One row per trading rate, one column per value of `key` (edge or k)."""
    shape = (len(STEPS), len(columns))
    on_x, none, captured, tree, ideal = (np.full(shape, np.nan) for _ in range(5))
    for i, step in enumerate(STEPS):
        for j, value in enumerate(columns):
            cell = records(results, step, **{key: value})
            if not cell:
                continue
            s = cell_summary(cell, step)
            on_x[i, j], none[i, j], captured[i, j] = s["on_x"], s["none"], s["captured"]
            tree[i, j], ideal[i, j] = s["sharpe"]["tree"], s["sharpe"]["ideal rule"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    axes = axes.ravel()
    heatmap(axes[0], on_x, "first split on the true feature x\n(% of the datasets)",
            lambda i, j: f"{on_x[i, j]:.0f}", "Greens", 0, 100, xlabels, xlabel)
    heatmap(axes[1], none, "no split kept after pruning\n(% of the datasets)",
            lambda i, j: f"{none[i, j]:.0f}", "Greys", 0, 100, xlabels, xlabel)
    heatmap(axes[2], captured, "test gain over the tree without splits,\nas % of the ideal "
            "rule's gain (blank: under 0.5% a year to gain)",
            lambda i, j: f"{captured[i, j]:.0f}", "RdYlGn", -50, 110, xlabels, xlabel)
    heatmap(axes[3], tree, "annualized test Sharpe ratio of the tree\n(ideal rule in brackets)",
            lambda i, j: f"{tree[i, j]:.2f}\n({ideal[i, j]:.2f})", "Blues", 0,
            max(1.0, np.nanmax(tree)), xlabels, xlabel)
    axes[0].set_ylabel("trading rate", fontsize=9.5)
    axes[2].set_ylabel("trading rate", fontsize=9.5)
    fig.tight_layout()
    return fig


def figure_thresholds_by_rate(results, plt):
    fig, axis = plt.subplots(figsize=(8.5, 4.6))
    offsets = np.linspace(-0.27, 0.27, len(EDGES))
    for offset, edge in zip(offsets, EDGES):
        color = EDGE_COLORS[edge]
        for i, step in enumerate(STEPS):
            cell = records(results, step, edge=edge)
            found = [root_of(r)[1] for r in cell if root_of(r)[0] == "x"]
            if found:
                axis.boxplot(found, positions=[i + offset], widths=0.22, showfliers=True,
                             boxprops={"color": color}, medianprops={"color": color, "lw": 2},
                             whiskerprops={"color": color}, capprops={"color": color},
                             flierprops={"marker": ".", "markeredgecolor": color, "ms": 4})
            axis.text(i + offset, 1.0, f"{len(found)}/{len(cell)}", ha="center", va="bottom",
                      fontsize=7, color=color, transform=axis.get_xaxis_transform())
        axis.plot([], [], color=color, lw=2, label=f"edge {edge:g} ({edge * BASE.volatility * DAYS_PER_YEAR:+.0%} a year)")
    axis.axhline(BASE.d, color="black", ls="--", lw=1, label=f"planted d = {BASE.d:g}, the mean of x")
    axis.set_xticks(range(len(STEPS)), [f"{RATE_NAMES[s]}\n(every {s} d)" for s in STEPS])
    axis.set_xlabel("trading rate (above: datasets whose first split is on x)")
    axis.set_ylabel("first threshold found on x (standard deviation of x = 1)")
    axis.set_ylim(-1.6, 1.6)
    axis.set_title(f"Threshold found against the trading rate and the strength of the drift "
                   f"(k = {BASE.k:g}, volatility {BASE.volatility:.0%} a day)", fontsize=10, pad=18)
    axis.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    return fig


def figure_settings_positions(results, plt, names, step=5):
    """Mean test position by x: the tree (median and quartiles over datasets) against the ideal rule."""
    fig, axes = plt.subplots(len(names), len(SETTING_EDGES), figsize=(8.5, 2.2 * len(names) + 0.8),
                             sharex=True, sharey=True)
    centers = (POSITION_BINS[:-1] + POSITION_BINS[1:]) / 2
    for j, name in enumerate(names):
        for i, edge in enumerate(SETTING_EDGES):
            axis = axes[j, i]
            cell = setting_records(results, name, step, edge)
            if not cell:
                continue
            tree = np.array([[np.nan if v is None else v for v in r["profile"]["tree"]] for r in cell])
            ideal = np.array([[np.nan if v is None else v for v in r["profile"]["ideal rule"]] for r in cell])
            low, mid, high = np.nanpercentile(tree, [25, 50, 75], axis=0)
            axis.fill_between(centers, low, high, color=ORANGE, alpha=0.25, lw=0,
                              label="tree: quartiles over datasets")
            axis.plot(centers, mid, color=ORANGE, lw=2, label="tree: median over datasets")
            axis.plot(centers, np.nanmedian(ideal, axis=0), color=BLUE, lw=1.6, ls="--",
                      label="ideal rule: median over datasets")
            axis.axvline(BASE.d, color="black", ls=":", lw=1)
            s = setting_summary(cell, step)
            axis.set_title(f"{name}, edge {edge:g}: {s['leaves']:.1f} leaves, Sharpe "
                           f"{s['sharpe']['tree']:.2f} ({s['sharpe']['ideal rule']:.2f})", fontsize=9)
            if i == 0:
                axis.set_ylabel("mean weight of the asset")
            if j == len(names) - 1:
                axis.set_xlabel("x on the decision day")
    axes[0, 0].set_ylim(-0.05, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Position taken on the test decisions, by x ({RATE_NAMES[step]} decisions, "
                 f"{DATASETS_PER_SETTING} datasets per panel)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return fig


def figure_settings_maps(results, plt, names, edge=0.10, title="The settings experiment"):
    shape = (len(SETTING_STEPS), len(names))
    captured, tree, ideal, leaves, turnover, turn_ideal = (np.full(shape, np.nan) for _ in range(6))
    for i, step in enumerate(SETTING_STEPS):
        for j, name in enumerate(names):
            cell = setting_records(results, name, step, edge)
            if not cell:
                continue
            s = setting_summary(cell, step)
            captured[i, j], leaves[i, j] = s["captured"], s["leaves"]
            tree[i, j], ideal[i, j] = s["sharpe"]["tree"], s["sharpe"]["ideal rule"]
            turnover[i, j], turn_ideal[i, j] = s["turnover"]["tree"], s["turnover"]["ideal rule"]
    labels = [n.replace(", ", "\n").replace(" + ", "\n+ ") for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    heatmap(axes[0], captured, "test gain over the tree without splits,\nas % of the ideal rule's gain",
            lambda i, j: f"{captured[i, j]:.0f}", "RdYlGn", -50, 110, labels, "", SETTING_STEPS)
    heatmap(axes[1], tree, "annualized test Sharpe ratio of the tree\n(ideal rule in brackets)",
            lambda i, j: f"{tree[i, j]:.2f}\n({ideal[i, j]:.2f})", "Blues", 0, max(1.0, np.nanmax(tree)),
            labels, "", SETTING_STEPS)
    heatmap(axes[2], leaves, "mean number of leaves kept", lambda i, j: f"{leaves[i, j]:.1f}",
            "Oranges", 1, 8, labels, "", SETTING_STEPS)
    heatmap(axes[3], turnover, "turnover per decision of the tree\n(ideal rule in brackets)",
            lambda i, j: f"{turnover[i, j]:.2f}\n({turn_ideal[i, j]:.2f})", "Purples", 0,
            max(1.0, np.nanmax(turnover)), labels, "", SETTING_STEPS)
    for axis in axes[:2]:
        axis.set_ylabel("trading rate", fontsize=9.5)
    fig.suptitle(f"{title} at edge {edge:g} ({DATASETS_PER_SETTING} datasets per cell)", fontsize=11)
    fig.tight_layout()
    return fig


def figure_scores(results, plt, names, step=5, edge=0.10):
    """The tree's score for the asset as a function of x: median and quartiles over datasets."""
    rows_ = (len(names) + 1) // 2
    fig, axes = plt.subplots(rows_, 2, figsize=(9, 3.1 * rows_), sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[len(names):]:
        axis.set_visible(False)
    for axis, name in zip(axes, names):
        cell = setting_records(results, name, step, edge)
        if not cell:
            continue
        scores = np.array([r["score_profile"] for r in cell])
        low, mid, high = np.percentile(scores, [25, 50, 75], axis=0)
        axis.fill_between(SCORE_GRID, low, high, color=ORANGE, alpha=0.25, lw=0,
                          label="quartiles over datasets")
        axis.plot(SCORE_GRID, mid, color=ORANGE, lw=2, label="median over datasets")
        axis.axhline(0, color="black", lw=0.6)
        axis.axvline(BASE.d, color="black", ls=":", lw=1)
        s = setting_summary(cell, step)
        axis.set_title(f"{name}\nmedian threshold {s['median']:+.2f}, spread {s['spread']:.2f}", fontsize=9)
        axis.set_xlabel("x on the decision day")
    for axis in axes[0::2]:
        axis.set_ylabel("score of the asset given by the tree")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"What the tree says as a function of x ({RATE_NAMES[step]} decisions, edge {edge:g}, "
                 f"{DATASETS_PER_SETTING} datasets per panel)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def figure_thresholds_by_setting(results, plt, names, step=5):
    """Histograms of the first threshold on x, one row per setting, one column per edge."""
    fig, axes = plt.subplots(len(names), len(SETTING_EDGES), figsize=(8.5, 2.1 * len(names) + 0.8),
                             sharex=True, sharey="row")
    bins = np.linspace(-1.5, 1.5, 31)
    for j, name in enumerate(names):
        for i, edge in enumerate(SETTING_EDGES):
            axis = axes[j, i]
            cell = setting_records(results, name, step, edge)
            if not cell:
                continue
            first = [th for r in cell for f, th in [root_of(r)] if f == "x"]
            axis.hist(np.clip(first, bins[0], bins[-1]), bins=bins, color=ORANGE, label="first split on x")
            axis.axvline(BASE.d, color="black", ls="--", lw=1.2, label=f"planted d = {BASE.d:g}")
            s = setting_summary(cell, step)
            axis.set_title(f"{name}, edge {edge:g}\non x {s['on_x']:.0f}%, median {s['median']:+.2f} "
                           f"[{s['quartiles'][0]:+.2f}, {s['quartiles'][1]:+.2f}], within 0.25: {s['within']:.0f}%",
                           fontsize=8)
            if i == 0:
                axis.set_ylabel("datasets")
            if j == len(names) - 1:
                axis.set_xlabel("first threshold on x")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"Where the first threshold lands ({RATE_NAMES[step]} decisions, "
                 f"{DATASETS_PER_SETTING} datasets per panel)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return fig


def figure_split_shares(results, plt, names):
    """Per setting and trading rate: first split on x, no split kept, a second-level split kept (% of datasets)."""
    fig, axes = plt.subplots(len(SETTING_EDGES), 3, figsize=(12, 3.4 * len(SETTING_EDGES)), sharey=True)
    keys = [("on_x", "first split on x"), ("none", "no split kept"), ("second", "second-level split kept")]
    width = 0.8 / len(names)
    for i, edge in enumerate(SETTING_EDGES):
        for j, (key, label) in enumerate(keys):
            axis = axes[i, j]
            for k, name in enumerate(names):
                values = [setting_summary(setting_records(results, name, step, edge), step)[key]
                          if setting_records(results, name, step, edge) else np.nan for step in SETTING_STEPS]
                axis.bar(np.arange(len(SETTING_STEPS)) + (k - (len(names) - 1) / 2) * width, values, width,
                         label=name, color=plt.cm.Oranges(0.3 + 0.6 * k / max(1, len(names) - 1)))
            axis.set_xticks(range(len(SETTING_STEPS)), [RATE_NAMES[s] for s in SETTING_STEPS], fontsize=8)
            axis.set_title(f"{label} (edge {edge:g})", fontsize=9.5)
            if j == 0:
                axis.set_ylabel("% of datasets")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(names), fontsize=8, frameon=False)
    fig.suptitle(f"What the trees keep ({DATASETS_PER_SETTING} datasets per cell)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return fig


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(OUT, exist_ok=True)
    grid = x_grid(BASE)
    expected = expected_return_curve(BASE, grid, n_paths=100000)     # smooth enough to draw
    single = run_case(BASE, seed=42, curve=(grid, expected))
    print(f"single dataset: splits kept {single['kept']}; expected return changes sign at "
          f"x = {sign_change(grid, expected):.3f}")
    for name, path in single["paths"].items():
        figures = annual(path, BASE)
        print(f"   test {name:22s} return {figures['return']:.1%}  Sharpe {figures['sharpe']:.2f}  "
              f"turnover {figures['turnover']:.2f} per decision")
    figures = {"recovery_world": figure_world(single, plt),
               "recovery_learned": figure_learned(single, grid, expected, plt),
               "recovery_decisions": figure_decisions(single, plt)}
    if os.path.exists(RESULTS):
        with open(RESULTS) as handle:
            results = json.load(handle)
        print_summary(results)
        edge_labels = [f"edge {e:g}\n({e * BASE.volatility * DAYS_PER_YEAR:+.0%} a year)" for e in EDGES]
        k_labels = [f"k = {k:g}\n(half-life {np.log(2) / k:.0f} d)" for k in K_VALUES]
        figures.update({
            "recovery_histograms": figure_histograms(results, plt),
            "recovery_maps": figure_maps(results, plt, "edge", EDGES, edge_labels,
                                         "drift / volatility per day (drift per year)"),
            "recovery_maps_by_k": figure_maps(results, plt, "k", K_VALUES, k_labels,
                                              "reversion speed of x per day"),
            "recovery_thresholds_by_rate": figure_thresholds_by_rate(results, plt)})
    if os.path.exists(SETTINGS_RESULTS):
        with open(SETTINGS_RESULTS) as handle:
            settings = json.load(handle)
        print_settings_summary(settings, list(SETTINGS), "settings experiment")
        figures.update({"recovery_settings_positions": figure_settings_positions(settings, plt, list(SETTINGS)),
                        "recovery_settings_maps": figure_settings_maps(settings, plt, list(SETTINGS))})
    if os.path.exists(ENHANCEMENTS_RESULTS):
        with open(ENHANCEMENTS_RESULTS) as handle:
            enhancements = json.load(handle)
        names = list(ENHANCEMENTS)
        print_settings_summary(enhancements, names, "enhancements experiment")
        figures.update({
            "recovery_enhancements_scores": figure_scores(enhancements, plt, names),
            "recovery_enhancements_thresholds": figure_thresholds_by_setting(enhancements, plt, names),
            "recovery_enhancements_maps": figure_settings_maps(enhancements, plt, names,
                                                              title="The enhancements experiment")})
    if os.path.exists(PRUNING_RESULTS):
        with open(PRUNING_RESULTS) as handle:
            pruning = json.load(handle)
        names = list(PRUNING)
        print_settings_summary(pruning, names, "pruning experiment")
        figures.update({
            "recovery_pruning_splits": figure_split_shares(pruning, plt, names),
            "recovery_pruning_thresholds": figure_thresholds_by_setting(pruning, plt, names),
            "recovery_pruning_maps": figure_settings_maps(pruning, plt, names,
                                                         title="The pruning experiment")})
    for name, fig in figures.items():
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=130)
        print("wrote", name)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "time":
        tasks = all_tasks()
        total = 0.0
        for step in STEPS:
            task = next(t for t in tasks if t[1].step == step)
            started = time.perf_counter()
            job(task)
            one = time.perf_counter() - started
            count = sum(t[1].step == step for t in tasks)
            total += one * count
            print(f"one fit every {step} days: {one:.1f}s, {count} fits", flush=True)
        print(f"{len(tasks)} fits, about {total / 3600:.1f} core-hours; on about 6 effective "
              f"cores: about {total / 6 / 60:.0f} minutes")
    elif command == "run":
        run(all_tasks(), RESULTS)
    elif command == "run-settings":
        run(settings_tasks(), SETTINGS_RESULTS)
    elif command == "run-enhancements":
        run(enhancement_tasks(), ENHANCEMENTS_RESULTS)
    elif command == "run-pruning":
        run(setting_tasks("pruning", PRUNING), PRUNING_RESULTS)
    elif command == "plot":
        plot()
    else:
        print(__doc__)
