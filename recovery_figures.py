#!/usr/bin/env python3
"""Experiments and figures for the planted-threshold test on daily data (see recovery_test.py).

  python recovery_figures.py time    one job per trading rate, to estimate the run time
  python recovery_figures.py run     every experiment, in parallel; saves report/figures/recovery_results.json
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


def job(task):
    tag, case, seed = task
    grid, expected = curve(case)
    result = run_case(case, seed, curve=(grid, expected))
    paths = result["paths"]
    return {
        "tag": tag, "step": case.step, "edge": case.edge, "k": case.k,
        "volatility": case.volatility, "seed": seed, "sign_change": sign_change(grid, expected),
        "grown": result["grown"], "kept": result["kept"],
        "test_mean": {name: float(path.net_returns.mean()) for name, path in paths.items()},
        "test_sharpe": {name: float(path.sharpe) for name, path in paths.items()},
        "test_turnover": {name: float(path.turnover.mean()) for name, path in paths.items()},
    }


def all_tasks():
    tasks = []
    for step in STEPS:                          # the daily jobs are the slowest: submit them first
        for edge in EDGES:
            n = DATASETS_PER_HISTOGRAM if edge in HISTOGRAM_EDGES else DATASETS_PER_CELL
            tasks += [("edge", replace(BASE, step=step, edge=edge), seed) for seed in range(n)]
        for k in K_VALUES:
            if k != BASE.k:
                tasks += [("speed", replace(BASE, step=step, k=k), seed)
                          for seed in range(DATASETS_PER_CELL)]
    return tasks


def run():
    tasks, results, start = all_tasks(), [], time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 50 == 0 or done == len(tasks):
                print(f"progress: {done}/{len(tasks)} fits after {time.perf_counter() - start:.0f}s",
                      flush=True)
    with open(RESULTS, "w") as handle:
        json.dump(results, handle)
    print(f"saved {RESULTS}")


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


def heatmap(axis, values, title, text, cmap, vmin, vmax, xlabels, xlabel):
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axis.set_xticks(range(len(xlabels)), xlabels, fontsize=9)
    axis.set_yticks(range(len(STEPS)), [f"{RATE_NAMES[s]}\n(every {s} d)" for s in STEPS], fontsize=9)
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
        run()
    elif command == "plot":
        plot()
    else:
        print(__doc__)
